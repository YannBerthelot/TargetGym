# Using the tuned PIDs

Every environment ships a tuned PID controller. It is the baseline every
published number is measured against, so it is useful in three ways: as a
working controller to run, as the reference your agent's score is paired
against, and as a prior to build learning on. This page shows where to get it,
how to run it, how to evaluate against it, and how to plug it into training.

## Where to get one

Each environment builds its own PID, and the registry builds the same one by
name, which is handier for code that loops over environments:

```python
from target_gym import CSTR
from target_gym.registry import REGISTRY

pid = CSTR().make_pid()                 # from the environment
pid = REGISTRY["cstr"].make_pid()       # the same controller, by registry name

spec = REGISTRY["cstr"]
print(spec.has_pid, spec.baselines_note or "no caveat")
```

All 23 environments ship one (`spec.has_pid` is true everywhere). Where a
baseline comes with a caveat, `spec.baselines_note` states it. The gains were
tuned by `scripts/tune_pid.py` and live in
`src/target_gym/data/pid_gains.json`; you do not need to read that file to use
them. [Baselines](baselines.md) describes how each one is structured and tuned.

## The interface

A PID is a small stateful object:

- `pid(obs)` takes the observation as a NumPy array and returns the action, in
  the environment's own action units and already clipped to its bounds.
- `pid.reset()` clears the integrator and derivative memory. Call it at the
  start of every episode, or the previous episode's integral leaks in.
- It reads the observation only, as a plant controller does. It never sees the
  hidden state, which is what makes it a fair peer for a learned policy. (The
  MPC is the opposite: it reads the true state and is an upper bound, not a
  peer.)

It runs on the host, in plain Python, so call it outside `jax.jit`. See
[Inside a jitted loop](#inside-a-jitted-loop) for the JAX-traceable form.

## Running it in a loop

```python
import jax
import numpy as np
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env = spec.make_env()
params = spec.make_test_params()   # the configuration the baselines are scored on
pid = env.make_pid()

key = jax.random.PRNGKey(0)
obs, state = env.reset(key, params)
pid.reset()

total = 0.0
for _ in range(params.max_steps_in_episode):
    action = pid(np.asarray(obs))
    obs, state, reward, terminated, truncated, _ = env.step(key, state, action, params)
    total += float(reward)
    if terminated or truncated:
        break

print("PID return:", total)
```

Use `spec.make_test_params()` rather than the default params. On most
environments they are the same, but the plane variants share one parameter
class and are scored over different episode lengths, which only the test
params resolve.

## Evaluating against it

### Episode returns, paired on seeds

The recorded PID returns are in `src/target_gym/data/baseline_returns.json`,
one per seed. `runners.rollout` is the loop they were recorded with, so running
your policy through it on the same seeds gives a paired comparison without
re-running the PID:

```python
import numpy as np
from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy, rollout

spec = REGISTRY["cstr"]
params = spec.make_test_params()
recorded = load_recorded_baselines()["cstr"]
seeds = range(recorded["seeds"])


def episode_return(make_policy, seed):
    return float(np.sum(rollout(spec, params, make_policy(), seed=seed)[2]))


# A fresh, reset PID per episode reproduces the recorded returns exactly.
pid_returns = [episode_return(lambda: baseline_policy(spec, "pid", params), s) for s in seeds]
assert np.allclose(pid_returns, recorded["pid_returns"])

# Your policy goes in the same slot: any callable obs -> action
# (or (obs, state) -> action) built fresh for each episode.
mine = [episode_return(lambda: baseline_policy(spec, "pid", params), s) for s in seeds]
wins = sum(m > p for m, p in zip(mine, recorded["pid_returns"]))
print(f"beats the PID on {wins} of {len(mine)} seeds")
```

### The protocol score

Returns mix the cost of reaching a target with the cost of holding it, in a
proportion set by the episode length. The [RL protocol](rl-protocol.md) scores
the two apart, and its headline is the long-run cost while holding (the
*gain*) and the normalised expert advantage, NEA, which is 0 at PID parity and
1 at the achievable floor. The PID's protocol row is in
`src/target_gym/data/protocol_results.json`, so a policy of yours is scored
against it like this:

```python
import json
from importlib.resources import files

from target_gym.eval import evaluate, run_episode, scored_burn_in
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy

name = "cstr"
spec = REGISTRY[name]
params = spec.make_test_params()
row = json.loads((files("target_gym") / "data" / "protocol_results.json").read_text())[name]

make_policy = lambda: baseline_policy(spec, "pid", params)   # swap in your own
episodes = [run_episode(spec, params, make_policy(), seed=s) for s in range(row["seeds"])]
score = evaluate(
    episodes,
    burn_in=scored_burn_in(name, params),
    rho_floor=float(params.rho_floor),
    rho_ref=row["pid"]["gain"],
)
print(f"gain {score['gain']:.4g} (PID {row['pid']['gain']:.4g}), NEA {score['nea']:.3f}")
```

With the PID itself in the slot, NEA comes out at 0. For a shipped controller
there is a one-line shortcut, `target_gym.eval.evaluate_controller("cstr", "pid")`.

## Using it in learning

### As a residual base

The structure the [RL protocol](rl-protocol.md#10-the-expert-based-arm) treats
as primary: the agent learns a bounded correction on top of the PID, so a zero
correction recovers the PID exactly.

```python
import numpy as np
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
space = env.action_space(params)
low, high = np.asarray(space.low), np.asarray(space.high)
alpha = 0.25 * (high - low) / 2      # the correction spans a quarter of the range


def residual_policy(pid, correction):
    """``correction(obs)`` is your network's output, in [-1, 1]."""

    def act(obs):
        return np.clip(pid(obs) + alpha * correction(obs), low, high)

    return act


pid = env.make_pid()
pid.reset()
act = residual_policy(pid, lambda obs: np.zeros_like(low))   # untrained: the PID
```

The protocol sweeps the scale over 0.1, 0.25 and 1.0 and reports all three, so
a reader can tell "learning improved the PID" from "learning ignored it".

### As demonstrations

To pre-fill a replay buffer or pretrain by behavioural cloning, record the
PID's own transitions:

```python
import jax
import numpy as np
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
pid = env.make_pid()

transitions = []
for seed in range(3):
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key, params)
    pid.reset()
    for _ in range(params.max_steps_in_episode):
        action = np.atleast_1d(pid(np.asarray(obs)))
        next_obs, state, reward, terminated, truncated, _ = env.step(key, state, action, params)
        transitions.append((np.asarray(obs), action, float(reward), np.asarray(next_obs)))
        obs = next_obs
        if terminated or truncated:
            break

print(len(transitions), "transitions")
```

### Inside a jitted loop

A fully jitted learner (a `jax.lax.scan` over environment steps, as in
PureJaxRL-style agents) cannot call the host-side PID. Some environments also
expose `env.expert_policy`, a JAX-traceable form that is batched and threads
its own state:

```python
import jax
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
expert = env.expert_policy

expert_state = expert.init_state(num_envs=1)
obs, state = env.reset(jax.random.PRNGKey(0), params)
actions, expert_state = expert(expert_state, obs[None])   # (num_envs, action_dim)
```

It is the scored PID on six environments only: `cstr`, `first_order`,
`four_tank`, `glass_furnace`, `hvac` and `patrol`.
`tests/test_registry_rules.py` checks this list against `make_pid` every run.
Elsewhere `expert_policy` is either missing or a different controller from the scored one,
so on those environments compute the PID's actions on the host, or compare
against the recorded numbers rather than a jitted re-run.
