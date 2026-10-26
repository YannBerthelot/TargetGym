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

A PID remembers its integral and its last error, so something has to hold that
memory from one step to the next. Pick the form by where your loop runs:

| Your loop | Use | Each step | At each episode start |
|---|---|---|---|
| Python, NumPy, outside `jax.jit` | `pid = env.make_pid()`, which keeps its own memory | `action = pid(obs)` | `pid.reset()` |
| `jit`, `vmap`, `lax.scan`, GPU | `ctrl = spec.make_pid_controller()`; you carry its memory | `action, pid_state = ctrl.act(pid_state, obs)` | `pid_state = ctrl.init_state` |

For many environments at once, `env.expert_policy` is the second form already
batched: start from `pid_state = env.expert_policy.init_state(num_envs)`, then
`actions, pid_state = env.expert_policy(pid_state, obs_batch)`.

!!! warning "Always pass the state to `env.expert_policy`"
    `env.expert_policy(obs)`, with one argument, starts from a fresh state on
    every call. It never builds an integral, so it is not the tuned PID and
    does not reproduce its scores.

Both forms act alike and return the action in the environment's own units,
already clipped to its bounds. They read the observation only, never the hidden
state, which is what makes the PID a fair peer for a learned policy. (The MPC
reads the true state, so it is an upper bound, not a peer.)

Two rules keep the memory right:

- **Reset it at the start of every episode**, or the previous episode's
  integral leaks in. In a jitted loop nothing does this for you; see
  [Episode boundaries](#episode-boundaries).
- **Call `pid(obs)` exactly once per environment step.** Each call advances its
  memory, so an extra call, for logging or on replayed observations, counts as
  a step that never happened. `ctrl.act` advances only the state you keep.

## Running it in a loop

The host object, as the baselines run it:

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

Use `spec.make_test_params()` rather than the default params. On every
aircraft task and the reactor the defaults run a longer episode, and some
aircraft tasks also set a different reference, so only the test params
reproduce the scored numbers.

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

Because the PID has memory, three things matter for a residual agent:

- **Store the PID's action with each transition.** It depends on the PID's
  memory, not only on `obs`, so recomputing it from a replayed observation
  gives a different action and disturbs the live controller.
- **Feed the PID's action to the network.** The integral is not in `obs`, so
  without it the agent cannot see what the base is about to do.
- **Watch for windup at large scales.** The PID cannot see the correction. When
  the correction pushes the sum past a bound, the PID's integral keeps growing.
  At zero correction it is exactly the PID.

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

### Inside a jitted loop, on GPU

A fully jitted learner (a `jax.lax.scan` over environment steps, as in
PureJaxRL-style agents) uses the pure form from [The interface](#the-interface)
and keeps `pid_state` in the scan's carry, which is where the integral lives.
It matches the host object up to float32 rounding, close enough to reproduce
the recorded returns, as the example checks.

```python
import jax
import numpy as np
from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
ctrl = spec.make_pid_controller()


def episode_return(key):
    obs, env_state = env.reset_env(key, params)

    def step(carry, _):
        obs, env_state, pid_state = carry
        action, pid_state = ctrl.act(pid_state, obs)
        obs, env_state, reward, _, _ = env.step_env(key, env_state, action, params)
        return (obs, env_state, pid_state), reward

    carry = (obs, env_state, ctrl.init_state)
    _, rewards = jax.lax.scan(step, carry, None, length=params.max_steps_in_episode)
    return rewards.sum()


keys = jax.numpy.stack([jax.random.PRNGKey(s) for s in range(10)])
returns = jax.jit(jax.vmap(episode_return))(keys)   # ten episodes at once

recorded = load_recorded_baselines()["cstr"]["pid_returns"]
assert np.allclose(returns, recorded, rtol=1e-3)     # the host-scored numbers
```

For a batch of environments, `env.expert_policy` does the same with one state
per environment, as in the next example.

#### Episode boundaries

Nothing calls `pid.reset()` inside a scan. `env.step` restarts an environment
when its episode ends, but the `pid_state` you carry keeps the old integral
unless you reset it too. `reset_where_done` does that, for one environment or a
batch:

```python
import jax
from target_gym.experts import reset_where_done
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
policy, num_envs = env.expert_policy, 4
keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys, params)
step_all = jax.vmap(env.step, in_axes=(0, 0, 0, None))


def step(carry, _):
    obs, env_state, pid_state = carry
    actions, pid_state = policy(pid_state, obs)
    obs, env_state, reward, terminated, truncated, _ = step_all(keys, env_state, actions, params)
    # env.step has already restarted the environments that finished; restart their PIDs too.
    pid_state = reset_where_done(terminated | truncated, pid_state, policy.init_state(num_envs))
    return (obs, env_state, pid_state), reward


carry = (obs, env_state, policy.init_state(num_envs))
run = jax.jit(lambda c: jax.lax.scan(step, c, None, length=2 * params.max_steps_in_episode))
_, rewards = run(carry)   # two episodes per environment, back to back
```

For a single environment, pass a scalar `done` and `ctrl.init_state`. Reset on
`terminated | truncated` only: in a continuing environment a trip restarts the
plant without raising either flag, and the PID keeps its memory through it, as
the scored baselines do.
