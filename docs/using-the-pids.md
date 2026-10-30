# Using the tuned PIDs

Every environment ships a tuned PID controller. It is the baseline every
published number is measured against, so it is useful in three ways: as a
working controller to run, as the reference your agent's score is paired
against, and as a prior to build learning on. This page shows where to get it,
how to run it, how to evaluate against it, and how to plug it into training.

## Where to get one

`target_gym.make_expert` builds it by registry name:

```python
import target_gym as tg
from target_gym.registry import REGISTRY

ctrl = tg.make_expert("cstr")   # the pure controller
pid = ctrl.host()               # the form the baselines are scored with

spec = REGISTRY["cstr"]
print(spec.has_pid, spec.baselines_note or "no caveat")
```

All 24 environments ship one (`spec.has_pid` is true everywhere). Where a
baseline comes with a caveat, `spec.baselines_note` states it. The gains were
tuned by `scripts/tune_pid.py` and live in
`src/target_gym/data/pid_gains.json`; you do not need to read that file to use
them. [Baselines](baselines.md) describes how each one is structured and tuned.

## The interface

A PID remembers its integral and its last error, so something has to hold that
memory from one step to the next. Pick the form by where your loop runs:

| Your loop | Use | Each step | At each episode start |
|---|---|---|---|
| Python, NumPy, outside `jax.jit` | `pid = tg.make_expert(name).host()`, which keeps its own memory | `action = pid(obs)` | `pid.reset()` |
| `jit`, `vmap`, `lax.scan`, GPU | `ctrl = tg.make_expert(name)`; you carry its memory | `action, pid_state = ctrl.act(pid_state, obs)` | `pid_state = ctrl.init_state` |

For many environments at once, `policy = tg.make_expert(name).batched()` is the
second form already batched: start from `pid_state = policy.init_state(num_envs)`,
then `actions, pid_state = policy(pid_state, obs_batch)`.

!!! warning "Always pass the state to the batched expert"
    `policy(obs)`, with one argument, starts from a fresh state on
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
import target_gym as tg
from target_gym.benchmark import episode_keys
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env = spec.make_env()
params = spec.make_test_params()   # the configuration the baselines are scored on
pid = tg.make_expert("cstr").host()

# Seed 0's keys as the scored episodes take them: one to reset, a fresh one
# every step.
T = int(params.max_steps_in_episode)
reset_keys, step_keys = episode_keys(jax.random.PRNGKey(0), 1, T)
obs, state = env.reset_env(reset_keys[0], params)
pid.reset()

total = 0.0
for t in range(T):
    action = pid(np.asarray(obs))
    obs, state, reward, terminated, _ = env.step_env(step_keys[t, 0], state, action, params)
    total += float(reward)
    if terminated:
        break

print("PID return:", total)
```

Use `spec.make_test_params()` rather than the default params. On every
aircraft task and the reactor the defaults run a longer episode, and some
aircraft tasks also set a different reference, so only the test params
reproduce the scored numbers.

## Evaluating against it

### Episode returns, paired on seeds

The recorded PID returns are in `src/target_gym/data/baseline_returns/`,
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
proportion set by the episode length. The [RL protocol](rl-protocol.md#4-metrics)
scores a policy against the expert on three numbers: how many times cheaper it
is over the episode (`ratio`, with the share of seeds it wins), where it sits
between the expert (0) and the ceiling (1) while holding (`score`), and how much
room the task leaves (`headroom`, in decades). One call runs your policy and the
expert on the same seeds:

```python
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy
from target_gym.scoring import score_policy

name = "cstr"
spec = REGISTRY[name]
params = spec.make_test_params()

make_policy = lambda: baseline_policy(spec, "pid", params)   # swap in your own
m = score_policy(name, make_policy, seeds=3)                 # 10 for a result
print(f"{m['ratio']:.3g}x cheaper than the expert on {m['win_rate']:.0%} of seeds")
print(f"score {m['score']:.2f}, room {m['headroom']:.2f} decades")
```

With the expert itself in the slot, the ratio comes out at 1 and the score at
0. A score of `nan` means the expert already tracks within the instrument's
resolution, so the task has no room to score.

## Using it in learning

### As a residual base

The structure the [RL protocol](rl-protocol.md#10-the-expert-based-arm) treats
as primary: the agent learns a bounded correction on top of the PID, so a zero
correction recovers the PID exactly.

```python
import numpy as np
import target_gym as tg
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


pid = tg.make_expert("cstr").host()
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
import target_gym as tg
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
pid = tg.make_expert("cstr").host()

transitions = []
for seed in range(3):
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = env.reset(reset_key, params)
    pid.reset()
    for _ in range(params.max_steps_in_episode):
        key, step_key = jax.random.split(key)   # a fresh key every step
        action = np.atleast_1d(pid(np.asarray(obs)))
        next_obs, state, reward, terminated, truncated, _ = env.step(step_key, state, action, params)
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
import target_gym as tg
from target_gym.benchmark import episode_keys
from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
ctrl = tg.make_expert("cstr")


def episode_return(key):
    # The scored episodes' keys: one to reset, a fresh one every step.
    reset_keys, step_keys = episode_keys(key, 1, int(params.max_steps_in_episode))
    obs, env_state = env.reset_env(reset_keys[0], params)

    def step(carry, step_key):
        obs, env_state, pid_state = carry
        action, pid_state = ctrl.act(pid_state, obs)
        obs, env_state, reward, _, _ = env.step_env(step_key, env_state, action, params)
        return (obs, env_state, pid_state), reward

    carry = (obs, env_state, ctrl.init_state)
    _, rewards = jax.lax.scan(step, carry, step_keys[:, 0])
    return rewards.sum()


keys = jax.numpy.stack([jax.random.PRNGKey(s) for s in range(10)])
returns = jax.jit(jax.vmap(episode_return))(keys)   # ten episodes at once

recorded = load_recorded_baselines()["cstr"]["pid_returns"]
assert np.allclose(returns, recorded, rtol=1e-3)     # the host-scored numbers
```

For a batch of environments, `tg.make_expert(name).batched()` does the same
with one state per environment, as in the next example.

#### Episode boundaries

Nothing calls `pid.reset()` inside a scan. `env.step` restarts an environment
when its episode ends, but the `pid_state` you carry keeps the old integral
unless you reset it too. `reset_where_done` does that, for one environment or a
batch:

```python
import jax
import target_gym as tg
from target_gym.experts import reset_where_done
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
policy, num_envs = tg.make_expert("cstr").batched(), 4
key, reset_key = jax.random.split(jax.random.PRNGKey(0))
obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
    jax.random.split(reset_key, num_envs), params
)
step_all = jax.vmap(env.step, in_axes=(0, 0, 0, None))


def step(carry, step_key):
    obs, env_state, pid_state = carry
    actions, pid_state = policy(pid_state, obs)
    keys = jax.random.split(step_key, num_envs)   # a fresh key every step
    obs, env_state, reward, terminated, truncated, _ = step_all(keys, env_state, actions, params)
    # env.step has already restarted the environments that finished; restart their PIDs too.
    pid_state = reset_where_done(terminated | truncated, pid_state, policy.init_state(num_envs))
    return (obs, env_state, pid_state), reward


carry = (obs, env_state, policy.init_state(num_envs))
n_steps = 2 * int(params.max_steps_in_episode)   # two episodes per environment
run = jax.jit(lambda c, k: jax.lax.scan(step, c, jax.random.split(k, n_steps)))
_, rewards = run(carry, key)
```

For a single environment, pass a scalar `done` and `ctrl.init_state`. Reset on
`terminated | truncated` only: in a continuing environment a trip restarts the
plant without raising either flag, and the PID keeps its memory through it, as
the scored baselines do.
