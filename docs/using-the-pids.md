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

expert = tg.make_expert("cstr")

spec = REGISTRY["cstr"]
print(spec.has_pid, spec.baselines_note or "no caveat")
```

All 25 environments ship one (`spec.has_pid` is true everywhere). Where a
baseline comes with a caveat, `spec.baselines_note` states it. The gains were
tuned by `scripts/tune_pid.py` and live in
`src/target_gym/data/pid_gains.json`; you do not need to read that file to use
them. [Baselines](baselines.md) describes how each one is structured and tuned.

## The interface

A PID remembers its integral and its last error. You carry that memory from
one step to the next, as you carry the environment's `state`:

| When | Call |
|---|---|
| At each episode start | `memory = expert.reset()` |
| Each step | `action, memory = expert.step(memory, obs)` |
| For `n` environments at once | `memory = expert.reset(n)`, then `actions, memory = jax.vmap(expert.step)(memory, obs)` |

The same `step` runs in a Python loop, under `jit` and inside `lax.scan`, on
CPU or GPU. It returns the action in the environment's own units, already
clipped to its bounds. It reads the observation only, never the hidden state,
which is what makes the PID a fair peer for a learned policy. (The MPC reads
the true state, so it is an upper bound, not a peer.)

Two rules keep the memory right:

- **Reset it at the start of every episode**, or the previous episode's
  integral leaks in. Under an auto-resetting `env.step` inside a scan, see
  [Episode boundaries](#episode-boundaries).
- **Keep the memory `step` returns, once per environment step.** Stepping on
  a replayed or logged observation is harmless as long as you discard the
  memory it returns.

`score_policy`, `run_suite` and `env.save_video` run a policy in their own
loop and take the expert as it is.

## Running it in a loop

With the keys the scored episodes use:

```python
import jax
import numpy as np
import target_gym as tg
from target_gym.benchmark import episode_keys
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env = spec.make_env()
params = spec.make_test_params()   # the configuration the baselines are scored on
expert = tg.make_expert("cstr")

# Seed 0's keys as the scored episodes take them: one to reset, a fresh one
# every step.
T = int(params.max_steps_in_episode)
reset_keys, step_keys = episode_keys(jax.random.PRNGKey(0), 1, T)
obs, state = env.reset_env(reset_keys[0], params)
memory = expert.reset()

total = 0.0
for t in range(T):
    action, memory = expert.step(memory, obs)
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
import target_gym as tg

policy = tg.make_expert("cstr")      # swap in your own: policy(obs) -> action
m = tg.score_policy("cstr", policy, seeds=3)   # 10 for a result
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


def residual_step(expert, correction):
    """``correction(obs)`` is your network's output, in [-1, 1]."""

    def step(memory, obs):
        action, memory = expert.step(memory, obs)
        return np.clip(action + alpha * correction(obs), low, high), memory

    return step


expert = tg.make_expert("cstr")
step = residual_step(expert, lambda obs: np.zeros_like(low))   # untrained: the PID
memory = expert.reset()
```

The protocol sweeps the scale over 0.1, 0.25 and 1.0 and reports all three, so
a reader can tell "learning improved the PID" from "learning ignored it".

Because the PID has memory, three things matter for a residual agent:

- **Store the PID's action with each transition.** It depends on the PID's
  memory, not only on `obs`, so recomputing it from a replayed observation
  gives a different action.
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
expert = tg.make_expert("cstr")

transitions = []
for seed in range(3):
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = env.reset(reset_key, params)
    memory = expert.reset()
    for _ in range(params.max_steps_in_episode):
        key, step_key = jax.random.split(key)   # a fresh key every step
        action, memory = expert.step(memory, obs)
        next_obs, state, reward, terminated, truncated, _ = env.step(step_key, state, action, params)
        transitions.append((np.asarray(obs), np.asarray(action), float(reward), np.asarray(next_obs)))
        obs = next_obs
        if terminated or truncated:
            break

print(len(transitions), "transitions")
```

### Inside a jitted loop, on GPU

A fully jitted learner (a `jax.lax.scan` over environment steps, as in
PureJaxRL-style agents) keeps the expert's memory in the scan's carry, which
is where the integral lives. It matches the scored host PID up to float32
rounding, close enough to reproduce the recorded returns, as the example
checks.

```python
import jax
import numpy as np
import target_gym as tg
from target_gym.benchmark import episode_keys
from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
expert = tg.make_expert("cstr")


def episode_return(key):
    # The scored episodes' keys: one to reset, a fresh one every step.
    reset_keys, step_keys = episode_keys(key, 1, int(params.max_steps_in_episode))
    obs, env_state = env.reset_env(reset_keys[0], params)

    def step(carry, step_key):
        obs, env_state, memory = carry
        action, memory = expert.step(memory, obs)
        obs, env_state, reward, _, _ = env.step_env(step_key, env_state, action, params)
        return (obs, env_state, memory), reward

    carry = (obs, env_state, expert.reset())
    _, rewards = jax.lax.scan(step, carry, step_keys[:, 0])
    return rewards.sum()


keys = jax.numpy.stack([jax.random.PRNGKey(s) for s in range(10)])
returns = jax.jit(jax.vmap(episode_return))(keys)   # ten episodes at once

recorded = load_recorded_baselines()["cstr"]["pid_returns"]
assert np.allclose(returns, recorded, rtol=1e-3)     # the host-scored numbers
```

For a batch of environments, `expert.reset(n)` and `jax.vmap(expert.step)` do
the same with one memory per environment, as in the next example.

#### Episode boundaries

Nothing resets the memory inside a scan for you. `env.step` restarts an environment
when its episode ends, but the memory you carry keeps the old integral
unless you reset it too. `reset_where_done` does that, for one environment or a
batch:

```python
import jax
import target_gym as tg
from target_gym.experts import reset_where_done
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
expert, num_envs = tg.make_expert("cstr"), 4
key, reset_key = jax.random.split(jax.random.PRNGKey(0))
obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
    jax.random.split(reset_key, num_envs), params
)
step_all = jax.vmap(env.step, in_axes=(0, 0, 0, None))
expert_all = jax.vmap(expert.step)


def step(carry, step_key):
    obs, env_state, memory = carry
    actions, memory = expert_all(memory, obs)
    keys = jax.random.split(step_key, num_envs)   # a fresh key every step
    obs, env_state, reward, terminated, truncated, _ = step_all(keys, env_state, actions, params)
    # env.step has already restarted the environments that finished; restart their PIDs too.
    memory = reset_where_done(terminated | truncated, memory, expert.reset())
    return (obs, env_state, memory), reward


carry = (obs, env_state, expert.reset(num_envs))
n_steps = 2 * int(params.max_steps_in_episode)   # two episodes per environment
run = jax.jit(lambda c, k: jax.lax.scan(step, c, jax.random.split(k, n_steps)))
_, rewards = run(carry, key)
```

For a single environment, pass a scalar `done`. Reset on
`terminated | truncated` only: in a continuing environment a trip restarts the
plant without raising either flag, and the PID keeps its memory through it, as
the scored baselines do.
