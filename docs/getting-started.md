# Getting started

## Install

```bash
pip install target-gym
# or
uv add target-gym
```

Python 3.11 through 3.13. The full-state MPC oracles on nine tasks plan with
CasADi, which is optional: `pip install 'target-gym[mpc]'`. Videos and the
aircraft dashboards need `pip install 'target-gym[render]'`, and
`target-gym[all]` brings both. For development, see [CONTRIBUTING](https://github.com/YannBerthelot/TargetGym/blob/main/CONTRIBUTING.md).

## An episode, directly

Environments follow the [gymnax](https://github.com/RobertTLange/gymnax)
interface: pure functions taking a PRNG key, a state and parameters, and
returning the next state. Nothing is hidden in instance attributes, so the
whole thing composes with `jit`, `vmap` and `scan`. `tg.make` returns the
environment and its scored configuration; keyword arguments change single
fields of it.

```python
import jax
import jax.numpy as jnp
import target_gym as tg

env, params = tg.make("cstr", max_steps_in_episode=100)

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key, params)

step = jax.jit(env.step_env)
total = 0.0
for _ in range(params.max_steps_in_episode):
    action = jnp.array([0.0])
    obs, state, reward, terminated, info = step(key, state, action, params)
    total += float(reward)
    if bool(terminated):
        break
print(total)
```

`step_env` reports **natural termination only** -- the plant reaching a state
it cannot come back from. The time limit is separate, and gymnax's six-value
`env.step` applies it, returning `terminated` and `truncated` separately. Use
`step_env` when you want to reason about the physics, and `step` when you want
the standard RL episode boundary.

## Many episodes at once

The reason for the functional interface: rollouts vectorise.

```python
import jax
import jax.numpy as jnp
import target_gym as tg

env, params = tg.make("cstr", max_steps_in_episode=100)

def episode_return(key):
    _, state = env.reset_env(key, params)

    def body(carry, _):
        state, total = carry
        _, state, reward, _, _ = env.step_env(key, state, jnp.zeros(1), params)
        return (state, total + reward), None

    (_, total), _ = jax.lax.scan(body, (state, 0.0), None, params.max_steps_in_episode)
    return total

keys = jax.random.split(jax.random.PRNGKey(0), 256)
returns = jax.jit(jax.vmap(episode_return))(keys)
print(returns.mean())
```

## The Gymnasium interface

For libraries that expect the classic API, any task comes wrapped, on its
scored configuration:

```python
import target_gym as tg

env = tg.make_gymnasium("plane")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step((0.8, 0.0))
```

The wrapper is **stateful** and returns NumPy, so unlike `step_env` it cannot
be wrapped in `jax.jit`.

## Experts and oracles

Every task ships an expert, a copy of the controller practitioners deploy,
which reads the observation only. Pick its form where you use it:

```python
import jax
import numpy as np
import target_gym as tg

env, params = tg.make("boiler_drum")
obs, state = env.reset(jax.random.PRNGKey(0), params)

pid = tg.make_expert("boiler_drum").host()      # keeps its own memory
action = pid(np.asarray(obs))                   # pid.reset() between episodes

policy = tg.make_expert("boiler_drum").batched()   # you carry its memory
memory = policy.init_state(1)
actions, memory = policy(memory, obs[None])
```

`tg.make_oracle(name)` returns the full-state MPC, which reads the true state
and so shows what a perfect model could do. `tg.list_envs()` gives every name;
the [environment reference](environments.md) describes each task, and
[Using the PIDs](using-the-pids.md) covers evaluating against and training on
the experts. For code that needs more than this, `target_gym.registry.REGISTRY`
holds each task's `EnvSpec`.

## Rendering

Each environment renders a control-room dashboard: a live plant schematic, an
instrument stack with limit and setpoint markers, and strip charts. Quantities
the controller cannot measure are marked, so a frame shows both what the agent
knows and what it is up against.

Full guide, including the two rendering toolkits, headless setup and how to
regenerate the shipped media: **[Rendering](rendering.md)**.

```python
import target_gym as tg

env, params = tg.make("plane", max_steps_in_episode=1_000)
env.save_video(
    lambda obs: (0.8, 0.0), seed=42,
    folder="videos", episode_index=0, params=params, format="gif",
)
```

## The multi-agent formation

In `formation` every aircraft of a three-ship formation is a learner, scored
by one team cost. `FormationMARL` gives each aircraft its own observation and
action, in the JaxMARL style; see [Multi-agent formation](multi-agent.md).

```python
import jax
import target_gym as tg

env = tg.FormationMARL()                     # a lead and two wingmen
_, params = tg.make("formation")
key = jax.random.PRNGKey(0)

obs, state = env.reset(key, params)          # obs = {"lead": ..., "wingman_0": ..., ...}
actions = {agent: env.action_space(agent).sample(key) for agent in env.agents}
obs, state, rewards, dones, info = env.step(key, state, actions, params)
# rewards/dones are dicts keyed by agent (+ dones["__all__"]); reward is shared.
```

## Wind and turbulence

Every plane-based env (`Plane`, `Plane3D`, `PlanePatrol`, `Formation`)
inherits a wind model applied to the air-relative aerodynamics. It is an
**unobservable** disturbance by default; pass `observe_wind=True` for a
fully-observable baseline. Formations feel a single shared gust field.

```python
from target_gym import Plane, PlaneParams

params = PlaneParams(
    wind_x=-15.0,          # steady mean wind (m/s), world frame
    wind_shear_x=0.02,     # + linear altitude shear: +0.02 m/s per metre above...
    shear_ref_alt=5000.0,  # ...this reference altitude
    turbulence_sigma=3.0,  # + Ornstein-Uhlenbeck gusts (0 = off); theta = turbulence_theta
)

hidden = Plane()                     # wind is a hidden disturbance (POMDP)
baseline = Plane(observe_wind=True)  # appends the realized wind to the observation
```
