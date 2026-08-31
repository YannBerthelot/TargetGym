# Getting started

## Install

```bash
pip install target-gym
# or
uv add target-gym
```

Python 3.11 through 3.14. For development, see [CONTRIBUTING](../CONTRIBUTING.md).

## An episode, directly

Environments follow the [gymnax](https://github.com/RobertTLange/gymnax)
interface: pure functions taking a PRNG key, a state and parameters, and
returning the next state. Nothing is hidden in instance attributes, so the
whole thing composes with `jit`, `vmap` and `scan`.

```python
import jax
import jax.numpy as jnp
from target_gym import CSTR, CSTRParams

env = CSTR()
params = CSTRParams(max_steps_in_episode=100)

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
from target_gym import CSTR, CSTRParams

env, params = CSTR(), CSTRParams(max_steps_in_episode=100)

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

For libraries that expect the classic API, every environment has a wrapper:

```python
from target_gym import GymnasiumPlane

env = GymnasiumPlane()
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step((0.8, 0.0))
```

`gym_wrapper_factory` builds the same wrapper for any of the JAX environments.
Note that the wrapper is **stateful** and returns NumPy -- unlike `step_env`,
it must not be wrapped in `jax.jit`.

## Working from the registry

The registry is how the library refers to environments generically -- it is
what the test suite, the baselines and the figure generation are all driven
by, and the most convenient way to write code that works across environments.

```python
from target_gym.registry import REGISTRY

spec = REGISTRY["boiler_drum"]
env, params = spec.make_env(), spec.params_cls()
pid = spec.make_pid()          # the shipped baseline, or None
pid.reset()
```

See the [environment reference](environments.md) for every registry name.

## Rendering

Each environment renders a control-room dashboard:

```python
from target_gym import Plane, PlaneParams

env = Plane()
params = PlaneParams(max_steps_in_episode=1_000)
env.save_video(
    lambda obs: (0.8, 0.0), seed=42,
    folder="videos", episode_index=0, params=params, format="gif",
)
```
