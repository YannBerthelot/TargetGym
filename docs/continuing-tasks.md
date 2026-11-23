# Continuing tasks

Every task is a plant that keeps running. `tg.make` gives you one window of it,
which ends at a time limit. `tg.make_stream` gives you the plant itself, with no
end, for algorithms that optimise the average cost per step.

```python
import jax
import jax.numpy as jnp
import target_gym as tg

stream = tg.make_stream("first_order")
obs, state = stream.reset(jax.random.PRNGKey(0))

action = jnp.zeros(stream.action_space().shape)
obs, state, reward, terminated, truncated, info = stream.step(
    jax.random.PRNGKey(1), state, action
)
```

## What a stream is

- **It never ends.** `terminated` and `truncated` are always false. A trip
  restarts the plant inside the task and is charged its cost, and
  `info["tripped"]` says when it happened.
- **Scenarios take turns.** A task's operating conditions (`stream.scenarios`)
  follow each other at random, and the share of time spent in each one is its
  weight. `info["scenario"]` is the index of the current one.
- **Events arrive at their own rate.** A setpoint move, a load swing or a new
  sortie starts when the plant's record says such things happen.
  `info["event"]` marks the step that starts one.
- **The observation is the task's own.** It holds no clock and no scenario
  label, so a policy cannot read where it is in a schedule.
- **The seed is the world.** Every draw comes from the key given to `reset`.
  Two policies on the same key meet the same scenarios, the same events and
  the same disturbances, whatever they do, and the key passed to `step` is not
  used. Compare policies on the same reset keys.

## Average cost per step

The reward is minus a cost, so the average cost per step over a run is the
quantity to minimise. A long run under `lax.scan`:

```python
import jax
import jax.numpy as jnp
import target_gym as tg

stream = tg.make_stream("first_order")
action = jnp.zeros(stream.action_space().shape)


@jax.jit
def average_cost(key, steps=2000, skip=500):
    _, state = stream.reset(key)

    def body(state, step_key):
        _, state, reward, _, _, _ = stream.step(step_key, state, action)
        return state, -reward

    _, cost = jax.lax.scan(body, state, jax.random.split(key, steps))
    return cost[skip:].mean()  # leave out the start-up


print(float(average_cost(jax.random.PRNGKey(0))))
```

Replace the constant action with your policy. `skip` leaves out the steps
where the plant still remembers its start. How long that takes differs by
task: seconds on `first_order`, days on the glass furnace.

## With Gymnasium

```python
import numpy as np
import target_gym as tg

env = tg.make_gymnasium("first_order", stream=True)
obs, info = env.reset(seed=0)
for _ in range(300):
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, np.float32)
    )
```

The seed of `reset` fixes the world, as the key does above. A Gymnasium stream
does not render yet.

## A finite stream

`tg.make_stream(name, length=n)` truncates after `n` steps, for a training
loop that needs an end.

## What is not here yet

The published score of a task is its average cost on pinned scoring runs: one
scenario at a time, after a hidden burn-in, with a fixed number of events. The
function that runs them, and each task's recorded burn-in and reference cost,
are not public yet. Until they are, numbers measured on a stream are yours to
compare between your own runs, on the same keys.
