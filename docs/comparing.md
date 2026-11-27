# Comparing an agent with the reference

Four tasks are scored, `tg.FLAGSHIP`: `reactor`, `hvac`, `compressor_surge` and
`ph_neutralization`. On each, a result is a saving over the task's **reference**,
`1 - J_agent / J_reference`, where `J` is the cost per step over every scenario of
the task, each weighted by its share of operating time. `tg.score` computes it the
way the charter does, so a number from it means the same thing in every paper.

Each scored task's page gives its lines and its room:
[reactor](environments/reactor.md#reference-and-ceiling),
[hvac](environments/hvac.md#reference-and-ceiling),
[compressor_surge](environments/compressor_surge.md#reference-and-ceiling) and
[ph_neutralization](environments/ph_neutralization.md#reference-and-ceiling).

## The reference

The reference is a controller built by one fixed recipe on every task: PI loops
tuned by SIMC from bump tests on the plant, then feedforward from the plant's
balances, a cascade or an override where they pay, keeping the cheapest rung
that trips nothing. On the reactor it is the published rod procedure, the
shipped expert. Its numbers are frozen in the package, so it is the same object
on every machine:

```python
# doc: records -- builds a frozen reference
import target_gym as tg

ref = tg.make_reference("hvac")
print(ref.rung)                     # the rung the charter kept
memory = ref.reset()                # stepped as tg.make_expert's expert is
```

Beating `tg.make_expert(name)` is a weaker claim: on the ladder's tasks the
shipped expert is not the reference, and it costs more than the reference does,
on some of them by a large margin.

## Train on the scenarios

`tg.make(name)` gives the task's scored configuration. The score reads all of
the task's scenarios, weighted by their share of operating time, which
`tg.list_scenarios(name)` gives:

```python
import target_gym as tg

print(tg.list_scenarios("ph_neutralization"))   # {'normal': 0.8, 'bad_day': 0.15, ...}
env, params = tg.make("ph_neutralization", scenario="bad_day")
```

To train on the weighted mix, step `tg.make_stream(name)`, which rotates the
scenarios by weight and never ends, or draw a scenario for each episode by
those shares. [Continuing tasks](continuing-tasks.md) explains the stream.

## Choose on the validation seeds, score once

Choose checkpoints and settings on the validation seeds, 300 to 349, and read the
scoring seeds, 200 to 209, once, for the result you report:

```python
# doc: records -- builds a frozen reference
import jax.numpy as jnp
import target_gym as tg
from target_gym import charter


def policy(obs):  # any pure policy(obs) -> action; both valves at mid-stroke here
    return jnp.zeros(2)


check = tg.score("ph_neutralization", policy, seeds=charter.VALIDATION_SEEDS[:2])
m = tg.score("ph_neutralization", policy, seeds=charter.ANCHOR_SEEDS[:2])  # all 10
print(f"saving {m['saving']:+.1%}, 95 % interval {m['saving_ci']}")
for row in m["scenarios"]:
    print(f"{row['scenario']:12s} weight {row['weight']:.2f} saving {row['saving']:+.1%}")
```

`tg.score` runs a pure `policy(obs) -> action`, or one in the experts' form
(`reset`, `step(memory, obs)`), compiled once for every seed. A policy that
keeps its own Python memory, or any with `jit=False`, runs on the host, one
episode at a time. A list of policies is one per training run, and the interval
then covers the variation between runs as well as between seeds.

## Read the result

- `saving` and `saving_ci`, its 95 % interval. `beats_reference` holds when the
  interval is above 0 and the agent trips no more often than the reference in
  any scenario.
- `probability_of_improvement`, the weighted share of seeds the agent wins, and
  `ties`, the share within the task's noise floor: two code paths of one
  controller differ by that much on a seed, so it is no win.
- `scenarios`, one row per scenario: its weight, both costs, the saving, the
  trips and the room. On the reactor, `days` gives the saving on baseload days
  and on load-following days apart.
- `travel_ratio` and `reversal_ratio`: how much more the agent moves its
  actuators than the reference. They are not part of the verdict, but a saving
  that wears a valve out ten times faster is one a plant would not take.
- `lines`: the charter's other lines from the task's record, for context: the
  best null policy, the MPC, which plans with the true state and can lose to the
  reference, and the ceiling, the best known plan with foresight. `room` is the
  reference's cost over the ceiling's. `current` says whether the record was
  measured on this plant.
- `reference`: its rung, and how often it leaves its zones on seeds it was
  never placed on.
- `read_state`: true for a `tg.FullState` policy, which is given the
  simulator's true state and so is not comparable with the reference.

The room is an upper bound on what a policy can save, since the ceiling knows
what no policy observes: each task's page names what its ceiling reads.

## Outside the scored suite

`tg.score` refuses a task that is not scored: its records are not held current.
`tg.score_policy` still runs any registered task against its shipped expert,
on the protocol [Using the PIDs](using-the-pids.md#the-protocol-score) describes.
