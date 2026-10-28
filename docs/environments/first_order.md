# First order

<p align="center"><img src="../../videos/first_order/pid_output.gif" width="480px"/></p>

Drive a single first-order lag to a target drawn at reset. It is the suite's sanity check, not a physical plant: linear, fully observed, with no disturbance and no way to trip, so it is the place to check a new algorithm or wrapper against a known answer before moving on.

```python
import target_gym as tg

env, params = tg.make("first_order")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((2,))` |
| Tracked variable(s) | x |
| Episode length | 150 steps of 0.05 s (7.5 s) |
| Cite as | `first_order-v3` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/first_order/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | input | -1 | 1 |

## Observation space

2 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[1]`.

| # | meaning |
|---|---|
| 0 | `x`, the system's output |
| 1 | target `x` |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of three non-negative costs,
`-(tracking_cost + running_cost + failure_cost)`. The running cost
charges consumption and the failure cost charges trips out of the
operating envelope. The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 4 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 150 steps.

## Baselines

Recorded over 10 seeds of the 100-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -4.26e+04 | 426 |
| MPC | -4.111e+04 | 411.1 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("first_order", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 0.05 | s |
| `max_steps_in_episode` | 150 | steps |
| `K` | 1 | - |
| `tau` | 0.5 | s |
| `u_min` | -2 | - |
| `u_max` | 2 | - |
| `x_min` | -3 | - |
| `x_max` | 3 | - |
| `reward_version` | 2 | - |
| `e_floor` | 0.006 | - |
| `e_tol` | 0 | - |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 2e+06 | reward |
| `restart_steps` | 100 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |
| `precision_floor` | 0.006 | - |

