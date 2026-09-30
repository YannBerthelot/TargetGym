# First order

<p align="center"><img src="../../videos/first_order/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((2,))` |
| Tracked variable(s) | x |
| Episode length | 100 steps (5 s at 0.05 s per step) |
| Import | `from target_gym import FirstOrderSystem, FirstOrderParams` |
| Cite as | `first_order-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | input | -1 | 1 |

## Observation space

2 values. Indices (0,) carry the tracked variable(s) that the
reward scores.

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

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 100 steps.

## Baselines

Recorded over 10 seeds of the 100-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.018e+05 | 1018 |
| MPC | -1.002e+05 | 1002 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.05 |
| `max_steps_in_episode` | 100 |
| `K` | 1 |
| `tau` | 0.5 |
| `u_min` | -2 |
| `u_max` | 2 |
| `x_min` | -3 |
| `x_max` | 3 |
| `reward_version` | 2 |
| `e_floor` | 0.006 |
| `e_tol` | 0 |
| `tracking_exponent` | 2 |
| `failure_cost` | 2e+06 |
| `restart_steps` | 100 |
| … | 3 more, see the params dataclass |

