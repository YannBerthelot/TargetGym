# Building HVAC

<p align="center"><img src="../../videos/hvac/pid_output.gif" width="480px"/></p>

Building HVAC — single thermal zone, ISO 13790 5R1C reduced-order model.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | zone air temperature (deg C) |
| Episode length | 720 steps (648000 s at 900 s per step) |
| Import | `from target_gym import BuildingHVAC, HVACParams` |
| Cite as | `hvac-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | commanded heating power | -1 | 1 |

## Observation space

7 values. Indices (0,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 10 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The plant keeps its state through a trip instead of restarting, and every step spent outside the envelope is charged the trip cost and sets `info["tripped"]` until the controller brings it back. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 720 steps.

## Baselines

Recorded over 10 seeds of the 720-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -14.15 | 0.01966 |
| MPC | -7.659 | 0.01064 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 900 |
| `max_steps_in_episode` | 720 |
| `A_floor` | 150 |
| `lambda_at` | 4.5 |
| `f_class` | 2.5 |
| `cm_per_area` | 165000 |
| `h_is` | 3.45 |
| `h_ms` | 9.1 |
| `A_wall` | 120 |
| `U_wall` | 0.28 |
| `A_roof` | 150 |
| `U_roof` | 0.18 |
| `A_window` | 25 |
| `U_window` | 1.3 |
| … | 29 more, see the params dataclass |

