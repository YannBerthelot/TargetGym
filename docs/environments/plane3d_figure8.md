# 3D figure-8

<p align="center"><img src="../../videos/plane3d_figure8/pid_output.gif" width="480px"/></p>

3D airplane environment state, parameters, and transition logic.

| | |
|---|---|
| Action space | `Box((3,))`, all actions in [-1, 1] |
| Observation space | `Box((19,))` |
| Tracked variable(s) | altitude (m) |
| Episode length | 400 steps (400 s at 1 s per step) |
| Import | `from target_gym import Plane3DFigureEight, PlaneParams3D` |
| Cite as | `plane3d_figure8-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |
| 2 | aileron | -1 | 1 |

## Observation space

19 values. Indices (2,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 27 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 400 steps.

## Baselines

Recorded over 10 seeds of the 400-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.878e+07 | 4.695e+04 |
| MPC | -2709 | 6.772 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 400 |
| `gravity` | 9.81 |
| `initial_mass` | 73500 |
| `thrust_output_at_sea_level` | 240000 |
| `air_density_at_sea_level` | 1.225 |
| `frontal_surface` | 12.6 |
| `wings_surface` | 122.6 |
| `C_x0` | 0.095 |
| `C_z0` | 0.9 |
| `initial_fuel_quantity` | 19088 |
| `specific_fuel_consumption` | 0.0175 |
| `power_response_rate` | 0.05 |
| `stick_response_rate` | 0.9 |
| … | 60 more, see the params dataclass |

