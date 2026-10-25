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

**Termination.** None. `terminated` is always false: the plant keeps its state through a trip instead of restarting, and every step spent outside the envelope is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]` until the controller brings it back. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 720 steps.

## Baselines

Recorded over 10 seeds of the 720-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -14.15 | 0.01966 |
| MPC | -7.659 | 0.01064 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 900 | s |
| `max_steps_in_episode` | 720 | steps |
| `A_floor` | 150 | m^2 |
| `lambda_at` | 4.5 | - |
| `f_class` | 2.5 | - |
| `cm_per_area` | 165000 | J/(K m^2) |
| `h_is` | 3.45 | W/(m^2 K) |
| `h_ms` | 9.1 | W/(m^2 K) |
| `A_wall` | 120 | m^2 |
| `U_wall` | 0.28 | W/(m^2 K) |
| `A_roof` | 150 | m^2 |
| `U_roof` | 0.18 | W/(m^2 K) |
| `A_window` | 25 | m^2 |
| `U_window` | 1.3 | W/(m^2 K) |
| `g_window` | 0.55 | - |
| `air_changes_per_hour` | 0.5 | 1/h |
| `room_height` | 2.6 | m |
| `Q_heat_max` | 7200 | W |
| `emitter_tau` | 900 | s |
| `T_out_mean` | 5 | degC |
| `T_out_amplitude` | 5 | K |
| `T_out_noise_std` | 3 | K |
| `solar_peak` | 350 | W/m^2 |
| `gain_occupied` | 8 | W/m^2 |
| `gain_unoccupied` | 2 | W/m^2 |
| `occupied_start_h` | 7 | h |
| `occupied_end_h` | 22 | h |
| `setpoint_occupied` | 21 | degC |
| `setpoint_setback` | 17 | degC |
| `T_air_min` | 5 | degC |
| `precision_floor` | 0.1 | K |
| `T_air_max` | 35 | degC |
| `comfort_band` | 1 | K |
| `energy_weight` | 0 | - |
| `reward_version` | 2 | - |
| `comfort_tolerance` | 0.5 | K |
| `comfort_price` | 0.03 | EUR/(K^2 h) |
| `gas_price` | 0.1 | EUR/kWh |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 4.335 | reward |
| `restart_steps` | 4 | steps |
| `rho_floor_tracking` | 0.001 | reward |
| `rho_floor` | 0.001 | reward |

