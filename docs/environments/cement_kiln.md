# Cement Kiln

<p align="center"><img src="../../videos/cement_kiln/pid_output.gif" width="480px"/></p>

Cement rotary kiln — 1-D axial model with counter-current gas.

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((8,))` |
| Tracked variable(s) | discharge free lime (%) |
| Episode length | 700 steps (21000 s at 30 s per step) |
| Import | `from target_gym import CementKiln, CementKilnParams` |
| Cite as | `cement_kiln-v3` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel | -1 | 1 |
| 1 | kiln_speed | -1 | 1 |

## Observation space

8 values. Indices (0,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 11 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. Both failure modes are irrecoverable in practice. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 700 steps.

## Baselines

Recorded over 10 seeds of the 700-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -4458 | 6.368 |
| MPC | -283.4 | 0.4048 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 30 | s |
| `max_steps_in_episode` | 700 | steps |
| `reward_version` | 2 | - |
| `e_floor` | 0.0005 | - |
| `e_tol` | 0 | - |
| `tracking_exponent` | 2 | - |
| `c_hold` | 1.824 | kg/s |
| `running_weight` | 1 | - |
| `failure_cost` | 14112 | reward |
| `restart_steps` | 2880 | steps |
| `rho_floor_tracking` | 0.00956484 | reward |
| `rho_floor` | 0.00956484 | reward |
| `diameter` | 4 | m |
| `length` | 60 | m |
| `slope` | 0.035 | - |
| `n_zones` | 16 | - |
| `w_bed_gas` | 2.906 | m |
| `w_wall_gas` | 9.313 | m |
| `w_wall_bed` | 3.253 | m |
| `refractory_thickness` | 0.2 | m |
| `rho_refractory` | 2200 | kg/m^3 |
| `cp_refractory` | 900 | J/(kg K) |
| `U_shell` | 4 | W/(m^2 K) |
| `raw_meal_nominal` | 53.82 | kg/s |
| `caco3_fraction` | 0.79 | - |
| `calcination_upstream` | 0.92 | - |
| `cp_solid` | 1000 | J/(kg K) |
| `cp_gas` | 1150 | J/(kg K) |
| `bulk_density` | 1400 | kg/m^3 |
| `angle_of_repose` | 35 | deg |
| `h_calcination` | 1.78e+06 | J/kg |
| `h_clinkerisation` | 420000 | J/kg |
| `A_calcination` | 151990 | 1/s |
| `E_calcination` | 170000 | J/mol |
| `A_clinker` | 4.7308e+06 | 1/s |
| `E_clinker` | 280000 | J/mol |
| `fuel_lhv` | 2.6e+07 | J/kg |
| `air_fuel_ratio` | 10.5 | - |
| `excess_air` | 1.15 | - |
| `flame_length` | 15 | m |
| `emissivity_gas` | 0.25 | - |
| `h_convective` | 30 | W/(m^2 K) |
| `h_wall_bed` | 500 | W/(m^2 K) |
| `T_feed` | 1073 | K |
| `T_secondary_air` | 1373 | K |
| `T_ambient` | 300 | K |
| `fuel_min` | 1.2 | kg/s |
| `precision_floor` | 0.0005 | - |
| `fuel_max` | 2.4 | kg/s |
| `rpm_min` | 2 | rpm |
| `rpm_max` | 4.5 | rpm |
| `fuel_nominal` | 1.78 | kg/s |
| `rpm_nominal` | 3 | rpm |
| `feed_noise_std` | 2 | kg/s |
| `lime_band` | 0.006 | - |
| `T_bz_max` | 1900 | K |
| `T_bz_min` | 1620 | K |
| `lime_max` | 0.05 | - |
| `fuel_weight` | 0 | - |

