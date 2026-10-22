# Altitude and airspeed

<p align="center"><img src="../../videos/plane_energy/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((10,))` |
| Tracked variable(s) | altitude (m) |
| Episode length | 1200 steps (1200 s at 1 s per step) |
| Import | `from target_gym import Airplane2D, PlaneParams` |
| Cite as | `plane_energy-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |

## Observation space

10 values. Indices (1,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 17 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. The aircraft trips when its altitude leaves ``[min_alt, max_alt]``. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1200 steps.

## Baselines

Recorded over 10 seeds of the 1200-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.639e+06 | 1366 |
| MPC | -8.785e+04 | 73.21 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 1200 | steps |
| `gravity` | 9.81 | m/s^2 |
| `initial_mass` | 73500 | kg |
| `thrust_output_at_sea_level` | 240000 | N |
| `air_density_at_sea_level` | 1.225 | kg/m^3 |
| `frontal_surface` | 12.6 | m^2 |
| `wings_surface` | 122.6 | m^2 |
| `C_x0` | 0.095 | - |
| `C_z0` | 0.9 | - |
| `initial_fuel_quantity` | 19088 | kg |
| `specific_fuel_consumption` | 0.0175 | kg/(kN.s) |
| `cl_alpha` | 0.08786 | 1/deg |
| `cl0` | 0.2 | - |
| `cd0` | 0.02 | - |
| `k` | 0.045 | - |
| `aoa_stall` | 15 | deg |
| `CL_max` | 1.5 | - |
| `CL_min` | -1 | - |
| `aoa_stall_width` | 3 | deg |
| `aspect_ratio` | 9.48 | - |
| `M_crit` | 0.8 | - |
| `k_shock_stall` | 4 | - |
| `shock_stall_floor` | 0.25 | - |
| `precision_floor` | 1 | m |
| `power_response_rate` | 0.05 | 1/s |
| `stick_response_rate` | 0.9 | 1/s |
| `k_drag` | 5 | - |
| `I` | 9000000 | kg.m^2 |
| `moment_arm_stabilizer` | 15 | m |
| `moment_arm_wings` | 1.5 | m |
| `stabilizer_surface` | 27 | m^2 |
| `elevator_surface` | 10 | m^2 |
| `reward_version` | 2 | - |
| `e_floor` | 4.55 | m |
| `e_tol` | 0 | m |
| `tracking_exponent` | 2 | - |
| `c_hold` | 5.23 | m/s |
| `running_weight` | 1 | - |
| `failure_cost` | 1.43601e+07 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 1 | reward |
| `rho_floor` | 1 | reward |
| `min_alt` | 0 | m |
| `max_alt` | 12191.4 | m |
| `speed_weight` | 0.5 | - |
| `target_speed` | 230 | m/s |
| `speed_precision_floor` | 0.5 | m/s |
| `speed_envelope` | 60 | m/s |
| `target_pattern` | 1 | - |
| `target_amplitude` | 900 | m |
| `target_period` | 240 | s |
| `target_steps` | 8 | - |
| `target_chirp_octaves` | 2 | - |
| `initial_z_dot` | 0 | m/s |
| `initial_x_dot` | 200 | m/s |
| `initial_theta_dot` | 0 | deg/s |
| `initial_theta` | 0 | deg |
| `initial_power` | 1 | - |
| `initial_stick` | 0 | deg |
| `wind_x` | 0 | m/s |
| `wind_z` | 0 | m/s |
| `turbulence_sigma` | 1.2 | m/s |
| `turbulence_theta` | 0.2 | 1/s |
| `impulse_prob` | 0 | 1/step |
| `wind_shear_x` | 0 | 1/s |
| `shear_ref_alt` | 0 | m |

