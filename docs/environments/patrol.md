# Patrol - wingman

<p align="center"><img src="../../videos/patrol/pid_output.gif" width="480px"/></p>

Fly as a wingman: reach and hold a slot fixed in the lead aircraft's frame, behind, beside and above or below it, while the lead flies a patrol route that turns onto a new rate every minute. The slot moves whenever the lead turns, so the reference is non-stationary, the heading difference to the lead is charged too, and coming too close to the lead trips the plant.

```python
import target_gym as tg

env, params = tg.make("patrol")
```

| | |
|---|---|
| Action space | `Box((3,))`, all actions in [-1, 1] |
| Observation space | `Box((26,))` |
| Tracked variable(s) | slot error (m) |
| Episode length | 200 steps of 1 s (3.33 min) |
| Cite as | `patrol-v3` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/patrol/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |
| 2 | aileron | -1 | 1 |

## Observation space

26 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[24]` against `obs[25]`.

| # | meaning |
|---|---|
| 0-9 | the wingman's own state, as in `plane3d_heading` |
| 10-12 | offset from the slot in the lead's frame: back, right, up (m) |
| 13-15 | velocity relative to the lead in the lead's frame: forward, right, up (m/s) |
| 16-18 | the slot in the lead's frame: back, right, up (m) |
| 19 | the lead's heading relative to the wingman's (rad) |
| 20 | the lead's speed (m/s) |
| 21 | throttle, 0 to 1 |
| 22 | elevator deflection (rad) |
| 23 | aileron deflection (rad) |
| 24 | slot error, the distance to the slot (m) |
| 25 | target slot error, always 0 |

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

`reset` samples the initial condition and the target; state has 12 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 200 steps.

## Baselines

Recorded over 10 seeds of the 200-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2.845e+04 | 142.3 |
| MPC | -3907 | 19.54 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("patrol", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 200 | steps |
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
| `power_response_rate` | 0.05 | 1/s |
| `stick_response_rate` | 0.9 | 1/s |
| `aileron_response_rate` | 0.9 | 1/s |
| `aspect_ratio` | 9.48 | - |
| `aileron_effectiveness` | 0.4 | - |
| `cl_alpha` | 0.08786 | 1/deg |
| `cl0` | 0.2 | - |
| `cd0` | 0.02 | - |
| `k` | 0.045 | - |
| `aoa_stall` | 15 | deg |
| `CL_max` | 1.5 | - |
| `CL_min` | -1 | - |
| `aoa_stall_width` | 3 | deg |
| `M_crit` | 0.8 | - |
| `k_shock_stall` | 4 | - |
| `shock_stall_floor` | 0.25 | - |
| `k_drag` | 5 | - |
| `I` | 9000000 | kg.m^2 |
| `I_x` | 2500000 | kg.m^2 |
| `moment_arm_stabilizer` | 15 | m |
| `moment_arm_wings` | 1.5 | m |
| `stabilizer_surface` | 27 | m^2 |
| `elevator_surface` | 10 | m^2 |
| `wingspan` | 35.8 | m |
| `aileron_surface` | 6 | m^2 |
| `moment_arm_aileron` | 14 | m |
| `C_lp` | -0.4 | - |
| `min_alt` | 0 | m |
| `max_alt` | 12191.4 | m |
| `precision_floor` | 1 | m |
| `heading_precision_floor` | 0.0087 | rad |
| `position_precision_floor` | 3 | m |
| `reward_version` | 2 | - |
| `e_floor_altitude` | 1.44 | m |
| `e_tol_altitude` | 0 | m |
| `e_floor_heading` | 0.0087 | rad |
| `e_floor_path` | 3 | m |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 760790 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 0.0363037 | reward |
| `rho_floor` | 0.0363037 | reward |
| `figure8_altitude_amplitude` | 200 | m |
| `initial_z_dot` | 0 | m/s |
| `initial_x_dot` | 200 | m/s |
| `initial_y_dot` | 0 | m/s |
| `initial_theta_dot` | 0 | deg/s |
| `initial_theta` | 0 | deg |
| `initial_phi` | 0 | deg |
| `initial_phi_dot` | 0 | deg/s |
| `initial_heading` | 0 | rad |
| `initial_power` | 1 | - |
| `initial_stick` | 0 | deg |
| `initial_aileron` | 0 | deg |
| `wind_x` | 0 | m/s |
| `wind_y` | 0 | m/s |
| `wind_z` | 0 | m/s |
| `turbulence_sigma` | 1.2 | m/s |
| `turbulence_theta` | 0.2 | 1/s |
| `wind_shear_x` | 0 | 1/s |
| `wind_shear_y` | 0 | 1/s |
| `shear_ref_alt` | 0 | m |
| `slot_tolerance` | 60 | m |
| `heading_tolerance` | 0.5236 | rad |
| `min_separation` | 25 | m |
| `max_slot_error` | 1500 | m |
| `slot_precision_floor` | 3 | m |
| `e_floor_slot` | 3 | m |
| `e_tol_slot` | 0 | m |
| `lead_leg_seconds` | 60 | s |
| `follower_spawn_noise` | 40 | m |

