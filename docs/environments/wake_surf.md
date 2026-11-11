# Wake surfing - A320 in trail

<p align="center"><img src="../../videos/wake_surf/pid_output.gif" width="480px"/></p>

Surf a lead airliner's wake: hold an A320 at a station 3 km behind another, where the lead's upwash lifts it, and pay for fuel and time as an operator does. The station sits 0.87 of a span outboard and below by the vortex's sink; inside a 30 m box about it only the economy counts, so the cost falls where the follower finds the upwash, and the vortex core a span inboard can roll it past 60 deg, an upset (plane3d/PHYSICS.md, Amendment 4, wake surfing).

```python
import target_gym as tg

env, params = tg.make("wake_surf")
```

| | |
|---|---|
| Action space | `Box((3,))`, all actions in [-1, 1] |
| Observation space | `Box((26,))` |
| Tracked variable(s) | slot error (m) |
| Episode length | 1800 steps of 1 s (30 min) |
| Cite as | `wake_surf-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/patrol/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | throttle | -1 | 1 |
| 1 | pitch demand | -1 | 1 |
| 2 | bank demand | -1 | 1 |

## Observation space

26 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[24]` against `obs[25]`.

| # | meaning |
|---|---|
| 0-9 | the follower's own state, as in `plane3d_heading` |
| 10-12 | offset from the station in the lead's frame: back, right, up (m) |
| 13-15 | velocity relative to the lead in the lead's frame: forward, right, up (m/s) |
| 16-18 | the station in the lead's frame: back, right, up (m) |
| 19 | the lead's heading relative to the follower's (rad) |
| 20 | the lead's speed (m/s) |
| 21 | throttle, 0 to 1 |
| 22 | elevator deflection (rad) |
| 23 | aileron deflection (rad) |
| 24 | distance to the station (m) |
| 25 | target distance, always 0 |

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

`reset` samples the initial condition and the target; state has 15 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1800 steps.

## Baselines

No MPC ships with this environment, so there is no recorded comparison. No MPC ships yet: the patrol's twin-autopilot oracle plans the slot, not the station's economy, and the shared solver is not wired to this task (owner, 2026-11-10).

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("wake_surf", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 1800 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
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
| `power_response_rate` | 0.588235 | 1/s |
| `idle_power` | 0.05 | - |
| `stick_response_rate` | 16.6667 | 1/s |
| `aileron_response_rate` | 16.6667 | 1/s |
| `surface_rate_limit` | 0.523599 | rad/s |
| `load_factor_max` | 2.5 | g |
| `load_factor_min` | -1 | g |
| `ap_load_increment` | 0.2 | g |
| `vmo_eas` | 180.056 | m/s |
| `mmo` | 0.82 | - |
| `buffet_mach` | 0.3 | - |
| `buffet_slope` | 1.4 | - |
| `bank_limit` | 1.0472 | rad |
| `aspect_ratio` | 9.48 | - |
| `aileron_effectiveness` | 0.4 | - |
| `cl_alpha` | 0.08378 | 1/deg |
| `sweep_half_chord` | 0.38186 | rad |
| `cl0` | 0.2 | - |
| `cd0` | 0.02 | - |
| `k` | 0.045 | - |
| `aoa_stall` | 15.5 | deg |
| `CL_max` | 1.5 | - |
| `CL_min` | -1 | - |
| `aoa_stall_width` | 3 | deg |
| `M_crit` | 0.8 | - |
| `k_shock_stall` | 4 | - |
| `shock_stall_floor` | 0.25 | - |
| `k_drag` | 5 | - |
| `pitch_gyration_radius` | 8.666 | m |
| `roll_gyration_radius` | 5.039 | m |
| `moment_arm_stabilizer` | 13.36 | m |
| `moment_arm_wings` | 0.822 | m |
| `stabilizer_surface` | 31 | m^2 |
| `tail_cl_alpha` | 0.07021 | 1/deg |
| `tail_aspect_ratio` | 5 | - |
| `tail_sweep_half_chord` | 0.411 | rad |
| `tail_k` | 0.0796 | - |
| `tail_CL_max` | 1 | - |
| `tail_aoa_stall` | 15 | deg |
| `tail_efficiency` | 0.9 | - |
| `elevator_effectiveness` | 0.5 | - |
| `downwash_ratio` | 1 | - |
| `stabilizer_setting` | 0.02417 | rad |
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
| `failure_cost` | 377468 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |
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
| `turbulence_sigma` | 0.36 | m/s |
| `turbulence_theta` | 0.2 | 1/s |
| `turbulence_scale_length` | 533.4 | m |
| `gust_roll_rate` | 0 | rad/s |
| `wind_shear_x` | 0 | 1/s |
| `wind_shear_y` | 0 | 1/s |
| `shear_ref_alt` | 0 | m |
| `target_cas` | 149.189 | m/s |
| `target_mach` | 0.78 | - |
| `speed_band` | 5.14444 | m/s |
| `mach_band` | 0.02 | - |
| `e_tol_heading` | 0 | rad |
| `e_tol_path` | 0 | m |
| `assigned_speed_mode` | 0 | - |
| `holding_rate` | 0.0523599 | rad/s |
| `holding_bank` | 0.523599 | rad |
| `hold_design_bank` | 0.436332 | rad |
| `cost_index` | 30 | kg/min |
| `economy_weight` | 1 | - |
| `economy_scale` | 1.242 | kg/s |
| `heading_clearance_period` | 0 | s |
| `protected_bank` | 0.523599 | rad |
| `holding_split_altitude` | 4267.2 | m |
| `holding_leg_low` | 60 | s |
| `holding_leg_high` | 90 | s |
| `holding_cap_low` | 118.322 | m/s |
| `holding_cap_high` | 136.328 | m/s |
| `slot_tolerance` | 30 | m |
| `heading_tolerance` | 0.5236 | rad |
| `min_separation` | 40 | m |
| `max_slot_error` | 1500 | m |
| `slot_precision_floor` | 3 | m |
| `e_floor_slot` | 30 | m |
| `e_tol_slot` | 30 | m |
| `lead_leg_seconds` | 300 | s |
| `wake_strength` | 1 | - |
| `wake_core_ratio` | 0.0582 | - |
| `wake_decay_rate` | 0.2 | - |
| `follower_spawn_noise` | 10 | m |
| `upset_bank` | 1.0472 | rad |

