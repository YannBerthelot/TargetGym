# Patrol - wingman

!!! note "Under review"
    Patrol - wingman is registered and runs like every other task, but it is not scored in this release. It is under review, so its reference, its ceiling and its room are not measured and no result is published for it. It can join the scored suite in a later release, once it passes the bar the scored tasks passed.

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
| Cite as | `patrol-v6` |
| Status | Under review, not scored |
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
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `tracking`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 15 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 200 steps.

## Baselines

Not measured. A PID and an MPC ship and run, but this task is outside the scored suite, so its recorded numbers are not kept current and none is shown.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("patrol", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 200 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `gravity` | 9.81 | m/s^2 |
| `initial_mass` | 15109.2 | kg |
| `thrust_output_at_sea_level` | 95192 | N |
| `air_density_at_sea_level` | 1.225 | kg/m^3 |
| `frontal_surface` | 12.6 | m^2 |
| `wings_surface` | 37.1612 | m^2 |
| `C_x0` | 0.095 | - |
| `C_z0` | 0.9 | - |
| `initial_fuel_quantity` | 3000 | kg |
| `specific_fuel_consumption` | 0.0229433 | kg/(kN.s) |
| `power_response_rate` | 0.905981 | 1/s |
| `idle_power` | 0.05 | - |
| `stick_response_rate` | 16.6667 | 1/s |
| `aileron_response_rate` | 16.6667 | 1/s |
| `surface_rate_limit` | 0.523599 | rad/s |
| `load_factor_max` | 7.5 | g |
| `load_factor_min` | -3 | g |
| `ap_load_increment` | 7.5 | g |
| `vmo_eas` | 360.111 | m/s |
| `mmo` | 0.85 | - |
| `buffet_mach` | 0.3 | - |
| `buffet_slope` | 1.4 | - |
| `bank_limit` | 1.0472 | rad |
| `aspect_ratio` | 3.50064 | - |
| `aileron_effectiveness` | 0.4 | - |
| `cl_alpha` | 0.0831 | 1/deg |
| `sweep_half_chord` | 0.226893 | rad |
| `cl0` | -0.0204 | - |
| `cd0` | 0.02 | - |
| `k` | 0.113661 | - |
| `aoa_stall` | 20.7028 | deg |
| `CL_max` | 1.7 | - |
| `CL_min` | -1 | - |
| `aoa_stall_width` | 3 | deg |
| `M_crit` | 0.88 | - |
| `k_shock_stall` | 4 | - |
| `shock_stall_floor` | 0.25 | - |
| `k_drag` | 5 | - |
| `pitch_gyration_radius` | 3.6846 | m |
| `roll_gyration_radius` | 1.43663 | m |
| `moment_arm_stabilizer` | 4.96413 | m |
| `moment_arm_wings` | -0.262908 | m |
| `stabilizer_surface` | 11.3995 | m^2 |
| `tail_cl_alpha` | 0.0474437 | 1/deg |
| `tail_aspect_ratio` | 2.4 | - |
| `tail_sweep_half_chord` | 0.610865 | rad |
| `tail_k` | 0.165786 | - |
| `tail_CL_max` | 1 | - |
| `tail_aoa_stall` | 15 | deg |
| `tail_efficiency` | 0.9 | - |
| `elevator_effectiveness` | 1 | - |
| `downwash_ratio` | 0.461959 | - |
| `stabilizer_setting` | -0.09309 | rad |
| `wingspan` | 11.4056 | m |
| `aileron_surface` | 7.95612 | m^2 |
| `moment_arm_aileron` | 4 | m |
| `C_lp` | -0.337 | - |
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
| `economy_weight` | 0 | - |
| `economy_scale` | 1.242 | kg/s |
| `heading_clearance_period` | 0 | s |
| `protected_bank` | 1.16937 | rad |
| `holding_split_altitude` | 4267.2 | m |
| `holding_leg_low` | 60 | s |
| `holding_leg_high` | 90 | s |
| `holding_cap_low` | 118.322 | m/s |
| `holding_cap_high` | 136.328 | m/s |
| `wind_tau` | 2010 | s |
| `slot_tolerance` | 3.048 | m |
| `heading_tolerance` | 0.5236 | rad |
| `min_separation` | 17.0688 | m |
| `max_slot_error` | 1500 | m |
| `slot_precision_floor` | 3 | m |
| `e_floor_slot` | 3 | m |
| `e_tol_slot` | 0 | m |
| `lead_leg_seconds` | 60 | s |
| `wake_strength` | 1 | - |
| `wake_core_ratio` | 0.0582 | - |
| `wake_decay_rate` | 0.2 | - |
| `follower_spawn_noise` | 10 | m |
| `upset_bank` | 0 | rad |

