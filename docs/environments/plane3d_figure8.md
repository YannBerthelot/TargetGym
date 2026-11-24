# 3D figure-8

!!! note "Under review"
    3D figure-8 is registered and runs like every other task, but it is not scored in this release. It is under review, so its reference, its ceiling and its room are not measured and no result is published for it. It can join the scored suite in a later release, once it passes the bar the scored tasks passed.

<p align="center"><img src="../../videos/plane3d_figure8/pid_output.gif" width="480px"/></p>

Follow a twisted figure-8 that climbs and descends, with the 3D airliner. The bank has to reverse through wings level at every crossover, and the observation gives the vector to the nearest point of the path and the path's heading there, but not its curvature. The half-width is 34 to 36 km, so the tips, of radius a / 3, take 25 deg of bank at 230 m/s and less at the speed flown. They used to be 2.7 to 4 km, 46 to 57 deg at 200 m/s. A lap is 183 km, so 1000 steps reach both tips at the slowest speed the schedule flies.

```python
import target_gym as tg

env, params = tg.make("plane3d_figure8")
```

| | |
|---|---|
| Action space | `Box((3,))`, all actions in [-1, 1] |
| Observation space | `Box((21,))` |
| Tracked variable(s) | distance to path (m) |
| Episode length | 1000 steps of 1 s (16.7 min) |
| Cite as | `plane3d_figure8-v7` |
| Status | Under review, not scored |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/plane3d/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | throttle | -1 | 1 |
| 1 | pitch demand | -1 | 1 |
| 2 | bank demand | -1 | 1 |

## Observation space

21 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[19]` against `obs[20]`.

| # | meaning |
|---|---|
| 0-9 | the aircraft's state, as in `plane3d_heading` |
| 10 | target altitude (m) |
| 11 | size of the figure-8 (m) |
| 12-14 | vector from the aircraft to the nearest point of the path, x, y and z (m) |
| 15 | heading of the path at the nearest point (rad) |
| 16 | throttle, 0 to 1 |
| 17 | elevator deflection (rad) |
| 18 | aileron deflection (rad) |
| 19 | distance to the path, the tracked output (m) |
| 20 | target distance to the path, always 0 (m) |

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

`reset` samples the initial condition and the target; state has 42 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1000 steps.

## Baselines

Not measured. A PID and an MPC ship and run, but this task is outside the scored suite, so its recorded numbers are not kept current and none is shown.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("plane3d_figure8", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 1000 | steps |
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
| `failure_cost` | 1.08889e+09 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 0.18407 | reward |
| `rho_floor` | 0.18407 | reward |
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
| `turbulence_sigma` | 0.564 | m/s |
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
| `protected_bank` | 0.523599 | rad |
| `holding_split_altitude` | 4267.2 | m |
| `holding_leg_low` | 60 | s |
| `holding_leg_high` | 90 | s |
| `holding_cap_low` | 118.322 | m/s |
| `holding_cap_high` | 136.328 | m/s |
| `wind_tau` | 2010 | s |

