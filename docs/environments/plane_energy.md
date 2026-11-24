# Altitude and airspeed

<p align="center"><img src="../../videos/plane_energy/pid_output.gif" width="480px"/></p>

Fly ATC's level and speed clearances with the same airliner as `plane`: TECS's job, altitude changes with the speed held and speed changes at a constant level [TECS]. Redesigned on 2026-11-08 under Amendment 2; see *Amendment 2: `plane_energy`* below.

```python
import target_gym as tg

env, params = tg.make("plane_energy")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((12,))` |
| Tracked variable(s) | altitude (m) |
| Episode length | 1800 steps of 1 s (30 min) |
| Cite as | `plane_energy-v10` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/plane/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | throttle | -1 | 1 |
| 1 | pitch demand | -1 | 1 |

## Observation space

12 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[1]` against `obs[6]`.

| # | meaning |
|---|---|
| 0 | `x_dot`, horizontal speed (m/s) |
| 1 | `z`, altitude (m) |
| 2 | `z_dot`, vertical speed (m/s) |
| 3 | `theta`, pitch angle (rad) |
| 4 | `theta_dot`, pitch rate (rad/s) |
| 5 | `gamma`, flight-path angle (rad) |
| 6 | target altitude (m) |
| 7 | throttle, 0 to 1 |
| 8 | elevator deflection (rad) |
| 9 | commanded airspeed (m/s) |
| 10 | airspeed, true, against the wind and gust (m/s) |
| 11 | normal load factor at the end of the step (g) |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `tracking`, `running`, `level_off`, `economy`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 29 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The aircraft trips only when it reaches the ground. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1800 steps.

## Baselines

No MPC ships with this environment, so there is no recorded comparison. No oracle ships until the 2D oracle is rebuilt for the Amendment 2 score (clearances on the AIM and RVSM bands, plane/PHYSICS.md): the shipped model plans the earlier cost, so plane_energy's room is not measured (charter rule 9).

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("plane_energy", max_steps_in_episode=...)`.

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
| `cl_alpha` | 0.08378 | 1/deg |
| `sweep_half_chord` | 0.38186 | rad |
| `cl0` | 0.2 | - |
| `cd0` | 0.02 | - |
| `k` | 0.045 | - |
| `aoa_stall` | 15.5 | deg |
| `CL_max` | 1.5 | - |
| `CL_min` | -1 | - |
| `aoa_stall_width` | 3 | deg |
| `aspect_ratio` | 9.48 | - |
| `M_crit` | 0.8 | - |
| `k_shock_stall` | 4 | - |
| `shock_stall_floor` | 0.25 | - |
| `precision_floor` | 1 | m |
| `power_response_rate` | 0.588235 | 1/s |
| `idle_power` | 0.05 | - |
| `stick_response_rate` | 16.6667 | 1/s |
| `surface_rate_limit` | 0.523599 | rad/s |
| `load_factor_max` | 2.5 | g |
| `load_factor_min` | -1 | g |
| `ap_load_increment` | 0.2 | g |
| `vmo_eas` | 180.056 | m/s |
| `mmo` | 0.82 | - |
| `buffet_mach` | 0.3 | - |
| `buffet_slope` | 1.4 | - |
| `k_drag` | 5 | - |
| `pitch_gyration_radius` | 8.666 | m |
| `moment_arm_wings` | 0.822 | m |
| `moment_arm_stabilizer` | 13.36 | m |
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
| `reward_version` | 2 | - |
| `e_floor` | 19.812 | m |
| `e_tol` | 19.812 | m |
| `tracking_exponent` | 2 | - |
| `c_hold` | 5.06 | m/s |
| `running_weight` | 1 | - |
| `failure_cost` | 377468 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |
| `min_alt` | 0 | m |
| `max_alt` | 11917.7 | m |
| `speed_weight` | 0 | - |
| `target_cas` | 149.189 | m/s |
| `target_mach` | 0.78 | - |
| `speed_precision_floor` | 0.5 | m/s |
| `speed_envelope` | 60 | m/s |
| `target_pattern` | 5 | - |
| `target_amplitude` | 800 | m |
| `target_period` | 240 | s |
| `target_steps` | 4 | - |
| `target_chirp_octaves` | 2 | - |
| `initial_z_dot` | 0 | m/s |
| `initial_x_dot` | 200 | m/s |
| `initial_theta_dot` | 0 | deg/s |
| `initial_theta` | 0 | deg |
| `initial_power` | 1 | - |
| `initial_stick` | 0 | deg |
| `wind_x` | 0 | m/s |
| `wind_z` | 0 | m/s |
| `turbulence_sigma` | 0.55 | m/s |
| `turbulence_theta` | 0.2 | 1/s |
| `gust_scale_length` | 533.4 | m |
| `speed_band` | 5.14444 | m/s |
| `mach_band` | 0.02 | - |
| `cost_index` | 30 | kg/min |
| `economy_weight` | 0 | - |
| `economy_scale` | 1.242 | kg/s |
| `clearance_period` | 300 | s |
| `level_change_rate_min` | 2.54 | m/s |
| `level_off_height` | 304.8 | m |
| `level_off_rate_max` | 7.62 | m/s |
| `level_off_rate_band` | 2.54 | m/s |
| `clearance_level_share` | 0.6 | - |
| `impulse_prob` | 0 | 1/step |
| `wind_shear_x` | 0 | 1/s |
| `shear_ref_alt` | 0 | m |

