# Formation

<p align="center"><img src="../../videos/formation/pid_output.gif" width="480px"/></p>

Fly a lead and two wingmen in a V, every aircraft controlled by the policy. The lead holds an altitude and a heading that follows a patrol route, and each wingman holds a slot fixed in the lead's frame. Because the lead is learned too, each wingman's reference is a co-player rather than a script; the team shares one cost, and a collision between any two aircraft trips all three. The action is the three aircraft's controls, the lead's first.

```python
import target_gym as tg

env, params = tg.make("formation")
```

| | |
|---|---|
| Action space | `Box((9,))`, all actions in [-1, 1] |
| Observation space | `Box((74,))` |
| Tracked variable(s) | lead altitude (m), wingman 0 slot error (m), wingman 1 slot error (m) |
| Episode length | 200 steps of 1 s (3.33 min) |
| Cite as | `formation-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/formation/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | lead power | -1 | 1 |
| 1 | lead stick | -1 | 1 |
| 2 | lead aileron | -1 | 1 |
| 3 | wingman 0 power | -1 | 1 |
| 4 | wingman 0 stick | -1 | 1 |
| 5 | wingman 0 aileron | -1 | 1 |
| 6 | wingman 1 power | -1 | 1 |
| 7 | wingman 1 stick | -1 | 1 |
| 8 | wingman 1 aileron | -1 | 1 |

## Observation space

74 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[2]` against `obs[10]`, `obs[46]` against `obs[47]`, `obs[72]` against `obs[73]`.

| # | meaning |
|---|---|
| 0-14 | the lead's observation as in `plane3d_heading`, its target heading being the one the route commands this step |
| 15 | the route's current turn rate (rad per step) |
| 16-18 | wingman 0's offset from its slot in the lead's frame: back, right, up (m) |
| 19-21 | wingman 1's offset from its slot, likewise (m) |
| 22-47 | wingman 0's observation, as in `patrol` |
| 48-73 | wingman 1's observation, as in `patrol` |

## Rewards

One team cost: the lead's squared altitude and heading errors against its route, plus each wingman's squared slot error and heading misalignment with the lead, each in units of its instrument's resolution (1 m, 0.5 deg, 3 m, 0.5 deg). Every aircraft receives the same reward.

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

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The formation trips when any aircraft leaves the altitude band, any two aircraft come within ``min_separation`` (25 m), or any wingman is ``max_slot_error`` (1500 m) from its slot. The whole formation restarts. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 200 steps.

## Baselines

Recorded over 10 seeds of the 200-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5.837e+04 | 291.9 |
| MPC | -5321 | 26.61 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

The PID flies the lead seat as a formation lead: the 3D heading task's course hold and pitch on height, aimed at the heading the route commands, at a set power, so its wingmen have a steady reference. Each wingman seat flies the patrol's formation hold. The MPC slot is the patrol oracle's design applied to the whole formation: the patrol's lead autopilot in the lead seat and its twin-autopilot law in each wingman seat, with one residual planner over every seat on the team cost.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("formation", max_steps_in_episode=...)`.

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
| `e_floor_altitude` | 1 | m |
| `e_tol_altitude` | 0 | m |
| `e_floor_heading` | 0.0087 | rad |
| `e_floor_path` | 3 | m |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 2.99072e+08 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 0.704494 | reward |
| `rho_floor` | 0.704494 | reward |
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
| `w_track` | 0.6 | - |
| `w_lead` | 0.4 | - |

