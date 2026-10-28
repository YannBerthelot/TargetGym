# Altitude hold

<p align="center"><img src="../../videos/plane/pid_output.gif" width="480px"/></p>

Hold an A320-class airliner at a target altitude in light-to-moderate turbulence with throttle and elevator. This is the suite's transparent aircraft: nearly the whole state is observed and only the gusts are hidden, so the difficulty is the plant itself, with slow climbs and pitch, speed and thrust all coupled.

```python
import target_gym as tg

env, params = tg.make("plane")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((10,))` |
| Tracked variable(s) | altitude (m) |
| Episode length | 280 steps of 1 s (4.67 min) |
| Cite as | `plane-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/plane/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |

## Observation space

10 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[1]` against `obs[6]`.

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

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The aircraft trips when its altitude leaves ``[min_alt, max_alt]``. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 280 steps.

## Baselines

Recorded over 10 seeds of the 280-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.058e+06 | 3778 |
| MPC | -3.138e+05 | 1121 |

The MPC beats the PID on 9 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("plane", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 280 | steps |
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
| `e_floor` | 1 | m |
| `e_tol` | 0 | m |
| `tracking_exponent` | 2 | - |
| `c_hold` | 5.06 | m/s |
| `running_weight` | 1 | - |
| `failure_cost` | 2.9729e+08 | reward |
| `restart_steps` | 3600 | steps |
| `rho_floor_tracking` | 0.209856 | reward |
| `rho_floor` | 0.209856 | reward |
| `min_alt` | 0 | m |
| `max_alt` | 12191.4 | m |
| `speed_weight` | 0 | - |
| `target_speed` | 230 | m/s |
| `speed_precision_floor` | 0.5 | m/s |
| `speed_envelope` | 60 | m/s |
| `target_pattern` | 0 | - |
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
| `turbulence_sigma` | 1.2 | m/s |
| `turbulence_theta` | 0.2 | 1/s |
| `impulse_prob` | 0 | 1/step |
| `wind_shear_x` | 0 | 1/s |
| `shear_ref_alt` | 0 | m |

