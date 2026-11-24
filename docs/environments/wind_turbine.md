# Wind Turbine

<p align="center"><img src="../../videos/wind_turbine/pid_output.gif" width="480px"/></p>

Run a 5 MW wind turbine above rated wind with blade pitch and generator torque: deliver the power the wind makes available, up to the rating, while sparing the tower and the pitch system. Pitching to hold speed changes the thrust, which swings the tower. The policy reads rotor speed, pitch, torque, power, the current rating, the nacelle's fore-aft acceleration and the nacelle wind speed. Each step is charged, in dollars, the production lost against the available power, the tower's fatigue damage and the pitch travel. Overspeed at 1.20 times rated speed trips the turbine, which is charged as a restart.

```python
import target_gym as tg

env, params = tg.make("wind_turbine")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | electrical power (MW) |
| Episode length | 2400 steps of 0.25 s (10 min) |
| Cite as | `wind_turbine-v6` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/energy/wind_turbine/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | pitch_raw | -1 | 1 |
| 1 | torque_raw | -1 | 1 |

## Observation space

7 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[3]` against `obs[4]`.

| # | meaning |
|---|---|
| 0 | rotor speed (rpm) |
| 1 | blade pitch (deg) |
| 2 | generator torque, percent of its maximum |
| 3 | electrical power (MW) |
| 4 | the current rating's power (MW) |
| 5 | nacelle fore-aft acceleration (m/s^2) |
| 6 | nacelle anemometer wind speed (m/s) |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `lost`, `tower`, `pitch`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 33 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. A trip is the safety system's feather [S4]: no other limit trips. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 2400 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("wind_turbine", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 0.25 | s |
| `max_steps_in_episode` | 2400 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `R` | 63 | m |
| `rho` | 1.225 | kg/m^3 |
| `dynamic_inflow` | 1 | - |
| `oye_k1` | 1.1 | - |
| `oye_k2` | 1.3 | - |
| `J_rotor` | 3.87592e+07 | kg m^2 |
| `J_gen` | 534.116 | kg m^2 |
| `N_gear` | 97 | - |
| `eta_gen` | 0.944 | - |
| `P_rated` | 5e+06 | W |
| `v_rated` | 11.4 | m/s |
| `omega_rated_rpm` | 12.1 | rpm |
| `v_cut_out` | 25 | m/s |
| `tower_frequency_hz` | 0.324 | Hz |
| `tower_modal_mass` | 436865 | kg |
| `tower_damping_ratio` | 0.01 | - |
| `pitch_min` | 0 | deg |
| `pitch_max` | 90 | deg |
| `pitch_rate_max` | 8 | deg/s |
| `torque_max` | 47402.9 | N m |
| `torque_rate_max` | 15000 | N m/s |
| `turbulence_iref` | 0.14 | - |
| `turbulence_lambda1` | 42 | m |
| `kaimal_factor` | 8.1 | - |
| `v_mean_tau` | 0 | s |
| `v_ref` | 50 | m/s |
| `gust` | 0 | - |
| `overspeed_factor` | 1.2 | - |
| `rating` | 1 | - |
| `derate_to` | 1 | - |
| `derate_hold` | 1e+09 | s |
| `reward_version` | 4 | - |
| `energy_price` | 60 | $/MWh |
| `capex_per_kw` | 1750 | $/kW |
| `turbine_share` | 0.75 | - |
| `tower_share` | 0.263 | - |
| `pitch_share` | 0.0266 | - |
| `design_life_years` | 20 | years |
| `tower_height` | 87.6 | m |
| `fatigue_exponent` | 4 | - |
| `tower_damage_design` | 212600 | (MN m)^m per step |
| `pitch_travel_design` | 0.1754 | deg per step |
| `restart_steps` | 28800 | steps |
| `reset_callout` | 1000 | $ |
| `failure_cost` | 0.0555556 | $ per step |
| `e_floor` | 1680 | W |
| `precision_floor` | 1700 | W |

