# Boiler Drum

!!! note "Under review"
    Boiler Drum is registered and runs like every other task, but it is not scored in this release. It is under review, so its reference, its ceiling and its room are not measured and no result is published for it. It can join the scored suite in a later release, once it passes the bar the scored tasks passed.

<p align="center"><img src="../../videos/boiler_drum/pid_output.gif" width="480px"/></p>

Keep a boiler's drum water level between its low and high trip levels while the load moves, using the feedwater and the firing, and deliver the steam the load plan asks for. The level first moves the wrong way after a steam or feedwater change (shrink and swell). The policy reads the level, the pressure, the steam flow, its two actuators, the load plan and the caps an upset puts on firing and feedwater. Each step is charged, in GBP, the steam delivered off the plan beyond the meter's resolution. A level past either trip level for more than 10 s trips the unit, which is charged as a hot restart.

```python
import target_gym as tg

env, params = tg.make("boiler_drum")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((12,))` |
| Tracked variable(s) | drum level (m), drum pressure (bar) |
| Episode length | 1800 steps of 2 s (60 min) |
| Cite as | `boiler_drum-v6` |
| Status | Under review, not scored |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/boiler_drum/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel | -1 | 1 |
| 1 | feedwater | -1 | 1 |

## Observation space

12 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[5]`, `obs[1]` against `obs[6]`.

| # | meaning |
|---|---|
| 0 | drum level, deviation from normal water level (m) |
| 1 | drum pressure (bar) |
| 2 | steam flow drawn by the turbine (kg/s) |
| 3 | firing, percent of its maximum |
| 4 | feedwater flow, percent of its maximum |
| 5 | level target, always 0 (m) |
| 6 | pressure setpoint (bar) |
| 7 | load now: the steam the turbine valve passes at nominal pressure (kg/s) |
| 8 | load the plan ramps to (kg/s); during an upset, the runback's load if lower |
| 9 | seconds until the ramp starts, 0 once it has |
| 10 | firing the plant can still deliver, percent of its maximum |
| 11 | feedwater the plant can still deliver, percent of its maximum |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `imbalance`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 28 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. A level trip: past either trip level for more than ``level_trip_delay``. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1800 steps.

## Baselines

Not measured. A PID and an MPC ship and run, but this task is outside the scored suite, so its recorded numbers are not kept current and none is shown.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("boiler_drum", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 2 | s |
| `max_steps_in_episode` | 1800 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `reward_version` | 4 | - |
| `steam_rate` | 3.5 | kg/kWh |
| `imbalance_price` | 80 | GBP/MWh |
| `steam_resolution` | 0.047 | kg/s |
| `restart_steps` | 4500 | steps |
| `restart_load` | 0.25 | fraction of nominal steam |
| `failure_cost` | 4.26743 | GBP per step |
| `e_floor_level` | 0.00267 | m |
| `e_floor_pressure` | 0.05 | bar |
| `level_precision_floor` | 0.001 | m |
| `pressure_precision_floor` | 0.05 | bar |
| `level_band` | 0.1 | m |
| `pressure_band` | 2 | bar |
| `V_t` | 85 | m^3 |
| `V_d` | 37 | m^3 |
| `V_r` | 37 | m^3 |
| `V_dc` | 11 | m^3 |
| `A_d` | 23 | m^2 |
| `m_metal` | 300000 | kg |
| `m_riser` | 100000 | kg |
| `m_drum` | 20000 | kg |
| `C_metal` | 650 | J/(kg K) |
| `k_e` | 200 | kg/s^2 per kg/m^3 per m^3 |
| `V_sd0` | 8 | m^3 |
| `beta` | 0.3 | - |
| `T_d_flow` | 600 | kg |
| `normal_level_height` | 1 | m |
| `h_feedwater` | 1.037e+06 | J/kg |
| `Q_max` | 2.4e+08 | W |
| `q_feed_max` | 200 | kg/s |
| `Q_nominal` | 1.6e+08 | W |
| `q_steam_nominal` | 93.35 | kg/s |
| `p_nominal` | 85 | bar |
| `event_kind` | 0 | - |
| `runback_load` | 0.5 | fraction of nominal steam |
| `feed_trip_capacity` | 0.5 | fraction of nominal steam |
| `event_hold` | 1e+09 | s |
| `level_trip_low` | -0.33 | m |
| `level_trip_high` | 0.254 | m |
| `level_trip_delay` | 10 | s |

