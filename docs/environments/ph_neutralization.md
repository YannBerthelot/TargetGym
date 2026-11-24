# pH neutralisation

<p align="center"><img src="../../videos/ph_neutralization/pid_output.gif" width="480px"/></p>

Keep the discharge of a two-stage acid neutralisation inside its permit, pH 6 to 8, with a reagent valve on stage 1 and a trim valve on stage 2, without overdosing. The policy reads both stages' pH probes, its two valves, the influent flow, the time spent outside the permit and the diversion basin's level. The buffer flow and the acid strength are hidden. Each step is charged, in litres of reagent, the reagent used, the time outside the permit, and the effluent diverted while the discharge is below pH 5. One excursion longer than 60 min, or an overrun basin, is a violation and is charged as a restart.

```python
import target_gym as tg

env, params = tg.make("ph_neutralization")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((8,))` |
| Tracked variable(s) | pH |
| Episode length | 1440 steps of 5 s (2 h) |
| Cite as | `ph_neutralization-v6` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/ph_neutralization/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 |  | -1 | 1 |
| 1 |  | -1 | 1 |

## Observation space

8 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[2]`.

| # | meaning |
|---|---|
| 0 | discharge pH, stage 2's probe |
| 1 | stage-1 reagent valve, percent of its range |
| 2 | pH set point |
| 3 | minutes outside the permit in this excursion |
| 4 | stage-1 pH, its probe |
| 5 | stage-2 trim valve, percent of its maximum |
| 6 | influent flow (mL/s) |
| 7 | diversion basin level, percent of its volume |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `reagent`, `excursion`, `diversion`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 25 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The trip is a permit violation: one excursion past the allowed 60 min [P8], or an overrun diversion basin (Amendment 3). See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1440 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("ph_neutralization", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 5 | s |
| `max_steps_in_episode` | 1440 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `V_s1` | 20000 | mL |
| `V` | 20000 | mL |
| `q1` | 16.6 | mL/s |
| `Wa1` | 0.003 | mol/L |
| `Wb1` | 0 | mol/L |
| `q2_nominal` | 0.55 | mL/s |
| `q2_noise_std` | 0.35 | mL/s |
| `q2_min` | 0 | mL/s |
| `q2_max` | 2.5 | mL/s |
| `Wa2` | -0.03 | mol/L |
| `Wb2` | 0.03 | mol/L |
| `q3_min` | 0 | mL/s |
| `q3_max` | 22 | mL/s |
| `q3_trim_max` | 2.2 | mL/s |
| `Wa3` | -0.00305 | mol/L |
| `Wb3` | 5e-05 | mol/L |
| `pK1` | 6.35 | - |
| `pK2` | 10.25 | - |
| `acid_upset_size` | 0.2 | - |
| `acid_upset_ramp` | 600 | s |
| `flow_swing` | 0.2 | - |
| `window_upsets` | 1 | - |
| `event_hold` | 1e+09 | s |
| `permit_low` | 6 | pH |
| `permit_high` | 8 | pH |
| `excursion_limit` | 3600 | s |
| `sewer_floor` | 5 | pH |
| `basin_volume` | 236.16 | L |
| `pumpback_flow` | 4.15 | mL/s |
| `pH_min` | 2 | pH |
| `precision_floor` | 0.01 | pH |
| `pH_max` | 12 | pH |
| `reagent_cost_weight` | 0 | - |
| `setpoint_pH` | 7 | pH |
| `reward_version` | 4 | - |
| `reagent_price` | 1 | L/L |
| `excursion_price` | 1.66667 | L/s |
| `diversion_price` | 2 | L/L |
| `failure_cost` | 9 | L |
| `restart_steps` | 720 | steps |
| `e_floor` | 1 | pH |

