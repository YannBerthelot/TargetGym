# CSTR

<p align="center"><img src="../../videos/cstr/pid_output.gif" width="480px"/></p>

Run a jacketed exothermic reactor for what it makes, with two moves: the cooling water and the feed. Each step is charged the production lost against the reactor's nameplate, and all of it while the temperature is more than 1 K from its setpoint, where the product is off spec. The policy reads the reactor and jacket temperatures, its two flows and the setpoint, and never the composition. A reactor that reaches 75 C trips its interlock, charged as 4 h of lost production. The reference keeps the band on the normal day and through a setpoint move, but not in the three upsets (catalyst loss, a richer feed, a short loss of cooling): there it leaves the band on every seed, and their rooms are reported as measured.

```python
import target_gym as tg

env, params = tg.make("cstr")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | T (K) |
| Episode length | 960 steps of 15 s (4 h) |
| Cite as | `cstr-v8` |
| Status | Scored |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/cstr/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 |  | -1 | 1 |
| 1 |  | -1 | 1 |

## Observation space

5 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[4]`.

| # | meaning |
|---|---|
| 0 | `T`, reactor temperature (K) |
| 1 | `T_j`, jacket temperature (K), 2 min late |
| 2 | cooling water flow, percent of its 40 to 136 g/s range |
| 3 | feed flow, percent of its 40 to 110 g/s range |
| 4 | reactor temperature setpoint (K) |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `lost_production`, `off_spec`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 15 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The interlock: high reactor temperature cuts the feed [S3]. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 960 steps.

## Reference and ceiling

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies | not current | |
| Reference | the controller to beat | not current | |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | not current | |
| Ceiling | the best cost found with the true model, the true state and the future: a limit, not a target | not current | |

Recorded on an earlier version of this task, so not shown: the best null policy, reference, MPC and ceiling lines, which show again once re-recorded. The room shows again with the reference and the ceiling.

## Baselines

The recorded comparison was measured on an earlier version of this task, so it is not shown; it shows again once re-recorded.

Version 8 (the reactor run for what it makes, owner's decision of 2026-11-24) ships no oracle: its planner model is not written for two moves and the production score, and the owner decided the same day that 0.8.0 does not need it. The ceiling is pc_gym.foresight's (cstr/oracle.py).

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("cstr", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 0.25 | min |
| `max_steps_in_episode` | 960 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `k0` | 1.2e+13 | 1/min |
| `EA_over_R` | 10700 | K |
| `deltaHr` | -200000 | J/mol |
| `Caf` | 1.5 | mol/L |
| `q` | 5.4 | L/min |
| `q_min` | 2.4 | L/min |
| `q_max` | 6.6 | L/min |
| `V` | 183.6 | L |
| `rho` | 1000 | g/L |
| `C` | 4.18 | J/(g*K) |
| `UA` | 46717 | J/(min*K) |
| `Fw_min` | 2400 | g/min |
| `Fw_max` | 8160 | g/min |
| `Tw_in` | 303.08 | K |
| `M_j` | 36720 | g |
| `feed_T_period` | 1440 | min |
| `T_setpoint` | 331.15 | K |
| `T_band` | 1 | K |
| `setpoint_ramp` | 60 | min |
| `valve_rest` | 0.95 | fraction of span |
| `T_trip` | 348.15 | K |
| `time_unit_seconds` | 60 | s |
| `setpoint_step` | 0 | K |
| `k0_drop` | 0 | - |
| `k0_decay` | 0 | min |
| `feed_rise` | 0 | - |
| `cooling_loss` | 0 | min |
| `event_hold` | 1e+09 | min |
| `reward_version` | 3 | - |
| `e_floor` | 0.1 | K |
| `failure_cost` | 1 | reward |
| `restart_steps` | 960 | steps |
| `rho_floor` | 0 | reward |
| `precision_floor` | 0.1 | K |

