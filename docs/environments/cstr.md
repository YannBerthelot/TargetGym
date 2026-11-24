# CSTR

<p align="center"><img src="../../videos/cstr/pid_output.gif" width="480px"/></p>

Hold the temperature of a jacketed exothermic reactor at its setpoint with the cooling water flow. The policy reads the reactor and jacket temperatures, its own water flow and the setpoint, and never the composition. Each step is charged the squared temperature error, counted in units of 0.1 K. A reactor that reaches 75 C trips its interlock, which is charged as a restart.

```python
import target_gym as tg

env, params = tg.make("cstr")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((4,))` |
| Tracked variable(s) | T (K) |
| Episode length | 960 steps of 15 s (4 h) |
| Cite as | `cstr-v7` |
| Status | Scored |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/cstr/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | cooling water | -1 | 1 |

## Observation space

4 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[3]`.

| # | meaning |
|---|---|
| 0 | `T`, reactor temperature (K) |
| 1 | `T_j`, jacket temperature (K) |
| 2 | cooling water flow, percent of its 40 to 136 g/s range |
| 3 | reactor temperature setpoint (K) |

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

`reset` samples the initial condition and the target; state has 14 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The interlock: high reactor temperature cuts the feed [S3]. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 960 steps.

## Reference and ceiling

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies (`hold reset`) | 1722 | 0 |
| Reference | the controller to beat | 98.37 | 0 |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | 70.64 | 0 |
| Ceiling | the best cost found with the true model, the true state and the future: a limit, not a target | 65.28 | 0 |

Room, the reference's cost divided by the ceiling's, is 1.51.

## Baselines

Recorded over 10 seeds of the 960-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5038 | 5.247 |
| MPC | 0 | -0 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

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
| `e_tol` | 0 | K |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 245000 | reward |
| `restart_steps` | 960 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |
| `precision_floor` | 0.1 | K |

