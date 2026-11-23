# Unstable CSTR

<p align="center"><img src="../../videos/unstable_cstr/pid_output.gif" width="480px"/></p>

Hold the temperature of an LDPE autoclave at its setpoint with the initiator feed, at an operating point that is unstable without control. The policy reads the temperature, its own initiator feed and the setpoint. Conversion and the initiator in the reactor are hidden. Each step is charged the squared temperature error, counted in units of 1 K. A reactor that reaches 573 K trips, which is charged as a restart; a reactor that cools down pays only the tracking cost.

```python
import target_gym as tg

env, params = tg.make("unstable_cstr")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((3,))` |
| Tracked variable(s) | T (K) |
| Episode length | 1800 steps of 1 s (30 min) |
| Cite as | `unstable_cstr-v7` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/unstable_cstr/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | initiator in the feed | -1 | 1 |

## Observation space

3 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[2]`.

| # | meaning |
|---|---|
| 0 | `T`, reactor temperature (K) |
| 1 | initiator feed, percent of its 0 to 1e-3 kmol/m3 range |
| 2 | reactor temperature setpoint (K) |

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

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. The high-temperature trip: decomposition's protection [L2; L3]. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1800 steps.

## Baselines

Recorded over 10 seeds of the 1800-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -64.59 | 0.03588 |
| MPC | 0 | -0 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("unstable_cstr", max_steps_in_episode=...)`.

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
| `V` | 0.498 | m^3 |
| `feed_mass` | 8 | kg/s |
| `rho` | 499 | kg/m^3 |
| `mw` | 28.05 | kg/kmol |
| `T_feed` | 353 | K |
| `feed_T_wander` | 1 | K |
| `feed_T_wander_period` | 600 | s |
| `P` | 1.6212e+08 | Pa |
| `kp0` | 1e+06 | m^3/(kmol*s) |
| `Ep` | 27100 | J/mol |
| `Vp` | -2.31e-05 | m^3/mol |
| `kt0` | 6e+08 | m^3/(kmol*s) |
| `Et` | 16500 | J/mol |
| `kd0` | 4.2e+15 | 1/s |
| `Ed` | 153500 | J/mol |
| `f_eff` | 0.5 | - |
| `dH` | -9.36e+07 | J/kmol |
| `cp` | 2500 | J/(kg*K) |
| `I_f_min` | 0 | kmol/m^3 |
| `I_f_max` | 0.001 | kmol/m^3 |
| `T_setpoint` | 513 | K |
| `T_trip` | 573 | K |
| `setpoint_step` | 0 | K |
| `feed_T_step` | 0 | K |
| `efficiency_drop` | 0 | - |
| `feed_flow_step` | 0 | - |
| `event_hold` | 1e+09 | s |
| `reward_version` | 3 | - |
| `e_floor` | 1 | K |
| `e_tol` | 0 | K |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 96800 | reward |
| `restart_steps` | 7200 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |
| `precision_floor` | 1 | K |

