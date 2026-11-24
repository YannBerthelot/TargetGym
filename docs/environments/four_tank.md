# Four-tank

!!! note "Extra task"
    Four-tank is kept in the repository but is not part of the benchmark, so it is out of the headline and the capability claims. The reason: a lab rig with no industrial job: Johansson's teaching process tracks levels to 1 mm, and no plant family stands behind its flow-split zero (charter Amendment 3).

<p align="center"><img src="../../videos/four_tank/pid_output.gif" width="480px"/></p>

Hold the levels of the two lower tanks of a four-tank rig at their setpoints with two pump voltages. Each pump fills the far upper tank more than its own lower one, so a level first moves the wrong way. The policy reads all four levels and both setpoints. Each step is charged the sum of the two squared level errors, counted in units of 1 mm. Nothing trips: an empty tank stops draining and a full one overflows.

```python
import target_gym as tg

env, params = tg.make("four_tank")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | h1 (m), h2 (m) |
| Episode length | 3300 steps of 1 s (55 min) |
| Cite as | `four_tank-v7` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/four_tank/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | pump 1 voltage | -1 | 1 |
| 1 | pump 2 voltage | -1 | 1 |

## Observation space

6 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[4]`, `obs[1]` against `obs[5]`.

| # | meaning |
|---|---|
| 0 | `h1`, level of lower tank 1 (m), controlled |
| 1 | `h2`, level of lower tank 2 (m), controlled |
| 2 | `h3`, level of upper tank 3 (m) |
| 3 | `h4`, level of upper tank 4 (m) |
| 4 | setpoint for `h1` (m) |
| 5 | setpoint for `h2` (m) |

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

`reset` samples the initial condition and the target; state has 20 fields.

## Episode end

**Termination.** None: nothing trips this plant. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 3300 steps.

## Baselines

Recorded over 10 seeds of the 3300-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -4.357e+05 | 132 |
| MPC | -1.047e+05 | 31.73 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("four_tank", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 3300 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `g` | 9.81 | m/s^2 |
| `gamma1` | 0.43 | - |
| `gamma2` | 0.34 | - |
| `k1` | 3.14e-06 | m^3/(V*s) |
| `k2` | 3.29e-06 | m^3/(V*s) |
| `a1` | 7.1e-06 | m^2 |
| `a2` | 5.7e-06 | m^2 |
| `a3` | 7.1e-06 | m^2 |
| `a4` | 5.7e-06 | m^2 |
| `A1` | 0.0028 | m^2 |
| `A2` | 0.0032 | m^2 |
| `A3` | 0.0028 | m^2 |
| `A4` | 0.0032 | m^2 |
| `v_min` | 0 | V |
| `v_max` | 10 | V |
| `v_nominal` | 3.15 | V |
| `h_max` | 0.25 | m |
| `precision_floor` | 0.001 | m |
| `setpoint_step` | 0.0125 | m |
| `leak_fraction` | 0.3 | - |
| `leak_duration` | 300 | steps |
| `cup_volume` | 0.0001 | m^3 |
| `window_events` | 1 | - |
| `event_hold` | 1e+09 | s |
| `reward_version` | 3 | - |
| `e_floor` | 0.001 | m |
| `e_tol` | 0 | m |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 0 | reward |
| `restart_steps` | 1 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |

