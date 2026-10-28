# pH neutralisation

<p align="center"><img src="../../videos/ph_neutralization/pid_output.gif" width="480px"/></p>

Hold the pH of a stirred tank, where strong acid meets strong base in a carbonate buffer, at a target by setting the base flow. The titration curve makes the process gain swing by more than an order of magnitude across the range, and the buffer flow that moves the curve drifts unmeasured, so the same pH reading can call for very different moves.

```python
import target_gym as tg

env, params = tg.make("ph_neutralization")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((3,))` |
| Tracked variable(s) | pH |
| Episode length | 300 steps of 5 s (25 min) |
| Cite as | `ph_neutralization-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/ph_neutralization/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | base flow | -1 | 1 |

## Observation space

3 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[2]`.

| # | meaning |
|---|---|
| 0 | pH |
| 1 | base flow valve, percent of its range |
| 2 | target pH |

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

`reset` samples the initial condition and the target; state has 7 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 300 steps.

## Baselines

Recorded over 10 seeds of the 300-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5.043e+04 | 168.1 |
| MPC | -1594 | 5.313 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("ph_neutralization", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 5 | s |
| `max_steps_in_episode` | 300 | steps |
| `V` | 2900 | mL |
| `q1` | 16.6 | mL/s |
| `Wa1` | 0.003 | mol/L |
| `Wb1` | 0 | mol/L |
| `q2_nominal` | 0.55 | mL/s |
| `q2_noise_std` | 0.35 | mL/s |
| `q2_min` | 0 | mL/s |
| `q2_max` | 2.5 | mL/s |
| `Wa2` | -0.03 | mol/L |
| `Wb2` | 0.03 | mol/L |
| `q3_min` | 10 | mL/s |
| `q3_max` | 22 | mL/s |
| `Wa3` | -0.00305 | mol/L |
| `Wb3` | 5e-05 | mol/L |
| `pK1` | 6.35 | - |
| `pK2` | 10.25 | - |
| `pH_min` | 2 | pH |
| `precision_floor` | 0.01 | pH |
| `pH_max` | 12 | pH |
| `tracking_band` | 1 | pH |
| `reagent_cost_weight` | 0 | - |
| `reward_version` | 2 | - |
| `e_floor` | 0.01 | pH |
| `e_tol` | 0 | pH |
| `tracking_exponent` | 2 | - |
| `c_hold` | 16.24 | mL/s |
| `running_weight` | 1 | - |
| `failure_cost` | 3.1e+06 | reward |
| `restart_steps` | 720 | steps |
| `rho_floor_tracking` | 0.0169 | reward |
| `rho_floor` | 0.0169 | reward |

