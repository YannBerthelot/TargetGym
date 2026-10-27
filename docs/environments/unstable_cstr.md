# Unstable CSTR

<p align="center"><img src="../../videos/unstable_cstr/pid_output.gif" width="480px"/></p>

Hold the concentration of A in a jacketed exothermic reactor on its open-loop unstable middle steady state, commanding the coolant through a lagging jacket while the feed temperature drifts unmeasured. Left alone the reactor runs away within tens of seconds, and the target steps five times per episode, so every move crosses the unstable region under control. A trip costs an hour of restart.

```python
import target_gym as tg

env, params = tg.make("unstable_cstr")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((4,))` |
| Tracked variable(s) | C_a (mol/L) |
| Episode length | 1200 steps of 3 s (60 min) |
| Cite as | `unstable_cstr-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/unstable_cstr/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | coolant command | -1 | 1 |

## Observation space

4 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[3]`.

| # | meaning |
|---|---|
| 0 | `C_a`, concentration of A (mol/L) |
| 1 | `T`, reactor temperature (K) |
| 2 | `T_j`, jacket temperature (K) |
| 3 | target `C_a`, the current block's level (mol/L) |

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

**Truncation.** After 1200 steps.

## Baselines

Recorded over 10 seeds of the 1200-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -6.857e+07 | 5.714e+04 |
| MPC | -5.014e+06 | 4178 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("unstable_cstr", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 0.05 | min |
| `max_steps_in_episode` | 1200 | steps |
| `q` | 100 | L/min |
| `V` | 100 | L |
| `rho` | 1000 | g/L |
| `C` | 0.239 | J/(g*K) |
| `deltaHr` | -50000 | J/mol |
| `EA_over_R` | 8750 | K |
| `k0` | 7.2e+10 | 1/min |
| `UA` | 50000 | J/(min*K) |
| `Ti` | 350 | K |
| `Caf` | 1 | mol/L |
| `precision_floor` | 0.0001 | mol/L |
| `time_unit_seconds` | 60 | s |
| `T_c_min` | 290 | K |
| `T_c_max` | 310 | K |
| `tau_j` | 0.1 | min |
| `T_trip` | 365 | K |
| `T_valid_min` | 250 | K |
| `Ti_sigma` | 2 | K |
| `Ti_tau` | 10 | min |
| `Ti_dev_reset_clip` | 3 | - |
| `block_steps` | 200 | steps |
| `initial_CA_offset` | 0.01 | mol/L |
| `initial_T_offset` | 1.5 | K |
| `reward_version` | 2 | - |
| `e_floor` | 0.0001 | mol/L |
| `e_tol` | 0 | mol/L |
| `tracking_exponent` | 2 | - |
| `CA_error_max` | 0.55 | mol/L |
| `failure_cost` | 6.05e+07 | reward |
| `restart_steps` | 1200 | steps |
| `rho_floor_tracking` | 0.00395641 | reward |
| `rho_floor` | 0.00395641 | reward |

