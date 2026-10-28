# CSTR

<p align="center"><img src="../../videos/cstr/pid_output.gif" width="480px"/></p>

Hold the concentration of reactant A in a cooled, exothermic stirred-tank reactor at a target drawn at reset, by setting the coolant temperature. The coolant reaches the concentration only through the reactor temperature and an exponential reaction rate, so the authority is narrow and nonlinear. Nothing is hidden and nothing is random after reset, which makes this the first step up from `first_order`.

```python
import target_gym as tg

env, params = tg.make("cstr")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((3,))` |
| Tracked variable(s) | C_a (mol/L) |
| Episode length | 100 steps of 15 s (25 min) |
| Cite as | `cstr-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/pc_gym/cstr/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | coolant temperature | -1 | 1 |

## Observation space

3 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[2]`.

| # | meaning |
|---|---|
| 0 | `C_a`, concentration of A (mol/L) |
| 1 | `T`, reactor temperature (K) |
| 2 | target `C_a` (mol/L) |

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

`reset` samples the initial condition and the target; state has 5 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 100 steps.

## Baselines

Recorded over 10 seeds of the 100-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -6.252e+05 | 6252 |
| MPC | -5.987e+05 | 5987 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("cstr", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 0.25 | min |
| `max_steps_in_episode` | 100 | steps |
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
| `T_c_max` | 302 | K |
| `T_c_min` | 295 | K |
| `T_max` | 350 | K |
| `T_min` | 300 | K |
| `C_a_min` | 0.7 | mol/L |
| `precision_floor` | 0.0001 | mol/L |
| `C_a_max` | 1 | mol/L |
| `initial_T` | 330 | K |
| `time_unit_seconds` | 60 | s |
| `reward_version` | 2 | - |
| `e_floor` | 0.0001 | mol/L |
| `e_tol` | 0 | mol/L |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 1.8e+07 | reward |
| `restart_steps` | 240 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |

