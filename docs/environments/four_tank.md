# Four-tank

<p align="center"><img src="../../videos/four_tank/pid_output.gif" width="480px"/></p>

Hold the levels of the two lower tanks of Johansson's quadruple-tank process with two pumps. Each pump sends most of its flow to the upper tank on the other side, which drains into the opposite lower tank, so pairing each pump with the tank below it is the wrong move and the response is non-minimum phase. All four levels are measured: the difficulty is the coupling, not what is seen.

```python
import target_gym as tg

env, params = tg.make("four_tank")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | h1 (m), h2 (m) |
| Episode length | 500 steps of 1 s (8.33 min) |
| Cite as | `four_tank-v2` |
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
| 0 | `h1`, level of lower tank 1 (m) |
| 1 | `h2`, level of lower tank 2 (m) |
| 2 | `h3`, level of upper tank 3 (m) |
| 3 | `h4`, level of upper tank 4 (m) |
| 4 | target `h1` (m) |
| 5 | target `h2` (m) |

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

`reset` samples the initial condition and the target; state has 9 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 500 steps.

## Baselines

Recorded over 10 seeds of the 500-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -4.158e+05 | 831.6 |
| MPC | -1.864e+05 | 372.8 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("four_tank", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 500 | steps |
| `g` | 9.81 | m/s^2 |
| `gamma1` | 0.2 | - |
| `gamma2` | 0.2 | - |
| `k1` | 0.00085 | m^3/(V*s) |
| `k2` | 0.00095 | m^3/(V*s) |
| `a1` | 0.0035 | m^2 |
| `a2` | 0.003 | m^2 |
| `a3` | 0.002 | m^2 |
| `a4` | 0.0025 | m^2 |
| `A1` | 1 | m^2 |
| `A2` | 1 | m^2 |
| `A3` | 1 | m^2 |
| `A4` | 1 | m^2 |
| `v_min` | 0 | V |
| `v_max` | 10 | V |
| `h_min` | 0.05 | m |
| `precision_floor` | 0.001 | m |
| `h_max` | 1.5 | m |
| `tracking_band` | 0.05 | m |
| `reward_version` | 2 | - |
| `e_floor` | 0.001 | m |
| `e_tol` | 0 | m |
| `tracking_exponent` | 2 | - |
| `failure_cost` | 292042 | reward |
| `restart_steps` | 600 | steps |
| `rho_floor_tracking` | 0 | reward |
| `rho_floor` | 0 | reward |

