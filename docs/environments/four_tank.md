# Four-tank

<p align="center"><img src="../../videos/four_tank/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | h1 (m), h2 (m) |
| Episode length | 500 steps (500 s at 1 s per step) |
| Import | `from target_gym import FourTank, FourTankParams` |
| Cite as | `four_tank-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | pump 1 voltage | -1 | 1 |
| 1 | pump 2 voltage | -1 | 1 |

## Observation space

6 values. Indices (0, 1) carry the tracked variable(s) that the
reward scores.

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

| parameter | default | unit |
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

