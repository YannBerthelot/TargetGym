# pH neutralisation

<p align="center"><img src="../../videos/ph_neutralization/pid_output.gif" width="480px"/></p>

pH neutralisation — CSTR with acid, buffer and base streams.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((3,))` |
| Tracked variable(s) | pH |
| Episode length | 300 steps (1500 s at 5 s per step) |
| Import | `from target_gym import PHNeutralization, PHParams` |
| Cite as | `ph_neutralization-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | base flow | -1 | 1 |

## Observation space

3 values. Indices (0,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 7 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 300 steps.

## Baselines

Recorded over 10 seeds of the 300-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.009e+05 | 336.3 |
| MPC | -2.736e+04 | 91.22 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
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

