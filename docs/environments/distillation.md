# Distillation

<p align="center"><img src="../../videos/distillation/pid_output.gif" width="480px"/></p>

Binary distillation — Skogestad's "Column A".

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | yD (mole fraction) |
| Episode length | 200 steps (12000 s at 60 s per step) |
| Import | `from target_gym import DistillationColumn, DistillationParams` |
| Cite as | `distillation-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | L_raw | -1 | 1 |
| 1 | V_raw | -1 | 1 |

## Observation space

6 values. Indices (0,) carry the tracked variable(s) that the
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

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 200 steps.

## Baselines

Recorded over 10 seeds of the 200-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5.245e+04 | 262.3 |
| MPC | -5979 | 29.9 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 1 | min |
| `max_steps_in_episode` | 200 | steps |
| `alpha` | 1.5 | - |
| `M_tray` | 0.5 | kmol |
| `M_drum` | 0.5 | kmol |
| `F` | 1 | kmol/min |
| `zF_nominal` | 0.5 | - |
| `qF` | 1 | - |
| `zF_noise_std` | 0.03 | - |
| `zF_min` | 0.4 | - |
| `zF_max` | 0.6 | - |
| `L_min` | 2.3 | kmol/min |
| `L_max` | 3.1 | kmol/min |
| `V_min` | 2.8 | kmol/min |
| `V_max` | 3.6 | kmol/min |
| `D_min` | 0.1 | kmol/min |
| `D_max` | 0.9 | kmol/min |
| `yD_floor` | 0.9 | - |
| `xB_ceiling` | 0.1 | - |
| `tracking_band` | 0.02 | - |
| `precision_floor` | 0.0001 | - |
| `boilup_cost_weight` | 0 | - |
| `time_unit_seconds` | 60 | s |
| `reward_version` | 2 | - |
| `e_floor_top` | 0.0001 | - |
| `e_floor_bottom` | 0.0001 | - |
| `e_tol` | 0 | - |
| `tracking_exponent` | 2 | - |
| `c_hold` | 3.282 | kmol/min |
| `running_weight` | 1 | - |
| `failure_cost` | 3.61e+06 | reward |
| `restart_steps` | 240 | steps |
| `rho_floor_tracking` | 8.53057e-05 | reward |
| `rho_floor` | 8.53057e-05 | reward |

