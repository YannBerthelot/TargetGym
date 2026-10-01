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

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 200 |
| `alpha` | 1.5 |
| `M_tray` | 0.5 |
| `M_drum` | 0.5 |
| `F` | 1 |
| `zF_nominal` | 0.5 |
| `qF` | 1 |
| `zF_noise_std` | 0.03 |
| `zF_min` | 0.4 |
| `zF_max` | 0.6 |
| `L_min` | 2.3 |
| `L_max` | 3.1 |
| `V_min` | 2.8 |
| … | 20 more, see the params dataclass |

