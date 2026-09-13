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

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 7 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 200 steps.

## Baselines

A tuned PID ships with this environment.

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
| … | 19 more, see the params dataclass |

