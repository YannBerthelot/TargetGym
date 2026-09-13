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

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 7 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 300 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 5 |
| `max_steps_in_episode` | 300 |
| `V` | 2900 |
| `q1` | 16.6 |
| `Wa1` | 0.003 |
| `Wb1` | 0 |
| `q2_nominal` | 0.55 |
| `q2_noise_std` | 0.35 |
| `q2_min` | 0 |
| `q2_max` | 2.5 |
| `Wa2` | -0.03 |
| `Wb2` | 0.03 |
| `q3_min` | 10 |
| `q3_max` | 22 |
| … | 18 more, see the params dataclass |

