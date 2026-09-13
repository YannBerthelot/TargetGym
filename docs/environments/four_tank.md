# Four-tank

<p align="center"><img src="../../videos/four_tank/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | h1 (m), h2 (m) |
| Episode length | 500 steps (500 s at 1 s per step) |
| Import | `from target_gym import FourTank, FourTankParams` |
| Cite as | `four_tank-v1` |

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

Mean of the two level-tracking scores.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 9 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 500 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 500 |
| `g` | 9.81 |
| `gamma1` | 0.2 |
| `gamma2` | 0.2 |
| `k1` | 0.00085 |
| `k2` | 0.00095 |
| `a1` | 0.0035 |
| `a2` | 0.003 |
| `a3` | 0.002 |
| `a4` | 0.0025 |
| `A1` | 1 |
| `A2` | 1 |
| `A3` | 1 |
| … | 7 more, see the params dataclass |

