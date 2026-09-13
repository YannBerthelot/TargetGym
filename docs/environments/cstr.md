# CSTR

<p align="center"><img src="../../videos/cstr/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((3,))` |
| Tracked variable(s) | C_a (mol/L) |
| Episode length | 100 steps (1500 s at 15 s per step) |
| Import | `from target_gym import CSTR, CSTRParams` |
| Cite as | `cstr-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | coolant temperature | -1 | 1 |

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

`reset` samples the initial condition and the target; state has 5 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 100 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.25 |
| `max_steps_in_episode` | 100 |
| `q` | 100 |
| `V` | 100 |
| `rho` | 1000 |
| `C` | 0.239 |
| `deltaHr` | -50000 |
| `EA_over_R` | 8750 |
| `k0` | 7.2e+10 |
| `UA` | 50000 |
| `Ti` | 350 |
| `Caf` | 1 |
| `T_c_max` | 302 |
| `T_c_min` | 295 |
| … | 14 more, see the params dataclass |

