# Building HVAC

<p align="center"><img src="../../videos/hvac/pid_output.gif" width="480px"/></p>

Building HVAC — single thermal zone, ISO 13790 5R1C reduced-order model.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | zone air temperature (deg C) |
| Episode length | 720 steps (648000 s at 900 s per step) |
| Import | `from target_gym import BuildingHVAC, HVACParams` |
| Cite as | `hvac-v1` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | commanded heating power | -1 | 1 |

## Observation space

7 values. Indices (0,) carry the tracked variable(s) that the
reward scores.

## Rewards

Comfort tracking minus energy use.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 10 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 720 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 900 |
| `max_steps_in_episode` | 720 |
| `A_floor` | 150 |
| `lambda_at` | 4.5 |
| `f_class` | 2.5 |
| `cm_per_area` | 165000 |
| `h_is` | 3.45 |
| `h_ms` | 9.1 |
| `A_wall` | 120 |
| `U_wall` | 0.28 |
| `A_roof` | 150 |
| `U_roof` | 0.18 |
| `A_window` | 25 |
| `U_window` | 1.3 |
| … | 20 more, see the params dataclass |

