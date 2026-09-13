# Cement Kiln

<p align="center"><img src="../../videos/cement_kiln/pid_output.gif" width="480px"/></p>

Cement rotary kiln — 1-D axial model with counter-current gas.

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((8,))` |
| Tracked variable(s) | discharge free lime (%) |
| Episode length | 700 steps (21000 s at 30 s per step) |
| Import | `from target_gym import CementKiln, CementKilnParams` |
| Cite as | `cement_kiln-v1` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel | -1 | 1 |
| 1 | kiln_speed | -1 | 1 |

## Observation space

8 values. Indices (0,) carry the tracked variable(s) that the
reward scores.

## Rewards

Free-lime tracking minus fuel.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 11 fields.

## Episode end

**Termination.** Both failure modes are irrecoverable in practice.

**Truncation.** After 700 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 30 |
| `max_steps_in_episode` | 700 |
| `diameter` | 4 |
| `length` | 60 |
| `slope` | 0.035 |
| `n_zones` | 16 |
| `w_bed_gas` | 2.906 |
| `w_wall_gas` | 9.313 |
| `w_wall_bed` | 3.253 |
| `refractory_thickness` | 0.2 |
| `rho_refractory` | 2200 |
| `cp_refractory` | 900 |
| `U_shell` | 4 |
| `raw_meal_nominal` | 53.82 |
| … | 34 more, see the params dataclass |

