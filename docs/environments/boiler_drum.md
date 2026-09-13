# Boiler Drum

<p align="center"><img src="../../videos/boiler_drum/pid_output.gif" width="480px"/></p>

Boiler drum — natural-circulation drum boiler with shrink-and-swell.

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | drum level (m), drum pressure (bar) |
| Episode length | 400 steps (800 s at 2 s per step) |
| Import | `from target_gym import BoilerDrum, BoilerDrumParams` |
| Cite as | `boiler_drum-v1` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel | -1 | 1 |
| 1 | feedwater | -1 | 1 |

## Observation space

7 values. Indices (0, 1) carry the tracked variable(s) that the
reward scores.

## Rewards

Level and pressure tracking, minus fuel.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 11 fields.

## Episode end

**Termination.** High level carries water to the turbine; low level uncovers the tubes.

**Truncation.** After 400 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 2 |
| `max_steps_in_episode` | 400 |
| `V_t` | 88 |
| `V_d` | 40 |
| `V_r` | 37 |
| `V_dc` | 11 |
| `A_d` | 20 |
| `A_dc` | 0.355 |
| `L_r` | 11 |
| `m_metal` | 300000 |
| `C_metal` | 550 |
| `k_friction` | 25 |
| `tau_sr` | 8 |
| `T_d` | 15 |
| … | 20 more, see the params dataclass |

