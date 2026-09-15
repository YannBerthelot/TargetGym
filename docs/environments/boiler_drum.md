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
| Cite as | `boiler_drum-v2` |

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

See the environment's `compute_reward`.

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
| `reward_version` | 2 |
| `e_floor_level` | 0.00267 |
| `e_floor_pressure` | 0.0283 |
| `e_tol` | 0 |
| `tracking_exponent` | 2 |
| `c_hold` | 1.566e+08 |
| `running_weight` | 1 |
| `failure_cost` | 4e+06 |
| `restart_steps` | 7200 |
| `rho_floor_tracking` | 2 |
| `rho_floor` | 2 |
| `V_t` | 88 |
| … | 31 more, see the params dataclass |

