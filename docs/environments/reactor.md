# Reactor

<p align="center"><img src="../../videos/reactor/pid_output.gif" width="480px"/></p>

Nuclear reactor — point-kinetics with delayed neutrons, xenon poisoning, thermal feedback, and rate-limited control rods.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((4,))` |
| Tracked variable(s) | neutron power (normalised) |
| Episode length | 864 steps (8640 s at 10 s per step) |
| Import | `from target_gym import Reactor, ReactorParams` |
| Cite as | `reactor-v3` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | rho_ext_norm | -1 | 1 |

## Observation space

4 values. Indices (0,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 13 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 864 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 864 |
| `Lambda_gen` | 0.0001 |
| `alpha_fuel` | -3e-05 |
| `alpha_coolant` | -5e-05 |
| `T_fuel_ref` | 900 |
| `T_coolant_ref` | 580 |
| `rho_ext_min` | -0.01 |
| `rho_ext_max` | 0.005 |
| `rod_speed_insert` | 0.0004 |
| `rod_speed_withdraw` | 0.0002 |
| `P_thermal_ref` | 3e+09 |
| `C_fuel` | 3.3e+07 |
| `C_coolant` | 7e+07 |
| … | 32 more, see the params dataclass |

