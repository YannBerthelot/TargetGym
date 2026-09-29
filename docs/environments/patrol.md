# Patrol - MARL formation

<p align="center"><img src="../../videos/patrol/pid_output.gif" width="480px"/></p>

Close-patrol (formation-keeping) environment: state, parameters and transition.

| | |
|---|---|
| Action space | `Box((3,))`, all actions in [-1, 1] |
| Observation space | `Box((26,))` |
| Tracked variable(s) | slot error (m) |
| Episode length | 200 steps (200 s at 1 s per step) |
| Import | `from target_gym import PlanePatrol, PatrolParams` |
| Cite as | `patrol-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |
| 2 | aileron | -1 | 1 |

## Observation space

26 values. Indices (24,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 12 fields.

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
| `gravity` | 9.81 |
| `initial_mass` | 73500 |
| `thrust_output_at_sea_level` | 240000 |
| `air_density_at_sea_level` | 1.225 |
| `frontal_surface` | 12.6 |
| `wings_surface` | 122.6 |
| `C_x0` | 0.095 |
| `C_z0` | 0.9 |
| `initial_fuel_quantity` | 19088 |
| `specific_fuel_consumption` | 0.0175 |
| `power_response_rate` | 0.05 |
| `stick_response_rate` | 0.9 |
| … | 69 more, see the params dataclass |

