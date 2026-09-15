# Altitude hold

<p align="center"><img src="../../videos/plane/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((10,))` |
| Tracked variable(s) | altitude (m) |
| Episode length | 280 steps (280 s at 1 s per step) |
| Import | `from target_gym import Airplane2D, PlaneParams` |
| Cite as | `plane-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |
| 1 | stick | -1 | 1 |

## Observation space

10 values. Indices (1,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 17 fields.

## Episode end

**Termination.** Return True if the episode should terminate.

**Truncation.** After 280 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 280 |
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
| `cl_alpha` | 0.08786 |
| `cl0` | 0.2 |
| … | 53 more, see the params dataclass |

