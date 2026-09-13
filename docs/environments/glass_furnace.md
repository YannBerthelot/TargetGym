# Glass Furnace

<p align="center"><img src="../../videos/glass_furnace/pid_output.gif" width="480px"/></p>

Glass furnace (float-glass process) — regenerative end-port fired furnace.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | crown temperature (K) |
| Episode length | 1600 steps (48000 s at 30 s per step) |
| Import | `from target_gym import GlassFurnace, GlassFurnaceParams` |
| Cite as | `glass_furnace-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel rate | -1 | 1 |

## Observation space

5 values. Indices (0,) carry the tracked variable(s) that the
reward scores.

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract: the reward is
`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in
`[0, 1]`, and reaches 1 only while the target is held exactly. See
[Reward shaping](../reward-shaping.md).

## Starting state

`reset` samples the initial condition and the target; state has 16 fields.

## Episode end

**Termination.** See `check_is_terminal`.

**Truncation.** After 1600 steps.

## Baselines

A tuned PID ships with this environment.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 30 |
| `max_steps_in_episode` | 1600 |
| `LHV` | 5e+07 |
| `AFR` | 17 |
| `excess_air` | 0.1 |
| `c_p_air` | 1150 |
| `c_p_gas` | 1200 |
| `fuel_min` | 0.513 |
| `fuel_max` | 0.698 |
| `flame_rad_fraction` | 0.55 |
| `C_regen_node` | 3e+07 |
| `eps_regen_node` | 0.8 |
| `reversal_period` | 1500 |
| `reversal_dead_time` | 40 |
| … | 55 more, see the params dataclass |

