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

Every environment in this suite scores on one contract. The reward is
minus the sum of three non-negative costs,
`-(tracking_cost + running_cost + failure_cost)`. The running cost
charges consumption and the failure cost charges trips out of the
operating envelope. The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 16 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1600 steps.

## Baselines

Recorded over 10 seeds of the 1600-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2163 | 1.352 |
| MPC | -438.9 | 0.2743 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

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
| … | 56 more, see the params dataclass |

