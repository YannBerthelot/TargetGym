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
| Cite as | `cement_kiln-v2` |

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

`reset` samples the initial condition and the target; state has 11 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Both failure modes are irrecoverable in practice. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 700 steps.

## Baselines

Recorded over 10 seeds of the 700-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -4458 | 6.368 |
| MPC | -1194 | 1.705 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 30 |
| `max_steps_in_episode` | 700 |
| `reward_version` | 2 |
| `e_floor` | 0.0005 |
| `e_tol` | 0 |
| `tracking_exponent` | 2 |
| `c_hold` | 1.824 |
| `running_weight` | 1 |
| `failure_cost` | 14112 |
| `restart_steps` | 2880 |
| `rho_floor_tracking` | 0.467856 |
| `rho_floor` | 0.467856 |
| `diameter` | 4 |
| `length` | 60 |
| … | 44 more, see the params dataclass |

