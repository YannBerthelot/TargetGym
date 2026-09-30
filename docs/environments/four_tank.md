# Four-tank

<p align="center"><img src="../../videos/four_tank/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | h1 (m), h2 (m) |
| Episode length | 500 steps (500 s at 1 s per step) |
| Import | `from target_gym import FourTank, FourTankParams` |
| Cite as | `four_tank-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | pump 1 voltage | -1 | 1 |
| 1 | pump 2 voltage | -1 | 1 |

## Observation space

6 values. Indices (0, 1) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 9 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 500 steps.

## Baselines

Recorded over 10 seeds of the 500-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5.834e+05 | 1167 |
| MPC | -1.723e+05 | 344.5 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 1 |
| `max_steps_in_episode` | 500 |
| `g` | 9.81 |
| `gamma1` | 0.2 |
| `gamma2` | 0.2 |
| `k1` | 0.00085 |
| `k2` | 0.00095 |
| `a1` | 0.0035 |
| `a2` | 0.003 |
| `a3` | 0.002 |
| `a4` | 0.0025 |
| `A1` | 1 |
| `A2` | 1 |
| `A3` | 1 |
| … | 15 more, see the params dataclass |

