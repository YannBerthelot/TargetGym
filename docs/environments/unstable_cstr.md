# Unstable CSTR

<p align="center"><img src="../../videos/unstable_cstr/pid_output.gif" width="480px"/></p>

Unstable CSTR: cstr's reactor held on its open-loop unstable middle branch.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((4,))` |
| Tracked variable(s) | C_a (mol/L) |
| Episode length | 1200 steps (3600 s at 3 s per step) |
| Import | `from target_gym import UnstableCSTR, UnstableCSTRParams` |
| Cite as | `unstable_cstr-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | coolant command | -1 | 1 |

## Observation space

4 values. Indices (0,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 7 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1200 steps.

## Baselines

Recorded over 10 seeds of the 1200-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -6.857e+07 | 5.714e+04 |
| MPC | -5.014e+06 | 4178 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.05 |
| `max_steps_in_episode` | 1200 |
| `q` | 100 |
| `V` | 100 |
| `rho` | 1000 |
| `C` | 0.239 |
| `deltaHr` | -50000 |
| `EA_over_R` | 8750 |
| `k0` | 7.2e+10 |
| `UA` | 50000 |
| `Ti` | 350 |
| `Caf` | 1 |
| `precision_floor` | 0.0001 |
| `time_unit_seconds` | 60 |
| … | 20 more, see the params dataclass |

