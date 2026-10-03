# CSTR

<p align="center"><img src="../../videos/cstr/pid_output.gif" width="480px"/></p>

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((3,))` |
| Tracked variable(s) | C_a (mol/L) |
| Episode length | 100 steps (1500 s at 15 s per step) |
| Import | `from target_gym import CSTR, CSTRParams` |
| Cite as | `cstr-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | coolant temperature | -1 | 1 |

## Observation space

3 values. Indices (0,) carry the tracked variable(s) that the
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

`reset` samples the initial condition and the target; state has 5 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 100 steps.

## Baselines

Recorded over 10 seeds of the 100-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -6.319e+05 | 6319 |
| MPC | -5.752e+05 | 5752 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 0.25 |
| `max_steps_in_episode` | 100 |
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
| `T_c_max` | 302 |
| `T_c_min` | 295 |
| … | 15 more, see the params dataclass |

