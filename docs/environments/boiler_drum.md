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

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. High level carries water to the turbine; low level uncovers the tubes. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 400 steps.

## Baselines

Recorded over 10 seeds of the 400-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2.568e+05 | 641.9 |
| MPC | -1.696e+04 | 42.4 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 2 |
| `max_steps_in_episode` | 400 |
| `reward_version` | 2 |
| `e_floor_level` | 0.00267 |
| `e_floor_pressure` | 0.05 |
| `e_tol` | 0 |
| `tracking_exponent` | 2 |
| `c_hold` | 1.566e+08 |
| `running_weight` | 1 |
| `failure_cost` | 440734 |
| `restart_steps` | 7200 |
| `rho_floor_tracking` | 1.32036 |
| `rho_floor` | 1.32036 |
| `V_t` | 88 |
| … | 31 more, see the params dataclass |

