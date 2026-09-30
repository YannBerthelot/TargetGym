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

`reset` samples the initial condition and the target; state has 12 fields.

## Episode end

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 864 steps.

## Baselines

Recorded over 10 seeds of the 864-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2.105e+04 | 24.37 |
| MPC | -1186 | 1.373 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

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

