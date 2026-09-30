# Battery

<p align="center"><img src="../../videos/battery/pid_output.gif" width="480px"/></p>

Grid battery storage — equivalent-circuit Li-ion pack tracking a dispatch signal.

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | delivered power (MW) |
| Episode length | 360 steps (1800 s at 5 s per step) |
| Import | `from target_gym import GridBattery, BatteryParams` |
| Cite as | `battery-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |

## Observation space

5 values. Indices (3,) carry the tracked variable(s) that the
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

**Truncation.** After 360 steps.

## Baselines

Recorded over 10 seeds of the 360-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.244 | 0.003457 |
| MPC | -0.4304 | 0.001196 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default |
|---|---|
| `delta_t` | 5 |
| `max_steps_in_episode` | 360 |
| `energy_nominal` | 7.2e+09 |
| `power_max` | 1e+06 |
| `n_series` | 192 |
| `capacity_As` | 9e+06 |
| `R0` | 0.02 |
| `R1` | 0.01 |
| `C1` | 20000 |
| `ocv_a` | 3 |
| `ocv_b` | 1.15 |
| `ocv_c` | 0.3 |
| `ocv_d` | 12 |
| `ocv_e` | 0.05 |
| … | 35 more, see the params dataclass |

