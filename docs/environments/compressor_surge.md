# Compressor Surge

<p align="center"><img src="../../videos/compressor_surge/pid_output.gif" width="480px"/></p>

Compressor surge: a Greitzer compression system feeding a header.

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | header pressure (kPa) |
| Episode length | 1200 steps (120 s at 0.1 s per step) |
| Import | `from target_gym import CompressorSurge, CompressorSurgeParams` |
| Cite as | `compressor_surge-v2` |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | speed command | -1 | 1 |
| 1 | recycle command | -1 | 1 |

## Observation space

7 values. Indices (0,) carry the tracked variable(s) that the
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

**Termination.** None. `terminated` is always false, and leaving the operating envelope trips the plant instead. The step that leaves the envelope is charged the trip cost and sets `info["tripped"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1200 steps.

## Baselines

Recorded over 10 seeds of the 1200-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2.148e+06 | 1790 |
| MPC | -1.937e+05 | 161.4 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 0.1 | s |
| `max_steps_in_episode` | 1200 | steps |
| `P01` | 101325 | Pa |
| `T01` | 288.15 | K |
| `R_gas` | 287.05 | J/(kg K) |
| `gamma` | 1.4 | - |
| `A_c` | 0.1 | m^2 |
| `L_c` | 4 | m |
| `V_p` | 15 | m^3 |
| `U_r` | 200 | m/s |
| `psi_c0` | 0.3 | - |
| `H` | 0.18 | - |
| `W` | 0.25 | - |
| `phi_join` | 0.78 | - |
| `k_d` | 0.12 | kg/(s Pa^0.5) |
| `dp_out` | 10000 | Pa |
| `k_r` | 0.15 | kg/(s Pa^0.5) |
| `check_width` | 50 | Pa |
| `N_min` | 0.7 | - |
| `N_max` | 1.05 | - |
| `drive_rate` | 0.03 | 1/s |
| `tau_N` | 1 | s |
| `valve_open_rate` | 0.5 | 1/s |
| `valve_close_rate` | 0.1 | 1/s |
| `setpoint_block_steps` | 300 | steps |
| `demand_block_steps` | 200 | steps |
| `demand_ramp_s` | 5 | s |
| `demand_sigma` | 0.02 | - |
| `demand_theta` | 0.2 | 1/s |
| `demand_dev_reset_clip` | 3 | - |
| `N_reset` | 1 | - |
| `x_reset` | 0.35 | - |
| `reset_newton_iters` | 6 | - |
| `soft_min_temperature` | 0.001 | - |
| `time_unit_seconds` | 1 | s |
| `reward_version` | 2 | - |
| `e_floor` | 0.0275 | kPa |
| `e_tol` | 0 | kPa |
| `tracking_exponent` | 2 | - |
| `dp_error_max` | 20.68 | kPa |
| `failure_cost` | 1.13101e+06 | reward |
| `restart_steps` | 9000 | steps |
| `c_hold` | 62270 | W |
| `running_weight` | 1 | - |
| `precision_floor` | 0.0275 | kPa |
| `rho_floor_tracking` | 1.40284e-05 | reward |
| `rho_floor` | 1.40284e-05 | reward |

