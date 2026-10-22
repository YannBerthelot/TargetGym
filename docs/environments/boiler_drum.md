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

| parameter | default | unit |
|---|---|---|
| `delta_t` | 2 | s |
| `max_steps_in_episode` | 400 | steps |
| `reward_version` | 2 | - |
| `e_floor_level` | 0.00267 | m |
| `e_floor_pressure` | 0.05 | bar |
| `e_tol` | 0 | m or bar |
| `tracking_exponent` | 2 | - |
| `c_hold` | 1.566e+08 | W |
| `running_weight` | 1 | - |
| `failure_cost` | 440734 | reward |
| `restart_steps` | 7200 | steps |
| `rho_floor_tracking` | 1.32036 | reward |
| `rho_floor` | 1.32036 | reward |
| `V_t` | 88 | m^3 |
| `V_d` | 40 | m^3 |
| `V_r` | 37 | m^3 |
| `V_dc` | 11 | m^3 |
| `A_d` | 20 | m^2 |
| `A_dc` | 0.355 | m^2 |
| `L_r` | 11 | m |
| `m_metal` | 300000 | kg |
| `C_metal` | 550 | J/(kg K) |
| `k_friction` | 25 | - |
| `tau_sr` | 8 | s |
| `T_d` | 15 | s |
| `f_carry` | 0.1 | - |
| `f_cd` | 0.25 | - |
| `c_pw` | 5700 | J/(kg K) |
| `gravity` | 9.81 | m/s^2 |
| `h_feedwater` | 1.037e+06 | J/kg |
| `Q_max` | 2.4e+08 | W |
| `q_feed_max` | 200 | kg/s |
| `Q_nominal` | 1.6e+08 | W |
| `q_steam_nominal` | 93.35 | kg/s |
| `p_nominal` | 85 | bar |
| `V_wt_nominal` | 58 | m^3 |
| `q_steam_noise_std` | 8 | kg/s |
| `level_band` | 0.1 | m |
| `pressure_band` | 2 | bar |
| `level_trip` | 0.25 | m |
| `pressure_min` | 65 | bar |
| `level_precision_floor` | 0.001 | m |
| `pressure_precision_floor` | 0.05 | bar |
| `pressure_max` | 105 | bar |
| `fuel_weight` | 0 | - |

