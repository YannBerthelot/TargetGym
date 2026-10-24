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

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 864 steps.

## Baselines

Recorded over 10 seeds of the 864-step episode (see [Baselines](../baselines.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2.105e+04 | 24.37 |
| MPC | -1186 | 1.373 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

| parameter | default | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 864 | steps |
| `Lambda_gen` | 0.0001 | s |
| `alpha_fuel` | -3e-05 | 1/K |
| `alpha_coolant` | -5e-05 | 1/K |
| `T_fuel_ref` | 900 | K |
| `T_coolant_ref` | 580 | K |
| `rho_ext_min` | -0.01 | - |
| `rho_ext_max` | 0.005 | - |
| `rod_speed_insert` | 0.0004 | 1/s |
| `rod_speed_withdraw` | 0.0002 | 1/s |
| `P_thermal_ref` | 3e+09 | W |
| `C_fuel` | 3.3e+07 | J/K |
| `C_coolant` | 7e+07 | J/K |
| `UA` | 6.6e+06 | W/K |
| `m_dot_cp` | 9e+07 | W/K |
| `T_inlet` | 555 | K |
| `sigma_phi0` | 7.95e-05 | 1/s |
| `gamma_ratio` | 0.049 | - |
| `rho_Xe_full` | 0.025 | - |
| `n_min` | 0.01 | - |
| `precision_floor` | 0.0001 | - |
| `n_max` | 1.5 | - |
| `T_fuel_max` | 1473 | K |
| `T_fuel_min` | 500 | K |
| `T_coolant_max` | 620 | K |
| `T_coolant_min` | 500 | K |
| `rod_motion_weight` | 0.02 | - |
| `reward_band` | 0.03 | - |
| `demand_theta` | 0.00015 | 1/s |
| `demand_sigma` | 0.0025 | 1/s^0.5 |
| `P_electric_GW` | 1 | GW |
| `spot_price_per_MWh` | 80 | $/MWh |
| `imbalance_factor` | 3 | - |
| `initial_T_fuel` | 900 | K |
| `initial_T_coolant` | 580 | K |
| `reward_version` | 2 | - |
| `e_floor` | 0.00451 | - |
| `e_tol` | 0 | - |
| `tracking_exponent` | 1 | - |
| `imbalance_multiple` | 1.25 | - |
| `rod_wear_weight` | 1 | - |
| `failure_cost` | 666.667 | reward |
| `restart_steps` | 17280 | steps |
| `rho_floor_tracking` | 1.25278 | reward |
| `rho_floor` | 1.25278 | reward |

