# Reactor

<p align="center"><img src="../../videos/reactor/pid_output.gif" width="480px"/></p>

Run a day of a pressurized water reactor at the start of its fuel cycle, with control rods and soluble boron, knowing the day's load plan: hold the average coolant temperature within 1.6 K of its program on either side and the axial offset within 0.05 of its reference, keep the rods above their insertion limit, and dilute for the fuel's burnup, at the least cost in lost output, rod wear and effluent. Six scenarios weigh baseload days against five kinds of load-follow day (REALITY.md sections 6 and 7).

```python
import target_gym as tg

env, params = tg.make("reactor")
```

| | |
|---|---|
| Action space | `Box((2,))`, all actions in [-1, 1] |
| Observation space | `Box((11,))` |
| Tracked variable(s) | Tavg (K) |
| Episode length | 2880 steps of 30 s (24 h) |
| Cite as | `reactor-v8` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/reactor/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | rod demand | -1 | 1 |
| 1 | boron flow | -1 | 1 |

## Observation space

11 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[4]` against `obs[5]`.

| # | meaning |
|---|---|
| 0 | `n`, power relative to rated |
| 1 | hot-leg temperature (K) |
| 2 | rod position, +1 fully out, -1 fully in |
| 3 | turbine load |
| 4 | `T_avg` (K) |
| 5 | `T_ref` (K) |
| 6 | `delta_I`, top minus bottom, a fraction of rated |
| 7 | boron in the core (ppm), as the boronmeter read it 20 min ago |
| 8 | hours until the planned ramp down starts, 0 once it has |
| 9 | hours until the planned ramp up starts, 0 once it has |
| 10 | the recent power the starting xenon is balanced with (rule 16: operators know their power history) |

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

`reset` samples the initial condition and the target; state has 37 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 2880 steps.

## Baselines

Recorded over 10 seeds of the 864-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -5736 | 6.639 |
| MPC | -1205 | 1.395 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("reactor", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 5 | s |
| `max_steps_in_episode` | 2880 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `Lambda_gen` | 0.0001 | s |
| `alpha_fuel` | -3e-05 | 1/K |
| `alpha_coolant` | -0.00012 | 1/K |
| `alpha_tilt` | -0.0003 | 1/K |
| `T_fuel_ref` | 1051.4 | K |
| `T_coolant_ref` | 580.2 | K |
| `rho_ext_min` | -0.015 | - |
| `rho_ext_max` | 0 | - |
| `rod_speed` | 0.00526316 | 1/s |
| `rod_stroke_steps` | 228 | steps |
| `rod_park` | 0.0131579 | - |
| `boron_ref` | 1690 | ppm |
| `boron_worth` | -8e-05 | 1/ppm |
| `dilution_flow` | 10 | kg/s |
| `boration_flow` | 3 | kg/s |
| `boric_acid_ppm` | 7800 | ppm |
| `primary_mass` | 250000 | kg |
| `sigma_x` | 2.72e-18 | cm^2 |
| `D_diffusion` | 4.99301 | cm |
| `alpha_power` | 3.64e-16 | cm^2 s |
| `Sigma_f` | 0.14 | 1/cm |
| `nu_Sigma_f` | 0.3904 | 1/cm |
| `gamma_x` | 0.0046 | - |
| `gamma_I` | 0.061 | - |
| `H_core` | 365.8 | cm |
| `Phi_full` | 2.69333e+13 | 1/cm^2/s |
| `P_thermal_ref` | 3e+09 | W |
| `C_fuel` | 3.3e+07 | J/K |
| `C_coolant` | 4.7e+08 | J/K |
| `C_cold` | 9.3e+08 | J/K |
| `UA` | 6.6e+06 | W/K |
| `m_dot_cp` | 9e+07 | W/K |
| `UA_sg` | 1.36364e+08 | W/K |
| `C_sg` | 1e+09 | J/K |
| `T_avg_no_load` | 564.8 | K |
| `T_avg_full_load` | 580.2 | K |
| `sigma_phi0` | 5.75372e-05 | 1/s |
| `gamma_ratio` | 0.0754098 | - |
| `rho_Xe_full` | 0.025 | - |
| `n_min` | 0.01 | - |
| `precision_floor` | 0.05 | K |
| `n_max` | 1.09 | - |
| `T_fuel_max` | 1473 | K |
| `T_fuel_min` | 500 | K |
| `T_coolant_max` | 610 | K |
| `T_avg_min` | 556 | K |
| `floor_allowed` | 900 | s |
| `rod_motion_weight` | 0.02 | - |
| `reward_band` | 0.03 | - |
| `load_min` | 0.5 | - |
| `load_ramp_rate` | 0.000333333 | 1/s |
| `demand_theta` | 0.00333333 | 1/s |
| `demand_sigma` | 0.000408 | 1/s^0.5 |
| `P_electric_GW` | 1 | GW |
| `spot_price_per_MWh` | 80 | $/MWh |
| `imbalance_factor` | 3 | - |
| `reward_version` | 2 | - |
| `e_floor` | 0.05 | K |
| `e_tol` | 1.6 | K |
| `tracking_exponent` | 1 | - |
| `imbalance_multiple` | 0.002875 | 1/K |
| `rod_step_price` | 1 | $/step |
| `ao_band` | 0.05 | - |
| `ao_ref` | 0 | - |
| `ao_allowed` | 3600 | s |
| `derate_to` | 0.5 |  |
| `derate_price_per_MWh` | 80 | $/MWh |
| `ril_full` | 0.22 |  |
| `ril_half` | 0.78 |  |
| `ril_allowed` | 7200 | s |
| `dnb_margin` | 2.83333 | K |
| `dnb_allowed` | 7200 | s |
| `effluent_price` | 0.0628 | $/kg |
| `failure_cost` | 666.667 | reward |
| `restart_steps` | 5760 | steps |
| `standby_steps` | 1800 | steps |
| `rho_floor_tracking` | 0.0958333 | reward |
| `rho_floor` | 0.0958333 | reward |

