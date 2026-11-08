# Reactor

<p align="center"><img src="../../videos/reactor/pid_output.gif" width="480px"/></p>

Hold a 3 GW pressurised-water reactor's average coolant temperature on its program through a day of load follow, using the control rods. The turbine goes from full load to half and back at 5 %/min, power follows it by itself, and xenon moves the rods' balance by hundreds of pcm. Rod travel wears the mechanism, and costs are in dollars.

```python
import target_gym as tg

env, params = tg.make("reactor")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((6,))` |
| Tracked variable(s) | Tavg (K) |
| Episode length | 1440 steps of 30 s (12 h) |
| Cite as | `reactor-v4` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/reactor/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | rho_ext_norm | -1 | 1 |

## Observation space

6 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[4]` against `obs[5]`.

| # | meaning |
|---|---|
| 0 | `n`, neutron power relative to nominal (1 is full power) |
| 1 | hot-leg temperature (K) |
| 2 | rod position, normalised to [-1, 1] |
| 3 | turbine load, relative to rated |
| 4 | `T_avg`, the mean of the hot and cold legs (K) |
| 5 | `T_ref`, the program at this load (K) |

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

`reset` samples the initial condition and the target; state has 22 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1440 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("reactor", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 5 | s |
| `max_steps_in_episode` | 1440 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `Lambda_gen` | 0.0001 | s |
| `alpha_fuel` | -3e-05 | 1/K |
| `alpha_coolant` | -0.0003 | 1/K |
| `T_fuel_ref` | 1064 | K |
| `T_coolant_ref` | 580.2 | K |
| `rho_ext_min` | -0.01 | - |
| `rho_ext_max` | 0.005 | - |
| `rod_speed` | 7.89474e-05 | 1/s |
| `rod_stroke_steps` | 228 | steps |
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
| `sigma_phi0` | 7.95e-05 | 1/s |
| `gamma_ratio` | 0.049 | - |
| `rho_Xe_full` | 0.025 | - |
| `n_min` | 0.01 | - |
| `precision_floor` | 0.05 | K |
| `n_max` | 1.09 | - |
| `T_fuel_max` | 1473 | K |
| `T_fuel_min` | 500 | K |
| `T_coolant_max` | 620 | K |
| `T_avg_min` | 556 | K |
| `rod_motion_weight` | 0.02 | - |
| `reward_band` | 0.03 | - |
| `load_min` | 0.5 | - |
| `load_ramp_rate` | 0.000833333 | 1/s |
| `demand_theta` | 0.00333333 | 1/s |
| `demand_sigma` | 0.000408 | 1/s^0.5 |
| `P_electric_GW` | 1 | GW |
| `spot_price_per_MWh` | 80 | $/MWh |
| `imbalance_factor` | 3 | - |
| `reward_version` | 2 | - |
| `e_floor` | 0.05 | K |
| `e_tol` | 0 | K |
| `tracking_exponent` | 1 | - |
| `imbalance_multiple` | 0.002875 | 1/K |
| `rod_step_price` | 1 | $/step |
| `failure_cost` | 87.34 | reward |
| `restart_steps` | 5760 | steps |
| `rho_floor_tracking` | 0.0958333 | reward |
| `rho_floor` | 0.0958333 | reward |

