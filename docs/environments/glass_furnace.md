# Glass Furnace

<p align="center"><img src="../../videos/glass_furnace/pid_output.gif" width="480px"/></p>

Hold the crown temperature of a regenerative float-glass furnace on a slowly walking setpoint by setting the fuel flow. The crown is read through a lagging thermocouple, the glass and regenerator temperatures are hidden, the firing side reverses on a fixed cycle, and the furnace responds over hours, so a few instruments have to steer a large, slow plant.

```python
import target_gym as tg

env, params = tg.make("glass_furnace")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | crown temperature (K) |
| Episode length | 1600 steps of 30 s (13.3 h) |
| Cite as | `glass_furnace-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/glass_furnace/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | fuel rate | -1 | 1 |

## Observation space

5 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[4]`.

| # | meaning |
|---|---|
| 0 | crown temperature as the thermocouple reads it (°C); it lags the true crown temperature the reward scores |
| 1 | combustion air preheat temperature (°C) |
| 2 | fuel flow, percent of maximum |
| 3 | regenerator phase: 0 while regenerator A preheats the air, 1 while B does |
| 4 | target crown temperature (°C) |

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

`reset` samples the initial condition and the target; state has 16 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1600 steps.

## Baselines

Recorded over 10 seeds of the 1600-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -2163 | 1.352 |
| MPC | -438.9 | 0.2743 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("glass_furnace", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 30 | s |
| `max_steps_in_episode` | 1600 | steps |
| `LHV` | 5e+07 | J/kg |
| `AFR` | 17 | - |
| `excess_air` | 0.1 | - |
| `c_p_air` | 1150 | J/(kg K) |
| `c_p_gas` | 1200 | J/(kg K) |
| `fuel_min` | 0.513 | kg/s |
| `fuel_max` | 0.698 | kg/s |
| `flame_rad_fraction` | 0.55 | - |
| `C_regen_node` | 3e+07 | J/K |
| `eps_regen_node` | 0.8 | - |
| `reversal_period` | 1500 | s |
| `reversal_dead_time` | 40 | s |
| `reversal_firing_floor` | 0.05 | - |
| `tau_thermocouple` | 120 | s |
| `charge_period` | 300 | s |
| `charge_duty` | 0.3 | - |
| `charge_mass_jitter` | 0.15 | - |
| `U_regen` | 0.3 | W/(m^2 K) |
| `A_regen` | 300 | m^2 |
| `m_pull` | 5.79 | kg/s |
| `batch_yield` | 0.83 | - |
| `T_batch_in` | 25 | degC |
| `dH_fusion` | 800000 | J/kg |
| `m_batch_full` | 39000 | kg |
| `batch_shield` | 0.85 | - |
| `m_pull_noise_std` | 0.4 | kg/s |
| `C_gas` | 240000 | J/K |
| `C_crown` | 2.5e+08 | J/K |
| `C_melt` | 7.8e+08 | J/K |
| `C_work` | 7.8e+08 | J/K |
| `c_p_glass_a` | 900 | J/(kg K) |
| `c_p_glass_b` | 0.35 | J/(kg K^2) |
| `eps_rad` | 0.8 | - |
| `A_crown` | 200 | m^2 |
| `A_melt` | 200 | m^2 |
| `A_work` | 10 | m^2 |
| `h_conv` | 30 | W/(m^2 K) |
| `U_wall` | 1 | W/(m^2 K) |
| `A_wall_crown` | 200 | m^2 |
| `A_wall_melt` | 200 | m^2 |
| `A_wall_work` | 200 | m^2 |
| `UA_work_cooling` | 3000 | W/K |
| `T_ambient` | 25 | degC |
| `T_crown_min` | 1427 | degC |
| `tracking_band` | 10 | K |
| `precision_floor` | 1 | K |
| `T_crown_max` | 1677 | degC |
| `T_glass_min` | 900 | degC |
| `T_glass_max` | 1727 | degC |
| `fuel_cost_weight` | 0 | - |
| `target_T_crown_trim` | 6 | K |
| `initial_T_crown_offset` | 4 | K |
| `initial_T_gas_offset` | 31 | K |
| `initial_T_melt_offset` | -35 | K |
| `initial_T_work_offset` | -151 | K |
| `initial_T_regen_hot` | 1383 | degC |
| `initial_T_regen_cold` | 544 | degC |
| `initial_m_batch` | 10753 | kg |
| `reward_version` | 2 | - |
| `e_floor` | 1 | K |
| `e_tol` | 0 | K |
| `tracking_exponent` | 2 | - |
| `c_hold` | 0.59 | kg/s |
| `running_weight` | 1 | - |
| `failure_cost` | 66978 | reward |
| `restart_steps` | 40320 | steps |
| `rho_floor_tracking` | 0.00376996 | reward |
| `rho_floor` | 0.00376996 | reward |

