# Battery

<p align="center"><img src="../../videos/battery/pid_output.gif" width="480px"/></p>

Follow a grid dispatch schedule, one setpoint per 5-minute market block, with a 1 MW / 2 MWh lithium-ion battery. Every MWh moved wears the pack, and the running cost prices that wear in dollars at every step, so tracking is a trade against capacity fade. Leaving the 5 to 95 % charge window trips the pack, though exact tracking stays inside it over a scored episode.

```python
import target_gym as tg

env, params = tg.make("battery")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((5,))` |
| Tracked variable(s) | delivered power (MW) |
| Episode length | 360 steps of 5 s (30 min) |
| Cite as | `battery-v2` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/energy/battery/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | power | -1 | 1 |

## Observation space

5 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[3]` against `obs[4]`.

| # | meaning |
|---|---|
| 0 | `soc`, state of charge (0 to 1) |
| 1 | cell voltage (V) |
| 2 | pack temperature (°C) |
| 3 | delivered power (MW), positive when discharging |
| 4 | target power (MW) |

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

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 360 steps.

## Baselines

Recorded over 10 seeds of the 360-step episode (see [Baseline results](../baseline-results.md)). The reward is a cost, so a return closer to zero is better.

| controller | mean return | cost per step |
|---|---|---|
| PID | -1.244 | 0.003457 |
| MPC | -0.3195 | 0.0008876 |

The MPC beats the PID on 10 of 10 seeds and does not trip the plant on any of them.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("battery", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 5 | s |
| `max_steps_in_episode` | 360 | steps |
| `energy_nominal` | 7.2e+09 | J |
| `power_max` | 1e+06 | W |
| `n_series` | 192 | - |
| `capacity_As` | 9e+06 | A s |
| `R0` | 0.02 | ohm |
| `R1` | 0.01 | ohm |
| `C1` | 20000 | F |
| `ocv_a` | 3 | V |
| `ocv_b` | 1.15 | V |
| `ocv_c` | 0.3 | V |
| `ocv_d` | 12 | - |
| `ocv_e` | 0.05 | V |
| `ocv_f` | 8 | - |
| `ocv_g` | 0.85 | - |
| `C_thermal` | 9e+06 | J/K |
| `UA_thermal` | 3000 | W/K |
| `T_ambient` | 25 | degC |
| `k_calendar` | 3e-09 | 1/s |
| `E_activation` | 20000 | J/mol |
| `T_ref_aging` | 298.15 | K |
| `k_cycle` | 1.5e-09 | 1/(kA s) |
| `soc_min` | 0.05 | - |
| `soc_max` | 0.95 | - |
| `T_max` | 60 | degC |
| `V_cell_min` | 2.7 | V |
| `V_cell_max` | 4.25 | V |
| `power_band` | 150000 | W |
| `precision_floor` | 1000 | W |
| `max_step_cost` | 0.03 | reward |
| `cost_weight` | 0.1 | - |
| `degradation_weight` | 200000 | reward |
| `soc_comfort_weight` | 0.1 | reward |
| `dispatch_block_seconds` | 300 | s |
| `dispatch_range` | 800000 | W |
| `dispatch_noise_std` | 2000 | W |
| `reward_version` | 2 | - |
| `e_floor` | 1596 | W |
| `e_tol` | 0 | W |
| `tracking_exponent` | 1 | - |
| `imbalance_price` | 100 | $/MWh |
| `c_hold` | 1.92e-08 | 1/step |
| `fade_price` | 300 | $/kWh |
| `pack_kWh` | 1692 | kWh |
| `failure_cost` | 0.5 | reward |
| `restart_steps` | 720 | steps |
| `rho_floor_tracking` | 0.000221667 | reward |
| `rho_floor` | 0.000221667 | reward |

