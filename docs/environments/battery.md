# Battery

<p align="center"><img src="../../videos/battery/pid_output.gif" width="480px"/></p>

Follow a regulation signal for 4 hours with a 1 MW / 1.69 MWh lithium-ion battery. The signal is an hourly level plus a regulation movement, and following it exactly runs out of energy on most seeds. Leaving the charge or cell-voltage window trips the pack, and deep cycles cost more wear per MWh than shallow ones, so how far to follow is a real decision.

```python
import target_gym as tg

env, params = tg.make("battery")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((7,))` |
| Tracked variable(s) | delivered power (MW) |
| Episode length | 2880 steps of 5 s (4 h) |
| Cite as | `battery-v6` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/energy/battery/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | inverter power | -1 | 1 |

## Observation space

7 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[3]` against `obs[4]`.

| # | meaning |
|---|---|
| 0 | `soc`, state of charge (0 to 1) |
| 1 | cell voltage (V) |
| 2 | pack temperature (°C) |
| 3 | power at the meter (MW), positive when discharging |
| 4 | target power (MW) |
| 5 | the next block's level (MW) |
| 6 | minutes until it goes live |

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

`reset` samples the initial condition and the target; state has 18 fields.

## Episode end

**Termination.** None by default. Leaving the operating envelope trips the plant: the step that leaves it is charged the trip cost and sets `info["tripped"]` and `info["trip_cost"]`, and the plant restarts as `reset_env` would while the episode clock keeps running. Episodic use is optional: with `continuing = False` the plant is lost and `terminated` is raised on that step instead. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 2880 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("battery", max_steps_in_episode=...)`.

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
| `energy_nominal` | 6.0912e+09 | J |
| `power_max` | 1e+06 | W |
| `n_series` | 192 | - |
| `capacity_As` | 9e+06 | A s |
| `pcs_loss` | 0.03 | - |
| `aux_load` | 15000 | W |
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
| `k_calendar` | 6.34e-10 | 1/s |
| `E_activation` | 20000 | J/mol |
| `T_ref_aging` | 298.15 | K |
| `k_cycle` | 2.66e-10 | 1/(kA s) |
| `stress_alpha` | 0.000524 | - |
| `stress_beta` | 2.03 | - |
| `life_fade` | 0.2 | - |
| `cycle_hysteresis` | 0.05 | - |
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
| `dispatch_block_seconds` | 3600 | s |
| `dispatch_range` | 800000 | W |
| `dispatch_noise_std` | 10000 | W |
| `regulation_tau` | 900 | s |
| `reward_version` | 2 | - |
| `e_floor` | 7979 | W |
| `e_tol` | 0 | W |
| `tracking_exponent` | 1 | - |
| `imbalance_price` | 100 | $/MWh |
| `c_hold` | 3.17e-09 | 1/step |
| `fade_price` | 1500 | $/kWh |
| `pack_kWh` | 1692 | kWh |
| `failure_cost` | 1.5 | reward |
| `restart_steps` | 720 | steps |
| `rho_floor_tracking` | 0.00110819 | reward |
| `rho_floor` | 0.00110819 | reward |

