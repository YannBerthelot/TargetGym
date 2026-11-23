# Battery

<p align="center"><img src="../../videos/battery/pid_output.gif" width="480px"/></p>

Run a grid battery that sells a frequency service: keep its reported stored energy inside the window the market requires at the start of each 30 min period, at the least cost of energy. The only lever is the declared baseline power, and a change reaches the meter 60 to 90 min after it is made. The policy reads the BMS's reported state of charge, cell voltage and temperature, the meter, the required power, the three locked baselines, the period's energy requirements and the day-ahead prices; the true state of charge is hidden. Each step is charged, in GBP, the availability payment of a lost period, the energy the unit loses, and the energy's cost against the base price. Nothing trips.

```python
import target_gym as tg

env, params = tg.make("battery")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((17,))` |
| Tracked variable(s) | delivered power (MW) |
| Episode length | 14400 steps of 1 s (4 h) |
| Cite as | `battery-v6` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/energy/battery/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | Operational Baseline | -1 | 1 |

## Observation space

17 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[3]` against `obs[4]`.

| # | meaning |
|---|---|
| 0 | state of charge as the BMS reports it |
| 1 | cell terminal voltage (V) |
| 2 | cell temperature (degC) |
| 3 | power delivered at the meter (MW), positive when discharging |
| 4 | power required now: baseline plus contracted response (MW) |
| 5 | contracted response now, as a fraction of the contracted power |
| 6 | baseline of the current period (MW) |
| 7 | baseline of the next period (MW) |
| 8 | baseline of the period after next (MW) |
| 9 | this period's export energy requirement (MWh) |
| 10 | this period's import energy requirement (MWh) |
| 11 | seconds to the next period |
| 12 | 1 once this period is lost, else 0 |
| 13 | day-ahead price of the current period (GBP/MWh) |
| 14 | day-ahead price of the next period (GBP/MWh) |
| 15 | day-ahead price of the period after next (GBP/MWh) |
| 16 | day-ahead price of the third period ahead (GBP/MWh) |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `availability`, `losses`, `trade`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 30 fields.

## Episode end

**Termination.** None: nothing trips this plant. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 14400 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("battery", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 1 | s |
| `max_steps_in_episode` | 14400 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `power_max` | 1.927e+06 | W |
| `energy_capacity` | 1.38744e+10 | J |
| `round_trip_efficiency` | 0.92 | - |
| `cell_capacity_Ah` | 280 | A h |
| `cell_nominal_V` | 3.2 | V |
| `cell_resistance` | 0.00025 | ohm |
| `V_cell_min` | 2.5 | V |
| `V_cell_max` | 3.65 | V |
| `T_charge_min` | 0 | degC |
| `T_charge_max` | 55 | degC |
| `T_discharge_min` | -20 | degC |
| `T_discharge_max` | 55 | degC |
| `ocv_plateau_slope` | 0.02 | V |
| `ocv_knee_width` | 0.03 | - |
| `T_cell_start` | 25 | degC |
| `soc_error_std` | 0.075 | - |
| `soc_drift_rate` | 8.2672e-08 | 1/s |
| `soc_temp_coeff` | 0.0051 | 1/degC |
| `T_bms_curve` | 25 | degC |
| `reserve_fraction` | 0.4 | - |
| `volume_hours` | 1 | h |
| `recovery_fraction` | 0.2 | - |
| `period_seconds` | 1800 | s |
| `gate_closure_seconds` | 3600 | s |
| `unavailable_fraction` | 0.001 | - |
| `shortfall_tolerance` | 1 | W |
| `signal_interval` | 2 | s |
| `signal_tau` | 300 | s |
| `signal_std` | 0.4 | - |
| `neutrality_tau` | 1800 | s |
| `price_base` | 74 | GBP/MWh |
| `price_event_periods` | 2 | periods |
| `availability_price` | 4.36 | GBP/(MW h) |
| `reward_version` | 3 | - |
| `restart_steps` | 0 | steps |
| `failure_cost` | 0 | GBP |
| `e_floor` | 330 | W |
| `precision_floor` | 330 | W |

