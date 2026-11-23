# Building HVAC

<p align="center"><img src="../../videos/hvac/pid_output.gif" width="480px"/></p>

Heat an office through a concrete slab by choosing the supply water temperature, so the room stays in its 21 to 26 C comfort band during occupied hours at the least electricity cost. The slab takes hours to respond, so the policy has to act ahead: it reads the room, slab and water temperatures, the outdoor temperature with its last-24 h mean and next-24 h forecast, the clock, and the tariff with the hours to its next change. Each step is charged, in euros, the heat pump's electricity at the tariff, and each occupied kelvin-hour of room outside the band or of floor above its limit. Nothing trips.

```python
import target_gym as tg

env, params = tg.make("hvac")
```

| | |
|---|---|
| Action space | `Box((1,))`, all actions in [-1, 1] |
| Observation space | `Box((12,))` |
| Tracked variable(s) | zone air temperature (deg C) |
| Episode length | 1344 steps of 900 s (14 days) |
| Cite as | `hvac-v6` |
| Physics | [contract](https://github.com/YannBerthelot/TargetGym/blob/main/src/target_gym/hvac/PHYSICS.md) |

## Action space

Actions are normalised to `[-1, 1]` and mapped onto the plant's real
actuator range inside the environment.

| # | meaning | min | max |
|---|---|---|---|
| 0 | supply water temperature | -1 | 1 |

## Observation space

12 values in the plant's own units. The space declares no bounds (it spans the float32 range), so scale them before training. The reward scores `obs[0]` against `obs[8]`.

| # | meaning |
|---|---|
| 0 | room air temperature (C) |
| 1 | outdoor air temperature (C) |
| 2 | mean outdoor temperature over the last 24 h (C) |
| 3 | slab core temperature (C) |
| 4 | return water temperature (C) |
| 5 | supply water temperature (C) |
| 6 | sine of the hour of day |
| 7 | cosine of the hour of day |
| 8 | centre of the comfort band (C) |
| 9 | forecast mean outdoor temperature over the next 24 h (C) |
| 10 | electricity price now (EUR/kWh) |
| 11 | hours until the tariff next changes |

## Rewards

See the environment's `compute_reward`.

Every environment in this suite scores on one contract. The reward is
minus the sum of non-negative costs: a tracking cost where the task
holds a target, a running cost for what the plant consumes or fails to
deliver, and a failure cost for trips out of the operating envelope.
This task's terms are `comfort`, `electricity`, `failure`, reported
each step in `info["reward_terms"]`.
The reward is never positive, and its scale differs
by orders of magnitude between plants. Some plants are
priced in dollars or euros and the rest are dimensionless; the
[per-plant summary](../reward-shaping.md#per-plant-summary) says which.
See [Reward shaping](../reward-shaping.md) for how each cost is built.

## Starting state

`reset` samples the initial condition and the target; state has 17 fields.

## Episode end

**Termination.** None: nothing trips this plant. See [Reward shaping](../reward-shaping.md#failure).

**Truncation.** After 1344 steps.

## Baselines

A PID and an MPC ship with this environment, but no comparison has been recorded for them yet.

## Arguments

The scored configuration, which `tg.make` returns. A keyword argument changes one field, as in `tg.make("hvac", max_steps_in_episode=...)`.

| parameter | value | unit |
|---|---|---|
| `delta_t` | 900 | s |
| `max_steps_in_episode` | 1344 | steps |
| `sensor_noise_scale` | 1 | - |
| `actuator_dead_band` | 0 | fraction of span |
| `actuator_stiction` | 0 | fraction of span |
| `action_noise` | 0 | fraction of span |
| `actuator_rate` | 0 | fraction of span per delta_t unit |
| `actuator_lag` | 0 | delta_t units |
| `A_floor` | 150 | m^2 |
| `lambda_at` | 4.5 | - |
| `f_class` | 2.5 | - |
| `cm_per_area` | 165000 | J/(K m^2) |
| `h_is` | 3.45 | W/(m^2 K) |
| `h_ms` | 9.1 | W/(m^2 K) |
| `A_wall` | 120 | m^2 |
| `U_wall` | 0.28 | W/(m^2 K) |
| `A_roof` | 150 | m^2 |
| `U_roof` | 0.18 | W/(m^2 K) |
| `A_window` | 25 | m^2 |
| `U_window` | 1.3 | W/(m^2 K) |
| `g_window` | 0.55 | - |
| `air_changes_per_hour` | 0.5 | 1/h |
| `room_height` | 2.6 | m |
| `slab_thickness` | 0.25 | m |
| `tabs_covering` | 0.8 | - |
| `R_t` | 0.08 | m^2 K/W |
| `water_flow` | 15 | kg/(h m^2) |
| `R_flooring` | 0.125 | m^2 K/W |
| `concrete_density` | 2400 | kg/m^3 |
| `concrete_heat_capacity` | 1000 | J/(kg K) |
| `concrete_conductivity` | 2 | W/(m K) |
| `h_floor` | 10.8 | W/(m^2 K) |
| `h_ceiling` | 6.5 | W/(m^2 K) |
| `c_water` | 4186 | J/(kg K) |
| `T_supply_max` | 49 | degC |
| `T_supply_min` | 15 | degC |
| `cop_carnot_fraction` | 0.52 | - |
| `T_brine` | 0 | degC |
| `frost_on` | 4.4 | degC |
| `frost_off` | 7.2 | degC |
| `T_out_mean` | 5 | degC |
| `T_out_amplitude` | 5 | K |
| `T_out_noise_std` | 3 | K |
| `T_out_mean_tau` | 21600 | s |
| `solar_peak` | 175 | W/m^2 |
| `gain_occupied` | 8 | W/m^2 |
| `gain_unoccupied` | 2 | W/m^2 |
| `occupied_start_h` | 8 | h |
| `occupied_end_h` | 18 | h |
| `band_low` | 21 | degC |
| `band_high` | 26 | degC |
| `floor_surface_max` | 29 | degC |
| `comfort_tolerance` | 2.5 | K |
| `precision_floor` | 0.1 | K |
| `reward_version` | 4 | - |
| `price_valle` | 0.139 | EUR/kWh |
| `price_llano` | 0.184 | EUR/kWh |
| `price_punta` | 0.263 | EUR/kWh |
| `productivity_per_K` | 0.02 | 1/K |
| `occupants_per_m2` | 0.1 | 1/m^2 |
| `labour_cost` | 31.8 | EUR/h |
| `comfort_price` | 9.54 | EUR/(K h) |
| `restart_steps` | 0 | steps |
| `failure_cost` | 0 | EUR per step |

