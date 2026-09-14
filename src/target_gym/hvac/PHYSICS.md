# Building HVAC — physics model, provenance and validation

Reference building: a **150 m² well-insulated single thermal zone**, heavyweight
("medium" construction class), hydronic heating, heating-season weather.

Contract for `target_gym.hvac`. Every constant is derived here, cited, or
flagged `TUNED — not sourced`; each row of §2 is asserted by a test in
`tests/hvac/test_hvac_env.py`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

**EN ISO 13790 "simple hourly method"** — the 5R1C reduced-order network: three
nodes (air, surface, mass), five conductances, one capacitance.

```
          Phi_ia + Q_heat        Phi_st            Phi_m
                |                  |                 |
 T_out --H_ve-- T_air --H_tr_is-- T_s --H_tr_ms-- T_mass
                                   |                 |
                                H_tr_w            H_tr_em
                                   |                 |
                                 T_out             T_out
```

Chosen because it is the standard reduced-order building model, is fully
specified in a public standard, and is the lineage BOPTEST-style benchmarks
build on. ISO 13790 has since been **superseded by EN ISO 52016-1**, which
resolves construction layer-by-layer; 5R1C remains the accepted lumped
approximation and is what keeps this environment cheap enough for RL.

Only `T_mass` carries capacitance. `T_air` and `T_surface` have none, so their
balances are two linear equations solved in closed form each step — no solver,
no stiffness. Two differential states total (`T_mass`, plus `Q_emitter` for the
heating system's lag).

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Multiple zones | Single zone. Multi-zone coupling is a different (much larger) control problem. |
| Humidity / latent load | Sensible heat only, as ISO 13790's simple method assumes. |
| Cooling / heat-pump COP | Heating only; the action is thermal power, not electrical. |
| Window opening, blinds, infiltration wind-dependence | Ventilation is a fixed air-change rate. |
| Layer-resolved construction | The point of the lumped 5R1C reduction. |

**Regime of validity.** Heating season, continuous occupancy schedule, indoor
temperatures 15–25 °C. Not valid for cooling, summer overheating studies, or
buildings whose construction class differs materially from "medium".

---

## 2. Validation targets

Derived from the parameter table, then checked against published ranges for a
well-insulated dwelling.

| Quantity | Target | Model | |
|---|---|---|---|
| Total heat loss coefficient `H` | — | 159.4 W/K = 1.06 W/(m²·K) | ✅ |
| Building time constant `τ = C_m/H` | 30–100 h (heavyweight) | **43.1 h** | ✅ |
| Design heat load at −10 °C | 30–50 W/m² (well-insulated) | **31.9 W/m²** | ✅ |
| Heater sizing vs design load | 1.3–2.0× | 1.5× (7.2 kW vs 4.8 kW) | ✅ |
| Overnight free-float drop (8 h, T_out 0 °C) | a few K for heavyweight | 20 → 16.6 °C | ✅ |
| Steady 20 °C at T_out = 0 °C | — | 3.19 kW | ✅ |
| Seasonal heating energy (PID) | — | ~106 Wh/(m²·day) | ⚠️ |

The time constant and design load are the load-bearing checks: together they
pin both the envelope quality and the thermal mass, and they are what make the
control problem what it is — a building that responds over *hours*.

---

## 3. Parameter table

### Geometry (ISO 13790 §7 standard factors)
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `A_floor` | 150 | m² | Reference zone size | ✅ |
| `lambda_at` | 4.5 | – | `A_tot/A_floor`, ISO 13790 standard value | ✅ |
| `f_class` | 2.5 | – | `A_m/A_floor`, "medium" class | ✅ |
| `cm_per_area` | 165 000 | J/(K·m²) | "medium" class internal heat capacity | ✅ |
| `h_is` | 3.45 | W/(m²·K) | Air↔surface film, ISO 13790 | ✅ |
| `h_ms` | 9.1 | W/(m²·K) | Surface↔mass, ISO 13790 | ✅ |

### Envelope
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `U_wall` | 0.28 | W/(m²·K) | Modern insulated wall | ✅ |
| `U_roof` | 0.18 | W/(m²·K) | Modern insulated roof | ✅ |
| `U_window` | 1.30 | W/(m²·K) | Double glazing | ✅ |
| `g_window` | 0.55 | – | Solar transmittance, double glazing | ✅ |
| `A_wall/roof/window` | 120/150/25 | m² | Geometry for a 150 m² zone | ⚠️ |
| `air_changes_per_hour` | 0.5 | 1/h | Typical dwelling ventilation rate | ✅ |

`H_tr_em` is obtained by the standard's *series* split of the opaque envelope:
`1/H_tr_op = 1/H_tr_em + 1/H_tr_ms`, giving 61.7 W/K.

### Systems and disturbances
| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `Q_heat_max` | 7200 | W | 1.5× the 4.78 kW design load | ✅ |
| `emitter_tau` | 900 | s | TUNED — radiator/water thermal lag | ⚠️ |
| `gain_occupied` | 8.0 | W/m² | People + lighting + equipment | ✅ |
| `solar_peak` | 350 | W/m² | Winter clear-sky on vertical glazing | ⚠️ |
| `T_out_mean/amplitude` | 5 / 5 | °C | Heating-season daily cycle | ⚠️ |
| `T_out_noise_std` | 3.0 | °C | OU weather deviation, ~12 h correlation | ⚠️ |

---

## 4. Task design

**Observation** `[T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]` —
what a building management system actually measures: zone and outdoor
temperature, its own heat output, a pyranometer, and the clock. **`T_mass`,
`T_surface` and `weather_dev` are hidden**, and the mass is precisely the state
governing the multi-hour response. The agent measures air temperature but is
really fighting a mass it cannot see.

**Setpoint schedule** — occupied 21 °C (sampled 20–22.5 °C per episode), night
setback 17 °C. Recovery from setback takes hours against a 43 h time constant,
so a controller that waits for the setpoint step is already late. This is the
structural gap MPC exploits and PID cannot close.

**Reward** `log_scaled_reward(|err|, precision_floor, envelope) − energy_weight·(Q/Q_max)`, with `energy_weight = 0` for the 0.6 line so the reward scores comfort alone. The energy term stays wired and raising the weight restores the trade; see the roadmap item on framing running cost.
The clip matters: without it the quadratic turns back upward past
`2·comfort_band` and *rewards* large errors — a bug that made the first MPC
stop heating entirely.

**Episode** 7 days at 15 min steps (672 steps).

---

## 5. Known deviations

**⚠️ D1 — synthetic weather.** Outdoor temperature is a daily sinusoid plus an
OU deviation, not a real weather file. It has the right magnitude and
correlation time but none of a real year's structure (fronts, multi-day cold
spells, cloud). Consequence: an agent cannot learn genuine climate patterns,
only the daily cycle plus noise.

**⚠️ D2 — no cooling.** Heating only, so summer operation and the
cooling/heating changeover are out of scope. Over-heating is penalised but
cannot be corrected except by backing off the heat.

**The 35 °C trip is still reachable only through the actuator**, which was
worth checking, because a terminal state the controller has no authority over
would be unlike every other trip in this suite. Peak free gain is 5.6 kW --
4.8 kW of solar through the glazing plus 0.8 kW of occupancy -- against a heat
loss coefficient of 159.4 W/K, so the *steady* rise above outdoor would be
35 K, which on a 10 °C afternoon points at 45 °C. It never gets there: solar is
zero for twelve hours a day and the 43 h thermal mass integrates, so with the
heater held fully off the zone peaks at 21.6 to 24.2 °C across seeds. Reaching
the trip requires the heater, so it is the controller's to avoid.

**⚠️ D3 — solar gain ignores orientation and shading.** A single glazing area
with one sinusoidal profile; a real zone has orientation-dependent gains and
self-shading.

**⚠️ D4 — lumped thermal mass.** The single `C_m` cannot represent a building
whose surface responds quickly while its core lags — the distinction EN ISO
52016-1 exists to capture. Adequate here, and the reason the model stays cheap.

---

## 6. Baselines

| controller | return | MAE | energy |
|---|---|---|---|
| MPC (24-step horizon, 6 h) | **135.8** | 1.72 °C | **39.6 kWh** |
| PID | 87.6 | 1.38 °C | 68.3 kWh |
| heating off | 34.4 | 2.97 °C | 0 |

3-day episode. The MPC scores **1.55×** the PID while using **42 % less
energy** — the anticipation advantage, from pre-heating ahead of setback
recovery and coasting on forecast solar rather than reacting to it. It accepts
slightly looser comfort (1.72 vs 1.38 °C MAE) for a large energy saving, which
is exactly the trade the reward asks for.

**Cheap to step**: one differential state with closed-form algebraic nodes,
which is what the 5R1C reduction buys. See
[docs/performance.md](../../../docs/performance.md) for measured throughput --
it depends on the machine and the batch size, so it is measured in one place
rather than claimed in fifteen.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `hvac` | 720 | 900 | 180.0 h | 1 in [-1, 1] | 7 | 9 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in euros per step

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `comfort_tolerance` | 0.5 K | occupied comfort band about the setpoint (dead-zone) |
| `setback_lower_bound_only` | True | **provisional design choice.** During the setback only the shortfall below 17 C is charged. This plant cannot cool: under its clear-sky gains the zone sits +4.3 K above the setback target every night and above the occupied band two thirds of the day whatever the controller does (`scripts/measure_hold.py`, signed errors for the shipped MPC over 720 hold steps after a 516-step burn-in), so a symmetric night target would score the weather. |
| `comfort_price` | 0.2 EUR per K^2 per hour | **provisional**; sweep 0.1 / 0.2 / 0.4 |
| `gas_price` | 0.10 EUR/kWh | charged on the emitter's heat in full: the bill the plant trades against comfort |
| `tracking_exponent` | 2 | quadratic outside the band |
| `failure_cost` | 2 x a 14 K excursion for an hour | freezing or gross overheating |
| `restart_steps` | 4 (1 h) | steps the plant is down after a trip before it restarts (reset after a freeze or overheat alarm; provisional); where a plant engineer would get it: the plant's restart procedure |

Floor: overheating sets it. The shipped MPC's lowest per-seed hold cost after burn-in (EUR 0.0273 per
step on the test episode, comfort 0.0206; the 3-seed means are 0.0427 and 0.0364 -- the weather moves it 2x between seeds; `scripts/evaluate_baselines.py`) is `rho_floor`, an upper bound; the certified
reduced-model optimum under this cost is not yet computed (the earlier
certified HVAC runs used the version-1 comfort cost). `e_floor` is not a
parameter of this priced reward; `rho_floor` carries the MPC reference.

`rho_floor_tracking` is the tracking cost per step at the floor in the reward's
units (the NEA floor for tracking) and `rho_floor` the full floor including
consumption charged in full; `floor_is_documented_minimum` records whether
`e_floor` is a measured/certified floor or a resolution used as a scale, and
where it is a resolution (a deterministic plant) both references are 0, since
exact hold is achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce. A trip never ends the window (`base.failure_kernel`):
the plant is frozen at that cost, with tracking and running cost zeroed, for
`restart_steps` steps and then restarts as `reset_env` would; a plant with no
restart stays down to the window's end. `terminated` is never raised;
`info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
