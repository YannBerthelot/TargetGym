# Boiler drum — physics model, provenance and validation

Reference plant: a **natural-circulation drum boiler**, 160 MW, 85 bar, with
the geometry of the Öresundsverket P16-G16 unit used by Åström & Bell.

Contract for `target_gym.boiler_drum`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

Global mass and energy balances over the whole water/steam circuit, plus two
states saying **where** the steam is:

```
d/dt[ρ_s V_st + ρ_w V_wt]                          = q_f − q_s
d/dt[ρ_s h_s V_st + ρ_w h_w V_wt − p V_t + m C t_s] = Q + q_f h_f − q_s h_s
dm_sr/dt = q_ct + q_flash,r − m_sr/τ_sr        (steam in the risers)
dm_sd/dt = f_carry·m_sr/τ_sr + q_flash,d − m_sd/T_d − q_cd   (bubbles in the drum)
```

with `V_st + V_wt = V_t`. The first two are linear in `(dp/dt, dV_wt/dt)` and
are solved as a 2×2 system each step. Level follows from where the water is:

```
level = [ (V_wt − V_dc − (1 − α_v)·V_r) + m_sd/ρ_s − V_ref ] / A_d
```

**Tracking steam as mass rather than quality is what makes the model work.**
The void fraction is `α_v = m_sr/(ρ_s V_r)`, so a falling pressure raises it
*instantly* at constant mass — bubbles expand, water is pushed out of the
risers into the drum, and the level rises. A formulation that relaxes riser
*quality* toward a quasi-steady value on the 18 s riser time constant
suppresses exactly this effect, and produces no swell at all.

**Flashing** is the other half. A falling pressure lowers the saturation
temperature, leaving the entire water inventory superheated so that it boils:
`q_flash = −(V ρ_w c_pw · dt_s/dp / h_c)·dp/dt`. For this boiler a 1 bar drop
flashes about 143 kg of water — 3.1 m³ of steam, roughly 16 cm of level.

Saturated steam properties are quadratic fits to IAPWS values over 60–110 bar
(residuals below 0.5 %, asserted in the tests). Fits rather than steam tables
because the model must stay `jit`/`vmap`-friendly.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Superheater and steam temperature | The drum is the control problem; steam temperature is a separate downstream loop. |
| Furnace-side combustion dynamics and fuel transport lag | Heat release follows the fuel command directly. A real boiler has a burner lag of seconds and a fuel-mill lag of minutes. |
| Spatial distribution along the risers | One lumped riser; the void fraction is a volume mean. |
| Feedwater valve and drum-level-gauge dynamics | Both are treated as instantaneous. |
| Steam separator detail | Represented only by `f_carry`, the fraction of riser steam passing below the water level. |
| Drum geometry | Level is linear in volume (`A_d` constant), valid for deviations around normal water level. |

**Regime of validity.** 60–110 bar (the property-fit range), near-normal water
level, loads between roughly 50 % and 130 % of nominal. Termination on drum
pressure keeps episodes inside the fit range.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Steam property fits vs IAPWS, 60–110 bar | < 1 % | **max 0.49 %** | ✅ |
| Saturation temperature at 85 bar | 299.3 °C | 299.27 | ✅ |
| ρ_water, ρ_steam at 85 bar | 711.5, 45.3 kg/m³ | 712.0, 45.5 | ✅ |
| Latent heat at 85 bar | 1410 kJ/kg | 1410.1 | ✅ |
| Circulation ratio | 5–15 (natural circulation) | **8.84** | ✅ |
| Circulation ratio falls with load | required | 10.2 → 8.1 | ✅ |
| Mean riser void fraction | 0.3–0.6 | 0.467 | ✅ |
| Drum volume closure | 40 m³ | **40.00** | ✅ |
| Metal share of pressure inertia | dominant | **58.5 %** | ✅ |
| Swell on a load step | 25–100 mm typical | 25 mm (5 %) → 106 mm (21 %) | ✅ |
| Shrink on a feedwater step | present, correct sign | −2.4 mm | ⚠️ |
| Level self-regulates | it must **not** | +9.3 cm per 10 min at 2 % bias | ✅ |

The drum volume closure is the load-bearing check, because nothing in the
model is fitted to it: the water, bubbles and steam space computed from the
solved inventory add to 40.00 m³ against a drum specified independently as
40 m³. Circulation ratio is what sets `k_friction`, the one free parameter in
the circulation loop.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `V_t`, `V_d`, `V_r`, `V_dc` | 88, 40, 37, 11 | m³ | Åström & Bell P16-G16 geometry | ✅ |
| `A_d` | 20 | m² | Drum area at normal water level | ✅ |
| `A_dc`, `L_r` | 0.355 m², 11 m | – | Downcomer area, riser height | ✅ |
| `m_metal`, `C_metal` | 300 t, 550 J/(kg K) | – | Tube and drum steel | ✅ |
| property fit coefficients | see params | – | Least squares on IAPWS saturation values | ✅ |
| `k_friction` | 25 | – | Set by the circulation ratio (validated 5–15) | ⚠️ |
| `tau_sr` | 8.0 | s | Riser steam transit; equals `m_sr/q_steam` at nominal | ⚠️ |
| `T_d` | 15.0 | s | Bubble residence below the water level | ⚠️ |
| `f_carry` | 0.10 | – | TUNED — riser steam passing below the level | ⚠️ |
| `f_cd` | 0.25 | – | TUNED — feedwater heat deficit paid by drum bubbles | ⚠️ |
| `h_feedwater` | 1.037 MJ/kg | – | Feedwater at 240 °C (59 K subcooled) | ✅ |
| `Q_nominal`, `q_steam_nominal` | 160 MW, 93.35 kg/s | – | `Q = q_s(h_s − h_f)`, self-consistent | ✅ |
| `level_trip` | ±0.25 | m | Carryover / dryout; real trips are ±150–300 mm | ✅ |
| `delta_t` | 2.0 | s | ~17 steps to the swell peak | ✅ |

`f_carry` and `f_cd` are the two genuinely tuned numbers. They were chosen
together against three targets at once — baseline bubble volume, swell
magnitude and shrink magnitude — not fitted to any single one.

---

## 4. Task design

**Why it is hard.** Drum level is the classic non-minimum-phase control
problem, and nothing else in this suite has that shape.

1. **Swell.** Open the turbine valve and pressure falls; every bubble expands
   and the water inventory flashes, so the level goes **up** while mass is
   leaving. A controller that believes the gauge cuts feedwater exactly when
   it should be adding it, and the level then collapses into a low-level trip.
2. **Shrink.** Subcooled feedwater collapses bubbles, so adding water makes the
   level fall before it rises.
3. **Level is an integrator.** There is no self-regulating steady state — a
   2 % feedwater bias walks the level 9 cm in ten minutes.
4. **Both limits are irrecoverable.** High level carries water into the
   turbine, low level uncovers the tubes; real boilers trip on both.
5. **Voidage is unmeasurable.** `m_sr`, `m_sd` and `V_wt` are hidden. No plant
   instrument reads riser void fraction, and it is exactly the state driving
   the inverse response.

**Observation** `[level, pressure, q_steam, fuel_pct, feed_pct, target_level,
target_pressure]` — drum level, drum pressure and steam flow are the three
measurements of classic three-element control, plus the controller's own
demands.

**Two coupled targets.** Firing and feedwater are not independent: firing
harder raises pressure, which *compresses* bubbles and lowers the level.

**Reward** = ½·level band + ½·pressure band − fuel.

---

## 5. Baselines

| controller | return (400 steps) | mean level error | pressure error |
|---|---|---|---|
| MPC (horizon 30 ≈ 60 s) | **318 – 364** | **0.1 – 0.6 cm** | 0.2 – 0.7 bar |
| Three-element PID | 215 – 296 | 2.7 – 7.3 cm | 0.01 – 0.05 bar |
| best constant action | 5.0 | trips in 5 – 117 steps | — |

Throughput ≈ 10.8 M steps/s.

**Three-element control is the right PID baseline**, and its structure is the
point: feedwater tracks *measured steam flow* as a feedforward, closing the
mass balance without going through the level gauge, and the level PI only
trims it slowly. That is what makes it immune to the level lying during a
transient. Single-element level control — the 2×2 diagonal PID available as
`env.expert_policy` — is the weaker controller precisely because it has no
such feedforward.

**MPC wins on horizon, not on tuning.** A 30-step horizon covers the ~35 s
swell peak, so the optimiser sees the level reverse and keeps feeding through
a swell. It buys a 10× reduction in level error for a small pressure trade.

Its objective is a **quadratic, not the environment's reward**. The reward's
tracking bands clip flat beyond `level_band`, which is exactly the situation
the controller is called on to fix; optimising it directly leaves no gradient
where one is most needed. The objective shares the reward's *minimiser*, not
its shape — the same lesson recorded for the pH environment.

---

## 6. Known deviations

**⚠️ D1 — feedwater shrink is weak.** A 10 kg/s (11 %) feedwater step produces
about 2.4 mm of shrink, at the low end of what is reported for real boilers.
The magnitude is bounded by the feedwater subcooling, which is already at a
realistic 59 K, and the two channels through which it acts (suppressed riser
boiling, collapsed drum bubbles) largely cancel — as conservation requires,
since the heat deficit is fixed however it is split. The **load-change** swell,
which is the effect that actually destabilises level control, is at documented
magnitude.

**⚠️ D2 — no combustion dynamics.** Heat release follows the fuel command
instantly. A real boiler has burner and mill lags of seconds to minutes, which
make pressure control materially harder than modelled here.

**⚠️ D3 — lumped riser.** One volume-mean void fraction stands in for a
distribution that varies along the riser height, and `τ_sr` is constant rather
than varying with circulation.

**⚠️ D4 — `f_carry` and `f_cd` are tuned.** They are not measurable plant
quantities. They were set against three behavioural targets jointly, but they
are the least defensible numbers in the table.

**⚠️ D5 — level is linear in volume.** `A_d` is constant, which is right for
deviations around normal water level and wrong for large excursions in a
cylindrical drum. Trips at ±25 cm keep the model inside the linear region.
