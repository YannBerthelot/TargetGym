# Cement rotary kiln — physics model, provenance and validation

Reference plant: a **3000 t/day dry-process rotary kiln** with preheater and
precalciner, 4 m × 60 m, the standard modern configuration.

Contract for `target_gym.cement_kiln`. Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model

A 1-D axial model: the kiln is cut into 16 slices, each carrying four dynamic
states — solid temperature, refractory temperature, residual calcination extent
and free lime — with the gas solved **quasi-steady** by a single counter-current
sweep from the burner end.

```
solid  ->  m c dT/dt = Q_gas→solid + Q_wall→solid − Q_reaction + advection
wall   ->  C dT/dt   = Q_gas→wall − Q_wall→solid − shell loss
alpha  ->  dα/dt     = k_calc(T)(1−α)               + advection
lime   ->  dL/dt     = −k_sint(T)·L                 + advection
```

**The transport delay is emergent.** There is no delay parameter anywhere in
the model; it is measured at ~22 minutes to half response, against a 24.9 minute
residence time. Solid states are advected down the kiln by upwind differencing at
a velocity set by kiln speed, so a fuel change reaches the discharge only after
the material does — and changing kiln speed changes the delay itself. That is
what makes speed a genuinely different input from fuel rather than a second
knob on the same response.

**Quasi-steady gas** is justified by the timescales: gas crosses the kiln in
about 8 s against 25 minutes for the solid, a factor of ~180. Solving it
algebraically each step removes the stiffest dynamics at no cost in fidelity.

**The flame is distributed**, decaying exponentially over ~15 m from the
burner. This is not cosmetic: a point source at the burner end dumps the whole
firing rate into one 3.75 m slice, exhausts the gas immediately, and leaves the
rest of the kiln colder than its own feed.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Preheater and precalciner | Represented by their outputs: meal arrives at 800 °C and 92 % calcined. |
| Clinker cooler | Represented by secondary air at 1100 °C. |
| Clinker coating on the refractory | See D1 — this is the model's main deviation. |
| Individual clinker phases (C3S, C2S, C3A, C4AF) | One free-lime variable stands for burn-out; free lime is what plants actually control to. |
| Dust recirculation, alkali cycles, ring build-up | Real operational problems, all outside a temperature/quality model. |
| Radial gradients, bed mixing, kiln eccentricity | One lumped bed per slice. |
| Fuel transport lag and burner dynamics | Heat release follows the fuel command directly. |
| Gas-phase pressure/draught dynamics | The induced-draught fan is assumed to hold flow. |

**Regime of validity.** Near-normal operation: 1.2–2.4 kg/s fuel, 2–4.5 rpm,
feed within ±50 % of nominal. Outside that the lumped-bed and
quasi-steady-gas assumptions degrade.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| L/D ratio | 13–17 (preheater kiln) | **15.0** | ✅ |
| Residence time at 3 rpm | 25–40 min (Sullivan) | **24.9 min** | ✅ |
| Kiln burner duty — two independent routes | must agree | **43.9 vs 44.4 MW** | ✅ |
| Energy closure on the converged profile | in = out | **0.00 %** | ✅ |
| Shell loss | 8–12 % of fuel | 9 % | ✅ |
| Specific heat consumption | 3.0–3.5 MJ/kg clinker | **3.30** | ✅ |
| Burning-zone material temperature | ~1450 °C | **1481 °C** | ✅ |
| Back-end gas temperature | 1000–1200 °C | **1030 °C** | ✅ |
| Free lime at discharge | 0.5–2 % | **1.71 %** | ✅ |
| Calcination complete before discharge | required | α = 1.000 | ✅ |
| Gas / solid timescale ratio | gas quasi-steady | **183×** | ✅ |
| Circulation of heat: solid heats monotonically | required | yes | ✅ |
| Refractory hot face | 1300–1500 °C (coated) | 1816 °C | ⚠️ D1 |

The **energy closure** is the load-bearing check: fuel plus secondary air in
(72.80 MW) equals exhaust plus shell loss plus heat into the charge
(72.80 MW), tying the gas sweep, the reactions and the losses to one another.

The **duty cross-check** is the second: 40 % of a published 3.2 MJ/kg specific
consumption gives 44.4 MW, and building it bottom-up over the *calcined* feed —
sensible heat 24.7 + residual calcination 4.1 + clinker formation 15.1 — gives
43.9 MW. Nothing is fitted to make those agree. Getting the bottom-up wrong is
how two errors surfaced: feeding the kiln raw meal rather than hot meal, and
omitting the clinker-formation term entirely.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `diameter`, `length`, `slope` | 4.0 m, 60 m, 3.5 % | – | Standard 3000 t/d geometry | ✅ |
| `w_bed_gas`, `w_wall_gas`, `w_wall_bed` | 2.91, 9.31, 3.25 | m | Circle geometry at 10 % fill; sums to the 12.57 m circumference | ✅ |
| `raw_meal_nominal` | 53.82 | kg/s | 3000 t/d at 1.55 kg meal per kg clinker | ✅ |
| `calcination_upstream` | 0.92 | – | Modern precalciners reach 90–95 % | ✅ |
| `h_calcination` | 1780 | kJ/kg CaCO₃ | Standard | ✅ |
| `A_calcination`, `E_calcination` | 1.52e5 /s, 170 kJ/mol | – | Anchored to ~2 min calcination at 950 °C | ⚠️ |
| `A_clinker`, `E_clinker` | 4.73e6 /s, 280 kJ/mol | – | Anchored to a 100× lime reduction over a 5 min burning zone | ⚠️ |
| `flame_length` | 15 | m | Real flames run 15–25 m | ⚠️ |
| `emissivity_gas` | 0.25 | – | CO₂/H₂O band emissivity | ✅ |
| `h_wall_bed` | 500 | W/(m² K) | TUNED — covered-arc contact | ⚠️ |
| `U_shell` | 4 | W/(m² K) | Gives ~9 % shell loss; real kilns lose 8–12 % | ✅ |
| `T_feed`, `T_secondary_air` | 800 °C, 1100 °C | – | Preheater exit; clinker-cooler air | ✅ |
| `fuel_nominal`, `rpm_nominal` | 1.78 kg/s, 3.0 | – | The operating point that hits every target above | ✅ |
| `delta_t` | 30 | s | 480 steps = 4 h ≈ 10 transport delays | ✅ |

---

## 4. Task design

**Why it is hard.** This is the suite's transport-delay problem.

1. **The product lags the input by a residence time.** Fuel heats the burning
   zone locally, so free lime starts moving within minutes — but half the
   eventual change takes ~22 minutes, about one full residence time, and 90 %
   takes ~80 minutes. A controller acting on the discharge assay is acting on
   substantially stale information.
2. **Kiln speed moves the delay itself.** Speed sets residence time and holdup
   together — 2.4 rpm gives 31 min and 0.8 % lime, 4.2 rpm gives 18 min and
   2.4 % lime, at nearly unchanged burning-zone temperature. The two inputs are
   qualitatively different.
3. **Free lime is ferociously temperature-sensitive.** With an activation
   energy of 280 kJ/mol, the reaction time constant runs from 20 min at
   1230 °C to under a minute at 1480 °C.
4. **The kiln is operated nearly blind.** 64 states are dynamic; the
   observation exposes eight numbers. Real kiln operators do run the process on
   a handful of readings.
5. **Both failure modes are irrecoverable.** Overheat and the charge sinters
   into rings that block the kiln; go cold and recovery takes hours, longer
   than an episode.

**Observation** `[lime_pct, T_burning_zone, T_exhaust, T_back_end, feed_rate,
fuel_pct, speed_pct, target_lime_pct]` — a burning-zone pyrometer, a back-end
gas thermocouple, a back-end material pyrometer, a weighfeeder, and the
discharge assay. The axial profile is hidden.

**Reward** = free-lime tracking band − fuel.

---

## 5. Baselines

| controller | return (240 steps) | mean lime error | s/episode |
|---|---|---|---|
| CEM MPC (horizon 40 ≈ 20 min) | **193 – 203** | **0.044 – 0.054 pp** | 6 |
| Cascade PID | 163 – 180 | 0.076 – 0.108 pp | 0.2 |
| best constant action | 37 | — | — |

Throughput ≈ 0.65 M steps/s — the slowest environment in the suite, and
honestly so: 64 coupled states with a sequential gas sweep.

**Cascade is the right PID structure**, and for a structural reason. The inner
loop puts fuel on the burning-zone pyrometer, which is fast and carries no
transport delay; the outer loop lets free lime trim that setpoint, slowly.
Kiln speed follows the measured feed so bed depth — and therefore the delay —
stays put. Gains reflect this: the outer loop is strongly proportional with
weak integral, because integral action on a measurement half an hour behind
the input oscillates at the delay period.

**The MPC is gradient-free, by necessity rather than preference.** Every other
JAX environment here uses a gradient MPC. This one cannot: free lime depends on
temperature through an Arrhenius term, that temperature is advected down the
kiln, and the resulting tangent system grows about two orders of magnitude per
step. Reverse-mode gradients overflow to NaN after roughly eight steps —
measured at 1.6e-3 over five steps against 8.3e3 over eight — while finite
differences on the identical objective stay clean and the forward rollout is
perfectly well behaved. `GradientMPC` silently replaces NaN gradients with
zero, so it sat at its initial action sequence and scored exactly what a
zero-action constant scores. The cross-entropy method samples instead, and
recovers the expected MPC-over-PID margin.

---

## 6. Known deviations

**⚠️ D1 — no clinker coating on the refractory.** In a real kiln the burning
zone is protected by a layer of solidified clinker, which holds the hot face
near 1300–1500 °C. Without it the modelled refractory reaches 1816 °C, well
above what real linings survive. The validated solid-side quantities —
burning-zone temperature, free lime, exhaust temperature and heat consumption —
are unaffected, but the refractory temperature itself should not be read as a
prediction, and the coating's insulating dynamics (which real kilns lose and
regain, changing the heat balance) are absent entirely.

**⚠️ D2 — reaction kinetics are anchored, not measured.** Both Arrhenius pairs
were set so the reactions take the time they are documented to take — ~2 min
for calcination at 950 °C, a 100× lime reduction over a 5 min burning zone.
The activation energies are literature-typical but the pre-exponentials are
fitted to those durations, so absolute rates outside the operating range are
not trustworthy.

**⚠️ D3 — one quality variable.** Free lime stands in for the whole clinker
mineralogy. Real burnability also depends on raw-mix chemistry (lime saturation
factor, silica and alumina ratios), which is fixed here.

**⚠️ D4 — no combustion or fuel-handling dynamics.** Heat release follows the
fuel command instantly. Real coal mills and feeders add lags of minutes, on top
of the transport delay this environment is about.

**⚠️ D5 — kiln speed acts on the holdup instantly.** Zone mass is computed
from the current speed, so a speed change redistributes the bed immediately. A
real bed takes time to find its new depth, which would add a lag of order a
residence time to the speed input.

**⚠️ D6 — 16 axial zones.** The burning zone spans only a few slices, so its
peak temperature is resolution-limited; a finer grid shifts it modestly.
