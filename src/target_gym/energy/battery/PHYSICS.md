# Grid battery — physics model, provenance and validation

Reference system: a **2 MWh / 1 MW lithium-ion grid battery** — a 2-hour
system, the common utility build — modelled as a first-order equivalent
circuit with lumped thermal and capacity-fade dynamics.

Contract for `target_gym.energy.battery`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model

No single published benchmark defines a grid BESS the way NREL defines a
turbine, so this is a **first-principles equivalent-circuit model** with
parameters sized to reproduce published *behaviour* — round-trip efficiency,
cell voltage window, thermal rise. Per the project's standard, that is
physically defensible rather than citation-backed, and §2 is where it earns
its keep.

```
V_term     = OCV(soc) − I R0 − v_rc
dsoc/dt    = −I / Q
dv_rc/dt   = I / C1 − v_rc / (R1 C1)
C_th dT/dt = I² R0 + v_rc I − UA (T − T_amb)
dq_loss/dt = calendar(T) + cycle(|I|, T)
```

The controller commands **power**, so current follows from `P = V_term·I` — a
quadratic whose physical (smaller) root is taken. That is not a detail: it
means deliverable power is bounded by `(OCV − v_rc)² / 4R0`, a limit that
tightens as the pack empties.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Cell-to-cell imbalance, balancing circuits | Single lumped pack. |
| Electrochemical (P2D/SPM) dynamics | An equivalent circuit is the standard reduction for control work and is orders of magnitude cheaper. |
| Voltage hysteresis, multiple RC branches | One RC branch; adequate for minute-scale dispatch. |
| Power-electronics switching, converter efficiency curve | Losses are ohmic only. |
| Calendar life beyond the episode | Fade accumulates but never fully depletes the pack in one episode. |

**Regime of validity.** Dispatch operation at up to ~1 C, 10–90 % state of
charge, near ambient temperature. Not valid for fast charging, deep discharge
or thermal-runaway conditions.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Round-trip efficiency at rated power | 88–95 % (Li-ion grid BESS) | **90.6 %** | ✅ |
| Cell voltage window | 2.7–4.2 V | **2.84–4.19 V** | ✅ |
| OCV monotone in state of charge | required | yes | ✅ |
| Steady thermal rise at rated power | ~10–20 K (actively cooled) | **14.6 K** | ✅ |
| Thermal time constant | tens of minutes | 50 min | ✅ |
| 10→90 % traverse at rated power | ≈ 96 min for a 2 h system | 96 min | ✅ |
| Coulomb counting: ∫I dt = ΔSoC·Q | exact | closes | ✅ |

Round-trip efficiency is the load-bearing check — it is what sizes `R0`, and
it is jointly constrained by pack voltage, capacity and rated power. Two
sizing errors were caught by these targets before any model code ran: an `R0`
of 0.05 Ω gives 79 % round-trip (far below the band), and a passive `UA` of
250 W/K implies a **438 K** temperature rise, which is absurd — grid packs are
actively cooled.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `energy_nominal`, `power_max` | 2 MWh, 1 MW | – | 2-hour system, 0.5 C | ✅ |
| `n_series`, `capacity_As` | 192, 2500 Ah | – | Gives ~800 V nominal | ✅ |
| `R0` | 0.02 | Ω | Sized for 90.6 % round-trip | ✅ |
| `R1`, `C1` | 0.01 Ω, 20 kF | – | TUNED — diffusion time constant ~200 s | ⚠️ |
| OCV coefficients | see params | – | NMC-like fit; validated on window and monotonicity | ⚠️ |
| `C_thermal`, `UA_thermal` | 9e6 J/K, 3000 W/K | – | ~10 t pack; active cooling | ⚠️ |
| `k_calendar`, `k_cycle`, `E_activation` | 3e−9, 1.5e−9, 20 kJ/mol | – | TUNED — Arrhenius calendar plus throughput cycling | ⚠️ |
| `delta_t` | 5.0 | s | 720 steps = 60 min | ✅ |

---

## 4. Task design

**Dispatch tracking against a finite energy budget.** The tension is
structural rather than tuned: following the grid's request drains charge, and
running the pack empty or full **ends the episode irrecoverably**. Unlike a
thermal plant, the battery cannot hold a setpoint indefinitely — tracking now
costs the ability to track later.

**Observation** `[soc, V_cell, T_cell, P_MW, target_P_MW]` — what a battery
management system actually reports. The diffusion voltage `v_rc` and the
accumulated capacity fade `q_loss` are hidden: neither is directly measurable,
and the fade in particular is the cost the controller is implicitly trading
against.

**Efficiency depends on state.** Losses scale with current squared, and the
current needed for a given power depends on state of charge through the OCV
curve, so the same dispatch costs more when the pack is low.

**Reward** = dispatch tracking − degradation − a weak pull toward mid charge.
The last term is deliberately weak: it should bias toward keeping headroom in
both directions without overriding the dispatch the battery is paid to follow.

---

## 5. Baselines

| controller | return | power error | episodes completed |
|---|---|---|---|
| PID + charge guard | **155.7** | 0.062 MW | 360/360 |
| constant ±0.5 / 0 | 11.6 – 20.5 | 0.35 – 0.63 MW | 360/360 |

The guard matters more than the gains. Fading the demand out *only in the
direction that would breach a limit* — throttling discharge near empty and
charge near full, untouched in the middle — is what keeps the controller out
of the terminal states while leaving normal dispatch alone.

---

## 6. Known deviations

**⚠️ D1 — no published reference system.** Unlike the wind turbine, the
parameters are sized to reproduce published *behaviour* rather than taken from
a specific documented machine. The efficiency, voltage window and thermal rise
are right; the particular cell chemistry is generic NMC-like.

**⚠️ D2 — single RC branch.** Real packs show relaxation over several
timescales. One branch captures the dominant minute-scale polarisation and
misses both faster and slower components.

**⚠️ D3 — simplified ageing.** Calendar plus throughput with an Arrhenius
temperature factor. Real fade depends on depth of discharge, C-rate history and
state-of-charge dwell in ways this does not represent, so the degradation cost
is directionally right but not quantitatively trustworthy.

**⚠️ D4 — no cell imbalance.** A lumped pack cannot represent the
weakest-cell behaviour that actually determines real pack limits.
