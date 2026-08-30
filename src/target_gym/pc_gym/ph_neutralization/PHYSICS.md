# pH neutralisation — physics model, provenance and validation

Reference process: a **continuous stirred-tank pH neutralisation reactor** —
strong acid neutralised by strong base in the presence of a carbonate buffer.
The canonical extreme-nonlinearity benchmark in process control.

Contract for `target_gym.pc_gym.ph_neutralization`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

**Reaction-invariant formulation** (Gustafsson & Waller; Henson & Seborg).
Acid–base reactions are fast enough to sit at equilibrium, so the
thermodynamic state is fully determined by two quantities that the reaction
*cannot change* and which therefore obey ordinary CSTR mixing:

```
Wa = [H+] − [OH−] − [HCO3−] − 2[CO3²−]      (charge-related invariant)
Wb = [H2CO3] + [HCO3−] + [CO3²−]            (total carbonate)

V dWa/dt = q1(Wa1 − Wa) + q2(Wa2 − Wa) + q3(Wa3 − Wa)
V dWb/dt = q1(Wb1 − Wb) + q2(Wb2 − Wb) + q3(Wb3 − Wb)
```

pH is then the root of the charge balance — an *implicit algebraic* equation:

```
Wa + 10^(pH−14) − 10^(−pH) + Wb·(1 + 2·10^(pH−pK2))
                             / (1 + 10^(pK1−pH) + 10^(pH−pK2)) = 0
```

This split is what makes the model both cheap and brutal: two **linear** mixing
states, with every bit of the nonlinearity in a scalar root-find. Solved by
44-step bisection on [0, 14] — the residual is monotone in pH so bisection
cannot fail, and a fixed step count keeps it `jit`/`vmap`-friendly. Newton is
unsuitable: the residual is near-vertical at the equivalence point and Newton
overshoots the bracket.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Temperature dependence of pK1/pK2 | Isothermal operation, as in the benchmark. |
| Mixing dynamics / imperfect stirring | Perfect mixing is the standard assumption; a real tank adds a transport lag. |
| Electrode dynamics and drift | The pH probe is treated as instantaneous and exact. Real probes lag seconds and drift. |
| Acid/base concentration variation | Feed concentrations are fixed; only flows vary. |

**Regime of validity.** Near-neutral operation (pH 4–10) with the carbonate
buffer present. Outside that, the single-buffer titration curve is a poorer
approximation of a real multi-species effluent.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Nominal steady pH at q3 = 15.6, q2 = 0.55 | ≈ 7 (benchmark design point) | **7.026** | ✅ |
| Residence time V/q_total | — | 88.5 s | ✅ |
| pH bounded to [0, 14] | required | yes | ✅ |
| pH monotone increasing in base flow | required | yes | ✅ |
| Titration curve S-shaped, steepest near equivalence | required | yes | ✅ |
| Gain variation across operating range, nominal buffer | order of magnitude | **45×** | ✅ |
| Gain variation with no buffer | far larger | **462×** | ✅ |

The nominal design point matching 7.03 is the load-bearing check: it pins the
feed concentrations and flows jointly against the published benchmark.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `V` | 2900 | mL | Benchmark reactor volume | ✅ |
| `q1`, `Wa1`, `Wb1` | 16.6, 3.0e−3, 0 | mL/s, M | Acid feed (HNO₃) | ✅ |
| `q2_nominal`, `Wa2`, `Wb2` | 0.55, −3.0e−2, 3.0e−2 | mL/s, M | Buffer (NaHCO₃) | ✅ |
| `Wa3`, `Wb3` | −3.05e−3, 5.0e−5 | M | Base (NaOH + NaHCO₃) | ✅ |
| `q3_min`, `q3_max` | 10, 22 | mL/s | Spans pH ≈ 4.0–10.2, bracketing equivalence with failure margin both ways | ✅ |
| `pK1`, `pK2` | 6.35, 10.25 | – | Carbonic acid dissociation constants | ✅ |
| `q2_noise_std` | 0.35 | mL/s | TUNED — buffering disturbance amplitude | ⚠️ |
| `delta_t` | 5.0 | s | ≈ 18 steps per residence time | ✅ |

---

## 4. Task design

**Why it is hard.** Three compounding difficulties, all physical:

1. **The titration curve is savagely nonlinear.** Steady-state gain varies 45×
   across the operating range at nominal buffering. A fixed-gain controller is
   either sluggish on the flat shoulders or unstable through the steep middle.
2. **Buffering is the disturbance, and it is unmeasured.** Buffer flow shifts
   the operating point *and* flattens the curve by an order of magnitude:

   | buffer q2 (mL/s) | operating pH | gain ratio |
   |---|---|---|
   | 0.00 | 4.16 | 462× |
   | 0.55 | 7.03 | 45× |
   | 4.00 | 7.87 | 8× |

   It drifts as an Ornstein–Uhlenbeck process with a ~500 s correlation time —
   several residence times, so it reads as a changing operating condition
   rather than noise.
3. **pH does not determine the state.** The same pH can arise from different
   (Wa, Wb) pairs whose local gain differs substantially, so the plant is
   genuinely partially observed rather than merely noisy.

**Observation** `[pH, q3_pct, target_pH]` — a plant has a pH electrode and
knows its own valve position. It does not have an on-line assay of carbonate
speciation, so `Wa`, `Wb` and the buffer flow are hidden.

**Reward** `clip(1 − |err|/tracking_band, 0, 1)² − reagent_cost_weight·q3_norm`.

---

## 5. Baselines

| controller | return | tracking error |
|---|---|---|
| MPC (horizon 20 ≈ 100 s) | **278.9** | **0.030 pH** |
| PID | 258.0 | 0.085 pH |
| constant valve | 7.3 | 0.885 pH |

Throughput ≈ 2.1 M steps/s — the bisection dominates, and is the price of
having the nonlinearity be exact rather than approximated.

**The MPC objective is a quadratic in the error, not a copy of the reward.**
This is worth stating because getting it wrong failed in *both* directions
during development. Copying the reward without its clip makes the quadratic
turn back upward past the band, so large errors score better and the
controller gives up entirely. Copying it *with* the clip makes the objective
flat out there, so IPOPT sees no gradient at all, optimises the only live term
(reagent cost) and rails the valve shut — driving pH away from setpoint at
~3.9 pH mean error. What the objective must share with the reward is its
**minimiser**, not its shape.

---

## 6. Known deviations

**⚠️ D1 — ideal pH measurement.** The probe is instantaneous and noise-free.
Real electrodes have a seconds-scale lag, drift, and fouling — all of which
matter for pH control specifically, since the measurement is the whole game.

**⚠️ D2 — single buffer species.** Carbonate only. Real effluents carry several
weak acid/base pairs, giving a titration curve with multiple inflections
rather than one.

**⚠️ D3 — flows are exact.** No valve dynamics, hysteresis or flow measurement
error on the manipulated stream.
