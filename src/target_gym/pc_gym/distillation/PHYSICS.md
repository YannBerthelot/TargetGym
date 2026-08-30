# Binary distillation — physics model, provenance and validation

Reference process: **Skogestad's "Column A"** — a 40-tray binary distillation
column with a total condenser, the standard benchmark for ill-conditioned
multivariable control.

Contract for `target_gym.pc_gym.distillation`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

41 equilibrium stages (1 = reboiler … 41 = condenser), feed on stage 21.
Per-stage component balance with vapour–liquid equilibrium
`y = αx / (1 + (α−1)x)`:

```
tray i  :  M dx_i/dt = L_{i+1} x_{i+1} + V y_{i-1} − L_i x_i − V y_i
reboiler:  M dx_1/dt = L_2 x_2 − V y_1 − B x_1
condenser: M dx_N/dt = V y_{N-1} − (L + D) x_N
```

Constant molar flows with a saturated-liquid feed, so the liquid rate below the
feed is `L + F` while vapour is constant throughout; `D = V − L` and
`B = L + F − V` follow from the overall balance.

Assumptions, all from the reference model: binary mixture, constant pressure,
constant relative volatility, equilibrium on every stage, total condenser, no
vapour holdup.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Liquid flow (hydraulic) dynamics | The reference adds a linearised tray hydraulic lag; omitted here, so flow changes propagate down the column instantly. See D1. |
| Energy balance / varying molar flows | Constant molar overflow is the standard simplification and is what makes the model tractable. |
| Pressure dynamics, tray efficiency < 1 | Constant pressure, ideal stages. |
| Multicomponent separation | Binary only. |

**Regime of validity.** High-purity operation near the nominal point
(yD ≈ 0.99, xB ≈ 0.01) with D/F ≈ 0.5. Far from it the constant-relative-
volatility assumption degrades.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| yD at nominal L = 2.706, V = 3.206 | 0.99 (published) | **0.99000** | ✅ |
| xB at nominal | 0.01 (published) | **0.01001** | ✅ |
| D/F at nominal | 0.500 (published) | **0.500** | ✅ |
| Overall component balance `D·yD + B·xB = F·zF` | exact | closes to 5 dp | ✅ |
| Composition monotone increasing up the column | required | yes | ✅ |
| RGA(1,1) at nominal | ≫ 1 (Skogestad reports ~35) | **≈ 52** | ✅ |
| Condition number of the gain matrix | ≫ 1 (reference ~142) | **≈ 200** | ✅ |
| `dyD/dL + dxB/dL` | 1.96 from the mass balance | **1.94** | ✅ |

Reproducing yD = 0.99 / xB = 0.01 *simultaneously* from the published flows is
the load-bearing check: it pins α, the stage count, the feed location and the
flows jointly. The mass-balance identity is an independent cross-check on the
gains — and it is how a first, badly-converged gain computation was caught,
since it gave 0 rather than 1.96.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| trays / stages | 40 / 41 | – | Column A | ✅ |
| feed stage | 21 | – | Column A | ✅ |
| `alpha` | 1.5 | – | Relative volatility, Column A | ✅ |
| `F`, `zF_nominal`, `qF` | 1.0, 0.5, 1.0 | kmol/min, –, – | Saturated liquid feed | ✅ |
| `M_tray`, `M_drum` | 0.5, 0.5 | kmol | Stage holdup | ✅ |
| nominal `L`, `V` | 2.706, 3.206 | kmol/min | Operating point A | ✅ |
| `L_min/max`, `V_min/max` | 2.30–3.10, 2.80–3.60 | kmol/min | ±~0.4 around nominal | ⚠️ |
| `zF_noise_std` | 0.03 | – | TUNED — feed disturbance amplitude | ⚠️ |
| `delta_t` | 1.0 | min | 600 steps ≈ 3 dominant time constants | ✅ |

---

## 4. Numerics

**16 RK4 substeps are a stability requirement, not a refinement.** Tray holdup
0.5 against flows of ~3 gives a tray time constant of ~0.17 min, so the fastest
eigenvalue is `|λ| ≈ (L+V)/M ≈ 11.8 min⁻¹`. RK4 is stable only for
`h·|λ| < 2.78`, needing `h ≤ 0.235 min`:

| substeps | h·λ | result |
|---|---|---|
| 2 | 5.91 | profile saturates to 0/1 in one step |
| 4 | 2.96 | diverges |
| 8 | 1.48 | xB collapses to 0 |
| **16** | **0.74** | yD = 0.99000, xB = 0.01001 ✅ |

The reference implementations use adaptive implicit solvers, which hides this.
It costs throughput: **≈ 0.65 M steps/s**, the slowest environment in the
library — 41 states × 64 RHS evaluations per step. The fast modes are
individual tray holdups (~10 s residence) while the profile of interest evolves
over ~194 min, so this is stiffness, and an implicit integrator would be the
way to buy it back.

---

## 5. Task design

**Observation** `[yD, xB, L_pct, V_pct, target_yD, target_xB]` — a real column
has analysers on the two product streams, not on all 40 trays. **39 of the 41
stage compositions are hidden**, and that interior profile is the column's
memory. Feed composition is hidden too and drifts as an OU process.

**Reward** multiplies the two purity terms rather than summing them: hitting
one specification while losing the other is not half a success, it is an
off-spec column.

**Reset** starts from the converged nominal profile, computed once at import
from the same dynamics rather than hard-coded. With a ~194 min dominant time
constant an arbitrary initial profile would spend the whole episode relaxing.

---

## 6. Baselines

| controller | return | yD error | xB error |
|---|---|---|---|
| MPC (gradient, horizon 15) | **190.0** | **4e−5** | **1.4e−4** |
| PID (LV pairing) | 172.8 | 9.6e−4 | 4.0e−4 |
| constant flows | 37.7 | 7.1e−2 | 6.5e−3 |

The MPC's ~20× tighter top-composition tracking is the expected payoff on an
ill-conditioned plant: it makes *coordinated* reflux/boilup moves, which is
precisely what two independent diagonal loops cannot do when the useful
direction is a small difference between two large, nearly-cancelling effects.

PID gains come from a grid search. Return rises monotonically to Kp ≈ 400 with
no instability, but 150 captures 98 % of it at a quarter of the control
activity — the safer margin given RGA ≈ 50.

---

## 7. Known deviations

**⚠️ D1 — no liquid hydraulic lag.** The reference model adds a linearised tray
flow dynamic (and the K2 vapour effect). Without it a reflux change reaches
every tray instantly, so the column responds faster to L than a real one and
the L→composition path is missing a lag of order a minute per tray.

**⚠️ D2 — ideal composition measurement.** Product analysers are instantaneous
and exact. Real ones have dead time of minutes, which on a plant this
ill-conditioned materially changes achievable performance.

**⚠️ D3 — constant molar flows.** Standard for this benchmark, but it means
energy balance effects (varying latent heats, subcooling) cannot appear.
