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

**Reward** `log_scaled_reward(|err|, precision_floor, envelope) − reagent_cost_weight·q3_norm`, with `reagent_cost_weight = 0` for the 0.6 line so the reward scores tracking alone. The reagent term stays wired; see the roadmap item on framing running cost.

---

## 5. Baselines

| controller | return | tracking error |
|---|---|---|
| MPC (horizon 20 ≈ 100 s) | **278.9** | **0.030 pH** |
| PID | 258.0 | 0.085 pH |
| constant valve | 7.3 | 0.885 pH |

The bisection dominates the step cost, and is the price of having the
nonlinearity be exact rather than approximated. Its step count was cut from 44
to 20 on measurement, which roughly doubled throughput; see
[docs/performance.md](../../../docs/performance.md).

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

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `ph_neutralization` | 300 | 5 | 25 min | 1 in [-1, 1] | 3 | 6 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 0.0146 pH | the shipped MPC's long-run mean |error| under the shipped buffer-flow disturbance (`scripts/measure_hold.py`, 900 hold steps after a 108-step burn-in, 3 seeds; PID 0.032). An upper bound on the achievable floor: no reduced-model optimum exists for this plant. |
| `e_tol` | 0 | **provisional.** The discharge permit band (typically pH 6-9 on an outfall) is a regulatory number the plant would supply. |
| `tracking_exponent` | 2 | quadratic |
| `c_hold` | 16.24 mL/s | reagent flow while holding, PID and MPC alike (`scripts/measure_hold.py`) |
| `running_weight` | 1 | one floor-width of pH error is worth the hold-phase reagent flow again; sweep 0.5 / 1 / 2 |
| `failure_cost` | 9.4e5 | twice the span's cost, (10 / 0.0146)^2 |

`rho_floor_tracking` is the tracking cost per step at the floor in the reward's
units (the NEA floor for tracking) and `rho_floor` the full floor including
consumption charged in full; `floor_is_documented_minimum` records whether
`e_floor` is a measured/certified floor or a resolution used as a scale;
`failure_cost` is charged per step in a terminal state and exceeds the largest
tracking cost the envelope can produce. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
