# Binary distillation: physics model, sources, limits and deviations

The plant is Skogestad's Column A [D1, D2], a 40-tray binary column with a
total condenser, the standard model of an ill-conditioned high-purity column.
Its real counterpart with plant data is an industrial propylene/propane
splitter [D4]. `REALITY.md` here is the record, with sources [D1] to [D6];
`COLD_REVIEW.md` lists how the earlier task differed. This page follows
charter Amendments 1 and 2 (`docs/proposals/task-charter.md`).

**Named property: feed steps and late analysers on an ill-conditioned column.**
A 20 % feed-rate step is the large upset of [D1, Fig. 18]; the products are
read by analysers 5 to 10 min late [D5]; reflux and boilup push both products
almost the same way (RGA 35.94 [D1, eq. 31]). All three are in the model
(rule 18) and tested.

---

## 1. Model and provenance

41 equilibrium stages (1 = reboiler … 41 = condenser), feed on stage 21.
Per-stage component balance with vapour–liquid equilibrium
`y = αx / (1 + (α−1)x)`:

```
tray i  :  d(M_i x_i)/dt = L_{i+1} x_{i+1} + V y_{i-1} − L_i x_i − V y_i
           dM_i/dt = L_{i+1} − L_i
           L_i = L0_i + (M_i − M0) / tau_L
reboiler:  M dx_1/dt = L_2 x_2 − V y_1 − B x_1
condenser: M dx_N/dt = V y_{N-1} − (L + D) x_N
```

Constant molar overflow with a saturated-liquid feed, so steady liquid below the
feed is `L + F` while vapour is constant throughout. The feed rate `F` and its
composition `zF` are the plant's state: each steps once an episode (section 6). The feed adds to stage 21's
balances. `D = V − L` and `B = L_2 − V` follow from the two level loops, taken as
perfect, so bottoms follows the liquid that reaches the reboiler.

Assumptions, all from the reference model: binary mixture, constant pressure,
constant relative volatility, equilibrium on every stage, total condenser, no
vapour holdup, linearised tray hydraulics.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Vapour effect on liquid flow (K2) | The reference's `lambda` is 0 for column A, and in all its examples. |
| Reboiler and condenser level dynamics | The reference's P loops have gain 10 /min. With them the RGA below is the same to two decimals, so they are taken as perfect, and the two drums stay at fixed holdup. |
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
| Steady gain matrix `d[yD, xB]/d[L, V]` | [[0.8754, −0.8618], [1.0846, −1.0982]] ([tutorial](https://skoge.folk.ntnu.no/publications/1997/dist_plenary/submitted_version/dist_plenary.pdf), eq. 31) | matches to 4 dp | ✅ |
| RGA(1,1) at nominal | 35.94 (eq. 31) | **35.94** | ✅ |
| Condition number of the gain matrix | no published figure for this model | **145.5** | ✅ |
| `dyD/dL + dxB/dL` | 1.960 from the mass balance | **1.960** | ✅ |
| Dominant time constant | 194 min, the linearised slowest mode (tutorial, sections 4.3 and 4.4) | **193.9 min** | ✅ |
| Reflux change reaching the reboiler | 39 × 0.063 = 2.46 min (section 4.1) | step response of 39 lags, figure 4's shape | ✅ |
| RGA(1,1) magnitude at high frequency | approaches 1 above 0.41 rad/min (section 4.7.3) | **1.30** at 0.41, **0.57** at 1, **1.00** at 10 rad/min | ✅ |

Reproducing yD = 0.99 / xB = 0.01 *simultaneously* from the published flows is
the load-bearing check: it pins α, the stage count, the feed location and the
flows jointly. The mass-balance identity is an independent cross-check on the
gains — and it is how a first, badly-converged gain computation was caught,
since it gave 0 rather than 1.96.

These figures are a float64 Newton solve and the Jacobian, reproduced by a separate
NumPy model written from `colamod.m`. The 52, 200 and 1.94 ticked here until
0.8.0 were central differences at ±0.01 kmol/min, too wide on a plant this
nonlinear: ±0.003 gives 38, 153 and 1.953. The ~35 and ~142 the table once quoted
are 35.1 and 141.7, from the rounded two-by-two model in
[Skogestad and Morari (1987)](https://skoge.folk.ntnu.no/publications/1987/ifac87/old/ifac87.text.pdf),
not from this column. Eq. 31's own rounded entries give 36.07 and 146.0.

---

## 3. Numbers: cited or ours

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| stages, feed stage | 41, 21 | - | cited [D2] |
| `alpha` | 1.5 | - | cited [D2] |
| `F`, `zF_nominal`, `qF` | 1.0, 0.5, 1.0 | kmol/min, -, - | cited [D2] |
| `M_tray`, `M_drum` | 0.5, 0.5 | kmol | cited [D2]; D2 suggests 10 and 32.1 kmol drums "for more realistic studies" (deviation D3) |
| `tau_L` | 0.063 | min | cited [D1, p. 13; D2] |
| `L_nominal` | 2.706 | kmol/min | cited [D2] |
| `feed_step` | 0.2 | - | cited: F 1 to 1.2 [D1, Fig. 18] |
| `zF_step` | 0.1 | - | cited: zF 0.5 to 0.6 [D1, Fig. 18] |
| `step_window` | 30, 300 | min | ours: each step once, with room after it for the 40 min recovery of [D1, Fig. 18] |
| `L_min`, `L_max` | 2.00, 3.40 | kmol/min | ours: hold both specs after any sourced feed step, plus 5 % |
| `V_min`, `V_max` | 2.35, 4.15 | kmol/min | ours: as `L`; flooding [D6] has no figure |
| `D_min`, `D_max` | 0.1, 0.9 | kmol/min | ours: keeps the balance physical at the box's corners (deviation D5) |
| `yD_spec`, `xB_spec` | 0.99, 0.01 | - | cited: Column A's products [D2] |
| `delta_t` | 1 | min | cited: the splitter's MPC samples at 1 min [D4, p. 5] |
| `max_steps_in_episode` | 400 | steps | ours: about two dominant time constants |
| analyser cycle, dead time | 5, 5 | min | cited range: 3 to 30 min in total [D5] |
| `boilup_price` | 1 | kmol | ours: the unit of the score |
| `reprocess_price` | 3.206 | kmol/kmol | ours: no price found; off-spec product is run through again, at the nominal V/F |
| `failure_cost` | 7.997 | kmol | never charged: no trip; above any surviving step |
| `restart_steps` | 1 | steps | never charged |
| `e_floor_top`, `e_floor_bottom` | 4e-4 | - | the analysers' scatter (section Readings), for the time-in-band KPI only |

Count: 12 cited, 11 ours.

---

## 4. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Distillate below yD 0.99 | Zone | Off-spec distillate is charged its reprocessing | spec [D2]; zone [D4, p. 5, Table 1] |
| Bottoms above xB 0.01 | Zone | As above, for bottoms | spec [D2]; zone [D4, p. 5, Table 1] |
| Flooding | Not modelled | Kind not found [D6]; `V_max` stands in, ours | [D6] |
| Reflux and boilup | Hardware | The flow box, clipped | ours (section 3) |

The earlier composition trips at yD 0.90 and xB 0.10, with a 4 h restart, had
no source and are gone. The task never trips.

---

## 5. Observation (rule 16)

`[yD, xB, L_pct, V_pct, spec_yD, spec_xB, T_15, F]`: the two product analysers
[D4, Fig. 1; D5], the stage-15 tray temperature [D5], the operator's own flows,
the specs, and the feed flow from its meter (FC 1 [D4, p. 5]). The feed
composition and the interior profile are hidden: no feed analyser was found.

---

## 6. Episode (rule 17)

The column starts at rest at its design point [D2]: the converged profile at L
2.706 and V 3.206, both products on spec. Then:

- **The feed rate steps by 20 %** [D1, Fig. 18], once, up or down, at a
  minute drawn in 30 to 300. The plant meters it.
- **The feed composition steps by 0.1** [D1, Fig. 18], once, up or down, at
  another minute drawn in 30 to 300. Hidden.

The earlier Ornstein-Uhlenbeck drift of the feed composition, whose size was
tuned, is gone: the sources show steps. Nothing else moves.

---

## 7. Score (rule 15)

The job, from REALITY.md: keep both product impurities inside their zones, at
the least reflux and boilup, without flooding [D4, p. 5; D6]. Per minute, in
kmol of boilup:

```
reward   = -(boilup + off_spec)
boilup   = V dt
off_spec = 3.206 (D [yD < 0.99] + B [xB > 0.01]) dt
```

Inside the zones purity earns nothing; it costs the boilup it takes.

---

## 8. Behaviour tests

`tests/pc_gym/distillation/test_distillation_reality.py` and, for tests 5 and 7,
`test_distillation_hydraulics.py`:

| Test | Source | Before | After |
|---|---|---|---|
| 1. Nominal products | [D2] | pass | pass |
| 2. Reflux step end values | [D1, p. 14, Fig. 5] | pass | pass |
| 3. Boilup step, Fig. 5 sizes | [D1, Fig. 5] | pass | pass |
| 4. Internal flows, smaller and faster | [D1, Fig. 6] | pass | pass |
| 5. Reflux reaches the reboiler after 39 lags | [D1, Fig. 4] | pass | pass |
| 6. Slowest modes 193.9, 12.0, 3.5 min | [D1, 4.4] | pass | pass |
| 7. Gains eq. 31, RGA 35.94 | [D1, eq. 31] | pass | pass |
| 8. 20 % feed step under D1's PI | [D1, Fig. 18, eq. 82] | fail (feed rate fixed) | pass |
| 9. zF step smaller than the feed step | [D1, Fig. 18] | fail (same) | pass |
| 10. Analyser 3 to 30 min late | [D5] | pass | pass |
| Product limits are zones, not trips | [D4, Table 1] | fail | pass |
| Inside spec, less boilup scores better | [D4, p. 5] | pass | pass |

**Findings on the audit's tests, from D1's own figures.** Three of the audit's
tolerances read D1's prose more tightly than D1's figures draw it, and the
model is D1's own `colamod.m` (its gains match eq. 31 to 1e-4 and its slowest
mode is 193.9 min). The tests now check what the figures show, and nothing in
the model moved:

- Test 2: the "time constant of about 194 minutes" is the linearised mode
  (D1 section 4.4 says so); the nonlinear response to the 0.1 % step reaches
  63 % at 160 min, with a 170 min tail, as Fig. 5 draws it. The 194 min is
  test 6.
- Test 3: "almost the same, but in opposite direction" is drawn in Fig. 5 as
  a boilup step 1.4 times the reflux step on xD at 500 min. The test reads
  Fig. 5's four values at 500 min, within the audit's 25 %.
- Test 4: the audit's 7 to 30 min is D1's case with no liquid flow dynamics
  (the dotted lines); with them, Fig. 6 rises fast and then creeps. The test
  asks for "much faster" as under a quarter of the external step's 63 % time
  (ours).
- Test 9: D1 adds that zF moves xD less than xB, so the 1e-3 size is read on
  xB.

---

## 9. Known deviations (rule 13)

**D1. Tray temperature ignores pressure.** Real tray temperature moves with
column pressure and needs a correction, and inferential control at alpha 1.5 is
doubtful [D3]. Here T_15 is an exact function of composition. Effect: the
temperature loop is easier than on a real column.

**D2. No pressure, level or DP readings.** Pressure and levels are held
perfectly, and DP, the flooding warning [D6], is not modelled. Effect: no
flooding to watch.

**D3. Small drums.** 0.5 kmol, Column A's; D2 suggests 10 and 32.1 kmol for
realism, which would add about 10 min of lag on yD. Kept at 0.5 so that
D1's published dynamics (tests 2 to 9) hold. Effect: the top product answers
faster than on a plant.

**D4. No move limits.** The real splitter's MPC moves reflux at most 0.13 t/h
per minute [D4, Table 2]. That is a controller's setting on another column,
not a plant limit, so it is not imposed. Effect: large coordinated moves are
possible.

**D5. Distillate clip.** Where `D` clips at the box's corners the balance does
not close; ours.

**D6. Column A is a model.** Its dead time from reflux to bottoms is 2.5 min;
the real splitter's is 111 min [D4, G2]. The task is Column A.

---

## 10. What the reference ladder row needs

- Bump tests at the design point with `feed_step` and `zF_step` at 0.
- PI by SIMC on the analysers (5 min sample, 5 min dead time), then
  feedforward from the measured feed flow (obs 7) through the overall balance
  (D = zF F / yD), then the stage-15 temperature cascade, then the static
  decoupler.
- The rung is scored on the zone and boilup cost of section 7.

---

## 4. Numerics

**16 RK4 substeps are a stability requirement, not a refinement.** The stiffest
mode comes from the Jacobian: `|λ|` is 32.5 min⁻¹ at the nominal point, near
`(√L + √(V K))² / M` with `L + F` below the feed and `K = dy/dx = 1.5` at the
bottom. Across the corners of the flow box it reaches 37.6 min⁻¹, at `L_min` and
`V_max`. RK4 is stable only for `h·|λ| < 2.78`. The ~11.8 min⁻¹ quoted here until
0.8.0 was the diagonal alone, which could not explain why 8 substeps fail.

| substeps | h·\|λ\|, nominal | h·\|λ\|, worst corner | result at nominal |
|---|---|---|---|
| 2 | 16.3 | 18.8 | profile saturates to 0/1 |
| 4 | 8.1 | 9.4 | profile saturates to 0/1 |
| 8 | 4.1 | 4.7 | xB runs to 0.5 |
| 12 | 2.7 | 3.1 | holds, but a corner is past the limit |
| **16** | **2.03** | **2.35** | yD = 0.99000, xB = 0.01001 ✅ |

The tray flows add no new limit. Each lags at `1/tau_L = 15.9 min⁻¹`, so
`h/tau_L = 0.99`, well inside the stiffest composition mode. The margin at the worst
corner is 15%. The step costs about 1.75 times what it did (3.4 to 5.9 µs per env
step at batch 256 on the owner's Mac), for 80 states against 41.

The reference implementations use adaptive implicit solvers, which hides this.
It costs throughput -- 80 states by 64 RHS evaluations per step makes this one
of the two or three most expensive environments here, and which one is *the*
most expensive depends on the batch size, so see
[docs/performance.md](../../../docs/performance.md) rather than a claim here. The fast modes are
individual tray holdups (~10 s residence) while the profile of interest evolves
over ~194 min, so this is stiffness, and an implicit integrator would be the
way to buy it back.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `distillation` | 400 | 60 | 6.7 h | 2 in [-1, 1] | 8 | 110 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Readings

The observation reports the two products and the tray temperature through their
instruments, and the product specs, the feed flow and the column's own reflux and boilup
exactly
(`docs/proposals/measurement-noise.md`). `sensor_noise_scale = 0` gives the
noise-free task. The 38 stages with no instrument stay exact, and `get_obs` does
not show them.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `x[-1]`, `yD` | 4e-4 mole fraction, sampled every 5 min, 5 min old | none | Process gas chromatograph | Confirmed | [GC8000](https://www.yokogawa.com/be-nl/solutions/products-and-services/measurement/analyzers/gas-analyzers/process-gas-chromatographs/process-gas-chromatograph-gc8000/) |
| `x[0]`, `xB` | 4e-4 mole fraction, sampled every 5 min, 5 min old | none | Process gas chromatograph | Confirmed | [GC8000](https://www.yokogawa.com/be-nl/solutions/products-and-services/measurement/analyzers/gas-analyzers/process-gas-chromatographs/process-gas-chromatograph-gc8000/) |
| `x[14]`, carries `T_15` | 3e-3 mole fraction, 0.04 K on `T_15` | none | Pt100 RTD tray sensor with HART head transmitter | Confirmed | [iTEMP TMT82](https://bdih-download.endress.com/file/2c4c2c5496b6c81881df53d319b2e2f5/TI01010TDE_2725-00.pdf) |

**The analyser's cycle is 5 minutes, and it is ours.** The GC8000 page gives a
maximum analysis period of six hours and no typical cycle. The proposal's 5 to
10 minutes comes from the research behind it, and the refuter could not find it
on that page. 5 minutes is the fast end of that range, and it sits inside the 3
to 6 minutes between samples that a distillation analyser patent claims, with a
4-minute analysis in its example
([US 5464504](https://patents.google.com/patent/US5464504)). Each analyser
holds its value and its noise for those five 1-minute steps, and what it reports
is one cycle old, the time the analysis takes. So step `t` reads step
`5 * (t // 5) - 5`, and no earlier than the reset. A restart refills the analysers
from the restarted plant. The tray thermocouple stays on-line, which is why the
temperature loop is the fast one.

`interval` is in the units of `delta_t`, so it is 5.0 here and not 300.

The 4e-4 follows a range we chose. The page states ±2 % of full scale at 2 sigma
for liquids, under conditions it does not fix, and the 0.04 full scale is twice
our worst target impurity. A vapour sample would give 2e-4.

The tray figure assumes a Pt100 RTD, 0.04 K divided by the curve's 13.5 K per
mole fraction. Section 5 and `env.py` speak of thermocouples on the trays; a
type J or K would scatter about three times more.

The expert reads these readings, and so does every tuner. The oracle plans and
acts on the true state, so it stays a full-state bound. `reward`, trips and the
score read the true state. `info["measured_reward"]` is the same cost on the
readings, with any trip taken from the true state, and a learner can train on
it.

The feed meter (FC 1 [D4, p. 5]) reads exactly: its accuracy was not found.
`e_floor_top` and `e_floor_bottom` are the analysers' scatter, read only by the
benchmark's time-in-band KPI.

## Written exception to check 2

The credibility check on a sourced real figure has nothing to read here. The owner
accepted a class exception on 2026-11-06: no published figure could be found or opened
after two searches with a re-fetch check. The plan in docs/proposals/industrial-fixes.md
says what was tried.
