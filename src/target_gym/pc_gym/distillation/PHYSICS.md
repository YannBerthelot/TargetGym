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
It costs throughput -- 41 states by 64 RHS evaluations per step makes this one
of the two or three most expensive environments here, and which one is *the*
most expensive depends on the batch size, so see
[docs/performance.md](../../../docs/performance.md) rather than a claim here. The fast modes are
individual tray holdups (~10 s residence) while the profile of interest evolves
over ~194 min, so this is stiffness, and an implicit integrator would be the
way to buy it back.

---

## 5. Task design

### Task: `distillation`

Hold both product purities of a 40-tray binary distillation column (Skogestad's
Column A) with reflux and boilup. The two purities move together far more
easily than apart, which makes the plant ill-conditioned. Only the two products
are analysed, plus one tray temperature: the tray compositions that hold the
column's memory, and the drifting feed composition, are hidden. The reward
scores both purities.

| obs | meaning |
|---|---|
| 0 | `yD`, light-component mole fraction in the distillate (top product) |
| 1 | `xB`, light-component mole fraction in the bottoms |
| 2 | reflux flow, fraction of its range (0 to 1) |
| 3 | boilup, fraction of its range (0 to 1) |
| 4 | target `yD` |
| 5 | target `xB` |
| 6 | `T_15`, stage 15's temperature (K) |

**Observation.** A real column has analysers on the two product streams and
thermocouples on a few trays, not an assay of all 40. **39 of the 41 stage
compositions are hidden**, and that interior profile is the column's memory.
The one tray temperature, `T_15`, is stage 15's on column A's linear
boiling-point curve, T = 341.9 x + 355.4 (1 - x) K (Skogestad, distillation
mailing list, 19 May 1999): the stage the maximum gain rule picks in the bottom
section (220 K per kmol/min of boilup at the nominal point, derived), and the
location Skogestad recommends for column A ("Distillation control: dos and
don'ts", section 6.3). It was added in v3 for the expert's temperature loop.
Feed composition is hidden too and drifts as an OU process.

**Reward** charges each purity's tracking term, so holding one specification
does not excuse losing the other.

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

The PID in the table is the version-2 expert, two diagonal composition loops
grid-searched on the scored reward. Section 8 has the base layer that
replaced it.

**The MPC plans on the mean feed, with a decaying step.** Until the oracle
audit (2026-10-01) the feed-composition noise was not declared in the spec's
`noise_fields`, so the plan followed one fixed noise path, the plant's own on
protocol seed 0. Declared, the plan follows the mean, and that alone made the
oracle 15% worse: the fixed normalised step cannot place reflux and boilup
closer than about `lr` to the optimum, and that was the whole hold error. A
step decaying from 0.08 to 0.004 over 80 iterations scores 0.0125 on the
protocol seeds, against 0.297 before (zero trips). What is left is seed 0's
running cost (about 0.037 per step on that seed, 0.0122 in the three-seed
mean); the tracking left is far below the analyser's
resolution. (The table above predates the version-2 reward and this change.)

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

---

## 8. Expert

On columns an MPC over the temperature loops is the norm; under it runs the
base layer this expert copies, the LV-configuration with a temperature loop
using boilup ("Distillation control: dos and don'ts", sections 6.3 to 6.5,
which calls it widely used in industry). Boilup holds the stage-15
temperature, reflux holds the distillate and the temperature setpoint holds
the bottoms (`make_distillation_pid_controller`, `base_layer_step`). It starts
bumplessly from the observed flows and temperature. The observation-only MPC
that practice adds on top is PR 2 of the experts refactor.

SIMC (Skogestad and Grimholt 2012), from step tests on training seeds 100 to
109 against unstepped runs on the same feed noise, by
`scripts/tune_process_experts.py --envs distillation`:

- **Temperature loop.** Boilup stepped 0.005 kmol/min. Over the first 20 min
  the stage-15 temperature rises at k' = 1.19 K/min per kmol/min with no lag
  beyond the 1 min sample, theta = 0.5 min; its time constant, about 160 min,
  is over 300 delays, so the loop is lag-dominant and takes the integrating
  rule. Tight SIMC would move boilup by its whole range on the setpoint
  moves the bottoms loop makes (up to 2.5 K on the training seeds), and the
  column then runs off, so tau_c is the smallest that keeps the first move
  within boilup's room from the nominal point, 0.39 kmol/min (SIMC's input
  usage rule, eq. 5.35): tau_c = 4.8 min, Kp_T = 0.160 kmol/min per K,
  Ti_T = 21 min.
- **Distillate loop**, yD on reflux with the temperature loop closed: a
  0.005 kmol/min step. Median fit k = 0.0209, tau1 = 1.98 min,
  theta = 1.31 min; the input bound does not bind, so tau_c = theta:
  Kp_D = 36.2, Ti_D = 1.98 min.
- **Bottoms loop**, xB on the temperature setpoint: a 0.2 K step. Median fit
  k = -0.00526 per K, tau1 = 2.18 min, theta = 0.93 min; tau_c = theta:
  Kp_B = -223 K, Ti_B = 2.18 min.
- **Interaction.** With the temperature loop closed the steady-state relative
  gain of the remaining pairing is about 0.85, against about 50 for LV
  without it, the reduction the dos and don'ts describe (measured on the
  nominal profile).
- **Held-out check** (seeds 200 to 209). It holds 0.155 per step (worst seed
  0.29) against a ceiling of 2, settles in 46 steps and never trips
  (`tests/pc_gym/test_expert_specs.py`). The two diagonal composition loops
  it replaced held 391 on the protocol seeds. Over 600-step episodes they
  also tripped 3 of the 10 training seeds.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `distillation` | 200 | 60 | 3.3 h | 2 in [-1, 1] | 7 | 46 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, one tracking term per product, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_top`, `e_floor_bottom` | 1e-4, 1e-4 mole fraction (the MPC holds 7.6e-7 / 5.2e-7) | the lowest per-seed long-run mean \|error\| the shipped MPC held on each product under the shipped feed-composition disturbance (`scripts/measure_hold.py`, 1200 hold steps after a 582-step burn-in, 3 seeds; PID down to 5.7e-5 and 1.3e-4; before the oracle audit it held 1.35e-5 / 3.3e-5). Per-seed minima; upper bounds Below the instrument resolution the plant's own table cites, and measurement noise is not modelled, so the resolution sets the scale: a hold the instrument cannot see is not a floor. |
| `e_tol` | 0 | **provisional.** The purity specifications a column is run against come from the sales contract. |
| `tracking_exponent` | 2 | quadratic |
| `c_hold` | 3.282 kmol/min | boilup while holding, PID (MPC 3.292) (`scripts/measure_hold.py`) |
| `running_weight` | 1 | sweep 0.5 / 1 / 2 |
| `failure_cost` | 3.61e6 | twice the reachable two-product excursion's cost, 2 x 2 x (0.095 / 1e-4)^2: a 0.98 top target to the 0.90 trip, a 0.02 bottom target to the 0.10 trip, per trip step |
| `restart_steps` | 240 (4 h) | restart time priced into a trip, `restart_steps x failure_cost` (column shutdown and re-establishment of the profile; provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the oracle's best hold for tracking -- the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution -- and `rho_floor` the same with consumption
charged in full; `floor_is_documented_minimum` records whether `e_floor` is a
measured/certified floor or a resolution used as a scale, and where it is a
resolution on a deterministic plant both references are 0, since exact hold is
achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
