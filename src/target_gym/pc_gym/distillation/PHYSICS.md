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
tray i  :  d(M_i x_i)/dt = L_{i+1} x_{i+1} + V y_{i-1} − L_i x_i − V y_i
           dM_i/dt = L_{i+1} − L_i
           L_i = L0_i + (M_i − M0) / tau_L
reboiler:  M dx_1/dt = L_2 x_2 − V y_1 − B x_1
condenser: M dx_N/dt = V y_{N-1} − (L + D) x_N
```

Constant molar overflow with a saturated-liquid feed, so steady liquid below the
feed is `L + F` while vapour is constant throughout. The feed adds to stage 21's
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
| Dominant time constant | 194 min (tutorial, section 4) | **194.0 min** | ✅ |
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

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| trays / stages | 40 / 41 | – | Column A | ✅ |
| feed stage | 21 | – | Column A | ✅ |
| `alpha` | 1.5 | – | Relative volatility, Column A | ✅ |
| `F`, `zF_nominal`, `qF` | 1.0, 0.5, 1.0 | kmol/min, –, – | Saturated liquid feed | ✅ |
| `M_tray`, `M_drum` | 0.5, 0.5 | kmol | Stage holdup | ✅ |
| `tau_L` | 0.063 | min | Per tray, tutorial table and `colamod.m`'s `taul` | ✅ |
| nominal `L`, `V` | 2.706, 3.206 | kmol/min | Operating point A | ✅ |
| `L_min/max`, `V_min/max` | 2.20–3.40, 2.70–3.90 | kmol/min | Ours, to hold the target box (below) | ⚠️ |
| `zF_noise_std` | 0.03 | – | TUNED — feed disturbance amplitude | ⚠️ |
| `delta_t` | 1.0 | min | 400 steps ≈ 2 dominant time constants | ✅ |

**The flow limits are ours, for the owner to accept.** The reference gives no
limits, and these come from the target box, not from a column design. At the
nominal feed (0.995, 0.005) needs L 3.353 and V 3.853, and (0.98, 0.02) needs
L 2.286 and V 2.786. At feeds two sigma either side the extremes are L 2.27 and
3.36, V 2.73 and 3.90.

| limits | nominal feed | over the reset feed | at feed 0.41 |
| --- | --- | --- | --- |
| 2.30–3.10, 2.80–3.60, before | 2.1% unreachable | 3.0% | 12.7% |
| maximums raised only | 0.3% | 1.0% | 10.9% |
| 2.20–3.40, 2.70–3.90 | 0% | 0% | 0.1% |

So raising the maximums alone leaves the low corner out, and both ends move. Trimming
the box instead would change what the task asks for, and would not help a drifting
feed. The price is an action span 1.5 times wider, so a raw action moves the flows
further, and `D` clips over more of the action box, 44% against 25% (D1 below).
V 3.9 is 1.22 times nominal; a column built to run at about 80% of flood has about
that much room, which is our reading and not a sourced figure.

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
The tray flows start at the sampled reflux, so a reset has no hydraulic transient.

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

**D1: liquid hydraulics, closed.** Each of the 39 trays carries its own flow
state, the reference's `L_i = L0_i + (M_i − M0) / tau_L` with `tau_L = 0.063 min`
and `lambda = 0` (tutorial eq. 47, `colamod.m`). That is 0.063 min per tray, not
the minute per tray this entry used to say, and 2.46 min to the reboiler.

A single lag on `L` was the simpler choice, and it cannot work. It delays the top
as much as the bottom, and that only multiplies a column of the transfer matrix,
which leaves the RGA unchanged at every frequency. It would stay at 3.2 to 3.7
above 0.1 rad/min. What decouples the two ends is that liquid reaches the bottom
later than the top, so the delay has to be spread down the column. The cascade is
the reference's own model, and adds no new stiffness (section 4).

With it the RGA magnitude falls from 3.4 to 3.7 to 1.30 at 0.41 rad/min, and stays
between 0.52 and 1.30 up to 10 rad/min. It is still 3.10 at 0.1 and 2.39 at 0.2
rad/min, so the review's 0.7 to 1.4 above 0.1 rad/min did not reproduce, on this
model or with the reference's level loops. The tutorial's figure, about 1 above
1/θ_L = 0.41 rad/min, does. The steady state is unchanged.

`D` stays clipped to `D_min` and `D_max`, which breaks the balance where it binds.
At `L_max` and `V_min` the column settles on a profile that is not monotone, and it
trips long before that.

**D2 — analyser dead time, closed.** Product analysers report a sample one
5-minute cycle old, held until the next (section Readings). On a plant this
ill-conditioned that changes what is achievable: at 5 minutes the expert's gains,
tuned without it, lose the column, so its tuning is redone under the expert
protocols.

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

The tests hold a steady feed, on training seeds 100 to 109, by
`scripts/tune_process_experts.py --envs distillation`, section 9 of
`docs/proposals/expert-protocols-process.md`. Nobody leaves a column in manual, so a
P loop of 0.16 kmol/min per K on the tray holds it (ours).

- **Pairing.** RGA(1,1) from the steady gains in float64: 35.9 for yD and xB on L and
  V, and 0.83 for yD and xB on L and the tray's setpoint with the tray closed, the
  pairing the expert runs. The tray alone with V keeps 43 at steady state, so the
  tray loop has to be tight and the composition loops slower.
- **Temperature loop.** Integrating, k' = 1.15 K/min per kmol/min, no lag beyond the 1
  min sample, theta = 0.5 min. The filter reaches its cap, Tf_T = 0.4 min, and Kp_T =
  0.360 kmol/min per K, Ti_T = 9.7 min.
- **Setpoint ramp.** 0.70 K/min, the steepest the tray loop follows on boilup's room
  from the published point, so a composition move never steps it.
- **Distillate loop**, yD on reflux, tray closed, on the chromatograph. A step of
  0.25 kmol/min moves 1.5 bands. k = 0.0144, tau1 = 1.4 min, theta = 10.5 min, so
  Kp_D = 16.1, Ti_D = 4.9 min.
- **Bottoms loop**, xB on the tray's setpoint. A step of 1.05 K moves 2.6 bands. k =
  -0.00624 per K, tau1 = 0.87 min, theta = 7.5 min, so Kp_B = -36.0 K, Ti_B = 3.4 min.
- **Gate.** Each analyser loop acts on an update whose reading has moved past its band
  (0.0024) from the one it held, or sits past it from the target, then waits 2
  updates, 10 min. That is N8. The loops are slower than the tray, so the cascade is
  no longer inverted.
- **Held-out check** (seeds 200 to 209, 400 steps). It holds 7,636 per step (worst
  seed 24,880) against a ceiling of 2, with 170 trips on 3 of the 10 seeds. The best
  null is a constant, ratio 0.04, 2 of 10 wins.
- **Episode.** 400 steps, doubled from 200 on 2026-11-06, since the hold at 200 was
  2.81 times a doubled run's. The ratio at 400 is 0.57 on training seeds and 0.65
  held out, so the rule passes, but only because the longer run does worse: 298 trips
  on 5 seeds at 800 steps. At 200 steps it held 3,942 with no trips, so these gains
  lose the column past step 200. That is a finding, not a settled loop.
- **Scatter.** The recipe reproduces exactly at a fixed test length, which is why that
  length is now the recipe's, 200 steps (section 9, R7). A new noise path moves it a
  lot: over test lengths of 195 to 500 steps, `Kp_T` runs 0.12 to 0.36.
- **Levels.** 8,466 with 43 trips on 7 seeds at level 0.5, and 4,725 with none at
  level 2.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `distillation` | 400 | 60 | 6.7 h | 2 in [-1, 1] | 7 | 107 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, one tracking term per product, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_top`, `e_floor_bottom` | 1e-4, 1e-4 mole fraction (the MPC holds 7.6e-7 / 5.2e-7) | the lowest per-seed long-run mean \|error\| the shipped MPC held on each product under the shipped feed-composition disturbance (`scripts/measure_hold.py`, 1200 hold steps after a 582-step burn-in, 3 seeds; PID down to 5.7e-5 and 1.3e-4; before the oracle audit it held 1.35e-5 / 3.3e-5). Per-seed minima; upper bounds Below the instrument resolution the plant's own table cites, so the resolution sets the scale: a hold the instrument cannot see is not a floor. |
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

## Readings

The observation reports the two products and the tray temperature through their
instruments, and the setpoints and the column's own reflux and boilup exactly
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

A plant resting exactly on target pays 32 per step in measured reward,
`2 (4e-4 / 1e-4)^2`. `e_floor_top`, `e_floor_bottom` and `precision_floor` stay
at 1e-4, so the ceiling sits below anything this analyser could verify.
