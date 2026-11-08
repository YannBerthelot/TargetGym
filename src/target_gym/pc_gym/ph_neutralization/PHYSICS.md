# pH neutralisation: physics model, sources, limits and deviations

The plant is a continuous stirred-tank pH neutralisation: a strong acid
neutralised by a strong base with a carbonate buffer, the Hall and Seborg and
Henson and Seborg bench rigs, standing for an industrial effluent
neutralisation unit. `REALITY.md` in this folder is the plant's record, with
sources [P1] to [P10]; `COLD_REVIEW.md` lists how the earlier task differed.
This page follows charter Amendments 1 and 2 (`docs/proposals/task-charter.md`).

**Named property: the titration curve.** The process gain swings by an order
of magnitude across the operating range, and the unmeasured buffer moves the
curve [P1; P2]. It is in the model below (rule 18).

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

| Omitted | Why |
|---|---|
| Temperature dependence of pK1, pK2 | Isothermal, as the bench rig. |
| Mixing dynamics | One perfectly mixed tank; the mixing delay is lumped into the probe's dead time (section Readings). |
| Staging | Real units run two or more tanks or mixers [P3; P5, p. 2; P7]. One tank is the rig [P1, P2]. Deviation D4. |
| Electrode drift and fouling | Named failures [P5, pp. 2-3; P6], not modelled. Deviation D1. |
| Several buffer species | Carbonate only. Deviation D2. |

---

## 2. Numbers: cited or ours

Rule 1: every number is cited or labelled `ours` with a reason.

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| `V` | 2900 | mL | ours: the Henson and Seborg rig volume as the benchmark literature copies it; [P2] could not be opened (REALITY.md, owner list) |
| `q1`, `Wa1`, `Wb1` | 16.6, 3.0e-3, 0 | mL/s, M | ours: acid stream as copied from [P2], unverified |
| `q2_nominal`, `Wa2`, `Wb2` | 0.55, -3.0e-2, 3.0e-2 | mL/s, M | ours: buffer stream as copied from [P2], unverified |
| `Wa3`, `Wb3` | -3.05e-3, 5.0e-5 | M | ours: base stream as copied from [P2], unverified |
| `pK1`, `pK2` | 6.35, 10.25 | - | cited: carbonic acid constants at 25 C, as in [P2] |
| `q3_min` | 0 | mL/s | cited by absence: no source gives a floor on reagent flow, and a pump can stop (REALITY.md, "What they move") |
| `q3_max` | 22 | mL/s | ours: rig valve size not found; covers the acid upsets at +20 % |
| `q2_noise_std` | 0.35 | mL/s | ours: buffering changes are sourced [P1; P2], their size is not |
| `BUFFER_OU_THETA` | 2e-3 | 1/s | ours: shape and rate of buffer changes not found |
| `q2_min`, `q2_max` | 0, 2.5 | mL/s | ours: keeps the buffer flow physical |
| `acid_upset_size` | 0.2 | - | ours: large acid strength changes are sourced [P3, p. 4], their size is not |
| `acid_upset_ramp` | 600 | s | ours: the "calm to rocking during one lunch break" of [P3, p. 8] read as minutes |
| `permit_low`, `permit_high` | 6, 8 | pH | cited: the example spec pH 7 +- 1 [P7, p. 2]; a real permit's range is set by the guideline [P8] |
| `excursion_limit` | 3600 | s | cited: no single excursion over 60 min [P8] |
| `sewer_floor` | 5.0 | pH | cited: no discharge to a sewer below pH 5.0 [P9] |
| `setpoint_pH` | 7 | pH | ours: the centre of the permit range; the set point is the operator's [P3, p. 4] and not scored |
| `delta_t` | 5 | s | ours: twice the 2.5 s the 7 to 10 s mixing delay [P7, p. 2] needs to show as 2 steps |
| `max_steps_in_episode` | 1440 | steps | ours: 2 h, past the 60 min excursion [P8] |
| `reagent_price` | 1 | L/L | ours: the cost is reagent litres, the field plant's reagent to influent metric [P3, p. 9] |
| `excursion_price` | 1.667 | L/s | ours: no price found; a minute outside costs 100 minutes of hold reagent |
| `diversion_price` | 1 | L/L | ours: a diverted litre is treated again, about 1 L of this dilute base per litre |
| `failure_cost` | 8.443 | L | ours: one step outside at the full valve, the worst step a surviving plant pays, so the violation costs another hour of that |
| `restart_steps` | 720 | steps | ours: a violation is charged as 60 min more outside; no fine found |
| `e_floor` | 1.0 | pH | cited: the permit half-width [P7, p. 2]; read by the time-in-band KPI only, not by the reward |
| `precision_floor`, `pH_min`, `pH_max`, `reagent_cost_weight` | 0.01, 2, 12, 0 | - | the retired version-1 reward only |

Count: 7 cited (including the absence of a reagent floor), 18 ours.

---

## 3. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Discharge outside pH 6 to 8 | Zone, timed | Priced per second outside (`excursion`); one excursion past 60 min is a permit violation, the task's trip | [P8]; range [P7] |
| Discharge below pH 5.0 | Protected | Below 5.0 the effluent is diverted, not discharged: no trip, the excursion clock holds, and the diverted volume is charged for treatment (`diversion`) | [P9]; diversion [P3, pp. 4-5] |
| Reagent valve | Hardware | 0 to 22 mL/s, clipped | ours (section 2) |
| Level | Not modelled | Continuous overflow tank, fixed volume | [P4, p. 1] is a batch interlock |

The monthly allowance of 7 h 26 min outside [P8] is longer than an episode,
so only the 60 min rule is modelled. The violation restarts the plant on its
steady state at the set point (`base.failure_kernel`). A real violation does
not restart a plant; it is reported. Deviation D5.

---

## 4. Observation (rule 16)

| obs | meaning | source |
|---|---|---|
| 0 | tank pH, through the probe (section Readings) | probe in the tank [P5, p. 1; P6] |
| 1 | base valve, percent of its range | the operator's own valve |
| 2 | set point, pH 7 | the operator's own |
| 3 | minutes outside the permit in this excursion | the discharge monitor records it [P5, p. 2; P8] |

With one mixed tank the discharge sensor [P5, p. 2] reads the tank's pH. The
influent flow is constant here, so its meter [P3, Figure 2] reads a constant
and is left out. The buffer flow, the acid strength and the invariants are
hidden: no source shows them measured.

---

## 5. Episode (rule 17)

The tank starts on the steady state that holds pH 7 under the buffer flow it
meets, as a loop in automatic leaves it, with the reagent valve where that
needs it. Two things move:

- **Influent acid strength** [P3, p. 4]: two changes per episode at times
  drawn uniformly over it, each to a level within +-20 % of nominal, ramped
  over 10 min. Hidden. Size, rate and count are ours (section 2).
- **Buffer flow** [P1; P2]: an Ornstein-Uhlenbeck drift, mean 0.55 mL/s,
  std 0.35 mL/s, correlation time 500 s. Shape and size ours.

Nothing else moves: the set point does not, as no source shows it moved
from outside.

---

## 6. Score (rule 15)

The job, from REALITY.md: keep the effluent pH inside the discharge permit by
adding reagent [P5, p. 1], without overdosing [P5, p. 2]. Per step, in litres
of reagent:

```
reward = -(reagent + excursion + diversion + failure)
reagent   = q3 dt / 1000
excursion = 1.667 dt          while the discharge is outside pH 6 to 8
diversion = (q1 + q2 + q3) dt / 1000   while the tank is below pH 5.0
failure   = 720 x 8.443       on the step an excursion passes 60 min
```

Inside the permit only reagent counts: precision earns nothing, as no source
says it pays. A tank held near pH 6 uses less base than one at 7, and that is
the trade the field plant made by moving set points toward the limit [P3,
p. 4].

---

## 7. Behaviour tests

`tests/pc_gym/ph_neutralization/test_ph_reality.py`, from COLD_REVIEW.md's
proposals with their tolerances:

| Test | Source | Before | After |
|---|---|---|---|
| Mixing and measurement delay 5 to 20 s | [P7, p. 2] | pass | pass |
| Buffer shifts the curve by at least 0.1 pH | [P1; P2] | pass | pass |
| Every step below pH 5.0 is diverted | [P9; P3] | fail | pass |
| An excursion under 60 min is no violation | [P8] | pass | pass |
| An excursion past 60 min trips within one step | [P8] | fail | pass |
| Inside the band, less reagent scores at least as well | [P5; P3] | fail | pass |

---

## 8. Known deviations (rule 13)

**D1. Electrode drift and fouling.** The probe lags and reports late, with
noise, and stays that fast. Coating and junction blockage make real probes
sluggish and drive overdosing [P5, p. 2; P6]. Effect on the score: easier.

**D2. One buffer species.** Carbonate only. Real effluents carry several weak
acid and base pairs. Effect: a smoother curve than a real effluent's.

**D3. Ideal valve.** No resolution, stick-slip or limit cycles from split
range [P3, p. 4]; no on/off or timer dosing [P4]. Effect: easier near the
equivalence point.

**D4. One tank.** Real strong acid units are staged [P3; P7, p. 1]. Effect: a
single tank's gain, and the 88 s residence time of a bench rig, far under the
10 min that [P4, p. 2] asks of a continuous tank under on/off control.

**D5. A violation restarts the plant.** The framework's trip restarts on the
steady state at the set point. A real violation is reported and the plant runs
on. Effect: after a violation the episode resumes from a good state, at the
charged cost.

**D6. Rig constants unverified.** V, the flows and the concentrations are the
benchmark's copy of [P2], which could not be opened (REALITY.md, owner list).

---

## 9. What the reference ladder row needs

For `scripts/reference_ladder.py` (rule 3, not edited here):

- Bump tests at the steady state of reset, valve stepped by +-0.5 mL/s, read on
  the probe, with the acid upsets and the buffer drift off.
- PI by SIMC on the permit centre, then feedforward: none is available, since
  the influent flow is constant and its acid strength unmeasured.
- Cascade: no second measurement. Override: the excursion clock (obs 3) is
  available for a rung that drives back inside before 60 min.
- The rung is scored on the job cost of section 6, not on tracking.
- The oracle's stage (`oracle.py`) has no slope at the permit edge and must be
  rebuilt before rule 9 runs it.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `ph_neutralization` | 1440 | 5 | 2.0 h | 1 in [-1, 1] | 4 | 14 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Readings

The observation reports pH through its electrode, and the operator's own base
flow and the setpoint exactly (`docs/proposals/measurement-noise.md`).
`sensor_noise_scale = 0` gives the noise-free task, rounding included.

| reading | sigma | step | lag | dead time | instrument | status | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pH` | 0.01 pH | 0.01 pH | 10 s | 10 s | Glass electrode with industrial pH transmitter | Provisional | [McMillan](https://www.controlglobal.com/articles/2013/mcmillan-ph-response-accuracy-reliability) |

**This figure is weak, and it is the tracked quantity.** No noise source was
found. McMillan gives 0.02 pH only as an accuracy target that needs best
practice, with no coverage factor, so 0.01 as 2 sigma is ours. A transmitter's
±0.01 pH on simulated input, with no electrode, would be a lower bound. The
0.01 step is this project's `precision_floor`, which cites nothing.

**The lag is 10 s, for the owner to accept.** [McMillan's Tip 86](https://www.automation.com/article/tip-86-test-the-ph-response-time-to-determine-the-health-of-a-glass-electrode)
gives time constants for a clean electrode in a buffered solution: 0.75 and 1.5 s at
4 ft/s, 3 and 12 s at 0.5 ft/s, for a rising and a falling pH. 10 s sits inside the
0.5 ft/s pair, which is the speed here: this tank's 33 mL/s runs at 0.5 ft/s through
a 17 mm line (derived). The line is ours, and the tank is a fast place for it: a
probe in the mixed volume would see a different speed. It is not fouling, which is
D1. The same tip warns that electrodes approach their final value more slowly than a
first-order lag above 90 %, which this lag leaves out.

**The dead time is 10 s, for the owner to accept.** The tip calls the electrode's own
dead time usually negligible, so this is not the electrode's. It stands for mixing
and transport to the probe, which the tank here does not model. Faanes and Skogestad
([Comput. Chem. Eng. 28, 2004](https://skoge.folk.ntnu.no/publications/2004/faanes_cce_ph/ph.pdf))
take 10 s as "typical" and write that the mixing and measurement delay is "about 10
s or larger", with an example of 5 s of transport and 5 s of measurement. Their tanks
are industrial, and their figure lumps in measurement and valve dynamics, so the lag
and this delay together sit above it. At 33 mL/s, 10 s is also 330 mL of line, about
1 m of 20 mm pipe (derived). The step is 5 s, so this is exactly 2 steps. pH keeps
its Provisional status, which covers the noise.

The expert reads these readings, and so does every tuner. The oracle plans and
acts on the true state, so it stays a full-state bound. `reward`, trips and the
score read the true state. `info["measured_reward"]` is the same cost on the
readings, with any trip taken from the true state, and a learner can train on
it.

The score reads the true pH. A probe reading inside the band while the tank is
just outside is the operator's risk, as on the plant.

## Written exception to check 2

The credibility check on a sourced real figure has nothing to read here. The owner
accepted a class exception on 2026-11-06: no published figure could be found or opened
after two searches with a re-fetch check. The plan in docs/proposals/industrial-fixes.md
says what was tried.
