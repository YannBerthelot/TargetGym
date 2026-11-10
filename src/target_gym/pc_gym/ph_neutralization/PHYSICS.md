# pH neutralisation: physics model, sources, limits and deviations

The plant is a two-stage continuous pH neutralisation: a strong acid
neutralised by a strong base with a carbonate buffer, in two stirred tanks in
series, a member of the industrial effluent neutralisation family. Its
chemistry is the Hall and Seborg and Henson and Seborg bench rigs'; its
staging and retention are the family's. `REALITY.md` in this folder is the plant's record, with
sources [P1] to [P10]; `COLD_REVIEW.md` lists how the earlier task differed.
This page follows charter Amendments 1 to 3 (`docs/proposals/task-charter.md`).

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

Rule 1, with Amendment 3's rule 19: a number is cited, or ours inside the
family's documented range, or ours with a reason.

| Symbol | Value | Unit | Cited or ours |
|---|---|---|---|
| `V_s1`, `V` | 20000, 20000 | mL | ours, family: two stages [P3; P7], each over the 10 min retention [P4, p. 2] at the nominal 32.8 mL/s (10.2 min) |
| `q1`, `Wa1`, `Wb1` | 16.6, 3.0e-3, 0 | mL/s, M | ours: acid stream as the benchmark copies [P2], unverified |
| `q2_nominal`, `Wa2`, `Wb2` | 0.55, -3.0e-2, 3.0e-2 | mL/s, M | ours: buffer stream as copied from [P2], unverified |
| `Wa3`, `Wb3` | -3.05e-3, 5.0e-5 | M | ours: base stream as copied from [P2], unverified |
| `pK1`, `pK2` | 6.35, 10.25 | - | cited: carbonic acid constants at 25 C, as in [P2] |
| `q3_min` | 0 | mL/s | cited by absence: no source gives a floor on reagent flow |
| `q3_max` | 22 | mL/s | ours: valve size not found; covers the upsets |
| `q3_trim_max` | 2.2 | mL/s | ours, family: a tenth of stage 1's, as the 1 and 0.1 GPM final elements of [P4, p. 2] |
| `q2_noise_std`, `BUFFER_OU_THETA` | 0.35, 2e-3 | mL/s, 1/s | ours: buffering changes are sourced [P1; P2], their size and rate are not |
| `q2_min`, `q2_max` | 0, 2.5 | mL/s | ours: keeps the buffer flow physical |
| `acid_upset_size`, `acid_upset_ramp` | 0.2, 600 | -, s | ours: large acid changes are sourced [P3, p. 4], their size is not |
| `flow_swing` | 0.2 | - | ours: influent flow swings (Amendment 3), metered [P3, Figure 2]; size not found |
| `event_hold` | 2 | h (stream) | ours: how long an influent upset holds before it ramps back, until the producing unit's batch ends; no source gives it; a window never sets it, so its upsets never return |
| `window_upsets` | 1, 0 | - | the window's own upsets on (1), or off in a stream (0), which starts its own |
| `permit_low`, `permit_high` | 6, 8 | pH | cited: the example spec pH 7 +- 1 [P7, p. 2] |
| `excursion_limit` | 3600 | s | cited: no single excursion over 60 min [P8] |
| `sewer_floor` | 5.0 | pH | cited [P9] |
| `basin_volume` | 236.2 | L | ours, family "hours of flow" (Amendment 3): 2 h of nominal discharge |
| `pumpback_flow` | 4.15 | mL/s | ours: a quarter of the influent, so the basin drains in about 16 h |
| `setpoint_pH` | 7 | pH | ours: the permit centre; the set point is the operator's [P3, p. 4] |
| `delta_t`, `max_steps_in_episode` | 5, 1440 | s, steps | ours: 2 h, past the 60 min excursion [P8] |
| `reagent_price` | 1 | L/L | ours: the score is reagent litres, the field plant's metric [P3, p. 9] |
| `excursion_price` | 1.667 | L/s | ours: no price found; a minute outside costs 100 minutes of hold reagent |
| `diversion_price` | 2 | L/L | ours: above reagent (Amendment 3): a diverted litre is pumped back and treated again, about 1 L of this dilute base per litre, plus the pumping |
| `failure_cost` | 9.0 | L | ours: above the worst surviving step (outside, both valves full, diverting at full flow, 8.96 L) |
| `restart_steps` | 720 | steps | ours: a violation is charged another hour; no fine found |
| `e_floor` | 1.0 | pH | cited: the permit half-width [P7, p. 2]; read by the time-in-band KPI only |
| `precision_floor`, `pH_min`, `pH_max`, `reagent_cost_weight` | 0.01, 2, 12, 0 | - | the retired version-1 reward only |

---

## 3. Limits (rule 14)

| Limit | Kind | Model | Source |
|---|---|---|---|
| Discharge outside pH 6 to 8 | Zone, timed | Priced per second outside; one excursion past 60 min is a violation, the trip. The clock runs whether or not the effluent is diverted (Amendment 3) | [P8]; range [P7] |
| Discharge below pH 5.0 | Protected | Diverted into a finite basin, priced per litre at 2 L of reagent, pumped back to stage 1 when the discharge recovers; an overrun basin is a violation | [P9]; diversion [P3, pp. 4-5]; basin Amendment 3 |
| Reagent valves | Hardware | 0 to 22 and 0 to 2.2 mL/s, clipped | ours (section 2) |
| Level | Not modelled | Continuous overflow tanks | [P4, p. 1] is a batch interlock |

The monthly allowance of 7 h 26 min outside [P8] is longer than an episode,
so only the 60 min rule is modelled. A violation restarts the plant on its
steady state at the set point (`base.failure_kernel`); a real one is reported
and the plant runs on (D5).

---

## 4. Observation (rule 16)

| obs | meaning | source |
|---|---|---|
| 0 | discharge pH, stage 2's probe (section Readings) | [P5, p. 1; P6] |
| 1 | stage-1 reagent valve, percent | the operator's own |
| 2 | set point, pH 7 | the operator's own |
| 3 | minutes outside the permit in this excursion | the monitor [P5, p. 2; P8] |
| 4 | stage-1 pH, its probe | a transmitter per stage [P3, pp. 4-5] |
| 5 | stage-2 trim valve, percent | the operator's own |
| 6 | influent flow, mL/s | the waste flow transmitter [P3, Figure 2] |
| 7 | basin level, percent | ours: an operator reads a basin's level |

The buffer flow, the acid strength and the invariants are hidden: no source
shows them measured.

---

## 5. Episode (rule 17)

Both tanks start on the steady state that holds pH 7 under the buffer flow
they meet, the trim valve shut and the basin empty, as a loop in automatic
leaves them. What then moves is set by the scenario (section 6): influent acid
strength [P3, p. 4], influent flow, and the buffer's Ornstein-Uhlenbeck drift
[P1; P2]. Each upset comes twice, at times drawn over the episode, to a level
drawn within its size, ramped over 10 min.

---

## 6. Scenarios (rule 20)

`scenarios.py` in this folder, on `EnvSpec.scenarios`:

| Scenario | What moves | Source |
|---|---|---|
| `normal` | buffer drift only | [P5, p. 1]; [P2] |
| `acid_strength` | acid strength, +-20 % | [P3, p. 4] |
| `flow_swing` | influent flow, +-20 %, metered | [P3, Figure 2]; Amendment 3 |
| `buffer_loss` | buffering falls away (nominal buffer 0) | [P1]; [P2] |
| `bad_day` | acid strength and flow together | [P3, p. 8] |

Probe coating [P5, pp. 2-3; P6] is documented as a bad day but has no sourced
rate, so it is not a scenario yet (D1).

---

## 7. Score (rule 15)

The job: keep the discharge inside the permit by adding reagent [P5, p. 1],
without overdosing [P5, p. 2]. Per step, in litres of reagent:

```
reward    = -(reagent + excursion + diversion + failure)
reagent   = (q3 + q3_trim) dt / 1000
excursion = 1.667 dt                       while the discharge is outside 6 to 8
diversion = 2 (q1 + q2 + q3 + q3_trim) dt / 1000   while it is below 5.0
failure   = 720 x 9.0                      on a violation (60 min, or basin overrun)
```

Inside the permit only reagent counts. Adding no reagent now costs far more
than holding the band (section 8), where the review measured it within 1.1 to
1.4 times a perfect hold.

---

## 8. Behaviour tests

`tests/pc_gym/ph_neutralization/test_ph_reality.py`:

| Test | Source | Result |
|---|---|---|
| Each stage holds at least 10 min | [P4, p. 2]; [P3; P7] | pass |
| Mixing and measurement delay 5 to 20 s on the stage-1 probe | [P7, p. 2] | pass |
| Buffer shifts the curve by at least 0.1 pH | [P1; P2] | pass |
| Every step below pH 5.0 is diverted | [P9; P3] | pass |
| An excursion under 60 min is no violation | [P8] | pass |
| An excursion past 60 min trips within one step | [P8] | pass |
| Diverting does not stop the hour | Amendment 3 | pass |
| An overrun basin is a violation | Amendment 3 | pass |
| Inside the band, less reagent scores at least as well | [P5; P3] | pass |
| No reagent costs at least five times a hold, and trips | Amendment 3 (five times is ours) | pass |
| Influent flow swings are metered and move the plant | [P3, Figure 2] | pass |

`test_ph_experts.py` checks that the oracle plans on the buffer flow it reads.

---

## 9. Known deviations (rule 13)

**D1. Electrode drift and fouling.** The probes lag and report late, with
noise, and stay that fast. Coating and junction blockage make real probes
sluggish and drive overdosing [P5, p. 2; P6]. Effect: easier.

**D2. One buffer species.** Carbonate only. Effect: a smoother curve than a
real effluent's.

**D3. Ideal valves.** No resolution, stick-slip or split-range limit cycles
[P3, p. 4]; no on/off or timer dosing [P4]. Effect: easier near equivalence.

**D4. Bench chemistry at industrial retention.** The streams are the bench
rig's [P2] in tanks sized to the family's retention; the flows are millilitres
per second. The dynamics scale with residence time, so this picks a family
member, not a plant.

**D5. A violation restarts the plant.** A real violation is reported and the
plant runs on. Effect: after a violation the episode resumes from a good state,
at the charged cost.

**D6. Rig constants unverified.** The streams are the benchmark's copy of [P2]
(REALITY.md, owner list).

---

## 10. What the reference ladder row needs

- Zone: the discharge pH in 6 to 8 [P7]; the cheap side is low (pH 6), since
  less base holds a lower pH (19 % less at pH 6 than at 7 at nominal buffer).
  The reference sits 2 sigma above 6 (`ladder.zone_setpoint`, rule 22); stage
  1 has no zone and follows the same setpoint.
- Scenarios: the five of section 6.
- Two valves: stage 1 coarse, stage 2 trim. Bump tests at the reset steady
  state with `acid_upset_size`, `flow_swing` and `q2_noise_std` at 0.
- Feedforward is now available from the metered influent flow (obs 6), through
  the steady balance (`steady_base_flow`), as the field case names [P3, p. 3].
- The expert (`experts.py`) runs the linearised PI on stage 1 and a trim PI on
  the discharge with the same gains; its stored gains predate the staging.

## 11. Oracle needs

The oracle (`oracle.py`, NewtonMPC with `action_dim=2`) reads the true state,
the buffer included, and plans the buffer's mean path (tested). Its stage cost
is the reward's terms on the planner pH, so the excursion and diversion terms
have no slope at the permit's edges: before rule 9 runs it, the stage needs a
smooth hinge at 6, 8 and 5, and the 1440-step episode needs its horizon
re-measured for 20 min of residence. Not run.

---


### Scenario weights (charter J4)

`normal` 0.7, `acid_strength` 0.1, `flow_swing` 0.1, `buffer_loss` 0.05, `bad_day` 0.05: shares of operating time for the headline. No source says how often each
window occurs (`REALITY.md`), so each is ours, the normal day in the family
range 0.5 to 0.9 and the rest split among the upsets; the family sweep (B7)
draws them.

### The stream (docs/proposals/long-strata.md, S1)

As a continuing plant (`target_gym.stream`) the normal day lasts 24 h on average
and each upset regime a shift, 8 h. `acid_strength`, `flow_swing` and `bad_day`
bring one upset per visit on average (`events_per_h` 1/8); `buffer_loss` is the
regime itself and brings none. An upset ramps over `acid_upset_ramp` to a level
drawn within its scenario's size, holds `event_hold`, then ramps back to nominal,
so it runs to its return through a switch of regime; an upset of one kind never
ends one of the other. Each schedule counts its steps from its own onset, so a
ramp keeps its shape however long the plant runs. No source gives how often or
how long (`REALITY.md`), so these numbers are ours, set once without tuning and
open for the owner. In a window nothing changes: the window's upsets run as drawn,
the schedules count from step 0, and a scored window is bit-identical.

The permit monitor's excursion clock is a record, not part of the outside world:
a trip's restart keeps it through `restart_plant`, as before, and `OUTSIDE_WORLD`
names only what a policy cannot move (long-strata S2).

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `ph_neutralization` | 1440 | 5 | 2.0 h | 2 in [-1, 1] | 8 | 29 |

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
