# Reactor v7: physics review, handover to the implementation session

Status: review of `claude/reactor-axial-dd4617` at 41f5145 (reactor-v7), 2026-11-10.
No code changed. Written for the session that implements the reactor; the owner
decides every item marked **owner's call** before any code or rerun (charter J1: rules
are fixed before results are seen, never after).

The reviewer's brief: read the task as a PWR engineer, and flag clear mistakes in the
physics, missing phenomena that matter for control, and figures or decisions a
reviewer would question. The goal is control fidelity, not a model of one real plant.

## Verdict

Reactor is a good TargetGym task. Its control property is the real one: rods are fast
but tilt the power, boron is uniform but 5 to 15 min late and costs effluent, xenon
moves both the balance and the shape over hours, and the return to power is where the
three collide. Most of the physics holds up (section "Checked and holding", so it is
not redone). Five findings remain.

| # | Finding | Severity | Owner's call | Main files |
|---|---|---|---|---|
| 1 | Tavg above its program is free up to the DNB value, 583 K, at every power | High | yes, it reverses Amendment 3 decision 7 | `env.py`, `experts.py`, `oracle.py`, `PHYSICS.md` |
| 2 | The moderator coefficient is a late-cycle value on a beginning-of-cycle core | Medium | yes, which cycle point | `env.py`, tests with hand values |
| 3 | Every day starts with no axial xenon tilt | Low | yes, whether to do it now | `env_jax.py` reset |
| 4 | The hot-leg envelope, 620 K, sits above saturation, 618.0 K | Low | no | `env.py:177` |
| 5 | The public environment page and some docstrings describe reactor-v4 | Low | no | `docs/environments/reactor.md`, docstrings |

Severity: **High** gives an agent a freedom real operators lack and changes what a good
policy does. **Medium** shifts the difficulty materially without a loophole. **Low** is an
edge case or a visibility issue.

## How to work these

- **Test first** (TESTING.md, project memory): write each spec-derived test below, see it
  fail on 41f5145, then change the code. Expected values come from the sources or the
  hand derivations given here, never from running the env.
- **Batch 1 and 2.** Both change Tavg's economics, and any `env.py` edit stales the
  fingerprints, baselines and the ladder. Decide both with the owner, implement both,
  then rerun once: `scripts/reactor_ladder.py`, `scripts/reactor_nulls.py`, the ceilings
  (`scripts/reactor_ipopt.py`, `scripts/reactor_foresight.py`) and
  `scripts/reactor_sensitivity.py`, on bob.
- **The expert's gains** come from bump tests by a fixed recipe (SIMC, least gain kept).
  If finding 2 changes the plant, rerun the bump tests by that recipe; do not tune on
  cost.
- **4 and 5** are independent and cheap; they can land any time.
- Label every new number read, derived or ours, as PHYSICS.md does.

---

## 1. High: Tavg above its program costs nothing up to 583 K, at every power

### What the task does

- The tracking cost charges Tavg only below its program:
  `error = xp.maximum(state.T_ref - T_avg, 0.0)` ([env.py:963](../../src/target_gym/reactor/env.py)).
  The comment cites rule 15; the owner's Amendment 3 decision 7 set it ("Tavg above its
  program costs nothing below the DNB limit", reactor-axial.md).
- The only hot-side bounds are the DNB value, `tavg_dnb_limit` = full-power program plus
  2.83 K = **583.0 K absolute**, with 2 h allowed before hot standby, and the hot-leg
  envelope at 620 K.
- The expert uses that zone: `xenon_play`'s upper edge is the DNB limit
  ([experts.py:197](../../src/target_gym/reactor/experts.py)).

### Why it is wrong below full load

At part power the hot-side limit of a PWR is the steam side, not DNB. The Tavg program
slides down with load precisely to keep the steam pressure under the relief settings at
low load. The model's own steam generator reproduces this on the program, which makes
the off-program numbers credible:

| load | Tavg | T_sg (model) | steam pressure (IAPWS-IF97 saturation) |
|---|---|---|---|
| 1.0 | 580.2 K, program | 558.2 K | 989 psig |
| 0.0 | 564.8 K, program | 564.8 K | 1091 psig (Westinghouse no-load 1092 psig, HRTD 5.0) |
| 0.5 | 572.5 K, program | 561.5 K | 1039 psig |
| **0.5** | **583.0 K, the DNB value** | **572.0 K** | **1211 psig** |
| 0.75 | 583.0 K, the DNB value | 566.5 K | 1119 psig |

`T_sg = Tavg - P_thermal_ref x load / UA_sg`, at steady state.

1211 psig is past the 1185 psig main-steam design pressure and the safety valves' lift
settings of 1176 and 1190 psig (V.C. Summer technical specification bases, NRC
ML020580306, read from a search summary: nrc.gov blocks automated readers, so confirm
by hand). The atmospheric relief valves open lower still; a commonly quoted figure is
about 1125 psig, **not verified here**.

Both procedures the task already cites hold a symmetric band, not a one-sided one:

- EDF: the black rods move "if |Δθ| > 1.6 °C" (IAEA NP-T-3.23, annex II-5.1), on either
  side.
- Westinghouse: rod control has a ±1.5 °F dead band (NRC HRTD 8.1; `RodControlParams`
  in `experts.py` already encodes it).

And nothing fast stops a large hot excursion: at the hold, Tavg 20 K over its program
costs nothing for 2 h, where a real plant would lift its steam safeties within the first
few kelvin and its pressurizer would see the insurge.

### Impact

- At the 50 % hold, 10.5 K of free Tavg stores about **350 pcm** (hand derivation below),
  about a third of the **960 pcm** power defect from 100 % to 50 %. An agent can let Tavg
  ride hot through the ramp down and the xenon build-up, and skip boration and dilution
  that a real operator must do.
- The expert, the ladder, the room and the baseload finding were all measured with this
  freedom in place.

Hand derivation (all from `ReactorParams`): at steady state
`T_fuel = Tavg + 0.5 P/m_dot_cp + P/UA`, so at a fixed load of 0.5 a Tavg shift dT
shifts the fuel by dT too, and the reactivity moves by
`(alpha_coolant + alpha_fuel) dT = -33 pcm/K x 10.5 K = -348 pcm`. The power defect is
`alpha_fuel (T_fuel(0.5) - T_fuel(1)) + alpha_coolant (Tref(0.5) - Tref(1))`
= `-3e-5 x (-243.6 K) + -3e-4 x (-7.7 K)` = **+731 + 231 = 962 pcm**.

### Fix options (owner's call)

- **A, recommended: a symmetric dead band.** Charge Tavg past 1.6 K on either side of its
  program, at the same per-kelvin price. `R.tracking_cost` already takes `|error|` and
  subtracts `e_tol`, so the change is to drop the `xp.maximum(..., 0.0)` clamp at
  env.py:963 and pass the signed error. Then mirror `xenon_play`'s upper edge on its
  lower one: `hi = xp.maximum(dead_band / k - pull, 0.0)` against today's
  `lo = xp.minimum(-dead_band / k + pull, 0.0)`. It is the band both cited procedures
  keep, and it is already the number the cold side uses. Open sub-question for the owner: the charter prices lost output,
  and hotter steam gains a little output, so the hot side's price needs a stated reason
  (a procedure band priced like the cold one, as the offset band is a procedure band
  priced by its derate).
- **B: model the steam side.** Add steam relief when T_sg passes the relief setting,
  removing the excess heat, charged as lost output, and a trip at the safety-valve
  setting (T_sg about 570.5 K, 1185 psig). This is a missing phenomenon rather than a
  cost rule, so it fits "lost output only". But it leaves about 5 K free at the hold
  (with a 1125 psig relief setting) and needs the relief setting sourced. It also moves
  D1 ("the load goes straight to heat") slightly.
- A and B can be combined: A for the band, B's trip as the physical backstop.

Whatever is chosen, keep these consistent with the reward:

- `oracle.py`: its docstring says the objective is |Tref - Tavg| past the dead band, and
  it plans the DNB value as a hard bound (`Limit(hi=tavg_dnb_limit)`, line 162). Check the
  hinge and the full-day planners (`scripts/reactor_foresight.py`,
  `scripts/reactor_ipopt.py`) charge what the env charges.
- `PHYSICS.md`: the reward table's `e_tol` row, section 4's limits table, and section 6's
  "xenon tool's play" paragraph ("up to the DNB limit on the hot side").
- `reactor-axial.md`: the Amendment 3 table row for decision 7.

### Tests to write first

- **Cost on the hot side** (small): a state lined out at load 0.5 with Tavg 3 K over its
  program is charged a positive tracking cost under A. Expected: (3 - 1.6) K x $1.917
  per K = **$2.68 per 30 s control step**, from the reward table's $230/h per K. Fails
  today (charged $0).
- **Under B**: the steam pressure implied by T_sg never exceeds the safety-valve setting
  without a trip; expected values are IAPWS-IF97 saturation temperatures (table at the
  end), not the env's.
- **The expert stays inside the band through a load-follow day.** The existing check
  (`test_reactor_secondary.py:267`) is two-sided but only a median, on a quiet baseload
  with no ramp. Add one on an edf_daily seed that bounds the time Tavg spends more than
  1.6 K over its program during the hold.

---

## 2. Medium: a late-cycle moderator coefficient on a beginning-of-cycle core

### What the task does

`alpha_coolant = -30e-5` per K (-30 pcm/K, -16.7 pcm/°F,
[env.py:109](../../src/target_gym/reactor/env.py)), labelled "ours, inside -10 to -50
pcm/K, no figure opened". The rest of the core is at the beginning of a cycle:
`boron_ref` 1690 ppm (Krsko fig. 3), the dilution and boration rates at that boron, and
OAPS's 2 %/min ramps, which OAPS gives for the start of a cycle.

### Why it is inconsistent

- The MTC depends strongly on boron and is least negative at the start of a cycle, when
  boron is high; it "gradually loses significance as the boron concentration decreases"
  ([nuclear-power.com, MTC](https://www.nuclear-power.com/nuclear-power/reactor-physics/nuclear-fission-chain-reaction/reactivity-coefficients-reactivity-feedbacks/moderator-temperature-coefficient-mtc),
  opened). Its typical range there is 0 to -80 pcm/°C.
- Technical specifications allow a slightly positive MTC at the start of life at part
  power, which only makes sense because start-of-cycle values sit near zero.
- Values near -35 to -44 pcm/°F (-64 to -79 pcm/K) are end-of-life and 300 ppm
  surveillance limits (D.C. Cook Unit 2 Cycle 12 COLR, NRC ML17334A596, search
  summary only).
- At about 1700 ppm, a hot-full-power MTC near -5 to -10 pcm/°F (-9 to -18 pcm/K) is
  what the reviewer expects. **That range is engineering judgment, not a figure opened**;
  source it before using it (a Westinghouse startup physics test report or a COLR's
  beginning-of-life value).

### Impact

The MTC sets how far Tavg moves per pcm, which is the heart of Tavg control:

| | -30 pcm/K (now) | about -12 pcm/K |
|---|---|---|
| reactivity the 1.6 K dead band absorbs | ±53 pcm | about ±24 pcm |
| Tavg moved by 1 ppm of boron, fixed power | 0.24 K (8/33) | about 0.53 K (8/15) |

So Tavg control is roughly twice as forgiving as at the cycle point the rest of the task
describes, and the plant regulates itself more than it should.

### Fix options (owner's call)

- **A, recommended: stay at the beginning of the cycle**, MTC about -12 pcm/K once
  sourced. Every other sourced number (boron, rates, ramp rates, the damped axial mode)
  stays valid. One catch: `moderator_tilt` ([env.py:549](../../src/target_gym/reactor/env.py))
  scales with `alpha_coolant`, and that term carries the part-power axial shift that
  reproduces Krsko's +5 % delta I at half power. In a real core part of that shift is
  Doppler redistribution. So split it: give `moderator_tilt` its own lumped
  power-redistribution coefficient, kept at today's -30e-5 so the Krsko match holds, and
  change only the reactivity MTC. Then re-check the validation rows in PHYSICS.md
  section 2 (Krsko +5.4 %, 42 steps, 2 %/min, the inhour period).
- **B: move the core to mid-cycle**, boron about 800 to 1000 ppm, where -30 pcm/K fits.
  Dilution then runs at about half the rate, which is OAPS's own account of why load
  follow gets harder through the cycle. But OAPS's 2 %/min is a start-of-cycle figure,
  so the scenarios' ramp rates and the dilution validation would need mid-cycle sources
  too. Larger change.
- While there, `boron_worth` (-8 pcm/ppm, "ours") should be re-sourced at the same cycle
  point.

### Tests and hand values that move

- `test_reactor_physics.py:75` bounds `alpha_coolant` in [-50e-5, -10e-5]; tighten it to
  the chosen cycle point's sourced range.
- `test_reactor_reference.py:131` and `test_reactor_burnup.py` hold hand values built on
  -30e-5 (the 8/33 K per ppm rule). Re-derive them by hand from the new value, as the
  existing comments do; do not read them off the env.
- New: the power defect from 100 % to 50 % lies in a published range for the chosen
  cycle point (today 962 pcm, derivation in finding 1). Source the range first.

---

## 3. Low: every day starts with no axial xenon tilt

`reset_env` sets `I_tilt` and `Xe_tilt` to 0 ([env_jax.py:311](../../src/target_gym/reactor/env_jax.py)),
while the mean xenon carries the recent power (obs 10, "a unit that load-follows daily
is never quite at rest"). EDF units load-follow about 40 days a year, often on
consecutive days, and the axial mode lasts about 33 h. At this core's damping (-0.07 per
hour) about a fifth of yesterday's tilt survives a day (`exp(-0.07 x 24) = 0.19`), so
the effect is modest at the start of a cycle. But every load-follow day starts from the
best case.

**Fix (owner's call on timing):** draw the starting tilts from a stored table of
end-of-day states, per scenario, taken from the reference's runs on training seeds (not
held-out seeds). Optionally show a delta I history the way obs 10 shows the power
history; that changes the observation shape, so it is a separate decision. Can wait
until after 1 and 2.

---

## 4. Low: the hot-leg envelope sits above boiling

`T_coolant_max = 620.0` K ([env.py:177](../../src/target_gym/reactor/env.py)), commented
"the hot leg's saturation margin". Saturation at 2250 psia (15.51 MPa) is **618.0 K**
(IAPWS-IF97). Real plants keep the vessel exit below saturation with the overtemperature
ΔT trip. Today the DNB clock and the flux trip keep the hot leg near 600 K, so it rarely
binds, but while finding 1 is open a 2 h hot excursion can reach it.

**Fix:** about 610 K, labelled ours with that reason, or model the overtemperature ΔT
trip. **Test first:** the envelope sits below the IAPWS saturation temperature at the
nominal 2250 psia, 618.0 K, with the value derived in the test's comment.

---

## 5. Low: the public page describes a different reactor

- [docs/environments/reactor.md](../environments/reactor.md) is still reactor-v4: one
  action, 6 observations, 1440 steps (12 h), old parameters (`rho_ext_max` 0.005,
  `sigma_phi0` 7.95e-5, `gamma_ratio` 0.049, `T_fuel_ref` 1064) and old baselines
  (PID -5736, MPC -1205). It says "Cite as reactor-v4".
- Docstrings: [env.py:41](../../src/target_gym/reactor/env.py) and
  [env_jax.py:55](../../src/target_gym/reactor/env_jax.py) list 8 observations (it is
  11); [env_jax.py:92](../../src/target_gym/reactor/env_jax.py) says "the 12 h of a load
  cycle" (it is 24 h).
- PHYSICS.md section 4's limits table says "Tavg under 556 C"; it is 556 K.

**Fix:** bump the registry version and rerun `scripts/generate_env_pages.py` (its
`--check` mode is what CI reads); fix the three strings.

---

## Checked and holding (do not redo)

- **Xenon during load follow.** On the krsko_slow day (8 %/h down to 50 % and straight
  back), the model's mean xenon at the 50 % point is **1.142** times full-power
  equilibrium; Krsko reads 110 against 96, **1.146** (REALITY.md section 6). The retired
  scram-peak miss (1419 pcm) does not matter for a task without a scram, and the old burn
  rate (7.95e-5 /s) would overshoot Krsko, at 1.189. Keep `sigma_phi0`.
- **Steam generator.** 989 psig at full load and 1091 psig at no load (table in finding 1).
- **Power defect.** 962 pcm from 100 % to 50 % on the program, 731 of it Doppler:
  plausible for a PWR.
- **Boron.** About 4 ppm/min both ways at 1690 ppm (dilution `10/2.5e5 x 1690 x 60`,
  boration `3/2.5e5 x (7800 - 1690) x 60`), 32 to 35 pcm/min, which covers the power
  defect at about 2 %/min, as OAPS says.
- **Primary circuit.** 250 t of water matches `C_coolant + C_cold` (1.4e9 J/K at about
  5.5 kJ/kg/K); a 33 K rise and about 16 t/s of flow are typical of a 3 GW unit.
- **Two-point projections.** The rods' integral worth `y - sin(2 pi y) / (2 pi)` is the
  cos² integral over the rodded top, and their tilt weight `sin(pi y)^3` is the integral
  of cos x sin 2x over it, `(2H / 3 pi) sin^3(pi y)`: both correct first-order
  projections. In a two-node core the top-to-bottom coolant difference is half the rise
  whatever the power split, so leaving out a moderator feedback on the tilt is right.
- **Harmless at 30 s steps.** `Lambda_gen` 1e-4 s is 3 to 5 times a PWR's, and β = 0.0065
  is fresh U-235; neither shows at this step size. The pressurizer, decay heat and
  samarium do not bear on a day without a scram. Pooling EDF and Westinghouse limits is
  disclosed (REALITY.md) and fine for a family task.

## Reference values for tests

IAPWS-IF97 region 4 saturation temperatures (independent of the env):

| pressure | saturation temperature |
|---|---|
| 1092 psig (7.63 MPa abs), Westinghouse no-load steam | 564.9 K |
| 1176 psig (8.21 MPa abs), first safety-valve group, V.C. Summer | 570.0 K |
| 1185 psig (8.27 MPa abs), main-steam design pressure | 570.5 K |
| 2250 psia (15.51 MPa), nominal primary | 618.0 K |

## Sources

- [nuclear-power.com, moderator temperature coefficient](https://www.nuclear-power.com/nuclear-power/reactor-physics/nuclear-fission-chain-reaction/reactivity-coefficients-reactivity-feedbacks/moderator-temperature-coefficient-mtc): opened.
- [V.C. Summer technical specification bases, NRC ML020580306](https://www.nrc.gov/docs/ML0205/ML020580306.pdf): search summary only; the fetch was refused (403).
- [D.C. Cook Unit 2 Cycle 12 COLR, NRC ML17334A596](https://www.nrc.gov/docs/ML1733/ML17334A596.pdf): search summary only.
- [NRC generic fundamentals exam, PWR item 192004](https://www.nrc.gov/reactors/operator-licensing/history-rulemaking-activities/generic-fundamentals-examinations/pwr/pwr-files/192004.pdf): search summary only ("as boron increases, the MTC becomes less negative").
- The task's own sources (IAEA NP-T-3.23, OAPS, Krsko NENE 2022, NRC HRTD 5.0 and 8.1) as
  listed in REALITY.md.
