# Reactor rebuild: axial offset, rods and boron

Status: reactor-v9 (next section) is the current design, on top of v8, v7 and Amendment 3; the EDF rebuild
of 2026-11-08 and the first attempt of 2026-11-07 below it are history, on branch
`claude/reactor-axial-dd4617` (from `feat/sensor-noise` at a9c8ea8). Nothing is merged and
no shared baseline was re-recorded. The plant is described in
[`src/target_gym/reactor/PHYSICS.md`](../../src/target_gym/reactor/PHYSICS.md). This page
holds the measurements, under the rules of `task-charter.md`.

## Starting tilt and the family sweep, 2026-11-10 (reactor-v9)

**Finding 3** of the physics review: a day can start with yesterday's axial xenon tilt.
A tilt table of 10 end-of-day pairs comes from the reference on training seeds 0 to 99
of each load-follow scenario (`scripts/reactor_tilt_table.py`). A day draws one with
share 0.108, the share of a year's days that follow load (ours: 250 load-follow days
over a 5.5 + 50/60 day week, over 365). Delta I at reset is computed from the drawn
tilt. On seeds 100 to 109 the reference and null moved by a few dollars from v8.

### Family sweep (B7), 8 draws

`scripts/family_sweep.py` draws the four `ours` numbers that have ranges
(`reactor/ours.json`) on a Latin hypercube (seed 20261110). On each draw the reference
is re-tuned by its recipe's bump tests, and run against the null on seeds 100 to 109 in
every scenario. **The first sweep held on 2 of 8 draws**, 2 and 5, and draw 5 holds only
on medians: its reference trips on 1 to 10 seeds in every scenario.

| Draw | MTC, pcm/K | Boron worth, pcm/ppm | Reference beats null | Reference trips (of 10 seeds) |
|---:|---:|---:|---|---|
| 0 | -2.5 | -6.8 | 5 of 6, not fast_return | 0 to 10 |
| 1 | -19.3 | -9.6 | none | 10 everywhere |
| 2 | -16.9 | -10.5 | all | 0 |
| 3 | -13.5 | -7.9 | none | 10 everywhere |
| 4 | -7.9 | -8.8 | 3 of 6 | 10 everywhere |
| 5 | -3.7 | -8.5 | all, on medians | 1 to 10 |
| 6 | -12.4 | -7.5 | none | 10 everywhere |
| 7 | -8.4 | -10.4 | none | 10 everywhere |

The null at rest costs nothing on every draw (`at_rest_cost` 2e-6 dollars, no trip).

**Why the reference tripped (corrected):** the recipe's bump test for boron on Tavg
stepped 20 % of full dilution for an hour. That heats Tavg about 0.2 K a step, so the
plant tripped mid-test (the 610 K hot leg at full power, the DNB clock at half power on
some seeds), and the slope fit read across the restart. The gain then came out with
either sign: +0.015 to +0.024 instead of about -0.021 on draws 1, 3 and 7, so a cold
Tavg brought more boration and the floor's clock tripped the plant. The nominal plant
passed only because the design takes the gentler of the two operating points, and its
half-power reading was clean. The 610 K envelope came from the v8 fixes; the bump test
was never re-checked against it, which is amendment 5's inherited-limit case. A first
reading blamed the day's drift; a run without the step showed the drift is about 0.07 K
an hour and changes nothing.

### The fixes, owner's picks, and the re-sweep

- **The bump is sized from the curve book** (`tavg_bump` in `scripts/reactor_ladder.py`):
  the operator knows the boron worth and the MTC, so the dilution step makes the
  predicted rise over the hour after the line's delay end at the DNB limit, at most
  20 % of full flow. About 2.6 % on the nominal plant at full power. A paired run without
  the step takes out the drift. `tests/reactor/test_reactor_bumps.py` checks the slope
  against the hand value within 20 % and that the plant stays inside its limits.
- **MTC and boron worth move together** in the sweep, on one shared coordinate: the most
  negative MTC (-20 pcm/K) with the largest worth (-10.9 pcm/ppm), since both follow the
  core's boron (ours, from the physics). `tests/test_family_sweep.py` checks it.

The nominal gain barely moved (t_Kc -0.0209 to -0.0212), and the nominal ladder moved
under 1 % (edf_daily $4,581, baseload $239, fast_return still 2 trips). The room and
sensitivity below stand.

**Re-sweep: the property holds on 6 of 8 draws.**

| Draw | MTC, pcm/K | Worth, pcm/ppm | Holds | Reference trips (of 10) | fast_return, reference against null |
|---:|---:|---:|---|---|---|
| 0 | -19.1 | -10.7 | yes | none | $5,667 / $84.8M |
| 1 | -7.0 | -8.0 | no | fast_return 10, edf 1 | $95.74M / $95.74M |
| 2 | -10.2 | -8.7 | yes | fast_return 2 | $10,155 / $37.2M |
| 3 | -13.3 | -9.4 | yes | none | $7,762 / $84.0M |
| 4 | -4.0 | -7.4 | yes, on medians | fast_return 10, edf 3 | $62.6M / $64.3M |
| 5 | -16.3 | -10.1 | yes | none | $6,524 / $75.8M |
| 6 | -2.9 | -7.1 | no | fast_return 10, edf 5 | $50.02M / $50.02M |
| 7 | -11.5 | -9.0 | yes | none | $8,807 / $48.2M |

The failures sit at the weak-MTC, high-boron end (-2.9 to -7 pcm/K): the rods-on-Tavg
mode trips on every fast_return seed there, as the null does. With a weak MTC a load cut
releases little reactivity, and the plant's own feedback holds Tavg less, so the rod
program alone is not enough. Not fixed: whether that end of the range is a member that
runs 5 %/min returns at all is the owner's call (amendment 5, B5 scenarios).

### The trip guard and the narrowed MTC (owner, 2026-11-10)

The trip flag was raised during the bad bump; the recipe dropped it. Each step returns
`info["tripped"]`, and `closed()` kept only the states. It now raises
`TrippedDuringTest`, the sweep counts a refused tune as a failed draw, and a test checks
that the old 20 % step is refused. The same audit across the other tasks ran in its own
session (11 sites fixed, branch `claude/awesome-snyder-910825`; the cement kiln's
shipped inner loop was fitted across a restart).

The owner narrowed the MTC to -9 to -18 pcm/K, the physics review's judgment for this
member (finding 2, in PHYSICS.md before the sweep), still drawn with the boron worth.

**Re-sweep on the narrowed range: the property holds on 8 of 8 draws.** No tune was
refused, the null at rest costs nothing on every draw, and the reference has no trip
outside fast_return. fast_return still trips at the weak end:

| Draw | MTC, pcm/K | fast_return trips (of 10) | fast_return, reference / null medians |
|---:|---:|---:|---|
| 0 | -17.6 | 0 | $5,856 / $84.8M |
| 1 | -11.8 | 2 | $10,408 / $90.4M |
| 2 | -13.4 | 0 | $8,300 / $36.2M |
| 3 | -14.8 | 0 | $7,322 / $79.3M |
| 4 | -10.4 | 6 | $30.8M / $59.1M |
| 5 | -16.3 | 0 | $6,538 / $73.7M |
| 6 | -9.9 | 10 | $45.9M / $50.0M |
| 7 | -14.0 | 0 | $7,829 / $51.1M |

So at -10 pcm/K and weaker, the rods-on-Tavg mode loses most fast returns, and the
property holds there only because the null loses them all. Draw 4's edf_daily median is
$274,567 with no trip (other days about $4,000), not yet read. Both left as findings.

Also found: `test_a_plan_on_the_loads_ramp_replays_on_the_env` (medium tier, never run
in this rebuild) fails with a replay gap of 0.86 to 0.99, already on reactor-v6, before
any of today's changes. The IPOPT ceilings replay their plans on the env, so their
costs stand; the planner's model is not exact on that test.

### Ceilings, room and price sensitivity on reactor-v9

Adam foresight, 3000 iterations, seeds 100 to 109. Room is the reference's cost over
the ceiling, the mean over seeds.

| Scenario | Reference mean | Ceiling mean | Room |
|---|---:|---:|---:|
| edf_daily | $4,609 | $2,575 | 1.94 |
| edf_short_hold | $4,329 | $2,166 | 2.01 |
| krsko_slow | $1,740 | $750 | 2.32 |
| krsko_shallow | $1,157 | $416 | 2.79 |
| baseload | $240 | $90 | 2.65 |
| fast_return | $8.3M | $2.0M | 1.89 |

- fast_return's mean is the two tripping seeds (107 and 109, rooms 7.8 and 3.0); Adam
  stalls at its first plateau there. On the eight seeds without a trip the room is 1.00
  to 1.04: Adam finds nothing under the reference.
- edf_daily seed 101 also stalls (room 1.00).
- Room at a third and at three times each price:

| Scenario | Nominal | Rod step /3, x3 | Effluent /3, x3 | Derate /3, x3 |
|---|---:|---|---|---|
| edf_daily | 1.94 | 1.97, 1.86 | 3.10, 1.42 | 1.94, 1.94 |
| edf_short_hold | 2.01 | 2.05, 1.90 | 3.12, 1.47 | 2.01, 2.01 |
| krsko_slow | 2.32 | 2.42, 2.08 | 4.09, 1.53 | 2.32, 2.32 |
| krsko_shallow | 2.79 | 2.97, 2.39 | 4.81, 1.82 | 2.79, 2.79 |
| baseload | 2.65 | 2.77, 2.35 | 4.14, 1.98 | 2.65, 2.65 |
| fast_return | 1.89 | 1.89, 1.89 | 1.89, 1.89 | 1.89, 1.89 |

The effluent price (ours) sets most of the room; the derate price does not touch it.
IPOPT (the convergence check) crashed on the v7 task tuple in `scripts/reactor_ipopt.py`;
fixed and re-running.

## Physics review fixes, 2026-11-10 (reactor-v8)

From `reactor-physics-review.md`, the owner took the review's recommendations, each
written into PHYSICS.md before its run:

- **Finding 1:** Tavg is charged past 1.6 K on either side of its program, at the cold
  side's price; the xenon tool's free zone and the oracle's hinge are two-sided too.
  Amendment 3 decision 7 is reversed. Steam relief is not modelled (D10).
- **Finding 2:** the core stays at the start of its cycle with an MTC of -12 pcm/K
  (`ours`, range 0 to -20: no hot-full-power start-of-cycle figure was found). The
  part-power axial shift keeps -30 under its own coefficient. Boron worth stays -8
  pcm/ppm, now labelled with its range, -6.7 to -10.9 (BEAVRS measures 10.8 at hot
  zero power and lower boron). A ppm moves Tavg 8/15 K.
- **Finding 4:** the hot-leg envelope is 610 K, under the 618.0 K saturation at 2250 psia.
- **Finding 5:** docstrings fixed; the environment page regenerated as reactor-v8. Its
  baseline rows are the old records, which were not re-recorded.
- **Finding 3** (a starting axial tilt) waits.

The start-of-cycle MTC broke fast_return: a load cut releases about 820 pcm, boron
moves at most about 35 pcm/min, and with the rods on the offset Tavg ran 20 to 29 K hot
and the hot leg tripped on 7 seeds of 10. In the family fast ramps are carried by rods
on Tavg, so the owner chose a mode switch, written before the rerun: while the plan
ramps faster than boron can follow, the rods run Westinghouse's rod program on Tavg;
otherwise bank D holds the offset. The rods also keep above the insertion limit at the
power the plan reaches by the step's end. The reference's gains come from the v8 bump
tests by the same recipe.

### Reference and nulls on reactor-v8, seeds 100 to 109

| Scenario | Reference | Trips | Null |
|---|---:|---:|---|
| edf_daily | $4,603 | 0 | $63.7M, cheaper on no seed |
| edf_short_hold | $4,326 | 0 | $63.7M, cheaper on no seed |
| krsko_slow | $1,735 | 0 | $7.6M, cheaper on no seed |
| krsko_shallow | $1,152 | 0 | $6.9M, cheaper on no seed |
| fast_return | $9,200 to $11,000 on 8 seeds | 2 | $66.8M, cheaper on no seed |
| baseload | $240 | 0 | $122,845, cheaper on no seed |

No reference run sits past the insertion limit. Findings, left as they are:

- **fast_return seeds 107 and 109** trip on the DNB clock after the return to full
  power: Tavg sits about 3.7 K hot for 2 h while xenon burns out, and the boron loop,
  at its bump-test gain (half v7's), is too slow. Changing the gain would be tuning on
  results.
- **The Krsko-day validation fails:** delta I reaches 8.5 % against Krsko's 5 % plus
  1 %, with or without burnup. The test's fixed procedure (bank D inside Krsko's 35
  steps, a proportional boron trim on Tavg) now lets Tavg swing -17 to +24 K and pins
  the bank at the end of its travel.
- Four plant tests had setups that overran the new limits; their setups changed, their
  contracts did not (comments in each).

Not re-run on v8: ceilings, room and sensitivity.

## Burnup and the xenon tool's play, 2026-11-10 (reactor-v7)

This section updates the Amendment 3 results below it. Two owner decisions, each
written into PHYSICS.md before its run:

- **The xenon tool's play** (PHYSICS.md section 6). A diagnostic showed the
  reference's whole baseload cost, $90, was effluent from the xenon tool cancelling a
  drift that left alone costs $0.80; its Tavg loop never acted. The tool now keeps
  the boron it did not move as a debt and acts only on the part that would carry
  Tavg out of its free zone (1.6 K cold to the DNB limit hot). On v6 this alone did
  not save baseload ($93 against the null's $0.80): the Tavg loop then acted
  instead, and the slow Krsko days got dearer ($1,307 to $1,534, $627 to $994).
- **Fuel burnup** (PHYSICS.md D9). The fuel loses 2.3 to 3.3 ppm of boron's worth a
  day (BEAVRS, Table 24, measured letdown of two cycles of a Westinghouse 4-loop
  PWR), and the day starts with 0 to 1 day of it not yet diluted (ours: an operator
  dilutes at least daily). The Tavg it moves is 8/33 K per ppm: at a fixed power the
  fuel cools with the coolant, so both feedbacks count. A first hand value that
  counted the coolant alone (8/30) was wrong, and the six-hour test caught it
  (0.199 K measured, 0.200 K from the corrected rule).

### Reference and nulls on reactor-v7, seeds 100 to 109

| Scenario | Reference | Null | Null cheaper on |
|---|---:|---|---:|
| edf_daily | $2,551 | $8.04M, trips on all 10 | 0 |
| edf_short_hold | $2,420 | $8.04M, trips on all 10 | 0 |
| krsko_slow | $1,549 | $6.36M, trips on all 10 | 0 |
| krsko_shallow | $1,020 | $5.04M, trips on all 10 | 0 |
| fast_return | $4,318 | $8.52M, trips on all 10 | 0 |
| baseload | $81 | $142, no trips | 4 of 10 |

The reference trips nowhere and never sits past the insertion limit. On baseload the
null now loses on the mean and wins 4 seeds, short of the 6 the acceptance rule needs,
so no null passes for the reference. Baseload now has a real chore: dilution for
burnup and for the debt carried in.

Not re-run on v7: the ceilings, the room and its sensitivity. The tables below are
v6's.

## Amendment 3, 2026-11-08 to 2026-11-10 (reactor-v6)

This section supersedes everything below it. The owner's Amendment 3 decisions 1 to 10,
then option A for the reference, are in the code on this branch; sources are S1 to S17 in
[`REALITY.md`](../../src/target_gym/reactor/REALITY.md), and every unsourced number is
labelled `ours` with its family range in
[`PHYSICS.md`](../../src/target_gym/reactor/PHYSICS.md).

### What changed

| Change | Source or rule | Decided by |
|---|---|---|
| Offset band as a penalty clock: 1 h in 24 h free, then a derate to 50 % priced at $80/MWh | D.C. Cook and RAOC technical specifications; EDF scope | owner, decision 3 |
| Rod insertion limit, 22 % in at full power to 78 % at half, 2 h, then hot standby | KAIST's Bank D line; STS 3.1.6 | owner, decision 5 |
| Tavg's DNB limit, full-power program plus 5.1 F (2.833 K), 2 h, then hot standby | STS LCO 3.4.1; Braidwood COLR (search summary, nrc.gov blocks readers) | owner, "source the hot side" |
| Tavg above its program costs nothing below the DNB limit (reversed 2026-11-10: the physics review's finding 1, both procedures keep the band two-sided) | charter: not lost output | owner, decision 7 |
| Trips as lost production: scram 48 h ($3.84M), timed overrun 15 h ($1.2M) | `ours`, family 22 to 70 h | owner, decision 6 |
| Boronmeter 20 min late; obs[10] is the power the starting xenon is balanced with | Areva US 8,489,338 | owner, decision 8 |
| Six scenarios, weighted from EDF's 250 h a year of load following | IEA 2019 (Mercel, slide 7) | owner, decisions 9 and "weights from the record" |
| Reference: bank D on delta I with a +-2.5 % action band, boron on Tavg (1.6 K dead band) with plan and xenon feedforwards | Westinghouse/Krsko procedure, NENE 2022 | owner, option A; the band fixed before any run |
| Retired: the scram test, the null-exception test, the two-moves scorer test | TESTING.md section 10 | owner, 2026-11-08 |

### Reference and nulls, seeds 100 to 109

| Scenario | Weight | Reference | Null (rods parked, no boron) |
|---|---:|---:|---|
| edf_daily | 0.0216 | $2,479 | $8.40M, trips on all 10 seeds |
| edf_short_hold | 0.0216 | $2,320 | $8.40M, trips on all 10 |
| krsko_slow | 0.0216 | $1,307 | $7.08M, trips on all 10 |
| krsko_shallow | 0.0216 | $627 | $5.40M, trips on all 10 |
| fast_return | 0.0216 | $4,409 | $8.52M, trips on all 10 |
| baseload | 0.892 | $90 | **$1, wins 10 of 10** |

The reference trips nowhere and never sits past the rod insertion limit. Every other rung
of the ladder trips or costs over $600,000 a day on edf_daily.

Finding: on baseload the null wins. With no load move, the boron trim borates on the hot
side as xenon burns out (Tavg up to 2.4 K over program, under the 2.83 K limit), then
overshoots cold, and its rod steps and effluent cost $40 to $150 a day against the null's
$1. On the weighted headline baseload carries 89 % of the weight, so the question is which
family member the task represents: the load-follow share runs from 0.11 (EDF fleet
average, the weight used) to about 1.0 (Cattenom 2, 1.4 TWh curtailed in 2025, per Kpler).
This is a choice for the owner, not a tuning question (J1).

### Ceilings and convergence

IPOPT (multiple shooting, exact Hessian) on edf_daily's four corner seeds, 300 iterations
per width rung and 1500 on the last; Adam (3000 iterations, surrogate hinges, the DNB
limit priced at `failure_cost` per K) on all six scenarios. Both replay on the env.

| Seed | Reference | IPOPT replayed | KKT error | Capped |
|---|---:|---:|---:|---|
| 100 | 2,455 | 1,298 | 5.7e-3 | yes |
| 102 | 2,481 | 1,357 | 8.7e-9 | no, converged |
| 108 | 2,575 | 1,298 | 0.67 | yes |
| 109 | 2,468 | 1,332 | 7.7e-9 | no, converged |

All four plans are feasible (violation 1e-6) and replay with no trip. Seeds 100 and 108
stopped at the 1500-iteration cap, so their costs are true but possibly not optimal. Adam
sits about 33 % above IPOPT on edf_daily (mean $1,762 against $1,321 on the same seeds), so
the rooms on Adam-only scenarios are lower bounds.

### Room, with sensitivity

Room is the reference cost over the ceiling, the cheaper of each seed's plan and the
reference (`scripts/reactor_sensitivity.py`). Each price is moved to a third and three
times, one at a time.

| Scenario | Nominal | Rod step x1/3 | x3 | Effluent x1/3 | x3 | Derate x1/3 or x3 |
|---|---:|---:|---:|---:|---:|---:|
| edf_daily | 1.62 | 1.65 | 1.56 | 1.92 | 1.51 | 1.62 |
| edf_short_hold | 1.39 | 1.41 | 1.36 | 1.72 | 1.27 | 1.39 |
| krsko_slow | 2.68 | 2.95 | 2.19 | 2.53 | 2.78 | 2.68 |
| krsko_shallow | 5.42 | 7.73 | 3.18 | 3.13 | 7.89 | 5.42 |
| fast_return | 1.32 | 1.34 | 1.27 | 1.51 | 1.18 | 1.32 |
| baseload | 24.4 | 60.6 | 9.5 | 8.2 | 73.0 | 24.4 |

The derate price never moves the room: neither the reference nor the plans overrun the
offset clock. On the four EDF and fast days the room is 1.2 to 1.9 at any price; the
hard limits (insertion line, DNB) leave little between a good procedure and foresight.
On the Krsko days and baseload the room is large and price-driven, because the ceiling
there is a few hundred dollars or less.

Weighted over the suite (Adam ceilings, nominal prices): reference $321 a day, ceiling
$168, room 1.9. Baseload is a quarter of the reference's weighted cost; at a load-follow
share of 1.0 the weighted room would be about 1.5.

### Still open

- The baseload null win (above): the owner's call on the family member or the weight.
- Rule 4 acceptance and a learning check: not run. No shared baseline, null table or
  record was re-recorded.
- `scripts/reference_ladder.py`'s reactor row needs are in PHYSICS.md; the script was not
  edited.
- Unsourced numbers, labelled `ours` with family ranges: the trip durations (48 h, family
  24 to 70 h) and the 900 s floor allowance (family 15 to 30 min) in PHYSICS.md; the equal
  split of the five load-follow days in REALITY.md section 7; the 600 s feedforward lead;
  and the 2.5 % action band, half the published +-5 % band, set by the owner.

## EDF rebuild, 2026-11-08

This section supersedes the results below it, which describe the first attempt. After
the plant check (`reactor-plant-check.md`), the owner picked EDF practice and asked for
items 1 to 4. The coupling finding and the band sources are in `reactor-coupling.md`.

### Design changes, and who decided

| Change | Why | Decided by |
|---|---|---|
| The ramp schedule is in the observation: hours to the next ramp down and up | an operator borates ahead of a known plan | owner, item 1 |
| Tavg below 556 C is a priced limit with 900 s allowed, then a trip | the old floor tripped at once; 900 s is `ours`, after Salem, unconfirmed | owner, item 2 |
| Offset charged only outside +-0.05 of its reference | IAEA NP-T-3.23 Annex II-4.3, load follow | owner, item 3 |
| Tavg charged only outside a 1.6 K dead band | IAEA: black rods move past 1.6 C | owner, item 4 |
| D set so the halves couple at 283 pcm, 4.99 cm against the thesis 0.405 | the thesis core was 12 to 25 times too weakly coupled and unstable at half power; Ringhals-4, Pedersen et al., Table 3 | owner, "Yes, use 283 pcm" |
| Mode G (grey and black groups plus an R group) tried and reverted | the line-out pinned the offset at -0.99 | owner, "EDF Mode A, one bank" |
| Reference: the published rod program, plus rung 5's boron | see below | owner, "Yes, adopt it" |

Item 5, rod insertion limits, stays open: no sourced curve was found.

### Properties, with no controller

At 283 pcm the axial mode is damped at every hold: period 32.7 h and decay -0.070 per
hour at full power, 33.7 h and -0.050 at 75 %, 34.3 h and -0.057 at half power
(Ringhals-4: 28.3 h). Krsko's half-power offset reads +5.4 % against the published +5 %.
60 rod steps move delta I by -0.34 at once; the same reactivity in boron moves it by about
3e-6 and arrives 11 minutes later. The property holds.

### Reference ladder, seeds 100 to 109

Gains come from bump tests at both holds the episode visits, keeping the design that asks
the least gain (the PR #170 rule); no gain was tuned on costs.

| Row | Mean cost | Seeds tripped | Steps past offset band | Steps past dead band |
|---|---:|---:|---:|---:|
| Rung 1, boron PI on offset | $22.8M | 7 | 1645 | 2273 |
| Rung 2 | $17.9M | 10 | 1397 | 1928 |
| Rung 3, cascade with plan feedforward | $103,386 | 0 | 863 | 2090 |
| Rung 4, offset override | $18.0M | 8 | 1824 | 2391 |
| Rung 5, saturation override | $89,754 | 0 | 858 | 1703 |
| Rod program + rung 1 boron | $20.9M | 6 | 1655 | 1740 |
| **Rod program + rung 5 boron (reference)** | **$74,604** | **0** | **817** | **1037** |
| Null: rods parked, no boron | $161,561 | 0 | 1864 | 2622 |

Steps are out of 2,880 per day. The rod program is Westinghouse's published one (NRC
HRTD 8.1 and 5.0: a 1.5 F, 0.83 K, dead band, then 8 to 72 steps per minute, with the
power-mismatch channel). Paired with rung 1's
boron it trips, because that boron never dilutes for xenon or ahead of the return to
power. No null wins a seed against the reference.

### Ceiling and room

Two oracles, both with the true model, state and future. Adam plans the whole day by
gradient from rung 5's actions (3000 iterations on all ten seeds; 6000 and 12000 on seeds
100 and 101). The interior-point solve (IPOPT, multiple shooting, exact Hessian) runs on
the four corner seeds, 300 iterations per width rung, and its plan is replayed on the env.
The ceiling is the cheapest replayed plan; room is the reference cost over it.

| Seed | Reference | Adam 3000 | Adam longer | IPOPT replayed | Last KKT error | Room |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 117,111 | 2,480 | 2,424 (6000) | 1,722 | 0.26 | 68.0 |
| 101 | 109,745 | 2,601 | 2,526 (6000) | | | 43.4 |
| 102 | 77,256 | 2,414 | | 1,782 | 0.033 | 43.4 |
| 103 | 40,411 | 2,097 | | | | 19.3 |
| 104 | 50,694 | 2,311 | | | | 21.9 |
| 105 | 91,793 | 2,389 | | | | 38.4 |
| 106 | 87,562 | 2,909 | | | | 30.1 |
| 107 | 68,918 | 2,276 | | | | 30.3 |
| 108 | 42,335 | 2,413 | | 1,740 | 9e-6 | 24.3 |
| 109 | 60,215 | 2,613 | | 1,787 | 0.086 | 33.7 |

Convergence, plainly:

- Adam is not converged. Where IPOPT ran, Adam's 3000-iteration plan is 35 to 46 % above
  it. 6000 iterations help a little; 12000 did worse than 6000 on both seeds (2,522 and
  2,605), so a longer cosine schedule is not a fix. On the six Adam-only seeds the room is
  a lower bound.
- IPOPT hit its 300-iteration cap on every rung. Only seed 108 reached a KKT error near
  convergence (9e-6); the other three stopped between 0.03 and 0.26, with constraint
  violation below 5e-6. The plans are feasible and replay with no trip, so their costs are
  true costs, but they are local and possibly not optimal.

What the room is made of, replayed on the four corner seeds:

| Seed | Reference tracking | Reference running | IPOPT tracking | IPOPT running | Past offset band, ref / IPOPT | Past dead band, ref / IPOPT |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 107,106 | 9,649 | 62 | 1,660 | 1094 / 13 | 1428 / 85 |
| 102 | 68,664 | 8,511 | 73 | 1,709 | 773 / 20 | 1405 / 76 |
| 108 | 36,815 | 5,523 | 78 | 1,661 | 782 / 36 | 453 / 191 |
| 109 | 52,796 | 7,847 | 57 | 1,729 | 781 / 145 | 714 / 131 |

(Local replays on this Mac; the reference differs from the ladder by under 1 %, the known
run-to-run spread.) The oracle keeps the offset in band almost all day by moving boron
hours ahead, so its tracking cost is near zero and what it pays is rod wear and effluent.
The reference loses the band through every ramp and the xenon swing after it. Room of 19
to 68 says the task has a great deal of headroom over a textbook controller; it does not
say an agent that sees only the observation can reach the ceiling, and nothing here claims
it can.

### Open items

- Item 5, rod insertion limits: no sourced curve. A possible lead is KAIST's Bank D limit
  (22 % at full power to 78 % at half), not yet opened.
- The dead band: the cost uses IAEA's 1.6 C for the black rods, while the reference's rod
  program (and EDF's R group) starts moving at about 0.8 C. The two are not reconciled.
- The rod program is Westinghouse's, inside an EDF Mode A plant. The owner adopted it as
  the reference; the mix is disclosed here.
- Unsourced, labelled `ours`: 900 s allowed below the Tavg floor, the feedforward lead of
  600 s, and the saturation override's form.
- Rule 4 and a learning check not run. Registry version, null tables and records are
  stale; nothing was re-recorded.
- Known failing tests, kept as findings: peak xenon worth (1419 against 1800 pcm) and the
  two-moves scorer on seeds 100 and 101.

## First attempt, 2026-11-07

### Verdict

The property appeared. With no controller, the plant has a slow axial xenon mode close to
the published one, rods move the offset hard and at once, and boron moves it almost not at
all and arrives late. A textbook controller holds the plant with no trip on the ten
nominal seeds, and the foresight plan is at least 1.78 times cheaper on every seed, 2.8
to 3.0 times on the four seeds solved to convergence.

Three things are not clean. The ceiling is an interior-point solve on four corner seeds,
converged but local, and Adam's on the other six is about a third too high. Rule 4 and a
learning check were not run. Three tests fail, kept failing on purpose.

## What changed in the plant

The owner widened the brief: the rebuild may redefine more than the offset if that makes
the task relevant and it stays sourced. The design already did, to match how a plant runs
load follow, so it was kept as one attempt.

| | Before | Now |
|---|---|---|
| Core | point kinetics, one xenon | adds a two-point core: iodine and xenon tilts, offset solved algebraically (Onega and Kisner) |
| Action | rods | 2 inputs: rod demand, boron flow (dilute or borate) |
| Observed | 6 values | 8 values, adds delta I and boron |
| Boron | none | feed and bleed through a 5 to 15 min line, drawn at reset |
| Scored | Tavg | Tavg, plus delta I past +-5 %, rod travel and effluent |
| Load day | 12 h at 5 %/min | 24 h at 2 %/min, the top of OAPS's Mode A range |

Numbers the sources did not give are labelled `ours` with a reason in the parameter table
of `PHYSICS.md`: boron worth, boric acid strength, primary mass.

## Properties, measured with no controller

From `scripts/reactor_properties.py`: rods dipped for 2.5 h, then back, as in the Oconee II
test.

| Property | Measured | Published |
|---|---|---|
| Mode period, full power | 29.4 h | 28.3 h (Ringhals-4), about 28.5 h (Oconee II) |
| Mode decay, full power | -0.035 /h | -0.05 /h (Ringhals-4) |
| Mode period and decay, 75 % | 29.1 h, -0.0067 /h | no figure read |
| Mode period and decay, 50 % | 28.0 h, -0.0012 /h, near neutral | no figure read |

The same 179 pcm, moved by each actuator at full power:

| | Rods, 60 steps | Boron, full flow |
|---|---|---|
| Shows on Tavg after | 0.5 min | 11 min |
| Peak Tavg move | -5.5 K | -5.3 K |
| Delta I at that peak | -0.47 | about 1e-5 |

This matches the sources in kind: 5 to 15 min for boron (OAPS), and boron negligible on the
offset (KNS APR1400).

The core model itself, rebuilt from the thesis equations outside the env, gives a tilt of
.186 against .19, extrema at 23.4, 38.8 and 52.9 h against 23, 38 and 51.5 h, and an
undamped cross section of 2.92e-18 against 3.02e-18. That needs a total absorption of
2.44 x 0.16 = 0.39. The thesis's own Table I-1 value, 1.53, does not reproduce its results.

Where it does not match:

- The Krsko 50 % state gives delta I of +10 to 13 % against +5 %: right sign, about twice
  the size.
- Peak xenon worth is 1419 pcm against the 1800 floor of the existing test, because the
  flux now follows Oconee. That test still fails.

Holding delta I at zero at 50 % statically takes 42 steps and 58 pcm of rods, against 34
steps at Krsko. Both inputs at full flow give about 2 %/min of load, against 1 to 2 %/min
for Mode A in OAPS.

## Step cost

0.74 M steps/s against a median of about 0.72 M, so about 1 times the median and under
rule 7's 3 times. It was 1.17 M before. No reset or step runs a solver loop.

## Reference

Rule 3's recipe, ten nominal seeds, 100 to 109, from `scripts/reactor_ladder.py`. Rungs 5
to 8 are shown before and after the rod-loop fix described below. Rungs 1 to 4 and the
rod program do not use that loop and did not change.

| Row | Pairing | Mean cost | Seeds tripped | Steps past band of 2880 |
|---|---|---:|---:|---:|
| Rung 1 | Mode A | $20.2M | 10 | 2166 |
| Rung 2 | Mode A | $363k | 0 | 2222 |
| Rung 3 | Mode A | $31.2M | 10 | 2720 |
| Rung 4 | Mode A | $3.07M | 2 | 1534 |
| **Rung 5, reference** | rods on delta I, boron on Tavg | **$9,607** (was $15,427) | **0** | **24** (was 92.5) |
| Rung 6 | same | $14,918 (was $21,180) | 0 | 24 (was 103) |
| Rung 7 | same | $17,920 (was $24,147) | 0 | 24 (was 102) |
| Rung 8 | same | $17,923 (was $1.01M) | 0 (was 1) | 24 (was 83) |
| Rod program with boron trim | published | $25.2M | 10 | 2289 |

The reference's cost splits into $6,975 of tracking and $2,631 of running. It reaches 44
steps of rods and |delta I| of .077. Tavg still swings in the ramps. This is the best
textbook recipe, not a plant's practice.

Three corrections came after results were seen, against the charter's spirit of a fixed
recipe, so the earlier numbers are kept above and the reasons stated:

1. The first rod tune gave a limit cycle between 54 and 18 steps, from misreading zero dead
   time. The static SIMC case is now pure integral with one step of lag and one of delay,
   as `_tavg_integrator` already reads its bump.
2. The Mode A pairing failed, so rungs 5 to 8 were added. The reason is physical, not the
   score: boron's gain on delta I is about zero, so the relative gain of that pairing is
   zero and the pairing cannot work. That Mode A's pairing fails under this recipe is
   itself a finding, and fits OAPS naming the operator's struggle with the offset as the
   limit on load follow.
3. The rod loop on delta I locked into a cycle at half power. The owner approved fixing
   it as a bug. The next section gives the cause and the rule.

### The rod loop's limit cycle, and the fix

**Cause.** The rods' gain on delta I changes across the day. The tilt a bank puts in goes
as sin(pi y)^3, so a bank near the top barely tilts the flux, and at half power the weaker
power feedback lets the same tilt move the offset further. The recipe read the gain from
one bump, at full power from the top: 0.777 per unit of demand. In the half-power hold the
local gain, measured step by step along the reference's own run, was 2.7 to 4.0 times
that. The loop is a velocity PI with one step of delay on a plant that settles inside a
step. With g the gain times Kc, its poles solve z^2 - (1 - 2g) z - g = 0, which reaches
-1, a cycle of two steps, at g = 2/3. The tune set g = 1/4, so it had a gain margin of
2.67. On seed 100 the cycle started where the local gain reached 3.03 times, and on seed
102 at 2.74 times. Seed 108 sat at 2.70 times and stayed just stable.

It reproduces without noise or wobble: seeds 100, 102 and 103 then show 444, 34 and 115
two-step swings of delta I over 0.05, and the other seven none.

**Rule.** The rods' gain is read at each point the day holds, and the tune uses the
largest. Full power is read as before. Half power is read by a bump of 10 steps, from a
line-out with xenon balanced and the rods where their tilt cancels the coolant's, so
delta I sits at 0 (42 steps). That reads 1.711, 2.20 times the full-power gain, and SIMC
is applied to it unchanged. SIMC's margin, a gain margin near 3 for tc = theta (Skogestad,
J. Process Control 13, 2003, table 2), is a margin on the gain it is tuned for. Tuning for
the largest gain measured is our reading of that, not a sentence the paper states. Under
it, the worst local gain seen in operation, 4.0 times, gives g = 0.45, under the 2/3
limit. No cost was looked at to choose the bump point or its size: the hold is the day's
other operating point, and the bump is small because the gain is local.

**Result.** With no noise or wobble, no seed shows a two-step swing after the fix, and
|delta I| peaks at 0.068 to 0.070 on every seed. With them, as scored:

| Seed | Before | After | Change |
|---:|---:|---:|---:|
| 100 | $51,948 | $9,836 | -81 % |
| 101 | $10,811 | $10,982 | +2 % |
| 102 | $14,719 | $9,824 | -33 % |
| 103 | $13,744 | $9,380 | -32 % |
| 104 | $8,655 | $8,754 | +1 % |
| 105 | $8,936 | $8,942 | 0 % |
| 106 | $9,877 | $10,029 | +2 % |
| 107 | $14,027 | $9,307 | -34 % |
| 108 | $12,398 | $9,708 | -22 % |
| 109 | $9,157 | $9,304 | +2 % |

No seed trips, before or after, and that includes the four corner seeds 100, 102, 108 and
109. The seeds that did not cycle pay 0 to 2 % more, from the slower loop at full power.
Seeds 107 and 108 had shorter cycles with noise that the no-noise run did not show.

## Nulls

Ten seeds, against the reference, from `scripts/reactor_nulls.py`. This is the check of the
rebuild, not rule 4. The shared null table predates the two inputs, so the script fits its
own: a quadratic in the load through the reference's actions on training seeds 0 to 9.

| Null | Mean cost | Seeds tripped | Wins of 10 |
|---|---:|---:|---:|
| Constant low, middle, high | $5.7G, $28.9M, $413M | 10 each | 0 |
| Rods parked, no boron | $459,513 | 0 | 0 |
| Table of load | $42,498 | 0 | 0 |
| Table plus Tavg integrator | $491,400 | 0 | 0 |
| Witness: rod integrator | $1.02M | 0 | 0 |
| Witness: Tavg integrator | $562,067 | 0 | 0 |

Against the fixed reference no null wins a seed. Before the fix the table won one, seed
100, where the reference cycled. The table is worth reading, though. It keeps |delta I|
near .08, about as tight as the reference, because it sets the rods from the load and
hardly moves boron (9 ppm). It pays $41,962 of tracking for Tavg off its program. So the
offset alone is easy to hold if Tavg is given up. The task is having both.

## Ceiling and room

`scripts/reactor_foresight.py` plans the whole day by gradient through the env (direct
single shooting), with the true model, the true state, the future and the seed's noise,
starting from the reference's actions, so it can only lower that cost. The shared oracle's
5 minute horizon is shorter than the boron line and cannot see the problem. Adam with
cosine decay, ten seeds on bob's CPU. These runs started from the reference before its
fix. A plan's cost does not depend on where it started, so the ceilings stand. The room
uses the fixed reference.

| Seed | Reference, fixed | Ceiling, 3000 iters | Room | Ceiling, 6000 iters | Room |
|---:|---:|---:|---:|---:|---:|
| 100 | $9,836 | $5,289 | 1.86 | $4,164 | 2.36 |
| 101 | $10,982 | $5,778 | 1.90 | $4,345 | 2.53 |
| 102 | $9,824 | $5,369 | 1.83 | | |
| 103 | $9,380 | $4,425 | 2.12 | | |
| 104 | $8,754 | $4,842 | 1.81 | | |
| 105 | $8,942 | $4,564 | 1.96 | | |
| 106 | $10,029 | $5,642 | 1.78 | | |
| 107 | $9,307 | $4,818 | 1.93 | | |
| 108 | $9,708 | $4,391 | 2.21 | | |
| 109 | $9,304 | $4,654 | 2.00 | | |

At 3000 iterations the mean ceiling is $4,977, the median room 1.92 and the range 1.78 to
2.21.

**Adam did not converge.** Doubling the iterations cut the cost by 21 and 25 %. Each
run flattens near its end only because the learning rate decays.

### Interior-point check

`scripts/reactor_ipopt.py` solves the same problem by Newton steps: the shared oracle's
multiple-shooting NLP and IPOPT port, exact Hessian, stretched to the whole day, with
the seed's true load wobble, cold from the held state, then replayed through
`step_env`. The seeds are the ones nearest each corner of hold length and boron delay,
picked from those draws, not from any cost. Each plan is replayed through `step_env`,
and the replayed cost is the one reported.

| Seed | Hold, delay | Reference, fixed | Adam, 3000 | Interior point, replayed | Room | Room before the fix |
|---:|---|---:|---:|---:|---:|---:|
| 109 | short, short | $9,304 | $4,654 | $3,352 | 2.78 | 2.73 |
| 102 | short, long | $9,824 | $5,369 | $3,488 | 2.82 | 4.22 |
| 108 | long, short | $9,708 | $4,391 | $3,220 | 3.01 | 3.85 |
| 100 | long, long | $9,836 | $5,289 | $3,265 | 3.01 | 15.9 |

Every solve converged at the final width: KKT error about 5e-9, under the solver's
1e-8, in 113 to 276 iterations, with no trip on replay. The first, wider rung hit its
300-iteration cap on every seed and only served as a start. Each seed took 4.5 to 5.7
hours on 8 CPU threads.

The interior point is 27 to 38 % cheaper than Adam at 3000 iterations, and 22 % cheaper
on seed 100 than Adam at 6000. So Adam's ceilings are too high by about a third, and
its rooms too low by as much. On these four seeds the room is 2.8 to 3.0.

### Why seed 100's room was 15.9

It was the reference, not the plant. The solver brings seed 100 to $3,265, in line with
the other three ($3,220 to $3,488). The reference paid $46,544 of its $51,951 in hours 5
to 10, the hold at 50 % load, where its rod loop cycled with a period of 2 steps (1 min).
The rods swung between 25 and 52 steps and delta I between -0.25 and +0.24, against a band
of +-0.05. That cost about $11k an hour, at $1000 an hour per % past the band.

None of the other causes checked held. Power peaked at 1.020 against 1.09, the hot leg at
603.2 K against 620, Tavg's low at 559.2 K against 556, so there was no near-trip. The boron
command was at a bound on 1.3 % of steps, as on the other seeds, so boron did not
saturate. The cost is the reward's own terms, so it was not a scoring fault. The cycle
reproduced with no noise. The fix is in the reference section above, and seed 100's room
is now 3.01.

### Why the planned cost is below the replayed one

The smoothed boron corner, all of it. The planner smooths the split between dilution and
boration with a softplus of width w = 0.01. At a resting command of 0 that leaves each
branch open by w ln 2 = 0.0069 of its flow: 0.069 kg/s of water and 0.021 kg/s of acid
at once. The effluent is charged on the command, so the planner gets this mixing free and
uses it to trim Tavg. The plant has no flow at 0. Rolled through the planner's own model,
on the same plans:

| Seed | Planner, as solved | Rod corner exact | Boron corner exact | Both exact | Plant |
|---:|---:|---:|---:|---:|---:|
| 100 | $2,590 | $2,590 | $3,278 | $3,278 | $3,265 |
| 102 | $2,636 | $2,636 | $3,518 | $3,518 | $3,488 |
| 108 | $2,637 | $2,637 | $3,237 | $3,237 | $3,220 |
| 109 | $2,690 | $2,690 | $3,365 | $3,365 | $3,352 |

The Tavg term carries the whole difference, $573 to $589 smoothed against $1,176 to
$1,471 exact. The rod corner changes nothing. With both corners exact the planner is
within 0.4 to 0.9 % of the plant. What is left is the wobble inside each 30 s step: with
no wobble on either side the two agree to $0.1 on seed 109 ($3,625.0). Sensor noise,
precision and block size are not causes. Noise changes no cost, and the block is one
step on both sides.

So the replayed cost is the honest ceiling, and the planned one is not, since no plant
can reach it. These plans are optimal for a planner model that is wrong at rest, so the
plant's best is probably below the replayed cost. A re-solve that charges the effluent on
the smoothed flows, or takes the boron width toward 0, would close that, and was not run.

A converged solve is a local optimum of a nonconvex problem. It is a better upper bound
on the best cost, not a proof of it.

Two fixes to the reactor's oracle came out of this, both in the shared slot. The boron
move had no scale of its own, which would have crashed the oracle's solve. The planner
also now smooths the corner between dilution and boration, where a resting plan sits
and the solver otherwise jammed (KKT error stuck at 0.22 on a 20 minute test, 1e-10
after). At no width it is the env's exact mixing, which a test checks.

None of this says what an agent could reach. The plan knows the future and the noise.

## Not done

- Rule 4 acceptance: it runs once from a tag on hashed seeds. Nothing here is a pass.
- A learning check: no agent was trained.
- Registry version bump, stored null tables, records and the accepted-reason strings in
  `null_policies.py` are stale for this task. Records were not re-recorded, as asked.

Three tests fail and were left failing, as findings:

- `test_peak_xenon_worth_is_in_the_published_band`: 1419 pcm against 1800, above.
- `test_a_day_has_two_moves_for_the_scorer[100]` and `[101]`: the shared eval does not read
  the slower ramp. The charter drops that scorer, and `eval.py` is shared, so it was not
  touched. Retired by the owner on 2026-11-08: the weighted-saving headline does not use
  the move detector it checked.

Rewritten around the two inputs: the reactor's env, expert, oracle and physics tests, and
`tests/test_industrial_readings.py`. Four tests that only made sense for rods alone were
deleted, against the usual rule, because the plant they described is gone: three on the
rods-only budget and `test_tavg_stays_inside_the_deadband_on_held_out_seeds`. The owner
should confirm those deletions. `tests/reactor/test_reactor_axial.py` adds 18
spec-derived tests.

## Sources

Opened and read:

- Mastroianni, MS thesis, Virginia Tech, 1979: the Onega and Kisner two-point model and the
  Oconee II data. <https://vtechworks.lib.vt.edu/bitstreams/0055f2f6-a23e-4c40-b6cd-3f36f856aa0a/download>
- Pedersen, Demaziere, Vinai, Progress in Nuclear Energy 202: the Ringhals-4 axial mode, 28.3 h
  and 0.05 per hour, and nu 2.44.
- OAPS, ICAPP 2025, arXiv 2605.20169: boron delay, flows, Mode A rates.
- Krsko, NENE 2022 paper 911: offset, delta I, the +-5 % band, 1690 ppm.
- KNS 24A-359, APR1400: boron negligible on the axial shape index.

Could not be opened by an automated reader, and not bypassed: ScienceDirect, epj-n.org,
nrc.gov.
