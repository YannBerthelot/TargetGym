# Reactor rebuild: axial offset, rods and boron

Status: results of the one design attempt, 2026-11-07, on branch
`claude/reactor-axial-dd4617` (from `feat/sensor-noise` at a9c8ea8). Nothing is merged and
no shared baseline was re-recorded. The plant is described in
[`src/target_gym/reactor/PHYSICS.md`](../../src/target_gym/reactor/PHYSICS.md). This page
holds the measurements, under the rules of `task-charter.md`.

## Verdict

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
  touched.

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
