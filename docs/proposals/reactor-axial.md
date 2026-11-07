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
nominal seeds, and the foresight plan is at least 1.75 times cheaper on every seed, 2.7
times on the four seeds solved to convergence.

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

Rule 3's recipe, ten nominal seeds, 100 to 109, from `scripts/reactor_ladder.py`:

| Row | Pairing | Mean cost | Seeds tripped | Steps past band of 2880 |
|---|---|---:|---:|---:|
| Rung 1 | Mode A | $20.2M | 10 | 2166 |
| Rung 2 | Mode A | $363k | 0 | 2222 |
| Rung 3 | Mode A | $31.2M | 10 | 2720 |
| Rung 4 | Mode A | $3.07M | 2 | 1534 |
| **Rung 5, reference** | rods on delta I, boron on Tavg | **$15,427** | **0** | **92.5** |
| Rung 6 | same | $21,180 | 0 | 103 |
| Rung 7 | same | $24,147 | 0 | 102 |
| Rung 8 | same | $1.01M | 1 | 83 |
| Rod program with boron trim | published | $25.2M | 10 | 2289 |

The reference's cost splits into $11,266 of tracking and $4,161 of running. It reaches 52
steps of rods and |delta I| of .251. It is not good control. Some episodes chatter, with
delta I near -0.2 for hours at low load (seed 100 costs $52k), and Tavg swings +8 and -18 K
in the ramps. This is the best textbook recipe, not a plant's practice.

Two corrections came after results were seen, against the charter's spirit of a fixed
recipe, so both sets are kept above and the reasons stated:

1. The first rod tune gave a limit cycle between 54 and 18 steps, from misreading zero dead
   time. The static SIMC case is now pure integral with one step of lag and one of delay,
   as `_tavg_integrator` already reads its bump.
2. The Mode A pairing failed, so rungs 5 to 8 were added. The reason is physical, not the
   score: boron's gain on delta I is about zero, so the relative gain of that pairing is
   zero and the pairing cannot work. That Mode A's pairing fails under this recipe is
   itself a finding, and fits OAPS naming the operator's struggle with the offset as the
   limit on load follow.

## Nulls

Ten seeds, against the reference, from `scripts/reactor_nulls.py`. This is the check of the
rebuild, not rule 4. The shared null table predates the two inputs, so the script fits its
own: a quadratic in the load through the reference's actions on training seeds 0 to 9.

| Null | Mean cost | Seeds tripped | Wins of 10 |
|---|---:|---:|---:|
| Constant low, middle, high | $5.7G, $28.9M, $413M | 10 each | 0 |
| Rods parked, no boron | $459,513 | 0 | 0 |
| Table of load | $41,662 | 0 | 1 |
| Table plus Tavg integrator | $492,485 | 0 | 0 |
| Witness: rod integrator | $1.02M | 0 | 0 |
| Witness: Tavg integrator | $562,067 | 0 | 0 |

No null comes near half. The table is worth reading, though. It is past the band for only
6.7 steps, with |delta I| of .058, tighter than the reference, because it sets the rods
from the load and hardly moves boron (10 ppm). It pays $41,108 of tracking for Tavg off its
program, and its one win is seed 100, where the reference chatters. So the offset alone is
easy to hold if Tavg is given up. The task is having both.

## Ceiling and room

`scripts/reactor_foresight.py` plans the whole day by gradient through the env (direct
single shooting), with the true model, the true state, the future and the seed's noise,
starting from the reference's actions, so it can only lower that cost. The shared oracle's
5 minute horizon is shorter than the boron line and cannot see the problem. Adam with
cosine decay, ten seeds on bob's CPU.

| Seed | Reference | Ceiling, 3000 iters | Room | Ceiling, 6000 iters | Room |
|---:|---:|---:|---:|---:|---:|
| 100 | $51,948 | $5,289 | 9.82 | $4,164 | 12.47 |
| 101 | $10,812 | $5,778 | 1.87 | $4,345 | 2.49 |
| 102 | $14,711 | $5,369 | 2.74 | | |
| 103 | $13,757 | $4,425 | 3.11 | | |
| 104 | $8,655 | $4,842 | 1.79 | | |
| 105 | $8,935 | $4,564 | 1.96 | | |
| 106 | $9,877 | $5,642 | 1.75 | | |
| 107 | $14,027 | $4,818 | 2.91 | | |
| 108 | $12,401 | $4,391 | 2.82 | | |
| 109 | $9,157 | $4,654 | 1.97 | | |

At 3000 iterations the mean ceiling is $4,977, the median room 2.36 and the range 1.75 to
9.82.

**Adam did not converge.** Doubling the iterations cut the cost by 21 and 25 %. Each
run flattens near its end only because the learning rate decays.

### Interior-point check

`scripts/reactor_ipopt.py` solves the same problem by Newton steps: the shared oracle's
multiple-shooting NLP and IPOPT port, exact Hessian, stretched to the whole day, with
the seed's true load wobble, cold from the held state, then replayed through
`step_env`. The seeds are the ones nearest each corner of hold length and boron delay,
picked from those draws, not from any cost.

| Seed | Hold, delay | Reference | Adam, 3000 | Interior point, replayed | Room |
|---:|---|---:|---:|---:|---:|
| 109 | short, short | $9,157 | $4,654 | $3,352 | 2.73 |
| 102 | short, long | $14,716 | $5,369 | $3,488 | 4.22 |
| 108 | long, short | $12,403 | $4,391 | $3,220 | 3.85 |
| 100 | long, long | $51,951 | $5,289 | $3,265 | 15.9 |

Every solve converged at the final width: KKT error about 5e-9, under the solver's
1e-8, in 113 to 276 iterations, with no trip on replay. The first, wider rung hit its
300-iteration cap on every seed and only served as a start. Each seed took 4.5 to 5.7
hours on 8 CPU threads.

The interior point is 27 to 38 % cheaper than Adam at 3000 iterations, and 22 % cheaper
on seed 100 than Adam at 6000. So Adam's ceilings are too high by about a third, and
its rooms too low by as much. On these four seeds the room is 2.7 to 16.

Two limits on these numbers:

- The replayed cost is 20 to 30 % above the solver's own (about $2,600 a seed). Sensor
  noise is not the cause, since replaying without it gives the same cost. The likely
  causes are the planner's model of the wobble inside each 30 s step, and the
  smoothing of the boron corner. That gap was not traced further. The replayed cost is
  the one charged, so it is the one used.
- A converged solve is a local optimum of a nonconvex problem. It is a better upper
  bound on the best cost, not a proof of it.

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
should confirm those deletions. `tests/reactor/test_reactor_axial.py` adds 15
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
