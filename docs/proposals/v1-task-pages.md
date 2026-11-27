# v1 task pages

One page per candidate against the v1 task protocol the owner adopted on
2026-11-27, so the owner can decide in one batch which tasks to build to v1 and
what each needs. Findings and plans only: no code, test or record was changed.

The protocol's nine items, as adopted: (1) a one-sentence job a real operator is
paid to do, and the agent's decision; (2) a score in the job's units over a fixed
episode, with nothing at the end a policy can game; (3) one expert from a written
practitioner recipe, frozen as data (`tg.make_reference`); (4) published
baselines: doing nothing, the expert, an MPC on the true model, PPO and SAC at one
cheap budget (30 seeds, shipped settings); (5) room: the MPC or a learned agent
beats the expert, with an interval excluding zero; (6) five checks: one published
figure, tripping never pays, the expert beats doing nothing, deterministic runs,
`jit` and `vmap`; (7) an episode in under a second on a CPU; (8) one card and a
Gymnasium ID; (9) frozen as `TargetGym/<task>-v1` with a CI guard.

## The decision in one table

| Task | Proposal | Work of its own, beyond the shared work below |
|---|---|---|
| `reactor` | Build to v1 | Item 7: score 2-day runs instead of 7-day ones (S), or read item 7 per episode in a batch (no change) |
| `ph_neutralization` | Build to v1 | Item 2: charge the reagent the tanks lost by the episode's end, as hvac's store is charged (M) |
| `compressor_surge` | Build to v1 once its MPC line is in | Item 5: finish the MPC line already running on bob (S); if it trips or loses, park |
| `hvac` | Build to v1 only if a re-run MPC beats the reference; otherwise park | Item 5: re-run the MPC line on the fixed code (S); if it loses, one M attempt at a planner that reads the forecast, then park |
| `first_order` | Freeze as the labelled, unscored debugging task | None |

Sizes: S is a day of work or less, plus any recording run; M is a few days; L
is a week or more.

## Status by item

`pass`, `fail`, `n.m.` (not measured on the current plant) or `n.a.` (not
applicable to an unscored task).

| Task | 1 job | 2 score | 3 expert | 4 baselines | 5 room | 6 checks | 7 speed | 8 card, ID | 9 frozen |
|---|---|---|---|---|---|---|---|---|---|
| `reactor` | pass | pass | pass | fail | pass, stale | pass | fail | fail | fail |
| `hvac` | pass | fail (pass with PR 306) | pass | fail | n.m. (last record failed) | pass | pass | fail | fail |
| `compressor_surge` | pass | pass | pass | fail | n.m. | pass | pass | fail | fail |
| `ph_neutralization` | pass | fail | pass | fail | pass, stale | fail (pass with PR 306) | pass | fail | fail |
| `first_order` | n.a. | n.a. | pass | fail | n.a. | pass | pass | fail | fail |

Item 3 passes on all four scored tasks with one step owed: the frozen files in
`src/target_gym/data/reference/` are the working sample of PR 307, so they are
frozen again once, on the final tree (shared step W2).

## What the evidence is

- **Code:** `int/amendment3` at 874f5fb3 (PRs 307, 308, 309 and 311 merged). PRs
  306 (hvac's window edge, the scored defaults, pH's trip charge), 312 (the
  learning check made report-only) and 314 (docs) are open. PR 313 (reactor
  cycle strata, compressor decoupling) is closed as paused. On int's last
  finished CI run (46069938666, at 186478ff) everything passes but the records
  job (stale by design) and one benchmark-runner test.
- **Stored records** (`src/target_gym/data/charter/`, 2026-11-13 to 17) describe
  older plants and older scenario sets. They are used here only where said.
- **The dry run** of the recording session (2026-11-26 and 27, int dceb2537 plus
  a table fix, on bob and flanders under `~/scratch/2026-11-26-record-five/wt_dry`)
  is the newest measurement of the lines. It predates PR 306, which re-stamps all
  four plants but pins every scenario's parameters (`tests/test_scored_defaults.py`),
  so the trip-free lines of the reactor, compressor_surge and pH should not move;
  hvac's do, since its edge charge changes. Every number from it is marked
  "stale" and is re-recorded once (W2).
- **My runs** (flanders, two cores, JAX 0.11.2, a code copy of 874f5fb3): one
  scored episode timed per task, and end-of-episode probes on the reactor,
  compressor_surge and pH. Scripts and outputs are in flanders
  `~/scratch/2026-11-27-tg-v1-pages/` and in the shared notes folder under
  `v1-task-pages-2026-11-27/`.

## Shared work, done once for every task

These close items 4, 8 and 9 everywhere. None needs a task to change.

- **W1. Land PRs 306 and 312** (in flight). PR 306 fixes hvac's item 2 and pH's
  trip charge in item 6.
- **W2. Re-freeze and record once on the final tree** (S per task, plus
  compute): `scripts/reference_ladder.py --freeze`, then the nulls, reference
  and MPC lines with `scripts/record_charter.py`. The ceiling can be kept as
  context, as the protocol says. The compressor's MPC line alone takes more than
  a day on bob.
- **W3. PPO and SAC at one budget** (M, once): one script trains Ajax's PPO (the
  learning check's settings) and Ajax's SAC (`ajax/agents/SAC`) with shipped
  settings, 30 training seeds each, at one budget such as the learning check's
  204,800 steps. It picks checkpoints on the validation seeds 300 to 349 and
  scores the 30 policies as a list with `tg.score`, which already gives the
  two-level interval over runs. Today no task has a result: `rl_results.json` is
  empty, and the learning check runs 5 PPO seeds and no SAC on `tg.make`'s single
  configuration, against the shipped expert rather than the reference
  (`tests/rl/test_learning.py:42`).
- **W4. One card per task** (S each), on the practitioner review's template
  (`docs/proposals/practitioner-review.md`, issue 7): the job, the observations
  and actions with units, the cost terms with their prices and whether each is
  sourced, the disturbances, the limits, the evaluation, the current numbers
  with intervals, and the deviations that matter most. No card exists today, and
  PR 314 adds none.
- **W5. Gymnasium IDs** (S, once): register `TargetGym/<task>-v1` with
  `gymnasium.register`. `tg.make_gymnasium` already exists and passes
  Gymnasium's own API checker on every task
  (`tests/test_env_conformance.py::test_gymnasium_view_passes_its_api_checker`);
  only the registration is missing. The registered episode must be the scored
  one (the scored default scenario, and the reactor's scored run).
- **W6. Freeze guard** (S, once): a manifest per v1 task holding the plant's
  stamp, the frozen reference's hash and the scoring definition (scenarios,
  weights, seeds, window, edge charge, run length), and a test that fails on any
  change to it. Two gaps today: `tests/test_env_versions.py` lets a stamp be
  re-stamped in place, which v1 must forbid; and the scoring definition is in no
  fingerprint by design (`scorecard.py`, `scoring_runs.py`, `evaluation.py`).
- **W7. One expert, everywhere** (S, docs): the cards and the v1 docs name
  `tg.make_reference` as the expert. `tg.make_expert` stays the PID building
  block, as already decided, but it is not the reference on hvac,
  compressor_surge or pH. The noise floors behind `tg.score`'s tie tolerance were
  measured on `make_expert` (`data/charter/noise_floor/*.json`), so they are
  measured again on the frozen reference.

---

## reactor

**Job.** A PWR operator follows the day's load plan with the control rods, the
soluble boron and the turbine's loading rate, holding the average coolant
temperature, the flux difference and the rod insertion inside their limits.
Every 30 s the agent sets the rod demand, the boron flow and the turbine's
loading rate.

**Score.** Dollars a step (lost output, rod wear, effluent; a trip charges
$3.84 M), averaged over a 7-day run of the stream pinned to each of four
scenarios (baseload 0.89, edf_daily 0.043, krsko_slow 0.043, fast_return 0.022),
from the reset, with no priced end (`scoring_runs.py:61-67`). The environment's
own episode is one day, 2880 steps; the scored episode is 20,160 steps.

| Item | Status | Evidence |
|---|---|---|
| 1 | pass | `docs/environments/reactor.md:5` |
| 2 | pass | Dollars, a fixed 7-day run. My probe: boron off for the run's last 20 min saves 0.055 % [0.001, 0.104] of the score, the same stop mid-run costs 0.068 %, and both are under the task's noise floor of 0.11 %. Off for the last 2 h costs 9.5 %. On baseload days alone the stop saves 0.3 % (20 min) and 0.9 % (2 h): a small end effect that the headline does not show. |
| 3 | pass | The Westinghouse rod program on the average temperature, from the NRC's training manual (HRTD 8.1 and 5.0), with boron from the operator's curve book (`reactor/experts.py:17-30`). Frozen in `data/reference/reactor.json`; re-freeze owed (W2). |
| 4 | fail | Nulls, expert and MPC exist in the dry run (stale): best null 55.97 (rod integrator), reference 0.2674, MPC 0.1032, no trips. No PPO or SAC (W3). |
| 5 | pass, stale | The MPC saves 61.4 % [27.0, 68.6] over the reference, with no trip and every solve converged (dry run, seeds 100 to 109, the repository's `scoring.headline`). |
| 6 | pass | Figure: `tests/reactor/test_reactor_axial.py::test_full_power_mode_is_a_damped_oscillation_near_the_published_figures` (Ringhals-4's axial mode). Trips: `tests/test_failure_cost_bound.py`. Expert over doing nothing: 0.267 against 55.97 at best. Determinism, `jit` and `vmap`: `tests/test_env_conformance.py`. All pass on int's CI. |
| 7 | fail | The 7-day scored run takes 2.80 s with the reference at batch 1 (1.80 s with a zero action). One day takes about 0.40 s, by proportion. Read per episode in a batch of 256, the 7-day run costs about 0.2 s (the RL review's 0.090 M steps a second, keeping observations and rewards). |
| 8 | fail | W4, W5 |
| 9 | fail | W6 |

**Smallest changes.**
- Item 7, one of two, for the owner to pick: score 2-day runs instead of 7-day
  ones, about 0.8 s (S: one constant in `scoring_runs.PROTOCOLS`, then W2's
  re-record; it gives up the 2026-11-24 choice of 7 days, which scored one boron
  cycle of the reference); or read item 7 per episode in a batch, where the
  task already passes (no change). Making the step itself three times faster is L
  and uncertain.
- Item 2 needs nothing for the headline. If the baseload number stays published
  beside it, as `tg.score` reports it, its end effect goes on the card; charging
  the run for the boron it owes at its end, as pH's tanks would be charged, is
  the alternative (S to M).
- Items 4, 5, 8, 9: shared work only.

**Left for v2.** Grey-rod operation (EDF's Mode G); axial damping that changes
through the cycle; the xenon peak after a scram (1,419 pcm against a 1,800 to
3,200 test band); load-follow days drawn from real unit output; cycle points
stratified across the seeds (closed PR 313); an MPC that reads the observation,
not the true state; and the baseload end effect above.

---

## hvac

**Job.** A building energy manager sets the supply water temperature of a heat
pump feeding a concrete slab, so an office stays in its 21 to 26 C band in
occupied hours at the least electricity cost on a day-night tariff. Every 15 min
the agent sets the supply temperature.

**Score.** Euros a step (electricity at the tariff, occupied kelvin-hours outside
the band, the floor above its limit), averaged over days 4 to 10 of a 12-day
episode, plus a once-off charge for the heat the building's store lost across
those days (`hvac/scenarios.py:75-92`). Three scenarios: normal 0.50,
shoulder_season 0.42, cold_snap 0.08. Nothing trips.

| Item | Status | Evidence |
|---|---|---|
| 1 | pass | `docs/environments/hvac.md:5` |
| 2 | fail on int; pass with PR 306 | On int the store's end credit pays: full supply on the last scored night saves 0.91 %, where the same move mid-window costs 1.1 to 1.2 % (RL review, H2). PR 306 charges only a shortfall: every end move then loses more than the same move mid-window on the weighted score. One residual remains: on the cold snap a top-up at 04:00 to 08:00 of day 10 saves 0.56 % against 0.40 % on day 7, about 0.01 % of the headline (the fix session's note). |
| 3 | pass | The ladder's feedforward rung: an SIMC PI on the room's temperature, set point 22.0 C, over a heating curve on the observed 24-hour forecast (`reference/rows/hvac.py:82-84`), frozen in `data/reference/hvac.json`. Re-freeze owed (W2). |
| 4 | fail | Nulls and expert in the dry run (stale): reference 0.006158, best null 0.01634 (hold reset). The MPC line was stopped when PR 306 changed the window's cost, so it is not measured on this plant. No PPO or SAC. |
| 5 | n.m.; the last record failed | Last MPC line 0.00860 against a reference of 0.00744 (2026-11-17, an older plant and the cascade rung): 16 % worse. The reference is now stronger (feedforward, 0.00616). PPO loses to a constant action on hvac (learning check). The foresight ceiling (0.00519, 2026-11-17) is context only. |
| 6 | pass | Figure: `tests/hvac/test_hvac_reality.py::test_slab_heating_time_constant_is_a_concrete_core_tabs` (source [10], p. 7, Fig. 3). Trips: none exist, so tripping cannot pay. Expert over doing nothing: 0.00616 against 0.0163 (dry run). Determinism, `jit`, `vmap`: conformance tests, noise floor 1.1e-6. |
| 7 | pass | 0.016 s for the 12-day episode at batch 1 |
| 8 | fail | W4, W5 |
| 9 | fail | W6 |

**Smallest changes.**
- Item 2: land PR 306 (in flight). Report the cold snap's 0.01 % residual on the
  card and leave it.
- Item 5: re-run the MPC line on the fixed code (S, one recording run). If it
  still loses, one M attempt: the planner models the weather as its
  mean-reverting deviation (`hvac/oracle.py:116-120`), while the observation's
  forecast also carries the regime's level (`hvac/env.py:418-428`), so the
  planner can be given the forecast the policy reads. If the MPC still loses
  after that, park hvac: no line then shows room, since the ceiling only counts
  as context.

**Left for v2.** Real weather (a public TMY) with a forecast error of its own;
cooling; a second zone; a cross-check against BOPTEST's
`bestest_hydronic_heat_pump`.

---

## compressor_surge

**Job.** A compressor station operator holds the header pressure on its set
point while the consumers' demand ramps, with the drive speed and the
anti-surge recycle valve, and never lets the machine surge. Every 0.1 s the
agent sets the speed command and the recycle command.

**Score.** kJ a step (the compressor's work, 1,000 kJ/s outside the pressure
zone, 2,000 kJ a surge, a trip charging 1.98e7 kJ), averaged over a 2-minute
episode of 1,200 steps. Three scenarios: normal 0.79, slow_approach 0.16,
fast_approach 0.05.

| Item | Status | Evidence |
|---|---|---|
| 1 | pass | `docs/environments/compressor_surge.md:5` |
| 2 | pass | kJ, a fixed 2-minute episode. My probe: the speed at its minimum for the last 1 s saves 0.010 % of the score, and for the last 3 s costs 36 % (mid-episode: 0.14 % and 214 %). The recycle shut for the last 1 s costs 2.3 % (normal day +0.6 %, against +0.4 % mid-episode); shut for 3 s or more it trips the approach episodes wherever it is placed. Nothing pays at the end that does not pay mid-episode. |
| 3 | pass | The ladder's feedforward rung: an SIMC PI on the pressure through the speed, and the recycle set by the steady balance at the vendors' 10 % control line from the metered delivered flow (`reference/rows/compressor_surge.py:24-80`), frozen in `data/reference/compressor_surge.json`. It has no margin loop on the valve. The limit cycle both reviews describe needs that loop (it survives with the speed loop frozen, PR 313's probe), so it belongs to the shipped `make_expert` pair (`compressor_surge/PHYSICS.md:737-752`), not to this expert. Re-freeze owed (W2). |
| 4 | fail | Nulls and expert in the dry run (stale): reference 2.997, best null 111.1 (constant 0.5). The MPC line has been running on bob since 2026-11-26 21:52 (`pipe_compressor_surge`). The last stored MPC line tripped 19 of 20 approach episodes (J 492,035, 2026-11-17). No PPO or SAC. |
| 5 | n.m. | The MPC line is not finished. Indirect only: on the default configuration the MPC beat the shipped expert on 10 of 10 seeds with no trip (hold 4.41 against 6.39, flanders, 2026-11-27), but the shipped expert costs about 47 % more than the reference (RL review, C1), so this says nothing about the reference. |
| 6 | pass | Figure: `tests/compressor_surge/test_compressor_surge_physics.py::test_gravdahl_fig_2_8_deep_surge_orbit`. Trips: `tests/test_failure_cost_bound.py`. Expert over doing nothing: 2.997 against 111.1. Deterministic: identical sums on the Mac and flanders across JAX 0.6.2 and 0.11.2 (RL review probe); `jit`, `vmap`: conformance tests. |
| 7 | pass | 0.153 s at batch 1 |
| 8 | fail | W4, W5 |
| 9 | fail | W6 |

**Smallest changes.**
- Item 5: let bob's MPC line finish, then read its saving over the reference
  with `scoring.headline` (S). If the MPC trips or loses, park the task: no
  other line shows room, and PPO loses to a constant here too. The tie tolerance
  must come from the reference's own noise floor (W7): the 0.19 on file is the
  limit-cycling pair's.
- The paused decoupling work is not needed for v1, since the frozen reference has
  no anti-surge PI to decouple.

**Left for v2.** A drive with inertia; a multistage map; sourced prices for the
zone, the surge and the trip; longer windows with process upsets; a margin loop
in the reference, which practitioners would expect (the ladder's override rung
adds one; on the dry run's climb it cost 2 % more, a median of 3.01 against
2.95, and tripped nothing either).

---

## ph_neutralization

**Job.** A wastewater operator keeps the discharge of a two-stage acid
neutralisation inside its pH 6 to 9 permit with the least reagent. Every 5 s the
agent sets the stage-1 reagent valve and the stage-2 trim valve.

**Score.** Litres of reagent a step: the reagent used, 1.67 L for each second
outside the permit, and 2 L for each litre diverted below pH 5. A violation (an
excursion past 60 min, or an overrun basin) charges a restart: 9,360 L with PR
306. The score averages over a 2-hour episode of 1,440 steps. Three scenarios:
normal 0.80, bad_day 0.15, buffer_loss 0.05.

| Item | Status | Evidence |
|---|---|---|
| 1 | pass | `docs/environments/ph_neutralization.md:5` |
| 2 | fail | My probe: both valves shut for the episode's last 5 min saves 4.1 % on the normal day, where the same cut mid-episode costs 344 %. On the weighted score the cut gains 0.2 % [-4.5, 4.9], because it loses on the bad day and the buffer loss; a policy that cuts only when it reads enough margin could keep the normal day's gain. The tanks hold neutralising capacity, and an episode may end with it drawn down for free. |
| 3 | pass | The ladder's feedforward rung: a PI on each stage, set point 6.45, with the reagent fed forward from the metered influent and the trim resting shut (`reference/rows/ph_neutralization.py:74-82`), frozen in `data/reference/ph_neutralization.json`. On the new placement seeds 6.45 holds (the comparison session's note). Re-freeze owed (W2). |
| 4 | fail | Nulls, expert and MPC in the dry run (stale): best null 0.809 (table), reference 0.0687, MPC 0.0577, no trips. No PPO or SAC; in the learning check PPO trained on the measured reward beats the best constant on 2 of 5 runs (report-only). |
| 5 | pass, stale | The MPC saves 16.0 % [13.9, 18.3] over the reference with no trip (dry run, seeds 100 to 109). Two caveats for the card: the MPC reads the hidden buffer flow and acid strength from the true state, so most of this room is set-point placement, which the owner accepted as fair if said; and the MPC plans to the episode's end, so part of its saving may be the end effect of item 2. Re-record after that fix. |
| 6 | fail on int; pass with PR 306 | Figure: `tests/pc_gym/ph_neutralization/test_ph_env.py::test_nominal_operating_point_matches_the_benchmark` (Henson and Seborg's pH 7.03 at the published flows). Trips: on the normal day, the default after PR 306, random actions survive at up to 8,879 L, above the old 6,480 L trip, so tripping could pay; PR 306 raises the charge to 13 L a step (9,360 L), a 5 % margin on 8 random seeds. Expert over doing nothing: 0.0687 against 0.809. Deterministic (noise floor 2.2e-7, identical across machines); `jit`, `vmap`: conformance tests. |
| 7 | pass | 0.024 s at batch 1 |
| 8 | fail | W4, W5 |
| 9 | fail | W6 |

**Smallest changes.**
- Item 2 (M): give pH an edge charge like hvac's. The env reports the
  neutralising capacity the two stages hold, in litres of reagent, from the
  charge-balance invariants it already carries. The evaluation charges, once,
  any of it the episode lost and credits nothing (`evaluation.Evaluation`, with
  `edge_charge`, needs one unscored step before the window). The MPC's plan
  takes the same charge. Then re-record (W2) and repeat the end probe as a test,
  as PR 306 did for hvac.
- Item 6: land PR 306 (in flight).

**Left for v2.** Valve dead band and stiction (the plant has both knobs, set to
0); probe drift; a reagent delivery delay; industrial-scale streams; influent
statistics from data.

---

## first_order (labelled debugging task)

**Job.** None by design: it is the suite's sanity check, a linear first-order lag
driven to a target drawn at reset, fully observed, with no disturbance and no
trip, for checking an algorithm or a wrapper against a known answer
(`docs/environments/first_order.md:3-8`). The agent sets one input every 0.05 s,
over 150 steps. Items 1, 2 and 5 do not apply: the vision keeps it unscored.

| Item | Status | Evidence |
|---|---|---|
| 1, 2, 5 | n.a. | Unscored by the vision |
| 3 | pass | The shipped PID (`tg.make_expert`, gains in `data/pid_gains.json`). There is no frozen reference, and an unscored task needs none; the freeze pins the PID's gains. |
| 4 | fail | PID and MPC returns stored on 10 seeds (`data/baseline_returns/first_order.json`, 2026-11-05). No null, PPO or SAC line. For a debugging task the learned agents' numbers are its main content: they are the known answer a user checks a new algorithm against. |
| 6 | pass | A textbook check stands in for a published figure, which a fixture does not need (`tests/pc_gym/first_order/test_first_order_physics.py::test_step_response_reaches_63_percent_at_one_time_constant`); no trip exists; determinism, `jit`, `vmap`: conformance tests. |
| 7 | pass | 0.001 s at batch 1 |
| 8 | fail | Its page carries the "Test fixture" note; a card and an ID come with W4 and W5 |
| 9 | fail | W6 |

**Smallest changes.** W3 adds its PPO and SAC numbers and a doing-nothing line,
which costs almost nothing at 150 steps. W4 to W6 freeze it as
`TargetGym/first_order-v1`, labelled unscored.

**Left for v2.** Nothing.

---

## What I ran

On flanders, two cores (`taskset`), JAX 0.11.2, from a code copy of int at
874f5fb3, with the frozen references of PR 307:

- **Timing (item 7):** one scored episode per task at batch 1 with the reference
  in the loop, compiled once and then timed on three seeds (median). The reactor
  ran its scored 7-day pinned stream, `scoring_runs.stream("reactor", "baseload")`.
- **End probes (item 2):** the frozen reference with one forced move over a block
  of steps, placed at the episode's end or in its middle, each scored by
  `tg.score` on the scoring seeds 200 to 209 at the scenario weights. Reactor:
  boron flow off for 20 min and 2 h. compressor_surge: the recycle shut for 1, 3
  and 10 s, the speed at its minimum for 1 and 3 s. pH: both valves shut for 5,
  10, 20 and 40 min. Two end effects showed: pH's 5-minute cut (4.1 % on its
  normal day) and the reactor's boron stop on baseload days alone (0.3 to 0.9 %,
  at most 0.055 % of its headline).
- **Room (item 5):** the MPC's saving over the reference with its interval,
  computed with the repository's `scoring.headline` on the dry run's per-seed
  costs for the reactor and pH.
