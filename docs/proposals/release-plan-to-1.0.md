# Release plan to 1.0

A plan, 2026-11-27, from the practitioner review
(`docs/proposals/practitioner-review.md`). It sequences that review's nine
fixes and its path to 10 into five releases, each one usable on its own by
beta-testers.

**Adopted by the owner on 2026-11-27 as the roadmap to 1.0,** with these
answers to its open decisions: the four additions proposed for 0.8.0 beside
the comparison API already in flight go in before the re-record
(compressor_surge's reference decoupled by a sourced, fixed recipe with no
sweep; the reactor's cycle point stratified across the scoring seeds; task
cards and a known-limits page; the beta label); cstr is registered, not
scored, in 0.8.0; and scored tasks change only in one realism batch, 0.10.0.

Two rules shape it:

- **Additive work never breaks anyone's numbers.** New lines, baselines,
  reports and new tasks can ship in any release.
- **Changing a scored task's definition is expensive.** It forces a full
  re-record, oracles included. So those changes go into one batch release
  instead of trickling out.

Each release has a gate rather than a date, so the timings below are
estimates.

| Release | Theme | Rough timing | What a beta-tester can do with it |
|---|---|---|---|
| 0.8.0 | First scored suite, beta | 3 to 4 weeks (late December or early January) | Train on the scored tasks and get a frozen score, with an interval, that they can cite |
| 0.9.0 | Reachable room | 6 to 8 weeks later (February to March) | See how much of the room causal control reaches, and compare with early PPO and SAC curves |
| 0.10.0 | Realism batch | 8 to 10 weeks later (May) | Use new task versions whose disturbances and prices come from data, with the old versions still scored |
| 0.11.0 | Wider suite | 8 weeks later (July) | Work on about eight scored tasks, at several information levels, with offline datasets |
| 1.0.0 | Stable benchmark | Once three months pass with no change to a scored task | Cite a suite that is promised not to change |

## 0.8.0: first scored suite, beta

The road already set for 0.8.0 stays as it is: the open PRs, the refactor, the
re-record, acceptance, the learning check and the release workflow. Five cheap
items join it before the re-record. Each one changes a recorded number, so
doing them later would force 0.9.0 to break.

- **Fix compressor_surge's reference.** Decouple or slow its pressure loop: a
  quick probe beat the shipped pair by 10 % with no trips (review, issue 4).
- **Stratify the reactor's cycle point across the scoring seeds** (issue 9).
- **Ship `tg.score` and `tg.make_reference`** (already in flight), and take
  `score`, `headroom` and `win_rate` out of the README (issue 2).
- **Write the docs a beta-tester needs** (issue 7):
  - a one-page card per scored task;
  - a "known limits" page saying that the room is an upper bound, that part
    of it needs information a policy does not have, and that there is no
    causal line or RL result yet.
- **Label the release as a beta**, in the README and in PyPI's
  development-status classifier.

On cstr, the project's own rule already says a task with an unmeasured line
waits. If its lines are not all measured at the freeze, ship it as "under
review" instead of holding the release.

**Gate.** Every scored line is current and reproduced in CI from the tag. The
reference scored against itself reads exactly 1. The installed-package gate of
the release workflow passes.

**Ask beta-testers.** Can you install, train and score in an afternoon? Is the
number `tg.score` returns the one you would put in a paper?

## 0.9.0: reachable room

Additive only: no scored task changes, so every 0.8.0 number stays valid.

- **A causal line per scored task** (issue 3). An estimator on the
  observation feeds the existing economic MPC, with constraints backed off. The
  room is then reported in two parts: what control can win, and what
  information is worth.
- **A preliminary RL slice** (issue 6). PPO and SAC at a few budgets up to
  1e7 steps, with published default settings, labelled preliminary. At these
  speeds it is cheap, so it is re-run in every release. The full grid still
  waits for the freeze.
- **Training support:** a wrapper that handles trips (capped or terminal) and
  a training recipe per task.
- **Sensitivity tables from the family sweep** (issue 5), showing whether the
  ranking of the lines holds across the ranges of the project's own numbers.
- **New scored tasks, if they meet the bar by the freeze** (glass_furnace,
  cement_kiln, distillation). Adding a task breaks nobody.

Three things run in parallel during 0.9.0 and are not shipped in it:

- **Validation studies:** replay a reactor load-follow day, run hvac beside
  BOPTEST's `bestest_hydronic_heat_pump`, check the compressor against a
  measured map, check pH against Hall and Seborg's experiments.
- **Data sourcing:** ENTSO-E unit output and prices, typical-year weather and
  archived forecasts.
- **Outside reviewers:** one engineer per plant family. Ask them at 0.8.0,
  because volunteers take time to find.

**Gate.** The causal line is recorded on every scored task, and it beats the
reference or its failure is documented. The RL slice is reproducible from one
script. The 0.8.0 numbers are unchanged.

**Ask beta-testers.** Do your agents land between the reference and the
causal line?

## 0.10.0: realism batch

This is the one release that changes scored tasks. All of them change at
once, each under a new version. The previous versions stay registered and
scored for one more release, so experiments already running can finish.

- **Disturbances and prices from data** (issue 5). Reactor load-follow days
  from ENTSO-E unit output, hvac on real weather with forecast errors,
  electricity at market prices.
- **The fidelity gaps** (issue 8). pH valve dead band, stiction, reagent delay
  and probe drift; reactor axial damping that changes through the fuel cycle;
  compressor drive inertia.
- **Everything the 0.9.0 validation studies found**, fixed here rather than in
  a later breaking release.
- **Validation reports and outside reviews**, published with each task.
- **A full re-record**, oracles and causal lines included, and the RL slice
  re-run.

**Gate.** Every changed task has its validation report and sensitivity table,
and the old versions still score.

**Ask beta-testers,** domain testers especially: do the new versions behave
like the plants you know?

## 0.11.0: wider suite

- **About eight scored tasks.** Add glass_furnace, cement_kiln, distillation
  and wake_surf as each passes the bar. Replace cstr if it has not earned its
  place.
- **Information levels.** Each task at full state, observation and delayed
  observation, so estimation and control are scored separately.
- **More baselines and data:** one model-based learner, offline datasets, and
  documented gradients through the plants.
- **Reporting for the whole suite:** intervals and performance profiles, a
  results table with a submission protocol, and the paper.
- **The full RL grid starts on TPU Research Cloud**, now that definitions are
  stable.

**Gate.** Each added task passes the same bar as the first five, and the
information levels are covered by tests.

## 1.0.0: stable benchmark

Ship it once three months pass with no change to a scored task's definition.
At that point:

- the API is frozen, with a deprecation policy;
- the full RL grid is published;
- the stability promise is written down: a scored version never changes, and
  new versions are added beside it.

The last point of the review's path to 10, use by other groups, builds up from
here.

## Rules for every release

- **Patches fix bugs only.** Within a release line (0.8.x and so on), a fix
  that moves a recorded number becomes a minor release instead.
- **Only minor releases change a scored task**, under a new version and with
  upgrade notes. Tasks under review can change in any release, and their pages
  say so.
- **Every minor release regenerates** the results table and the versioned
  docs. The dev docs follow main in between.
- **Every minor release needs a reason a user can see.** That follows the rule
  against versions nobody uses, and works out to about one release every two
  months.

## Where each of the review's issues lands

| Issue in `practitioner-review.md` | Release |
|---|---|
| 1. No current, stable number to cite | 0.8.0 (freeze, re-record, gate); 1.0.0 (written promise) |
| 2. The public comparison does not measure what the pages promise | 0.8.0 |
| 3. The room mixes control with clairvoyance | 0.9.0 |
| 4. One reference loses to a two-setting retune | 0.8.0 |
| 5. Prices and disturbances are mostly the project's own | 0.9.0 (sensitivity tables); 0.10.0 (data) |
| 6. No learned-policy results | 0.9.0 (preliminary slice); 0.11.0 (full grid starts); 1.0.0 (published) |
| 7. Contracts read as decision logs | 0.8.0 (task cards), kept current each release |
| 8. Fidelity gaps on the named properties | 0.10.0 |
| 9. Thin slices and one uncontrolled draw | 0.8.0 (stratified seeds, per-scenario table) |
