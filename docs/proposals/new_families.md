# New task families: a proposal

The owner asked for candidate task families that TargetGym does not yet
represent, prioritising two of them: trajectory tracking in process plants, and
operation close to an operating envelope. This page answers that request (called
the handover below). It proposes ten candidate tasks across seven families and
asks for two decisions before any environment code is written: which two tasks
to build first, and how to register new tasks without changing what existing
users get from the registry. Nothing on this page is implemented.

It was written on 2026-09-29 on the branch `docs/new-families-proposal`, which
starts from `reward/floor-normalised`. Every new task has to use the version-2
reward and `base.failure_kernel`, and `main` has neither yet. They exist on
`reward/floor-normalised` and on `feature/run-policy-on-benchmark`, which is
built on it. Both have since been merged into `main`.

## Status

As of 2026-09-30, after the owner's decisions and the gate checks:

- New tasks join the core pool. They are appended to the registry after the 21,
  so every default accessor, the counts and the default benchmark include them,
  and the seeds of the 21 do not move. The registry tier field proposed under
  [Registering new tasks](#registering-new-tasks) was dropped: with every new
  task in the core pool it had nothing to do.
- Two pieces of that design remain, because the new tasks need them. A task
  whose physics imports another package's file declares it in
  `EnvSpec.fingerprint_sources`, so both of its fingerprints hash it. A task
  added after the 21 keeps its controllers in its own package's `experts.py`,
  which its baseline fingerprint hashes and its version stamp leaves out, since
  `experts/pid.py` and `experts/mpc.py` are in every task's baseline
  fingerprint. `tests/test_registry_rules.py` pins the 21 specs and holds new
  tasks to these rules.
- The build set is `unstable_cstr`, then `compressor_surge`.
- The gate checks were run before any code, each re-derived by a second agent.
  - `compressor_surge` passed: a controller that chases the pressure setpoint
    without watching the surge margin crosses the surge line. Its drafted
    expert trips in about 10 % of episodes, so the expert needs a wider margin
    before anything is recorded.
  - `grade_transition` is dropped for now. On the published constants every
    grade is open-loop stable, the feed is too dilute for a runaway (the
    reactor cannot exceed 370.4 K), and the relative gain is 1.3 to 1.6, not
    the 3.2 to 7.8 quoted below from recalled constants.
  - `batch_reactor` is deferred. On the published jacket values a plain PI
    never trips; it becomes hard only with a 2-minute actuator lag that no
    source gives.
- `maglev` is not in this round.

The sections below are the proposal as it was written, before these decisions.

## What this page asks you to decide

1. Which two tasks to build first. The recommendation is `unstable_cstr`, then
   `grade_transition`. See [Recommendation](#recommendation).
2. How to register new tasks. The recommendation is a `tier` field that keeps
   every default accessor returning exactly the 21 tasks it returns today. See
   [Registering new tasks](#registering-new-tasks).
3. The smaller points collected under [Questions](#questions) at the end.

No environment code will be written until the choice is made.

## Why new families

TargetFoundation benchmarks set-point tracking on TargetGym's 21 tasks, and its
measurements split them in two: 12 process and energy plants regulated around an
operating point, and 9 aircraft tasks where an underactuated vehicle tracks a
moving reference. The aircraft are where generic methods fail. Every adaptive or
learned method TargetFoundation has tried trips on most set-point cycles of the
aircraft path and pursuit tasks, while the shipped PID trips on none. No process
plant in the suite is hard in that way.

A survey of the 21 tasks shows which properties are missing:

| Property | Tasks among the 21 that have it |
| --- | --- |
| Operation at an open-loop unstable point | none (see the note below the table) |
| A process plant holding an operating point next to a reachable trip | none |
| A moving reference in a process plant combined with instability | none; the four process plants with moving references (glass furnace, reactor, hvac, battery) are stable and single-loop |
| Batch operation, with a finite recipe | none |
| Quantised or staged actuators | none |
| More than 5 actuators | none; the most is 3 |
| A physical parameter drawn per episode and hidden | none |
| A sinusoidal reference outside the aircraft | none; hvac repeats a daily schedule |

The shipped cstr's model does have unstable steady states, but its coolant range
of 295 to 302 K keeps the task on the stable branch. Check 7 of the [model
review checklist](../model-review-checklist.md) runs each plant with its
actuator at zero and finds no task whose state grows at an accelerating rate.

The first three rows are what the rest of this page calls TargetFoundation's
gap: a process plant that is hard in the way the aircraft are, because it is
unstable or strongly nonlinear, runs close to a trip it can reach, and follows a
moving reference. The other rows would be useful additions, but none of them
makes a task hard in that way.

## How this was put together

The contract documents, the registry and every consumer of the registry in
TargetGym and TargetFoundation were read first, without changing either
repository. Ten candidate tasks were then drafted, one or two per family. A
small numpy script checked each draft's numbers: steady states, eigenvalues,
steps to trip and time scales. One reviewer then checked each draft's physics
and method, another its value as a benchmark and its cost to deliver, and the
draft was revised. Three more reviewers ranked the revised drafts, one each on
fit to the gap, delivery risk and faithfulness to sources.

The sources were looked up once the owner allowed it. One agent found and read
each source, and a second re-opened every URL and checked each value against it.
The second agent found 533 of the 559 values at the source as stated, with the
same unit wherever the source prints one. The other 26 were contradicted by the
source or not visible at the cited URL, for example a boiler table read from the
wrong column. A few of those 26 are values we derived ourselves, which this page
labels as derived; the rest are not used here. No person has re-read the papers;
the value checks were done by the two agents.

The derived numbers on this page (steady states, eigenvalues, steps to trip,
costs) come from those scratch scripts. The scripts ran before the sources were
looked up, on TargetGym's shipped values where a task reuses them and otherwise
mostly on values the drafting agent recalled from papers without a source at
hand. This page calls those recalled values. The scripts have not been re-run on
the published values, which differ in places; the task sections say where. None
of these numbers comes from an environment.

The full drafts (equations, parameter tables, observation and action layouts,
rewards, disturbances, validation targets and risks), the scripts and the source
checks are not in the repository. They sit in a temporary working directory and
will be deleted unless question 13 keeps them. This page does not repeat their
equations, parameter tables, per-task rewards or validation targets, so letting
it stand alone means writing those again for each chosen task. Where a draft and
this page differ, this page is current.

Sources and values carry one of six labels:

- **read**: the value was read in an open copy of the original.
- **read in a reproduction**: the original is paywalled, and the value was read
  in an open paper that reproduces its table.
- **citation only**: the paper exists as cited, but its values were not read.
- **recalled**: a value the drafting agent gave from memory with no source at
  hand. It is unverified, and it must be replaced by a read value or declared as
  ours before the task is built.
- **derived**: our own arithmetic on the values above (steady states,
  eigenvalues, steps to trip, frequencies), to be shown in `PHYSICS.md`.
- **ours**: a value chosen for the task, not taken from a source. Its
  `PHYSICS.md` marks a model constant or disturbance size as `TUNED` (not
  sourced), as the physics methodology requires, and a restart time, trip limit,
  tolerance or price as provisional, as the reward-shaping page does.

Where a task below gives no label, its time step, episode length, reference
schedule, action range, disturbance sizes and restart time are ours. Each
episode length is meant as the task's `test_params` episode, which is the one
baselines are recorded and scored on, and it meets `docs/rl-protocol.md`'s rule
N >= max(10 tau_actuator, 1 T_period) where a time constant or period exists.
Three drafts set a shorter `test_params` to keep tests fast and fall below that
rule (`grade_transition` 240 steps, `supermarket_refrigeration` 720,
`staged_heating` 600); each is to be raised to the length given here.

## Summary

| Task | Family | Model and lead source | State / obs / action | dt | Episode | Trip | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `grade_transition` | 1 | MMA polymerisation CSTR, Doyle et al. (1995) | 8 / 12 / 2 | 60 s | 480 steps, 8 h | reactor at 370 K | 8.5 d |
| `batch_reactor` | 1 | jacketed batch reactor, Cott and Macchietto (1989) | 14 / 6 / 1 | 30 s | 600 steps, 5 h | reactor at 110 C | 7 d |
| `compressor_surge` | 2 | Greitzer compressor, Moore and Greitzer (1985) | 18 / 7 / 2 | 0.1 s | 1200 steps, 2 min | surge line crossed | 8.5 d |
| `unstable_cstr` | 3 | shipped cstr at its unstable point, Decardi-Nelson and Liu (2022) | 11 / 4 / 1 | 3 s | 1200 steps, 1 h | reactor at 365 K | 5 d |
| `maglev` | 3 | voltage-driven levitation, textbook values (recalled) | 4 / 4 / 1 | 10 ms | 1200 steps, 12 s | ball touches magnet or rest | 4.5 d before re-sizing on a rig |
| `glucose_insulin` | 4 | Cambridge type 1 model, Wilinska et al. (2010) | 17 / 5 / 1 on the draft | 5 min | 864 steps, 3 days | glucose below 54 mg/dL | 7.5 d on the draft |
| `supermarket_refrigeration` | 5 | Larsen et al. (2007), as given in Sager (2012) | 16 / 11 / 3 | 10 s | 1440 steps, 4 h | goods above 8 C or below -1 C | 9.5 d |
| `staged_heating` | 5 | hvac's 5R1C building with a three-stage boiler | 8 / 8 / 1 | 60 s | 1440 steps, 1 day | supply at 85 C | 8.5 d |
| `cd_paper_machine` | 6 | separable cross-direction model, stylised | 177 / 77 / 25 | 20 s | 360 steps, 2 h | sheet break | 7.5 d |
| `grid_inverter` | 7 | averaged LCL inverter, our values | 7 / 6 / 1 | 100 µs | 2400 steps, 12 grid cycles | overcurrent or overvoltage | 13 d |

Each task's Sources bullet, under [The candidates](#the-candidates), says which
of its values were read and which are citation only.

Effort is in working days on the TargetGym side: `PHYSICS.md`, the environment
and its dashboard (`rendering.py`), tests, PID, MPC, docs, version stamps and
the attended part of baseline recording. It is calibrated on how the last seven
environments went, including their follow-up corrections. It leaves out the
one-off registry work (about 3.5 days, shared by every task), TargetFoundation's
declarations (0.5 to 2 days per task) and machine time, since each recording
runs as its own overnight job. The figures for `unstable_cstr` (5 d) and
`supermarket_refrigeration` (9.5 d) still include about 0.25 days each of that
shared registry work, so each is about 0.25 days high.

## Recommendation

Build `unstable_cstr` first and `grade_transition` second. Build
`compressor_surge` third, once its gate check (below) has passed.

`grade_transition` is the best fit for the gap. It is a 2x2 process plant where
both references move and the quality measurement arrives 10 minutes late. On the
recalled constants the draft was sized on, one grade is also open-loop unstable,
a loss of cooling trips the plant in about 6 steps from that grade and in 12 to
16 steps from the other three, and the loops are coupled: with the natural
pairing (molecular weight on initiator flow, temperature on coolant flow) the
relative gain is 3.2 to 7.8 on the three stable grades, where 1 would mean no
coupling. Distillation is more coupled (about 52), but its targets are fixed.
Here both references move, so the coupling matters during every grade change.
The published constants describe a different regime, so these figures are
measured again before any code (second caveat below).

`unstable_cstr` is the best-sourced task on the list, and at 5 days the cheapest
process plant; only `maglev`, an electromechanical servo at 4.5 days before it
is re-sized on a published rig, costs less. It is unstable at every target. No P
or PI loop on concentration can stabilise it at any gain, and a setpoint switch
can leave the reactor 3.3 K below its point of no return, the temperature above
which even full cooling cannot stop it running away to the trip. Its model is
the shipped cstr's, which is Seborg's textbook reactor. An open 2022 paper
already controls this reactor around the same unstable point, and later safe-RL
work reuses that paper's bounds. Building it first also builds what every later
task needs: the registry tier, controllers kept out of the shared expert
modules, the check-7 entry for unstable plants, and a policy for version-1
rewards on tasks where avoiding a trip is the main difficulty.

The three rankings agreed on `unstable_cstr` and split on the rest. Scores are
out of 10. The faithfulness ranking was made before the sources were looked up,
and the lookups helped `grade_transition` and `compressor_surge` most.

| Task | Fit to the gap | Delivery risk (high is safe) | Faithfulness |
| --- | --- | --- | --- |
| `grade_transition` | 8 | 3 | 5 |
| `unstable_cstr` | 7 | 8 | 8 |
| `compressor_surge` | 6 | 3 | 6 |
| `batch_reactor` | 5 | 4 | 4 |
| `glucose_insulin` | 4 | 3 | 3 |
| `maglev` | 3 | 7 | 7 |
| `supermarket_refrigeration` | 2 | 1 | 3 |
| `staged_heating` | 2 | 3 | 2 |
| `grid_inverter` | 2 | 2 | 5 |
| `cd_paper_machine` | 1 | 5 | 4 |

This departs from the handover, which asks to prioritise families 1 and 2. The
plant model of `compressor_surge`, the only family-2 candidate, is sound, but it
is not yet clear that the task would be hard. Under the version-2 reward a surge
costs about 3e9, while running one percent closer to the surge line saves only
about 50 per episode in recycle energy, so the reward trades tracking speed
against the risk of a trip and gives energy almost no weight. TargetFoundation's
current methods would also leave the recycle valve at mid-range, where the
compressor cannot surge, and for them the task would reduce to an easy
single-loop pressure problem. A numeric check of about a day, not yet run, would
show whether a controller that chases the pressure setpoint crosses the surge
line often enough to make the task hard. To keep strictly to families 1 and 2,
build `grade_transition` and `compressor_surge`, with that check first.

Two caveats on the recommended pair:

- Both are jacketed exothermic reactors that can run away. Their difficulties
  differ. `grade_transition` has coupled loops, a delayed measurement and a
  reference that moves into an unstable grade. `unstable_cstr` is unstable for
  the whole episode and has a point of no return. Until the compressor is built,
  though, any TargetFoundation result on instability would rest on one physical
  mechanism.
- `grade_transition` is the riskier of the two to deliver. The sources confirm
  the isothermal model completely. The constants that add the energy balance,
  and with it the runaway, come from a preprint that lists most but not all of
  them. They describe a different regime from the draft's: five grades at 344.8
  to 348.1 K with conversion 0.12 to 0.46, in a 2.1 m³ reactor with a residence
  time of about 5 h, where the draft's grades sit at similar temperatures but at
  conversion 0.04 to 0.12 in a 0.1 m³ reactor with a 6-minute residence time.
  Before any code, a scratch check must show that the published regime still has
  an unstable grade and a reachable runaway. If it does not, the task keeps its
  coupling, delay and moving references and loses its instability.

## Registering new tasks

### What must not change

TargetFoundation's `scripts/benchmark_all.py` runs `list(REGISTRY)` by default
and derives each task's episode seeds from the task's position in that list.
TargetFoundation's tests assert that its per-task declarations cover exactly the
registry's names, and its golden test, which fails when any file it hashes
changes, hashes `registry.py`. Adding a new spec to `registry.py` beside the 21,
so that it appears in `REGISTRY`, would therefore change TargetFoundation's
benchmark and break those tests, and a spec inserted mid-list would shift the
episodes of every later task. The unmerged `target_gym.benchmark` module derives
seeds the same way.

### Proposed: a registry tier field

- `EnvSpec` gains `tier`, `"core"` by default. The 21 stay core. This registry
  tier says only whether default accessors return a task. It is unrelated to the
  six difficulty tiers of `docs/complexity.md`, and on this page "tier" always
  means the registry tier.
- New tasks are `"extended"`. Their packages live under
  `src/target_gym/extended/`, and their specs in `target_gym/extended/specs.py`,
  which is imported only when asked for.
- `target_gym/__init__.py` exports each extended environment and params class
  lazily, through a module-level `__getattr__`, so they are importable from the
  package root as `tests/test_public_api.py` requires, while `import target_gym`
  still runs no extended code. A broken extended task then cannot break the
  import for the 21.
- New groups live beside those specs, in `EXTENDED_GROUPS`, with any display
  names in `EXTENDED_DISPLAY_NAMES`. They never reuse one of the four core group
  names. A group is created with its first task (question 4).
- `REGISTRY`, `GROUPS`, `all_specs()`, `env_names()` and `specs_in_group()`
  return exactly what they return today, in the same order, for every existing
  caller. Code that wants the new tasks asks for them, with `all_specs("all")`,
  `registry.get(name)` or `specs_in_group()` given a new group's name. The
  mechanism widens `get` to search every tier and lets `specs_in_group` accept
  an extended group, which today raises `KeyError`.
- Each new task declares a fixed number, from 1000 up, that TargetGym's
  `benchmark.py` uses in place of the task's list position when it derives
  episode seeds. Adding a task, or promoting one later if that is allowed
  (question 15), then never shifts another task's episodes. TargetFoundation's
  `scripts/benchmark_all.py` derives seeds itself and has to change before it
  can run a new task (see What TargetFoundation will need).
- Physics shared by several new tasks lives in
  `src/target_gym/extended/common/`, which every extended task's version stamp
  and baseline fingerprint both hash. A test checks that an extended package
  imports only its own modules, `base`, `reward`, `utils`, `integration`,
  `extended.common`, and the shared experts from its own `experts.py`. Any other
  import is declared on the spec as an extra fingerprint source.
- TargetGym's own tests, version stamping and page generators iterate every tier
  and every group, so new tasks get the whole conformance suite and their own
  pages. A guard test checks that no registered spec is left out of the
  conformance suite. `record_baselines.py` can record every tier, with the core
  tier as the proposed default (question 12). Its clean-up step, which today
  deletes every row whose name is not in `REGISTRY`, checks names against every
  tier instead. `scripts/tune_pid.py` also looks tasks up in `REGISTRY` and
  moves to `registry.get`.
- The environment counts in the README, `environments.md` and `docs/index.md`,
  and the homepage's flagship clips, stay on the 21. Each new page needs an
  entry in the hand-written nav in `mkdocs.yml`. Three hand-maintained tables
  list every plant and no test checks them: the per-plant summary in
  `docs/reward-shaping.md`, the ladder in `docs/complexity.md` and the
  validation table in `docs/PHYSICS_METHODOLOGY.md`. Each new task adds a row to
  each, in the same change as its `PHYSICS.md`.
- The same change rewrites "Adding an environment" in `CONTRIBUTING.md` to
  match.

**Cost to the 21.** From reading the code, the 21 existing tasks are unaffected.
`registry.py` is in no fingerprint, and the fingerprint code's path for extended
tasks never runs for core tasks. The one fingerprint change that runs for every
task keeps files named `experts*` out of the version stamp, so retuning a new
task's own controller moves its baseline fingerprint but not its version. No
package of the 21 holds such a file, so the 21 hash exactly the files they hash
today. The change that adds the mechanism has to prove this by passing the
baseline and version-stamp tests without re-recording or re-stamping anything.

**Cost to TargetFoundation.** Its default run is unchanged. It sees new tasks
only after it moves its pin and opts in. That move needs one rehash of its
golden test, because `registry.py`, `benchmark.py` and `eval.py` change once.
After that, a new task changes no TargetGym file the golden test hashes, unless
it changes shared evaluation code: `grade_transition`'s delayed measurement
needs `eval.py` to split episodes on a switch flag the environment reports, so
that support should ship with the mechanism. TargetFoundation's own
`policies.py`, where each new task needs rows, is golden-hashed too, so each
batch of tasks it declares costs it a rehash whatever design is chosen here.

**Cost in time.** Running every tier adds CI and recording time that has not
been measured. `docs/testing.md` holds both CI jobs to ten minutes, and each
extended task adds conformance compiles, slow closed-loop tests and an MPC to
record. One option is to run extended conformance as its own CI job that still
blocks merges, so the core budget stays visible, and to have
`record_baselines.py` record the core tier unless asked for more (question 12).

**Where it starts.** `benchmark.py` exists only on
`feature/run-policy-on-benchmark`, whose commit TargetFoundation pins, so the
mechanism branch should start from that commit or wait until it is merged. That
branch is also checked out in a separate worktree that TargetFoundation's
benchmark jobs import, so the new branch is created from the pinned commit and
that worktree is left untouched.

The mechanism costs about 3.5 days, paid once for all new tasks.

### Alternatives considered

- **A separate registry module**, leaving every core TargetGym file byte for
  byte the same. It would spare TargetFoundation the rehash caused by changed
  TargetGym files. TargetFoundation would still rehash when it opts in, because
  its own `policies.py` gains rows. It also duplicates the benchmark loop and
  the evaluation code, and it needs a second baseline file, because the recorder
  deletes rows for names it does not know. A test that forgets to import the
  second module silently drops the new tasks.
- **Named, frozen suites** such as `targetgym-21`. They are good for citation,
  but `REGISTRY` would change meaning and every added task would cost
  TargetFoundation a rehash. A frozen name for the 21 is still worth considering
  at the next release.

## Rules for every new task

- **Names.** A new name must not start with an existing task name, which also
  rules out `plane` and `patrol`, and no existing task name or PID gains key may
  start with the new name. The baseline fingerprint collects every gains key
  that starts with the task's name, so a task called `cstr_unstable` would move
  `cstr`'s recorded baseline, and a task called `four` would take in
  `four_tank`'s gains. Runner code also treats anything starting with `plane` or
  `patrol` as an aircraft. That is why the task is `unstable_cstr`.
- **Controllers stay out of shared files.** `experts/pid.py` and
  `experts/mpc.py` are in every task's baseline fingerprint, so a new controller
  there would make every recorded baseline stale (every task has one,
  `patrol_bearing_only` too since its MPC slot was filled). A task's
  controllers go in its own package as `experts.py`, reached through a module
  argument on `registry._pid` and `_mpc`. The open roadmap item on scoping the
  baseline fingerprint to the code each environment reaches would remove this
  constraint; keeping new
  controllers in their own package avoids it without waiting. `base.py`,
  `reward.py`, `utils.py` and `integration.py` are in every version stamp, so
  new tasks use them and do not edit them.
- **Imported physics is fingerprinted.** A task's fingerprints hash only its own
  package, so physics imported from elsewhere is invisible to them unless it is
  declared (see the import test above). `unstable_cstr` is the one task that
  imports physics so far. `staged_heating` reuses hvac's building differently:
  its draft copies those functions into its own package with a test that fails
  if they stop agreeing with hvac's. Importing and declaring is the default;
  copying with an agreement test is allowed where a task wants to decide when to
  follow the original.
- **A version-1 reward.** The reward contract requires one on every task, and
  TargetFoundation's headline score is the version-1 reward, which it finds by
  the name `compute_reward_v1` in the task's `env.py`. New tasks have no earlier
  reward to keep, so each gets one in the suite's log-scaled form, declared as
  ours. Under version 1 a tripped step scores zero and the plant restarts near
  its target, so a trip costs a single step. On a task where avoiding a trip is
  the main difficulty, a policy that keeps tripping can then outscore one that
  holds safely. On `unstable_cstr` a hand estimate gives about 0.2 per step to a
  policy that heats fully and so runs away, trips and restarts over and over,
  and about 0.05 to one that lets the reactor go out and stay safely
  extinguished. This is measured per task before `PHYSICS.md`, and
  TargetFoundation should report trips beside the version-1 score.
- **Hold measurements.** Each task gets a row in `scripts/measure_hold.py` and
  in `src/target_gym/data/hold_measurements.json`. TargetFoundation's step test
  reads its duration from that file, and without a row it runs for one step.
- **Disturbances.** Each task declares two tuples on its spec.
  `disturbance_fields` names the state entries that hold a zero-mean random
  process, such as a drifting feed temperature; the conformance suite checks
  that they do not ratchet when the plant is stepped with a constant key.
  `noise_fields` names the parameters that set the size of that randomness; the
  MPC plans on a copy of the parameters with these set to zero. Fixed schedules,
  meals (which are not zero-mean) and values drawn once per episode go in
  neither. Per-step randomness is drawn from `fold_in(key, state.time)`. A
  vector-valued disturbance, which the conformance suite cannot check yet, gets
  its own test in the plant's suite.
- **Restart in place.** A task with `restart_in_place = True` keeps its state
  through a trip and is charged `restart_steps` times the failure cost on every
  step outside its envelope, so its restart time acts as a per-step price.
  `glucose_insulin` (3) and `staged_heating` (15) do this, and the in-place test
  in `tests/test_reward_contract.py`, written for hvac, is extended to them.
- **Hidden parameters and the MPC.** Every MPC in the suite reads the true
  state, and the planner's copy zeroes only the noise fields. On a task that
  draws a hidden parameter per episode (`glucose_insulin`'s patient,
  `batch_reactor`'s rate multiplier, `grid_inverter`'s grid impedance), the MPC
  therefore knows it and is an oracle ceiling. The gap from the PID to the MPC
  then includes the value of knowing the parameter, and `PHYSICS.md` and the
  baselines note say so.
- **Unstable plants and check 7.** Check 7, the unforced run in the model review
  checklist, has no allowlist, and whether an unstable plant passes it depends
  on where the restarts fall. Give it an allowlist with an entry for open-loop
  unstable plants that records the measured growth rate, and add a test in the
  plant's own suite that asserts the instability.
- **Append, never insert**, in whichever list the tasks end up in.
- **Version stamp.** Ship new tasks as `-v2`, the stamp under which 20 of the 21
  existing tasks carry the floor-normalised reward (the reactor carries it as
  `-v3`, because its `-v2` was an episode-length fix), so a new task's first
  published number is on that reward (question 6).
- **Counts.** The README says "All 21 environments are covered by fifteen
  contracts" (the aircraft variants share a plant), and a test in
  `tests/test_docs.py` checks that number against every `PHYSICS.md` under
  `src/`, which includes `src/target_gym/extended/`. The first new contract
  therefore breaks it: either the README's sentence about the 21 becomes false,
  or the test counts only core packages. The mechanism change makes the test
  count core packages and has the README report extended contracts in a sentence
  of their own. The test also reads number words only up to twenty, and its
  pattern cannot match "twenty-one".

## The candidates

### Family 1: trajectory tracking in process plants

#### `grade_transition`: grade changes in an MMA polymerisation reactor

- **Model.** A free-radical methyl methacrylate solution polymerisation CSTR
  with a cooling jacket: species balances for monomer, initiator and two moments
  of the chain-length distribution, reactor and jacket energy balances, and two
  disturbance states. The quality variable is number-average molecular weight
  (NAMW).
- **Sources.** The isothermal model is Doyle, Ogunnaike and Pearson (1995), read
  in a reproduction. The original is paywalled, and Lawrynczuk (2022,
  Neurocomputing 513, Table 1) reproduces it with units for every parameter and
  a steady state to five figures. Terrazas-Moreno, Flores-Tlacuahuac and
  Grossmann (2007, open preprint), read, give four grades from 15,000 to 45,000
  kg/kmol with their initiator flows and optimal transition times. The constants
  for the energy balance, from Silva-Beard and Flores-Tlacuahuac (1999), are
  read in Terrazas-Moreno et al. (2008, open preprint), which gives five grades
  with their steady states but not every heat-transfer input. Daoutidis, Soroush
  and Kravaris (1990) and Maner et al. (1996): citation only.
- **Size.** 8 states, 12 observations, 2 actions (initiator flow and coolant
  flow, each through an equal-percentage valve map). dt 60 s, 480 steps (8 h).
- **Reference.** A production campaign. The plant starts lined out on one grade
  and moves through adjacent grades, dwelling 90 to 150 steps on each. NAMW and
  temperature both step at a switch. The next grade and the steps to the switch
  are observed, so the switches can be anticipated. The NAMW measurement arrives
  10 minutes late and is scored against the grade its sample was taken under.
- **Disturbances.** `disturbance_fields = ("Tw0_dev", "CI_in_dev")`: slow
  zero-mean drift in coolant supply temperature (1.5 K, 30 min correlation) and
  in initiator activity (3 %, 60 min correlation), both ours, with `noise_fields
  = ("Tw0_sigma", "CI_in_sigma")`.
- **Trip.** When the reactor reaches 370 K, a runaway interlock injects a
  shortstop, a chemical that halts the polymerisation (ours). Fresh restart, 4 h
  restart time. A restart always lands on a grade next to the new target, so
  tripping at a switch never skips a transition.
- **Experts.** A decentralised 2x2 PI (NAMW on initiator, temperature on
  coolant) with grade feedforward, gains scheduled by grade and a
  high-temperature override. Tuning reuses the coordinate-descent search in
  `scripts/tune_pid.py` (`_tune_aircraft_search`), which takes a `TUNERS` row, a
  starting entry for the gains, a lookup through `registry.get`, and a
  controller that reads its gains through `experts.pid._load_gains`. A CasADi
  NMPC over 40 steps that previews the campaign and falls back to the PID when a
  solve fails.
- **What it adds.** A 2x2 process plant with two moving references, coupled
  loops, a delayed quality measurement, a runaway trip 20 to 26 K above the
  grades, and an unstable grade (growing at +3.7 per hour on the recalled
  constants) that the reference moves into.
- **Effort.** 7 days as drafted, plus the draft's 1.5-day allowance for
  re-deriving the grade table, valve ranges and trip on the published constants.
- **Open points.** The gate on the published regime described under
  Recommendation. The delayed measurement needs `eval.py` and
  `scripts/measure_hold.py` to split episodes on a switch flag the environment
  reports, with the NAMW flag set 10 steps after the temperature's; the
  `eval.py` part ships with the mechanism. The shipped PID leans on grade
  feedforward, which is model knowledge TargetFoundation's methods lack. A PID
  without it would make that comparison fair, but a spec holds one `make_pid`
  and the recorder records one PID per task, so it needs a second registered
  spec or an optional baseline slot on `EnvSpec` (question 16). Linearised on
  its unstable grade, the plant looks like the unstable plants drawn by
  TargetFoundation's generator, which makes the synthetic plants
  TargetFoundation pretrains on. The generator has nothing like its coupled
  loops, its up to three steady states for the same flows, or its delayed
  measurement.

#### `batch_reactor`: a temperature recipe, with runaway as the trip

- **Model.** Cott and Macchietto's jacketed batch reactor: A + B to C (wanted)
  and A + C to D, two Arrhenius rates with different activation energies, and
  reactor and jacket energy balances.
- **Sources.** Cott and Macchietto (1989), read in a reproduction. The original
  is paywalled, and its table is read in two open reproductions, Tan et al.
  (2011) and Sujatha and Pappa (2010). After unit conversion they agree on the
  rate constants, heats of reaction, heat capacities, heat-transfer coefficient
  and initial charges. They differ on the jacket: Tan et al. give a volume of
  0.6912 m³ and model a coolant flow, while Sujatha and Pappa give 0.6921 m³ and
  model a first-order setpoint loop. The reactants' heat capacities are kcal
  values converted to kJ, which settles a question the draft had to leave open.
  Both also give the jacket fluid's heat capacity as 0.45 kcal/(kg K), or 1.88
  kJ/(kg K), where the draft recalled water's 4.184. On the published value the
  jacket lag is about 1.43 min instead of 1.69, and the jacket removes about 15
  % less heat per kelvin (derived).
- **Size.** 14 state entries (6 ODE states plus the recipe and batch clock), 6
  observations, 1 action (jacket supply temperature, 20 to 120 C). dt 30 s, 600
  steps (5 h, two batches).
- **Reference.** A recipe per batch: a heat-up ramp of 1.5 to 3 K/min to a hold
  between 88 and 95 C, the hold until minute 120 of the batch, a cool-down, and
  a recharge at minute 150. The recipe's state is observed, so a policy without
  memory can anticipate it.
- **Disturbances.** None declared: `disturbance_fields = ()` and `noise_fields =
  ()`. Each batch draws its charge temperature, recipe and a hidden rate
  multiplier of 0.85 to 1.15 on both reactions, which moves the point of no
  return by about 4 K.
- **Trip.** Reactor at 110 C (ours). Sujatha and Pappa bound the reactor at 100
  C, and Tan et al. bound only the jacket. A trip loses the batch and a fresh
  batch starts, with a 3 h restart time. A sourced alternative is the
  cooling-failure temperature of Ubrich et al. (1999).
- **Experts.** A cascade PI with ramp feedforward (reactor temperature to jacket
  setpoint to supply temperature), tuned by coordinate descent. A CasADi NMPC on
  the same discrete map, with a 20-minute horizon.
- **What it adds.** The suite's first batch task: a finite recipe that restarts
  inside the episode, a hold that starts slightly unstable and becomes stable as
  the reactants run down, a trip that costs a whole batch, and a hidden
  conversion driving the heat release.
- **Effort.** 7 days.
- **Open points.** With the confirmed units, and the draft's recalled jacket
  heat capacity, the plant is milder than the handover hoped. At the start of
  the hold a small temperature error grows by a factor of e in 8 to 60 minutes,
  and by the end of the hold it decays. A constant mid-range supply never runs
  away, and trips come only from overshoot where the ramp meets the hold. The
  published jacket heat capacity makes the hold less mild than this, so these
  figures are re-run on it before anything else. One reproduction reports that
  with the heat-transfer coefficient 25 % low and the first reaction 25 %
  faster, a fixed PID overshoots to about 180 C. Widening the hidden variation
  to that size and adding a fouling factor on the heat-transfer coefficient
  would be a sourced way to make the task hard. Ubrich, Srinivasan, Stoessel and
  Bonvin (1999), read, is an open semi-batch benchmark with a sourced maximum
  temperature and the feed rate as a second input. It is the better-documented
  alternative if a primary source matters more than the canonical plant.
  Linearised, the heat-up and the start of the hold are a slow unstable mode
  behind a lag tracking a previewed ramp, which TargetFoundation's generator
  already draws. The batch restart, the hold that turns stable as the reactants
  run down, and a trip that costs a whole batch are beyond it.

### Family 2: operation close to an envelope

#### `compressor_surge`: header pressure through turndown, with an anti-surge valve

- **Model.** A Greitzer lumped compression system with the Moore-Greitzer cubic
  characteristic scaled by the fan laws, a consumer valve, a recycle valve, and
  rate-limited speed and valve actuators. The gas in the duct and the plenum
  oscillates like a mass on a spring (a Helmholtz resonator), and surge is that
  oscillation losing its damping and growing.
- **Sources.** The characteristic's coefficients are read in Moore and
  Greitzer's NASA report CR-3878 (1985) and in Gravdahl's 1998 thesis: a
  shut-off head of 0.3 and a cubic hump with semi-height H = 0.18 and semi-width
  W = 0.25 in flow. Gravdahl simulates surge with B = 1.8, where B is Greitzer's
  stability parameter. When B is large, a compressor pushed past the peak of its
  characteristic goes into surge, the whole-system oscillation this task trips
  on. At small B it settles into rotating stall instead, a local flow
  disturbance that circles the blades, which this model leaves out. Greitzer
  (1976), Moore and Greitzer (1986) and Gravdahl and Egeland (1999): citation
  only. For the geometry, Gravdahl et al. (2000), read, describe a measured rig,
  the TU/e turbocharger, with a 0.0203 m³ plenum and a 1.8 m duct, where surge
  was measured at 20 Hz and simulated at 23 Hz. The 10 % surge-control margin is
  industrial practice as described by Mirsky et al. (2015), read. The draft's
  geometry is ours. Moving to the TU/e rig would raise the Helmholtz frequency
  from the draft's 2.2 Hz to about 28 Hz (our arithmetic from its published
  volume, area and length), which needs a control step near 10 ms, so dt, the
  episode length and the recording cost would all be re-sized.
- **Size.** 18 state entries (3 ODE states plus actuator and schedule memory), 7
  observations, 2 actions (speed and recycle valve). dt 0.1 s, 1200 steps (2
  min).
- **Reference.** Header pressure setpoints in four 30 s blocks between 20 and 28
  kPa, while consumer demand ramps between levels as a disturbance. In about two
  thirds of the blocks the recycle valve has to open to stay off the surge line.
  Only the current setpoint is observed; the levels and switch times are not, so
  every step arrives unannounced.
- **Disturbances.** `disturbance_fields = ("demand_dev",)`: a zero-mean drift in
  the consumers' valve opening (2 % of opening, 5 s correlation), with
  `noise_fields = ("demand_noise_std",)`.
- **Trip.** The compressor's flow coefficient falling below the peak of its
  characteristic, which is the surge line, at any substep. Fresh restart, 15 min
  restart time (no source gives one).
- **Experts.** The standard industrial pair: a pressure PI on speed and an
  anti-surge PI on the recycle valve at a 10 % margin, with an override that
  opens the valve fully below 4 %. A gradient MPC that differentiates the plant,
  with `reward.proximity_cost` as its surge barrier.
- **What it adds.** A process plant whose most efficient operation lies next to
  a reachable trip on a variable it does not track. A second actuator, the
  recycle valve, exists only to keep the plant away from that trip. At 24 kPa
  the damping ratio of the Helmholtz oscillation falls from 0.62 at an 8 %
  margin to 0.1 on the surge line.
- **Effort.** 8.5 days.
- **Open points.** The gate under Recommendation, measured before anything else.
  TargetFoundation reaches the anti-surge problem only if it declares the
  distance to the surge-control line as an unscored output paired with the
  recycle valve. The generator has nothing like this plant; flag it if the
  generator gains a multivariable rung or a protection-loop structure.

### Family 3: open-loop unstable plants

#### `unstable_cstr`: holding the exothermic CSTR on its middle steady state

- **Model.** The shipped cstr's two balances and ten parameter values, imported
  from `pc_gym/cstr/env.py` so they cannot drift apart, plus a first-order
  jacket lag and a feed-temperature disturbance. The spec declares that file as
  an extra fingerprint source, so an edit there moves this task's version stamp
  and baseline as well as the shipped cstr's. The shipped `cstr` does not
  change.
- **Sources.** The values are the textbook set of Seborg, Edgar, Mellichamp and
  Doyle (Table 2.3), read in search snippets and in open reproductions, and they
  match PC-gym's code. Decardi-Nelson and Liu (2022, open preprint), read,
  control this reactor around the same unstable point with coolant between 285
  and 315 K, temperature bounds of 345 to 355 K and a 0.1 min step, and publish
  steady states on the unstable branch that the model reproduces. Bo et al.
  (2023) reuse that setup for safe RL. We derived that the middle steady state
  is a saddle, meaning a small disturbance dies out in one direction and grows
  in the other; at 300 K coolant its eigenvalues are -0.45 and +2.83 per minute.
  The agent that re-checked the sources reproduced this independently.
- **Size.** 11 state entries (3 ODE states, the disturbance, a six-level
  schedule and its clock), 4 observations (concentration, reactor and jacket
  temperatures, target), 1 action (coolant command, 290 to 310 K). dt 3 s, 1200
  steps (1 h).
- **Reference.** Six 10-minute blocks with concentration targets between 0.45
  and 0.65 mol/L, all on the unstable branch. The switch sizes are the same in
  every episode. Seeds differ in the starting level, drawn from 0.45 to 0.65
  mol/L, and in the order and direction of the switches. Only the live target is
  observed; later levels and switch times are not.
- **Disturbances.** `disturbance_fields = ("Ti_dev",)`: a zero-mean drift in
  feed temperature (2 K, 10 min correlation, ours), with `noise_fields =
  ("Ti_sigma",)`. It moves the coolant temperature that holds a target at steady
  state, so a controller needs integral action.
- **Trip.** Reactor at 365 K (ours). Falling cold is not a trip: the reactor
  drops to its extinguished branch, which tracking charges and which is always
  recoverable. Fresh restart, with the shipped cstr's 1 h restart time. The
  published 345 to 355 K band is a sourced alternative and much tighter. Its top
  edge is about 2 K above the hottest target, and its 345 K floor sits above the
  steady temperature of every target over about 0.59 mol/L, so adopting it would
  also shrink the targets to about 0.45 to 0.59 mol/L.
- **Experts.** A cascade: PI on concentration setting a rate-limited temperature
  setpoint, and PD on temperature setting the coolant. Without the rate limit,
  the drafted cascade tripped in 62 % of 302 episodes on the numpy copy of the
  plant, and with it in none. Relay tuning cannot run on an unstable loop, so
  tuning uses the coordinate-descent search. A CasADi NMPC over 30 steps, with a
  terminal cost and a bound kept under the point of no return, falls back to the
  PID when a solve fails.
- **What it adds.** Instability at every target (a small error grows at 1.4 to
  3.2 per minute) together with a moving reference, on a process plant. A switch
  can land the reactor 3.3 K below its point of no return. A hot excursion trips
  the reactor, while a cold one only drops it to the recoverable extinguished
  state.
- **Effort.** 5 days.
- **Open points.** The 6 s jacket lag is ours, and it sets how hard the hottest
  targets are to hold, so a sensitivity table on it comes before `PHYSICS.md`.
  The reactor is unstable because of the textbook heat capacity, 0.239 J/(g K),
  about a seventeenth of water's. That value is inherited from the textbook and
  recorded as a known deviation. The cstr is one of TargetFoundation's
  development plants, so the new task's held-out status needs a decision
  (question 8). When linearised at one target, the plant is one unstable pole
  behind a lag, which TargetFoundation's generator already draws. What the
  generator cannot make shows only once a controller slips: the runaway to the
  trip, the recoverable cold branch, and a steady-state gain opposite in sign to
  the short-term one.

#### `maglev`: a steel ball under an electromagnet

- **Model.** Voltage-driven single-axis magnetic suspension with three states:
  gap, velocity and coil current. The coil's inductance falls as the gap widens,
  and both the magnetic pull on the ball and the back-EMF (the voltage the
  moving ball induces in the coil) follow from that one inductance.
- **Sources.** The draft was sized on a textbook parameter set recalled from
  Khalil's Nonlinear Systems (2002), which the source search did not find, so
  every physical value is unverified. The 15 to 75 mm band, the rest at 0.1 m
  and the 10 ms step are ours, sized on that recalled plant, so they move with
  it. Two open rig documents with full parameter tables were read and are the
  candidates to replace it. The Quanser MAGLEV manuals give 14 mm of travel and
  an operating point near 6 to 7 mm. INTECO's MLS manual drives the coil through
  a current loop and uses an exponential force law, and Balko and Rosinova
  (2017) re-identified an INTECO rig independently. Either rig's unstable pole
  is three to four times the textbook one, so moving to a rig changes the gap
  band, dt and the episode length. Wong (1986): citation only.
- **Size.** 4 / 4 / 1. dt 10 ms, 1200 steps (12 s).
- **Reference.** Six gap levels of 2 s each, between 15 and 75 mm. Only the live
  level is observed.
- **Disturbances.** `disturbance_fields = ("f_dist",)`: a zero-mean force on the
  ball standing for air currents and vibration (ours), with `noise_fields =
  ("force_noise_sigma",)`.
- **Trip.** The ball touches the pole face or lands on its rest. Fresh restart,
  10 s restart time.
- **Experts.** A cascade PID (current inner loop, gap outer loop) with gains set
  by pole placement on the discrete plant. Relay tuning does not work here,
  because a relay test drops the ball or pulls it into the magnet before the
  loop can oscillate. A CasADi NMPC if a solve is fast enough, otherwise a
  gradient MPC.
- **What it adds.** Instability that no constant action survives, with a trip on
  both sides of the band.
- **Effort.** 4.5 days, before re-sizing on a rig parameter set.
- **Open points.** It is an electromechanical servo, so it does not fill the
  gap, which is about process plants. Linearised, it is the third-order unstable
  plant that TargetFoundation's generator already produces. It could still be
  built later as a cheap unstable task with well-understood physics, labelled in
  TargetFoundation as in-distribution because the generator already covers its
  structure.

### Family 4: medical dosing

#### `glucose_insulin`: an insulin pump in type 1 diabetes

- **Model.** The draft combined the Bergman minimal model with Hovorka's
  absorption chains. The source search then found a better base in the Cambridge
  type 1 model, as documented by Wilinska et al. (2010), read in full. It gives
  the complete equations (subcutaneous insulin, gut absorption, renal clearance,
  sensor lag), sampling distributions for every parameter that varies between
  patients, and a validation against a clinical study, so each episode's patient
  can be drawn from published distributions.
- **Sources.** Wilinska et al. (2010), read. Bergman, Phillips and Cobelli
  (1981), read, but its parameters come from non-diabetic subjects. Hovorka et
  al. (2004): citation only. Thresholds are read: 70 and 54 mg/dL (International
  Hypoglycaemia Study Group, 2017) and time in range 70 to 180 mg/dL (Battelino
  et al., 2019). Wilinska et al. also give the 15 g rescue dose used below. The
  UVA/Padova simulator is licensed and not used.
- **Size.** 17 / 5 / 1 on the draft, of which 7 are ODE states. The Cambridge
  model has 11 ODE states, so the count rises. dt 5 min, 864 steps (3 days, nine
  meals).
- **Reference.** One target glucose per episode, between 100 and 130 mg/dL.
  Unannounced meals are the disturbance; their times fall in daily windows the
  observed clock reveals, and their sizes are unknown.
- **Disturbances.** None declared, since meals are not zero-mean. Meal sizes
  scale with `noise_fields = ("meal_carb_scale",)`, zeroed in the planner's
  copy. Meals draw from a key held in the state, so they do not ratchet under a
  constant key.
- **Trip.** Glucose below 54 mg/dL. The patient keeps their state
  (`restart_in_place`, as hvac does) and is treated with 15 g of rescue
  carbohydrate inside the plant. Every step below 54 is charged three times the
  failure cost (`restart_steps = 3`, ours, from the 15-minute treat-and-recheck
  interval).
- **Experts.** A PID with insulin-on-board feedback, tuned by coordinate
  descent. A gradient MPC with a hinge barrier above the trip.
- **What it adds.** A hidden patient drawn per episode, which poses
  TargetFoundation's in-context adaptation problem on a physical plant. Insulin
  can only lower glucose, and a dose acts hours later with no way to take it
  back. The trip at 54 mg/dL is 46 to 76 mg/dL below the target, only a few
  times the tracking error the MPC is expected to hold (an estimated 10 to 30
  mg/dL, not yet measured).
- **Effort.** 7.5 days on the drafted model. The move to the Cambridge model is
  not costed and will add to it.
- **Open points.** Whether TargetGym should ship a dosing benchmark at all
  (question 9). If it does, the environment, its `PHYSICS.md`, its docs page and
  the README should say that it is a research simulator, not a medical device,
  and not to be used to choose or check doses. It brings neither instability nor
  a moving reference, so it should come after tasks that do. Flag it if the
  generator gains a one-sided actuator, a bilinear structure or a disturbance
  preview.

### Family 5: discrete or hybrid actuators

#### `supermarket_refrigeration`

- **Model.** The supermarket refrigeration benchmark of Larsen, Izadi-Zamanabadi
  and Wisniewski (2007): display cases on a shared suction header, on/off inlet
  valves, a staged compressor rack and limits on the goods' temperature. Each
  action is continuous in [-1, 1] and quantised inside the plant.
- **Sources.** Larsen et al. (2007): citation only. Sager's (2012) benchmark
  library of mixed-integer control problems, read, gives the complete equations,
  every parameter with units, and polynomial fits for the refrigerant's
  temperature, latent heat and density against suction pressure. Larsen's thesis
  is open too, but its parameter table is for an earlier version of the model.
  The draft was sized on recalled values.
- **Size.** 16 / 11 / 3 (rack and two valves). dt 10 s, 1440 steps (4 h).
- **Reference.** A suction pressure zone that rises at night on a fixed clock,
  and air temperature zones in the cases.
- **Disturbances.** `disturbance_fields = ("m_other_dev",)`: a zero-mean drift
  in the refrigerant flow drawn by the store's other cases (5 %, ours), with
  `noise_fields = ("other_load_sigma",)`.
- **Trip.** Goods above 8 C (a chilled-food limit recalled from regulations, not
  verified) or below -1 C (ours), or suction pressure at 2.2 bar, a guard on the
  range where the model's fits are trusted (ours). Fresh restart, 1 h.
- **Experts.** A neutral-zone PI on suction pressure with thermostats on the
  valves, and a sampling MPC guided by the PID.
- **What it adds.** Discrete actuators with interlocks, a hold that is only
  possible by switching, and compressor starts priced as wear.
- **Effort.** 9.5 days.
- **Open points.** On the recalled values the first draft's two-compressor rack
  needed 136 to 167 rack starts per hour to hold the night zone. That is far
  above the recalled 6 to 12 per compressor, and above the 10 per hour that
  Copeland's heat-pump compressor guideline recommends (read; its
  air-conditioning bulletin sets no fixed number, and no refrigeration-rack
  figure was read). The revised draft moves to four equal stages (ours), a 0.10
  bar neutral zone and a 90 s restart interlock that caps the rack at 40 starts
  per hour. A hand estimate puts the hold at 7 to 22 rack starts per hour, not
  yet measured. The task also has to be re-sized on the published benchmark
  before it can be judged. It covers little of the gap, and each of its loops is
  close to what TargetFoundation's generator makes once its quantiser is
  switched back on.

#### `staged_heating`

- **Model.** hvac's 5R1C building heated by a three-stage electric boiler with
  minimum dwell times and a manual-reset safety limit, through radiators
  throttled by thermostatic valves.
- **Sources.** The 5R1C constants hvac uses (3.45, 4.5 and 9.1) are read in the
  open ISO/FDIS 13790:2007 draft. Start limits and minimum off (anti-cycling)
  times are read in manufacturer guidance (Copeland, Vaillant); no source gives
  a fixed minimum on time. No published staged-heating benchmark was found;
  BOPTEST's heat pump modulates continuously. The boiler, its stages, its 180 s
  dwell and its safety limit are ours.
- **Size.** 8 / 8 / 1. dt 60 s, 1440 steps (1 day).
- **Reference.** A weather-compensated supply temperature with a night setback.
- **Disturbances.** hvac's `weather_dev`, with `noise_fields =
  ("T_out_noise_std",)`.
- **Trip.** Supply at 85 C, a manual-reset safety limit (ours). Restart in
  place: all stages drop and the plant keeps its state, and each step at or
  above 85 C is charged 15 times the failure cost (the 15 min reset time, used
  as a price).
- **Experts.** A PI with a stage positioner, and a sampling MPC.
- **What it adds.** An achievable floor set by the actuator's quantisation.
- **Effort.** 8.5 days.
- **Open points.** Not recommended as drafted. On the first draft's design a
  plain PI held within 1.1 times the quantisation floor with 21 K to spare below
  the trip, and following the supply-temperature curve drove the room to 22 to
  29 C, outside the building model's valid range. The revised draft adds the
  thermostatic valves and a heating design week to fix the second point; its
  numbers have not been run. TargetFoundation's generator also already includes
  a command quantiser. It reuses hvac's building too, so TargetFoundation would
  have to decide whether a variant of an existing simulator can count as held
  out (question 8). If family 5 is wanted, the refrigeration benchmark is the
  better base.

### Family 6: a high-dimensional distributed plant

#### `cd_paper_machine`

- **Model.** A separable cross-direction model. Cross direction means across the
  width of the sheet, and machine direction means along the way it travels. 24
  dilution actuators, valves spaced across the headbox that add water to thin
  the pulp where they sit, act on 48 scanner bins, the zones across the sheet
  that a traversing gauge reports on each pass. Every actuator has the same
  spatial response and the same first-order-plus-dead-time dynamics, and a
  machine-direction loop holds the sheet's average weight. The plant is
  stylised.
- **Sources.** The draft's values (24 actuators, 48 bins, a 20 s scan, the lags
  and the response width) are ours. Stewart, Gorinevsky and Dumont (2003) and
  Gorinevsky and Gheorghe (2003), open author copies, read, give parameter
  tables for real machines to move to, for example a 54-actuator slice lip (the
  adjustable lip of the headbox opening) with a 25 s scan. VanAntwerp,
  Featherstone and Braatz (2001), read, give a closed-form model of an
  industrial fine-paper machine.
- **Size.** 177 / 77 / 25. dt 20 s, 360 steps (2 h).
- **Reference.** A flat profile at a grade drawn per episode.
- **Disturbances.** `disturbance_fields = ("md_dev",)`, a scalar drift in the
  machine direction, with `noise_fields = ("md_sigma", "cd_sigma_smooth",
  "cd_sigma_streak", "cd_sigma_scan")`. The cross-direction disturbances are
  vectors, which the conformance test cannot check yet, so the plant's own tests
  cover them.
- **Trip.** A sheet break when the lightest bin falls below 75 % of target
  (ours). Fresh restart, 15 min.
- **Experts.** A PI on the machine direction and a regularised integral
  controller in actuator space, and a linear MPC solved as bounded least
  squares.
- **What it adds.** 25 actuators with an ill-conditioned, spatially structured
  gain. No current task has more than 3.
- **Effort.** 7.5 days. The draft also costed 1 day for the shared registry
  tier, which is counted once under Registering new tasks.
- **Open points.** The plant is linear and stable, its reference is static and
  its trip is far from any trajectory, so it has none of the gap's properties.
  TargetFoundation's decentralised methods can pair only the machine-direction
  loop, and that loop is already a typical draw from its generator. Worth
  building once TargetFoundation has a centralised multivariable method.

### Family 7: a periodic reference

#### `grid_inverter`

- **Model.** An averaged single-phase inverter with an LCL filter, on a grid
  with its own impedance and harmonics.
- **Sources.** The draft's rating (5 kW, 230 V, 50 Hz) and its filter values are
  ours. TI's reference design TIDUB21D, read, gives a complete single-phase LCL
  parameter set, a proportional-resonant controller with harmonic terms, and
  measured distortion, but for a smaller inverter (500 VA, 110 V, 60 Hz), so
  adopting it would change the reference frequency and every current level.
  Zhang, Wang and Blaabjerg (2015), read, give a few-kW, 230 V, 50 Hz set with
  controller gains, the closest to the draft. Teodorescu, Liserre and Rodriguez
  (2011) are read in the authors' companion slides. Liserre, Blaabjerg and
  Hansen (2005): citation only; its filter design rules are read as restated in
  open lecture slides and a master's thesis. IEEE 1547-2018 harmonic limits are
  read in a reproduction.
- **Size.** 7 / 6 / 1. dt 100 µs, 2400 steps (12 grid cycles).
- **Reference.** A 50 Hz current whose amplitude and phase step every two
  cycles. The live reference is observed; the block amplitudes, phases and
  switch times are not.
- **Disturbances.** `disturbance_fields = ("grid_dev",)`, a zero-mean drift in
  the grid voltage amplitude (ours), with `noise_fields = ("grid_dev_sigma",)`.
  Harmonics, frequency and grid impedance are drawn per episode and hidden.
- **Trip.** Overcurrent at twice rated peak (61.5 A on the drafted rating), or
  600 V at the filter node (both ours). Fresh restart, 5 s.
- **Experts.** Proportional-resonant control with active damping, and a linear
  MPC.
- **What it adds.** A sinusoidal reference outside the aircraft, a disturbance
  at several harmonics, and a hidden grid impedance that moves a lightly damped
  resonance.
- **Effort.** 13 days, the most of any candidate.
- **Open points.** The plant is linear within an episode and stable, so it does
  little for the gap. In the first draft the policy set the raw modulation, and
  TargetFoundation's methods, which get no grid-voltage feedforward, would all
  have tripped and scored the same. The revised draft puts that feedforward
  inside the plant and has the policy command a correction on top of it; that is
  our modelling choice and needs approval. The generator has no sinusoidal
  reference; flag it if its planned sine term is ever applied to targets.

## What TargetFoundation will need

The items below are TargetFoundation work, outside this proposal, and belong in
a separate task there.

- Every lookup keyed on `REGISTRY` moves to `registry.get` before
  TargetFoundation can run a new task, because `REGISTRY` keeps only the 21. In
  `scripts/benchmark_all.py` the task is read with `REGISTRY[task]` and its
  episode key comes from `list(REGISTRY).index(task)`, so a new task passed with
  `--tasks` gives an error row for every policy; its key should come from the
  task's fixed seed number, as in TargetGym's `benchmark.py`, which gives
  today's values for the 21. `src/tmdp/stage0.py` reads `REGISTRY` the same way,
  `src/tmdp/plants.py` raises `KeyError` for a new task, and the shipped-PID
  direction lookup in `src/tmdp/policies.py` silently finds nothing.
- Rows in `src/tmdp/policies.py` for each new task: declared outputs, loop
  pairings, and physics directions wherever the step test cannot read one. That
  includes every unstable plant, whose step response diverges or trips.
- Entries for each new task in `tests/test_policies.py`, and a row in
  `docs/benchmark_declarations.md`.
- **D12**, TargetFoundation's decision to keep TargetGym's simulators out of
  pretraining.
  - Status: recorded as decided on 2026-09-28 in the committed
    `docs/general_generator.md` (the uncommitted
    `docs/foundation_world_model.md` says "TargetGym is held out"), but still a
    recommendation in the decisions table of `docs/budget_axis_plan.md`.
  - Wording: it names "TargetGym's 21 simulators", so it needs widening to cover
    new tasks.
  - `unstable_cstr`: either keep TargetFoundation's development on `cstr` inside
    the shipped regime (coolant at most 302 K) and hold the unstable regime out,
    or label the task a transfer test on a development simulator.
- The 12/9 split in `scripts/benchmark_report.py` finds each task's sub-family
  with `REGISTRY.get`. Under the tier field `REGISTRY` still holds only the 21,
  so new tasks would silently drop out of its sub-family table. If it moves to a
  lookup that sees every tier, it sends every non-aircraft task to "process and
  energy", and the new tasks would be averaged with the easy regulation tasks
  unless they get their own sub-family.
- Trips reported beside the version-1 score on every new task.
- Generator guards while these tasks are held out, so that they do not leak into
  pretraining. The generator should not gain an unstable plant with gain
  scheduling, unannounced setpoint switches and a one-sided trip tuned to match
  `unstable_cstr`, or a 2x2 unstable plant with switching references tuned to
  match `grade_transition`.
- A golden-test rehash when the pin moves, and for each batch of `policies.py`
  declarations.

## Found along the way

These issues are outside this proposal, and each should get its own fix.

- **Fingerprint gaps.** A task's fingerprints hash only its own package and the
  gains keys that start with its name, so they miss these:
  - `plane3d` imports `plane.dynamics`.
  - `patrol` imports `plane.dynamics`, `plane3d.env` and `plane3d.dynamics`.
  - The patrol lead aircraft, in both `patrol` and `patrol_bearing_only`, flies
    a PID from `experts/pid.py` on the `plane3d_heading` gains. Neither task's
    version stamp covers the PID or the gains, and neither task's baseline
    fingerprint collects them, though both oracles also fly that autopilot
    (`env._lead_pid_params`). (Baseline side closed by the shared oracle
    machinery change, 2026-10: both tasks declare
    `gains_keys=("plane3d_heading",)`, so their baseline fingerprints now hash
    that entry. Their version stamps still do not cover the PID or the gains.)
  - `patrol_bearing_only` now has a recorded baseline and a protocol row, and
    its fingerprints also miss the `patrol` gains, which its PID reads through
    `make_patrol_pid` (`experts/pid.py`). Retuning those gains would stale
    `patrol`'s records and leave `patrol_bearing_only`'s passing as fresh.
    (Closed by the same change: the baseline fingerprint now also hashes the
    `tuned_gains_key` entry, which is `patrol` for `patrol_bearing_only`, so a
    retune of `patrol` stales both tasks' records.)
  - `plane_energy` and `plane_sine` fly on the `plane_cascaded` gains, which
    their baseline fingerprints do not collect, so retuning those gains would
    leave their recorded baselines passing as fresh. (Closed in the oracle
    audit, 2026-10: their oracles read copies under their own names,
    `plane_sine_cascaded` and `plane_energy_cascaded`, and a test holds the
    copies equal to `plane_cascaded`, so a retune has to update them, which
    stales both tasks' records. The shared oracle machinery change, 2026-10,
    also declares `plane_cascaded` in both tasks' `gains_keys`, so their
    baseline fingerprints hash it directly.)

  The open roadmap item on scoping the baseline fingerprint is the natural place
  to fix these.

- **The shipped cstr's `PHYSICS.md`.** It says PC-gym cites no source for the
  model; the PC-Gym paper cites Seborg et al. and Ingham et al. It gives the
  multiplicity window as 299 to 304 K, where a direct solve gives 298.1 to 303.2
  K. Its steady-state table lists an ignited state at 305 K coolant without
  saying that it is unstable. By our derivation the reactor spirals away from it
  into a sustained oscillation, and the state becomes stable only above about
  306.2 K coolant, where the spiral switches from growing to decaying (a Hopf
  bifurcation).
- **hvac's sources.** Its 5R1C constants can now be cited to clause and page in
  the open ISO/FDIS 13790:2007 draft. Its `PHYSICS.md` still gives old episode
  lengths, 7 days (672 steps) in one section and 3 days in another; the code
  runs 720 steps (7.5 days).
- **Documentation drift.** The check-8 table in the model review checklist gives
  step counts as seconds for the glass furnace and distillation. `docs/index.md`
  says 19 of 21 tasks ship an MPC; the registry has 20. `docs/complexity.md`
  gives the wind turbine 6 observations; the code has 5. `docs/target-mdp.md`
  lists `plane_sine` at 800 m (its test episode uses 300 m), and it lists the
  glass furnace's five-setpoint schedule and hvac's daily setback as static
  references. `CITATION.cff` says eighteen environments. The model review
  checklist says each automated check carries an allowlist, but check 7 has
  none.

## Questions

1. Which two tasks first? Recommended: `unstable_cstr`, then `grade_transition`.
   Keeping to families 1 and 2: `grade_transition` and `compressor_surge`, with
   the compressor's gate check first.
2. Registry: the tier field as described? If so, may the mechanism branch start
   from `feature/run-policy-on-benchmark`?
3. The tier field adds TargetGym files to the rehash that TargetFoundation
   already has to do for its own `policies.py` rows when it opts in. Accept
   that, or keep core TargetGym files unchanged with the separate-registry
   design and its duplicated benchmark code?
4. Group names: by control challenge (`process_unstable`, `process_trajectory`)
   or by domain (`biomedical`, `electromechanical`)?
5. Version-1 rewards written for each new task and declared as ours, with trips
   reported beside the version-1 score on TargetFoundation's side?
6. Version stamp at birth: `-v2` (recommended) or `-v1`?
7. Check 7: an allowlist entry for open-loop unstable plants with the measured
   growth rate, or a check 7 that ignores the steps where a plant trips or
   restarts, for the whole suite?
8. D12: does it cover new tasks, and for `unstable_cstr`, is the unstable regime
   of the cstr simulator held out too?
9. Should TargetGym ship a dosing benchmark at all?
10. For `unstable_cstr`: the 365 K trip (ours), or the published 345 to 355 K
    band, which would also narrow the targets to about 0.45 to 0.59 mol/L? Can
    you supply a jacket time constant, or accept 6 s as ours with a sensitivity
    table?
11. For `grade_transition`: keep the 10-minute analyser delay, which is closer
    to industry and needs the `eval.py` change, or observe NAMW directly?
12. Should TargetGym's tooling default to every tier? Proposed: extended
    conformance as its own CI job that still blocks merges, and baseline
    recording of the core tier unless asked for more.
13. Keep the full drafts, scripts and source checks in the repository (for
    example under `docs/proposals/new_families/`, marked as superseded where
    this page differs), or let this page stand alone and specify each chosen
    task again from it and its sources?
14. Every restart time on this page is ours; no source gives one. Is there a
    plant engineer's figure for any of them?
15. May an extended task later be promoted to core, and may the core ever grow
    past 21? With fixed seed numbers neither would move any episode, but either
    would change `REGISTRY`, and with it TargetFoundation's default benchmark,
    the README counts and D12's wording.
16. For `grade_transition`: record a second PID without grade feedforward, as a
    second registered spec or an optional baseline slot on `EnvSpec`, at the
    cost of one more recording? Or only say in its baselines note that the
    shipped PID uses feedforward?
