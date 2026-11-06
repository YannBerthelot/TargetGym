# Industrial fixes before 0.8.0

A proposal written on 2026-11-05, after an independent industrial review of the
tasks, the experts and the metrics. It changes the order of work for 0.8.0.

## Status

Decided on 2026-11-05, with the choices listed under [Decided](#decided). The
0.8.0 re-record is held until the plant fixes land. No fix has started.

## Why

The review rated 11 of the 24 tasks credible with caveats, 4 fit for a narrow
use, and 9 not credible as shipped. Three causes run through most of it:

1. **Measurement and actuation are close to instant.** With a 5 minute analyser
   delay, the `distillation` expert's hold goes from 0.155 to 63,096, with 137
   trips in 6 episodes. Noise alone is a smaller effect. It raises process holds
   but trips nothing (r), and moves the industrial and energy holds by about 1%,
   12% on glass. The aircraft figures are not usable yet (see [Risks](#risks)).
2. **Some tasks have no control problem.** An echo of the target beats the
   `battery` expert on 10 of 10 seeds. One rod integrator holds `reactor` at
   1.486 against the full-state oracle's 1.473. Leaving the heater off beats the
   `hvac` expert on 6 of 10 seeds.
3. **The scoring can miss what an owner cares about.** The scored output has no
   trip count, and a run that trips on 2 of 10 seeds keeps its ratio. There is
   no spec band and no actuator wear figure, and fuel savings cannot show.

The results above were reproduced on the real scorer. Numbers marked (r) below
come from a reviewer's run and have not been re-checked. Each gets a failing
test first, and a claim that does not reproduce is dropped.

The order matters. Plants feed the oracle horizons, the floors, the settles and
every expert protocol. Fixing them after the re-record would mean recording
everything twice, and the expensive oracles (C2 and D) are not built yet, so
this is the cheapest moment to change the plants.

## What does not change

- Expert parameters come from a practitioner procedure, run as a fixed recipe.
- Oracles follow the oracle rule.
- Held-out seeds only report. Tests follow TESTING.md: test first, and a miss is
  a finding, never a loosened bound.
- The ceiling stays the instrument resolution, as chosen on 2026-10-27.
- `seed_scores` stays the study's score.

## What "credible" will mean

Three checks, fixed now, before any result is seen. A task passes all three, or
its `PHYSICS.md` carries a written exception the owner accepted.

1. **No null policy beats the expert.** No null policy wins 6 or more of 10
   held-out seeds with `ratio` above 1. The null policies are:
   - constant action at the middle of the range, and at each bound;
   - target echo, where the action shares the target's units;
   - a static table `u = f(target)`, built on training seeds;
   - that table with an I-only trim, tuned by SIMC where the shared bump test
     has a valid fit;
   - any policy a review found to beat the expert, kept as a named witness. It
     needs no tuning rule, because a win proves the defect however it was tuned.
2. **The expert's result looks like the real thing.** Its hold is within a factor
   of 2 of a sourced real-world figure, or the source says why it may not be.
3. **The oracle's win is operable.** Every limit a real plant enforces is a trip
   or a hard bound, so an oracle cannot win by breaking one.

Check 1 runs as a large test, one strict xfail per failing task, naming the
finding. Each fix removes its own xfail.

The owner has decided what these checks decide: a task enters the benchmark only
if it passes all three or has a written exception the owner accepted. The harness
reads the list from PR 8's credibility record at the freeze.

## Order of work

PRs 3 to 9 target `feat/sensor-noise` from 509fd5b, the integration branch,
because the noise work stales all 24 stamps and has to land at one re-record.
That branch is local only, and its version stamps are branch stamps, to be redone
at the end.

Pull requests into the integration branch run lint and docs only, and merge by
hand, so resolving a conflict costs a minute of CI and sessions rerun nothing. The
full run, the large tier and the null-policy gate included, is one draft pull
request from that branch into `main`, which never merges by hand and runs after each
batch. A new push cancels the last run, so batches are the unit of checking.

`records` stays red there until the re-record. These exceptions in `python-app.yml`
come out when the branch lands: tests run past a failing `records`, the small tier
uses 2 workers, its timeout is 60 minutes, compiled programs are freed after each
module, memory and disk are logged, and two examples in `docs/using-the-pids.md` that
compare with stale records are deselected. The large tier takes the same
workers, freeing and logging, with a 120 minute timeout.

| PR | What | Target | Budget (code + tests) |
|---|---|---|---|
| 1 | Scoring: trips, travel, true consumption, spec band | `main`, stales nothing | 150 + 120 |
| 2 | Null-policy gate, red on the known failures | `main`, tests only | 120 + 60 |
| 3 | Sensor dead time, true sampling, actuator rate and lag; `distillation` pilot | integration | 200 + 200 |
| 3b | The expert interface, its own session | integration | its own |
| 4 | Process plants | integration | 250 + 240 |
| 5 | Thermal plants | integration | 300 + 250 |
| 6 | Energy plants, `battery` and `reactor` rebuilt | integration | 600 + 400 |
| 7 | Aircraft plants, with the fixed inner loop | integration | 500 + 400 |
| 7b | The patrol family | integration | 150 + 120 |
| 8 | Scoring on the new plants: hold window, consumption, constants | integration | 100 + 80 |
| 9 | Oracles C2 and D, then expert protocols E and F, as amended below | integration | as in the build plan |
| 10 | One re-record, then G and the release gate | `main` | none |

Then the benchmark rechecks its task list and opens its GPU window.

The owner asked for parallel sessions, one per plant group, plus one for the expert
machinery of PR 9, which does not depend on the plants. PRs 4 to 7 run that way. PR 3b
needs PR 3's applied action, and PRs 6, 7 and 9 rework experts on top of it. PR 7b
follows PR 7, whose dynamics it shares. C2 and D hold until their plants are done.

## Scoring

### PR 1: can land now

`scoring.py` and `eval.py` are not fingerprinted, and these add numbers without
moving a recorded one.

- **Verdict on the whole episode.** `ratio` drops the burn-in today, which is
  half the episode on 12 tasks, so 10 of 10 trips inside it are invisible. #140
  documented that as a long-run verdict. This changes it to the whole episode,
  as `seed_scores` reads it.
- **Trips in the output.** `compare` and `score_policy` return `trips` per
  episode for both sides.
- **A win is gated.** `beats_expert` needs `ratio` more than 1% above 1, the gap
  two recordings of one row can show, and no more trips than the expert. The IQM
  stays, because it answers a different question.
- **Actuator figures.** `Episode` gains `actions`, and the report gains `travel`
  and `reversals` for both sides.
- **True consumption.** `consumption_ratio` reads the physical quantity, not the
  clipped term, so a saving shows.
- **Spec band.** A `SPEC_BAND` table in `scoring.py` gives each side's off-spec
  share of the hold. It uses a tolerance the repository already cites, and ten
  floors elsewhere, labelled as ours. Each plant PR replaces its rule with a
  sourced number. This is a report only. It stays out of the reward, because bands
  hid real errors before.
- **`s_max`.** The best score the task's known bound allows, reported beside `S`,
  so the radar stops implying 1 is reachable everywhere. Cross-task claims use
  `seed_scores`.
- **`first_order` leaves the aggregates.** It stays as a sanity fixture.
- **Expert rows on 10 seeds,** from the next re-record. Three seeds miss the
  `ph_neutralization` cycling on seeds 3 to 5. Oracle rows stay on 3 because of
  their cost.
- **Stale text.** `docs/reward-shaping.md:28` still points to NEA.

### PR 8: moves recorded numbers

- **One hold window for all.** `hold_mask` removes a per-run anticipation today,
  so a policy's rising excursion leaves the hold. Replace it with a fixed count of
  steps before each move, per task.
- **Linear consumption.** `running_cost` becomes `w * c / c_hold`. The clip gives
  no credit for saving, and it charges fluctuation that a fuel bill does not. This
  adds about `w` to every step's cost on the 6 tasks it covers, which compresses
  `seed_scores` there. That needs the owner's decision before PR 8.
- **Constants, once.** See [Constants](#constants).
- **Null policies in the package.** They move out of `tests/`, with their tables
  and gains stored in records, because the bump-test evidence is outside the wheel.
  ExpertBench runs them as reference rows through `run_policy`.
- **A credibility record per task,** through `records`: each null's result, the
  expert's sourced figure and any accepted exception.
- **Settle counts that see ramps.** `FIXED_SETTLE["reactor"]` is 1, but a ramp is one
  move lasting 20 steps, so it needs at least 20. `wind_turbine`'s is 0, from when its
  target never moved; the expert settles in about 36 steps and the oracle in about 48.
- **Ceilings from the rebuilt prices.** `ceiling("reactor")` must be per control step,
  $0.0958 at the 0.05 K floor, and returns 0.0032 today. `tests/test_scoring.py` reads
  one sub-step, which is what hid this: it must read the step, and lose its strict
  xfail. `wind_turbine`'s floor note says 1 kW where the
  branch's resolution is 1.7 kW, which moves its ceiling from 6.94e-6 to 1.18e-5.

## Shared plant stages (PR 3)

This reverses noise decisions 1 and 5, and PR 8 reverses decision 7. The PR also
updates `docs/proposals/measurement-noise.md`, which still states them.

- **Dead time and sampling.** `Reading` gains `dead_time`, and `interval` holds
  the value as well as its noise. `get_obs` must stay a pure function of the
  state, so the state carries a fixed-length history of each delayed reading.
- **Rate and lag.** `actuate` gains `actuator_rate` and `actuator_lag`, on in the
  scored tasks. These are plant physics with sourced numbers.
- **Stiction and dead band stay opt-in.** No source gives a typical stiction
  range, so noise decision 1 stands for these.
- **Applied action.** `info["applied_action"]` reports what the actuator did, and
  `Episode.actions` reads it, so `travel` and `reversals` measure the actuator, not
  the command.
- **What scale 0 means.** Dead time, sampling and rate limits stay on at
  `sensor_noise_scale` 0, so it stops meaning the ideal plant. `measured_reward`
  reads the same delayed copy as the observation.

## Plants

Row order is by severity within each PR.

Each plant PR also sources its timing and actuator figures, because the benchmark's
readiness gates ask for it: dead times, sampling intervals, rate limits and lags. A
figure that cannot be sourced is labelled for the owner to accept. The tracked
readings the noise proposal left Provisional need no sourcing: they pass as labelled
under its decision 4, so each `PHYSICS.md` must keep their status.

### PR 4: process

| Task | Change |
|---|---|
| `distillation` | Analyser 5 min dead time and 5 min sample, tray temperature stays fast; lag on `L`; `V_max` 3.9 and `L_max` 3.4, or trim the target box, since 4 to 12% of it is unreachable (r); correct the RGA table to 35.9, 145 and 1.960 (r) |
| `ph_neutralization` | Probe 10 s lag plus 10 s dead time |
| `four_tank` | Unmeasured load steps, because a table with no feedback beats the expert on 9 of 10 seeds (r); relabel the numbers, which are not the rig's |
| `cstr` | Analyser 1 min dead time; unmeasured load steps; narrower target band; remove the runaway claim from its docs, since it cannot trip (r) |
| `unstable_cstr` | `T` lag of 10 to 20 s; `T_c` rate limit and valve lag; ramp the target; label the jacket as synthetic |

### PR 5: thermal

| Task | Change |
|---|---|
| `cement_kiln` | Free lime sampled hourly and 60 min old; hidden drift on burnability and coal heating value; steady feed, or make feed an action; longer episode; check the clinkering heat sign first (r). An exact calcination step lands here from its own session. If drift scales `A_clinker` by more than about 1.2, the kiln also needs `rk4_2`, at twice the step cost |
| `hvac` | `solar_peak` 175 or a cloud process, so that heating is needed |
| `boiler_drum` | Firing lag and rate limit, then re-identify; steam draw follows pressure |
| `glass_furnace` | Fuel slew limit; settled reset; observe pull; hide gas heating value or cullet |

### PR 6: energy

| Task | Change |
|---|---|
| `battery` | Rebuild: a regulation signal over a 4 h, energy-limited window, 2880 steps at 5 s, so state of charge is the problem; next block in the observation |
| `reactor` | Rebuild: steam generator node so turbine load drives `T_cold`; tracks Tavg against Tref(load), not neutron power, which the turbine now sets, with a kelvin price labelled ours; 100-50-100 ramps at 5%/min over 12 h at 30 s steps, 1440 steps, because 60 s breaks the rod program; correct rod speeds |
| `wind_turbine` | Published NREL 5 MW `Cp` surface; torque rate limit; setpoint moves; a rotor-speed cost |
| `compressor_surge` | Ramp setpoints; per-episode surge-line offset, so the margin is not known exactly |

Also in PR 6: charge actual travel on `reactor` and `wind_turbine`, whose wear
terms charge command lag today, and have the `battery` expert return `(n, 1)`
where it returns `(n,)` today.

### PR 7: aircraft

| Change | Tasks |
|---|---|
| Gust rolling moment | all 3D |
| Trips on load factor, EAS, Mach and bank; `CL` capped at a buffet line | all |
| Engine lag about 3 s, 5% idle; retire the patches for the old lag | all |
| Policies, experts and oracles all send 1 Hz bank and pitch demands, through a fixed inner loop in the plant, so they share an action space. Its gains come once from the accepted attitude procedure and stay task constants | all |
| Speed schedule in CAS and Mach | `plane` family |
| Tips at 25 deg at flown speed, `a` about 35 km at 230 m/s | `plane3d_figure8` |

### PR 7b: patrol family

| Change | Tasks |
|---|---|
| Gust delayed by the slot distance over airspeed, with a decorrelated part | `patrol`, `patrol_bearing_only`, `formation` |
| Slot separation about 40 m, since 25 m is under the 34 to 35.8 m span | same |
| Turn rates to +-0.03 rad/s, from 0.003 rad/s, which is 4 deg of bank | same |
| The 3D expert as lead, in place of the rejected autopilot | same |
| Rename, since exact range is not passive. Recommended: keep range, and drop "bearing only" from the name | `patrol_bearing_only` |

Wake is left out unless a usable model turns up. Say so in `PHYSICS.md`.

## Expert protocols, amended

C2 is landing on `main` as #141, ahead of these plants. Its solver code carries over,
and its six oracle rows re-record after the plant PRs. `distillation`'s expert loses
the column to the analyser delay from PR 3 until PR 9 retunes it.

Oracles on rate-limited plants plan through `actuate`. Its slope at a held command is
fixed on the integration branch, but the clip still has none past the limit, so they
plan by rate. On `glass_furnace` seed 2 both together reach 0.29 K, against 8.84 K as
shipped. The aircraft surface rate limit and attitude clip have the same gap.

PR 9 also applies `docs/proposals/expert-protocols-noise.md`, which removes the
twin-run bump test and makes bump tests single runs sized by a measured noise
band. N8 there must allow for dead time and true sampling. PR 9 reruns the noise
measurement on the changed plants first, and keys the gains store by task and
noise level, with today's entries as level 1. `make_expert(name,
sensor_noise_scale=level)` and `make_pid_controller` take the level, a level with
no gains raises, and no level means the shipped gains.

| Task | Amendment |
|---|---|
| `ph_neutralization` | Bump near `q2 = 0`, since the accepted recipe still totals 225,945 on seed 4 (r) |
| `distillation` | Tight temperature loop, ramped `T_sp`, composition tuned on the analyser delay; today's cascade is inverted (r) |
| `glass_furnace`, `boiler_drum` | Run only after the slew and firing limits exist |
| `compressor_surge` | Add a schedule check: a 10% line gave no trips, but margin fell to 3 to 7% (r) |
| `hvac` | Outdoor reset plus thermostat, since Guideline 36 targets commercial VAV |
| `cement_kiln` | Keep the rule base; report the 4 times cheaper cascade as a second bar (r) |
| `battery`, `reactor`, `wind_turbine` | Write afresh on the rebuilt plants |
| Aircraft | Run after PR 7 |

## Constants

Floors, `c_hold`, `rho_floor` and the fixed settle change once, in PR 8, as a
task change under CONTRIBUTING's rule, and freeze again. Instrument floors stay
where the instrument did not change.

Where a floor is finer than the instrument scatter, which the noise proposal
lists for `cstr`, `distillation`, `boiler_drum` and `glass_furnace`, it moves to
the instrument's figure.

## Deferred past 0.8.0

- Reward spec bands. `SPEC_BAND` stays a report for now.
- Pricing consumption from real tariffs on the dimensionless plants.

## Decided

Chosen by the owner on 2026-11-05.

1. **Noise decisions 1, 5 and 7 are reversed** for dead time, true sampling, rate
   limits and floors. Stiction and dead band stay opt-in.
2. **All 24 tasks get their own fixes before 0.8.0,** the five outside the
   benchmark included.
3. **The aircraft move to a fixed inner loop.** Policies send 1 Hz bank and pitch
   demands, and no longer touch the surfaces. The alternatives were a 4 Hz step,
   at four times the steps and oracle cost, or keeping 1 Hz with a label.
4. **`battery` and `reactor` are both rebuilt.**

Chosen by the owner on 2026-11-06.

5. **The aircraft engine lag is 1.7 s,** which meets 14 CFR 33.73(b)'s 5 s from idle
   to 95%, in place of 3 s, which takes 8.8 s.
6. **These stand as working values, labelled ours,** and can still change before the
   re-record: pH lag 10 s and dead time 10 s, coolant valve lag 2 s and rate 8 %/s,
   boiler firing lag 20 s, glass fuel slew 1 % per min, aircraft surface rate 30 deg/s
   and the buffet line.
7. **`unstable_cstr`'s `tau_j` of 6 s is a written exception under check 2.** It is
   synthetic, and its `PHYSICS.md` says so.

## Cost

New work, on top of what remains of the build plan (C2 to G). Ranges are rough.

| Part | Tokens | Model time | Local or CI runtime |
|---|---|---|---|
| PRs 1 and 2 | 120-220k | 1-2 h | Under 1 h |
| PRs 3 to 8 | 1.7-3.0M | 14-26 h | 4-8 h of tests and checks |
| The one re-record | Small | Small | About 6 h on CI, as today, more with long `reactor` runs |

Aircraft oracle recording stays the largest compute item, at 1.5 to 3.5 h per
task on CI. The GPU window moves back by however long this takes.

Edits, lint and targeted tests run on the Mac. Anything over about 10 minutes
runs on bob, such as the large tier and horizon measurements. The re-record
itself stays on CI.

## Risks

- **Experts may fail on the new plants.** That is the finding we want, but it can
  push tasks out of the benchmark, so its list is rechecked after the freeze.
- **The rebuilds are real modelling work.** `battery` and `reactor` need new
  sourced numbers and may surface their own errors.
- **Long `reactor` episodes** cost more per episode and leave fewer episodes in
  the benchmark's 1M steps.
- **The inner loop moves into the plant,** so aircraft policies no longer touch
  the surfaces. Say so in the docs.
- **ExpertBench depends on some of this.** Its `docs/targetgym-contract.md` lists
  what it calls. Tell it before `tg.make`'s noise override, the `info` keys,
  `seed_scores`, `ceiling` or `make_pid_controller().batched()` change, flag any
  `max_steps_in_episode` that moves by more than about 2 times, and any change of
  step such as `reactor`'s, and send it the freeze commit. It records "actuator
  defects are opt-in"; rate limits are plant physics, not defects, but its decisions
  page should be updated.
- **Reviewer-only claims may not hold.** Each gets a failing test first.
- **Earlier aircraft noise numbers are unreliable.** Until 16df296, seven plane
  and plane3d tasks gave every episode one noise path after step 1, so noise-on
  costs from before that commit do not hold. The large tier has not been rerun on
  those tasks since.
- **The integration branch has two known failures.** Two examples in
  `docs/using-the-pids.md` compare against recorded `cstr` returns and stay red
  until the re-record.
