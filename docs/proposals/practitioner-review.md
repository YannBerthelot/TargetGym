# TargetGym from a practitioner's chair

An independent review of the 0.8.0 candidate, `origin/int/amendment3` at
`dceb2537` (2026-11-27), written for the owner. It judges the tasks the way a
control engineer or an RL researcher choosing a benchmark would: are they
faithful to the plants they stand for, what do they add over existing
benchmarks, and would anyone trust a result on them. The project's charter, its
gates and its rules were read only to understand how a task is built and
scored; they are not the yardstick here.

## Verdict

| | Grade |
|---|---|
| **Overall, as a practitioner would use it today** | **5.5 / 10** |
| Fidelity to the real plants | 6.5 / 10 |
| Novelty over existing benchmarks | 7 / 10 |
| Utility for an RL paper or a controller study | 4 / 10 |

The plants are the library's strength. They are reduced-order models of real
equipment: a PWR core with point kinetics and a two-point xenon model, a
Greitzer compressor, the Henson and Seborg pH chemistry, a pilot-plant CSTR and
an ISO 13790 zone on a concrete slab. Each is checked against published
figures, with realistic actuator limits, dead times, sensor lags and operating
limits taken from technical specifications, permits and vendor practice. Three
of the five scored tasks (PWR load follow, anti-surge control, permit-bound
neutralisation with a diversion basin) have no open equivalent in this form
that I could find, and everything runs at millions of steps per second under
JAX.

What stops a practitioner from using it today is the measuring stick, not the
plants. No scored task has a current reference, MPC or ceiling on the
candidate. Each scored plant's code took 14 to 39 commits this month. The public
scoring function disagrees with the task pages. The "room" above the reference
mixes what a controller could achieve with what only a clairvoyant could, and
no line shows the reachable part. One reference loses to a two-setting retune,
and there are no learned-policy results at all. All of this can be fixed in
weeks; the ranked list below says how.

What already works for a user, and should be kept as it is:

- The gymnax interface, with `jit`, `vmap` and `scan` throughout, a Gymnasium
  wrapper and a continuing-plant mode. In my runs a whole expert episode
  compiled and ran under `lax.scan` in 1 to 6 s; for the scored five the
  project's throughput table gives 1.2 M (reactor) to 200 M (cstr) steps a
  second on a CPU.
- Costs in money or product, operating limits that trip and restart the plant,
  sensor noise, lags and dead times, and hidden state, all on one contract.
- Physics contracts that cite every number or label it, with validation tables
  asserted by tests. Few benchmarks are this transparent about their models.
- A `pip install` with no simulator to build and no server to run.

## Grades per task

### The scored five

| Task | Fidelity | Novelty | Utility | Overall | In one line |
|---|---|---|---|---|---|
| `reactor` | 7.5 | 9 | 5 | **7** | The best task: a credible load-follow core with real limits, held back by missing upper lines and a noisy headline |
| `compressor_surge` | 6.5 | 8 | 4 | **6** | A new and well-validated surge plant whose reference limit-cycles and loses to a two-setting retune |
| `ph_neutralization` | 6.5 | 5 | 6 | **6** | A classic chemistry in a credible permit frame, with a sound reference and a room that is mostly set-point placement |
| `hvac` | 5.5 | 4 | 5 | **5** | A fast TABS surrogate of what BOPTEST already offers, on synthetic weather |
| `cstr` | 6 | 3 | 3.5 | **4** | The most worn benchmark plant, with a three-day-old job, no oracle, and upsets only foresight can hold |

What lies behind each line:

- **`reactor`, 7 / 10.** The best task in the library. Point kinetics with six
  delayed groups, an Onega and Kisner two-point xenon core whose axial mode
  period (32.7 h) sits inside published models of Ringhals-4 (31.5 to 39.3 h;
  the plant measured 28.3 h), boron that reaches the core late, a
  turbine bounded by Stodola's law and the C-16 stop, and Westinghouse CAOC,
  rod-insertion and DNB limits taken from NUREG-1431. I found no open RL
  benchmark for PWR load follow with axial offset. It loses points for having
  no current MPC or ceiling, for a headline that five load-follow days at 2.2 %
  weight each carry almost entirely, for a 15 times spread between seeds that
  tracks the drawn cycle point, and for an 855-line contract. Its expert
  tripped nothing in my 60 episodes.
- **`compressor_surge`, 6 / 10.** A Greitzer model checked against Gravdahl's
  deep-surge orbit, the anti-surge architecture vendors use, a 0.15 s valve
  dead time, stroke limits, transmitter lags and a hidden ±5 % shift of the
  surge line. I found no open anti-surge RL benchmark. It is held back by a
  reference that limit-cycles (documented by the project) and that a
  two-setting retune beats by 10 % in my runs, by an MPC line that tripped 19
  of 20 approach episodes in its last record, by 120 s windows, and by prices
  that are all the project's own.
- **`ph_neutralization`, 6 / 10.** The Henson and Seborg chemistry reproduces
  their published operating point (pH 7.03 at 15.6 mL/s of base in my check,
  against 7.0), set in a credible industrial frame: two stages, a pH 6 to 9
  permit, a 60 min excursion rule and a diversion basin. The expert is what a
  practitioner would build (a PI linearised through the titration curve, with
  flow feedforward and a trim stage) and it is well placed: moving its set
  point from 6.45 to 6.30 raised its cost by 83 % in my run. The room is mostly
  set-point placement under a hidden buffer, worth about 10 % of reagent. The
  streams are bench scale, and the valves and probes are ideal.
- **`hvac`, 5 / 10.** A 5R1C zone with a TABS slab, a heat pump at a fixed
  share of Carnot and a day-night tariff. The slab's time constants pass sourced
  tests. But the weather is synthetic, its forecast carries no error, the plant
  heats only, and BOPTEST's `bestest_hydronic_heat_pump` already offers a
  validated single-zone heat pump with floor heating and price scenarios. Its
  edge over BOPTEST is speed. Its room is small (about 1.4 times in the last
  record), and its MPC line cost more than its reference there.
- **`cstr`, 4 / 10.** The most used benchmark plant in process control, here
  with PARSEX's numbers and a production objective, which is a sensible twist.
  But version 8 changed the job on 2026-11-24, ships no oracle or MPC, prices a
  1 K band of the project's choosing as all or nothing, and leaves its three
  upsets unheld by every feedback structure tried, so their room needs
  foresight of randomly timed events. Stiffer loops do not help: doubling the
  temperature loop's gain raised the cost by 21 % in my run.

### The other twenty

| Task | Status on the candidate | Grade | Why |
|---|---|---|---|
| `distillation` | under review | 6 | Skogestad's Column A with late analysers and an RGA of 36: a standard and useful process benchmark, the natural next scored process task |
| `glass_furnace` | under review | 6 | Novel and physically serious (a 2D melt with flow, fining and defects, 7-day episodes); its oracle does not converge, and one action narrows it |
| `cement_kiln` | under review | 6 | Novel; the transport delay emerges from a 1D axial model and moves with kiln speed |
| `wake_surf` | under review | 6 | Novel and real (extended formation flight for fuel); no MPC yet |
| `wind_turbine` | under review | 5 | NREL 5 MW above rated with tower fatigue priced; OpenFAST and ROSCO studies cover this ground at higher fidelity |
| `plane3d_racetrack` | under review | 5 | A real duty, holding scored on fuel; no oracle |
| `plane_energy` | under review | 4.5 | TECS's job on an A320-class model; no oracle |
| `boiler_drum` | Extra, solved | 4.5 | Classic shrink and swell; its reference pays nothing on a normal day |
| `plane`, `plane3d_heading` | under review | 4 | Credible airliner models on jobs an autopilot already does; no oracle |
| `formation` | Extra | 4 | A rare multi-agent interface in JAX, but its cost is mostly start transients |
| `unstable_cstr` | Extra | 4 | Its room measures state knowledge, not control |
| `patrol`, `patrol_range_bearing` | Extra, solved | 3 | The reference already meets the published station-keeping figure |
| `four_tank` | Extra | 3 | A teaching rig, already in PC-Gym |
| `battery` | Extra | 3 | Simple nulls beat the reference; CityLearn covers dispatch |
| `plane_sine`, `plane3d_circle`, `plane3d_figure8` | under review or Extra | 2.5 | No real duty behind the reference |
| `first_order` | test fixture | none | |

The suite as a whole: 25 tasks are advertised, five are benchmarked, nine are
Extra and most aircraft tasks are solved. The README's first line ("25 JAX
environments ... with tuned PID and MPC baselines", README.md:4) oversells what a
practitioner can measure today, though the bullets under it are honest.

## Core issues, ranked

Ranked by how much each undermines the library for an outside user. Sizes: S
is days, M is one to three weeks, L is more than a month.

### 1. There is no current, stable number to cite

Every line on every scored task page reads "not current" on the candidate
(for example docs/environments/reactor.md:89-92), and the records in
`src/target_gym/data/charter/` were taken on earlier plants. I ran each shipped
expert on the record's own protocol (seeds 100 to 109, sensor noise 1, each
task's scored window, its scenario weights), on the candidate's code:

| Task | Last record (date, scope) | Shipped expert on the candidate, my run |
|---|---|---|
| `reactor` (v10) | reference 1.034, 2026-11-13, no MPC or ceiling recorded | 0.543 |
| `compressor_surge` (v5) | reference 8.58, four scenarios | 4.41, three scenarios |
| `ph_neutralization` (v6) | reference 0.195, five scenarios, old permit | 0.0746, three scenarios |
| `cstr` (v8) | 98.4, recorded on v7, a different job | 0.348 |
| `hvac` (v6) | reference 0.00744, five scenarios, 14-day windows | 0.00743, three scenarios, days 4 to 10 |

On the reactor the expert is the reference, so its yardstick halved without a
new record. On the other four the shipped expert is not the recorded rung
(issue 2), so the table shows the size of the drift, not a like-for-like gap.

The plants are still moving. Since 2026-11-01, 39 commits touched the
reactor's `env.py` or `env_jax.py`, 16 touched cstr's, and 21, 15 and 14
touched the `env.py` of hvac, compressor_surge and ph_neutralization. cstr
changed from setpoint tracking to a production job on 2026-11-24; the
reactor's observation grew from 11 to 16 channels and its action from 2 to 3
within the month (src/target_gym/reactor/PHYSICS.md:334-347). README.md:369
says "Rewards, physics, baselines and the measurement protocol are settled."

**Fix (M).** Freeze the five plants and their evaluation at a tag. Run the
re-record once, in CI, from that tag: every line, every scenario, intervals
included. Make the release gate refuse any stale line (`scripts/stale_records.py`
already lists them), and publish one results table keyed by task version
(reference, MPC, ceiling, best null, each with its interval). After the release,
a scored plant changes only with a version bump and its re-record in the same
PR, and the previous version stays registered for a release. **For a user:** a
number with a version that they can cite, reproduce and compare with next year.

### 2. The public comparison does not measure what the task pages promise

A user meets two yardsticks that disagree, and the one they can call is the
weaker one.

- `tg.score_policy` runs the policy and `make_expert` on seeds 0 to n-1 of the
  default configuration and reads them against a "resolution ceiling"
  (src/target_gym/scoring.py:258, 587-667). The task pages define the result as
  the saving over the ladder's reference, on seeds 100 to 109, over weighted
  scenarios, against a foresight ceiling.
- On four of the five scored tasks `score` and `headroom` come back NaN
  (`JOB_SCORED`, scoring.py:98). On the reactor the resolution ceiling, $0.096 a
  step, sits 440 times above the expert's hold, so `headroom` reads 0. I ran
  `score_policy(name, make_expert(name), seeds=2)` on all five.
- On the four ladder tasks the scored reference is a rung built by
  `scripts/reference_ladder.py`, which the wheel does not ship
  (pyproject.toml:181), and it is not the shipped expert. On hvac the expert is
  the plain heating curve, one rung of the ladder; on cstr it is a
  valve-position pair, while the kept rung holds the feed still
  (src/target_gym/pc_gym/cstr/PHYSICS.md:337-347).
- Ties are not ties. The expert scored against itself won 2 of 2 seeds on cstr
  (ratio 1.000001), and on compressor_surge its ratio was 0.990 (interval 0.979
  to 1.002): the host NumPy path and the traced JAX path drift apart on the
  limit-cycling valve.

A fix (`make_reference` and `tg.score`) is being landed by another session, so
this item lists what it has to deliver.

**Fix (S to M).** Ship `tg.make_reference(name)` and one `tg.score(name, policy)`
inside the package, returning exactly the page's number: per-scenario costs,
the weighted saving, its interval, trips, on the page's seeds and windows. Run
the policy and the reference through the same code path, so a controller scored
against itself reads exactly 1 and wins nothing. Take `score`, `headroom` and
`win_rate` out of the README until they are defined on every scored task.
**For a user:** one call gives the number they would put in a paper's table.

### 3. The room above the reference mixes control with clairvoyance

The ceilings read hidden state and the future. The pH ceiling sits on the
permit's edge because it knows the hidden buffer and acid strength
(src/target_gym/pc_gym/ph_neutralization/PHYSICS.md:141-147). The cstr ceiling
holds the upsets by timing its moves ahead of events whose times are random
(cstr PHYSICS.md:266-268). The compressor's ceiling knows the machine's true
surge line, and hvac's knows the hour-by-hour weather. The pages say part of
the room may be out of reach, but no line shows how much is reachable.

The MPC line that should show it is weak or missing. In the last records
hvac's MPC costs 0.0086 against its reference's 0.0074, compressor_surge's MPC
tripped 19 of its 20 approach episodes (J 492,035 against 8.58), cstr-v8 ships
none, and the reactor's is not recorded. The README still calls the MPC "an
upper bound" (README.md:37). So a learner that closes 30 % of the room may have
done very well or poorly, and nobody can tell which.

**Fix (M per task, about five weeks for all five).** Add one non-anticipative
baseline per scored task, built the way a practitioner would: an estimator on
the observation (an EKF or a moving-horizon estimator) feeding the economic MPC
the project already has, with its constraints backed off. Report the room in
two parts: reference to causal MPC (what control can win) and causal MPC to
ceiling (what information is worth). Stop calling a line an upper bound when it
loses to the PID. **For a user:** a reachable target, and an honest split of the
gap.

### 4. One reference loses to a two-setting retune, so part of its room measures the reference

compressor_surge's PI pair limit-cycles (compressor_surge/PHYSICS.md:737-752,
the project's own finding, documented as a "weak expert"). On the record's
protocol, slowing the pressure loop ten times and moving the control line from
15 % to 12.5 % of surge flow cut the weighted cost from 4.41 to 3.98 (10 %),
and the normal day, which carries 79 % of the weight, from 1.81 to 1.21 (33 %),
with no trip and no surge on seeds 100 to 109. At the vendors' 10 % line the
normal day costs 0.87, under half the shipped pair's 1.81, though the fast
approaches then leave the pressure zone. In every variant the valve still
travels 2.6 to 3.4 full strokes per two-minute normal window, where the recycle
should sit nearly shut, so the cycle survives the retune: the cure is
structural (decoupling the pressure and anti-surge loops), not only gains.

The other references held up under the same kind of probe. pH: a set point of
6.30 instead of 6.45 raised the cost by 83 % through permit excursions. cstr:
doubling the temperature loop's gain raised it by 21 %, quadrupling by 35 %.
hvac: a parallel shift of the heating curve by 0.5 K either way cost 8 % more.
The reactor's expert tripped nothing in 60 episodes, though its contract
records a 3 to 4 K Tavg overshoot after a third of ramp-ups
(reactor/PHYSICS.md:583).

**Fix (S per task).** Before the re-record, put each reference through what a
commissioning engineer would do: a stability check, then a sweep of its two to
four main settings on training seeds (never the scoring seeds), shipping the
best variant. For compressor_surge, decouple or slow the pressure loop, remove
the cycle, then re-record. **For a user:** beating the reference then means
beating good practice, not a loop with a known fault.

### 5. The prices and disturbances that create the room are mostly the project's own numbers

"ours" appears 60 times in compressor_surge's PHYSICS.md, 36 in cstr's, 34 in
pH's, 32 in the reactor's and 10 in hvac's. Among them are the numbers that
decide the room: compressor_surge's zone price (1,000 kJ/s), surge price, trip
cost, demand levels and ramps; cstr's 1 K all-or-nothing band, its upset sizes
and its scenario weights (no source gives how often each upset happens); pH's
excursion price, upset sizes and weights; hvac's weather (a 5 C mean, a
sinusoid with noise, a forecast with no error of its own; hvac/PHYSICS.md:127
and 304); the reactor's rod step and effluent prices. The plants are sourced.
The tasks' economics are mostly choices, and a practitioner will ask whether
the order of the controllers survives other reasonable choices.

**Fix (M).** The project already draws its own numbers over family ranges (the
family sweep). Publish that sweep for each scored task as a sensitivity table:
the reference's saving over the best null, the room, and whether the order of
the lines (null, reference, MPC, ceiling) changes across the ranges. Separately
(S), replace hvac's synthetic weather with one or two public typical
meteorological years (PVGIS TMY data are free) and a forecast with a realistic
error. **For a user:** they know which conclusions hold under other prices.

### 6. There are no learned-policy results, and the raw tasks are hard on standard RL

`src/target_gym/data/rl_results.json` holds no result, and README.md:47 says
none are published. A trip is charged in one step: the reactor's costs $3.84 M
(666.67 times 5,760 steps) against a median step of $0.057 in my run,
compressor_surge's 1.98e7 kJ against about 10, pH's 6,480 L against 0.07,
cstr's 960 against 0.25. That is 10^3 to 10^7 times a normal step. `ScaledEnv`
(src/target_gym/scaled.py) maps observations to [-1, 1] and divides rewards by
one constant per task, so that ratio stays, and cstr's 2 K wide band covers 4 %
of its scaled temperature range. Horizons are long: a reactor day is 2,880 steps
with a 20 min boron delay and a 29 h axial mode; a glass week is 5,040. The
README's RL example trains SAC for 10,000 steps on an unscored aircraft task.
Without a single learning curve, a researcher cannot tell whether these tasks
are learnable, trivial or hopeless.

**Fix (M).** Run one JAX PPO and one SAC (PureJaxRL style, or the owner's own
JAX library) on the five scored tasks: five seeds, a fixed budget such as 1e7
steps, on a training wrapper that caps or ends the episode at a trip. Publish
the curves against the best null and the reference, with the discount (or
average-reward setting) recommended per task. **For a user:** evidence that the
tasks are learnable, and a training recipe that works.

### 7. The task contracts read as decision logs, not specifications

The scored PHYSICS.md files run 378 to 1,329 lines; "owner" appears 8 to 37
times in each; superseded versions, rulings and dates sit inside the current
contract (for example compressor_surge/PHYSICS.md:1-17, whose version-4 section
overrides parts of sections 1 to 4 below it). The reactor's task page lists
over a hundred parameters above a reference table that reads "not current".
The physics is there, but a reader needs an afternoon to extract the task.

**Fix (S).** Generate a one-page task card for each scored task: the job in two
sentences; observations and actions with units and ranges; cost terms with
their prices and whether each is sourced or ours; disturbances with their
statistics; limits; the evaluation (seeds, windows, weights); current numbers
with intervals; the five deviations that matter most. Move the history to the
changelog. **For a user:** ten minutes to understand a task, and a page they
can cite.

### 8. A few fidelity gaps sit on the property a task is meant to test

- **reactor.** The axial mode's stability does not change through the cycle
  (D11, reactor/PHYSICS.md:479-484), so offset control late in the cycle is
  easier than in real cores, on exactly the days that cost most. Grey-rod
  operation (EDF's Mode G, run by 48 of its 56 units for deep load follow,
  reactor/PHYSICS.md:468) is not modelled, and the single bank from the top
  cannot push power back up the core (D3). The xenon peak after a scram is
  1,419 pcm against a 1,800 to 3,200 pcm test band (PHYSICS.md:212), which
  matters for restarts.
- **ph_neutralization.** The reagent valves are ideal. Valve resolution and
  rangeability near the equivalence point, with probe drift (listed as D1 but
  not modelled), are what make pH loops hard in plants. The library already
  has `actuator_dead_band` and `actuator_stiction`, set to 0.
- **hvac.** Heating only, one zone, all slab heat to the air node, and
  synthetic weather (issue 5).
- **compressor_surge.** One low-pressure-ratio machine (1.32), with speed as a
  rate-limited first-order setpoint and no drive torque or inertia. That is
  fine for the surge loop, but in my runs the pressure loop's tuning decides
  the zone charges, and that loop rests on the drive model.

**Fix (S to M each).** Turn on sourced valve dead band and stiction in the pH
task, and add a slow probe drift. Make the reactor's axial damping a function
of the cycle point from a published trend. Add a drive inertia to the
compressor. A cooling or two-zone variant of hvac can wait. **For a user:** the
named difficulty is the real one.

### 9. The headline rests on thin slices and one uncontrolled draw

On the reactor, baseload days carry 89 % of the weight but 2 % of the
headline; the five load-follow days, at 2.16 % each, carry 98 % (my run: J
0.543, baseload 0.013 a step, load-follow days 0.58 to 8.65). Within one day
type the seeds' costs span 1.3 to 19.3 on `edf_daily`, in the same order on
every day type, which points at the core's cycle point. It is drawn uniformly
per seed (reactor/env_jax.py:343-350) and not stratified, whereas
compressor_surge already stratifies its machine (`machine_stratum`). With ten
seeds the reactor's headline largely reports which cycle points were drawn.
The statistics are under review elsewhere; this item is about the design.

**Fix (S).** Stratify the cycle point across the scoring seeds, as
compressor_surge does for its machine. Show the per-scenario table beside the
weighted headline, and on the reactor report baseload and load-follow days as
two numbers. **For a user:** a difference between two controllers that is not
a difference between two seed draws.

## Tasks that cannot be made credible cheaply

- **`cstr`.** Its job is three days old and ships no oracle or MPC for its two
  moves; its room in the upsets needs foresight of random event times; and the
  plant is the most worn benchmark in the field. I would take it out of the
  0.8.0 scored suite and keep it registered, or score only its normal day and
  setpoint move and report the upsets separately. `distillation` (Column A,
  late analysers, strong interaction) is the better process task to finish for
  a later scored release.
- **`hvac`.** It can be made credible, but not cheaply as a standalone
  benchmark, because BOPTEST already sets the bar. Keep it, present it as a fast
  TABS surrogate, and cross-check it once against BOPTEST's
  `bestest_hydronic_heat_pump` on the same weather (M). Real weather (S) comes
  first.
- **The solved aircraft tasks** (`plane`, `plane_sine`, `plane3d_heading`,
  `plane3d_circle`, `plane3d_figure8`, `patrol`, `patrol_range_bearing`). No
  amount of work gives them room, because the jobs are solved by the autopilots
  they model. Keep them as a sandbox for tracking and as sanity tasks, and spend
  aircraft effort on `wake_surf` and `plane3d_racetrack`, which have real
  economics.
- **`battery`, `unstable_cstr`, `four_tank`.** Leave them as Extra, out of the
  headline count.

## What I ran and what I only read

**Read:** the README and the environment reference; the five scored task
pages; the five scored PHYSICS.md files in full or nearly (model, numbers,
limits, scenarios, scores, experts, deviations); their experts and scenarios
in code; the records in `src/target_gym/data/charter/`; the evaluation and
scoring modules; the opening and status sections of the other twelve plant
contracts (`first_order`'s excepted). I did not read the COLD_REVIEW.md or
REALITY.md files, the earlier outside review, or other sessions' notes.

**Ran,** on flanders, from a code-only copy of `dceb2537`, with the shipped
library and the project's own protocol runner (`scripts/reference_ladder.py`'s
run path), seeds 100 to 109, sensor noise 1:

- `tg.score_policy(name, tg.make_expert(name), seeds=2)` on the five scored
  tasks, plus one jitted expert episode each for reward and observation scales.
- The shipped expert on each scored task's full scenario suite (60 reactor
  episodes, 30 to 50 on the others).
- Simple alternatives: compressor_surge's pair with slower pressure loops, a
  detuned valve loop, no rate action, and control lines of 12.5 % and 10 %;
  pH set points of 6.30 and 6.15; cstr with stiffer temperature loops, a
  temperature held nearer the band's top, the valve held at 99 %, and the feed
  forced full in the window's last 20 and 40 min; hvac's curve shifted by -0.5,
  +0.5 and +1.0 K; the reactor with boron flow forced off in the last 20 min
  and 2 h of the day.
- A closed-form check of the pH plant's titration curve against Henson and
  Seborg's published operating point.

The window-end probes came back clean and are not issues: forcing the cstr
feed full in the last 20 min lowered its cost by 0.7 %, and stopping reactor
boron in the last 20 min, which cannot reach the core inside the window, by
0.09 % (0.4 % over the last 2 h, with no trip). The window ends are not an
obvious loophole.

## Limits of this review

- I did not run any oracle or MPC, so every ceiling and MPC number above comes
  from records the project itself labels stale. The room on the current plants
  is unknown to me.
- My alternatives are probes, not tuned controllers: ten seeds, one noise
  level, no held-out seeds. The compressor result shows the reference can be
  beaten cheaply; it does not measure how much more a good engineer would get.
- I trained no RL agent, so issue 6 states the absence of evidence, not
  evidence that the tasks are unlearnable.
- I checked one published figure per plant at most, and mostly trusted the
  contracts' own validation tables, which are unusually thorough. A domain
  expert per family (a reactor physicist, a turbomachinery control engineer)
  would find more than I did.
- Other sessions are landing fixes to the statistics, hvac's window end, the
  scored defaults and the comparison API on this branch. This review describes
  `dceb2537` and did not wait for them.

## Sources looked up for novelty

- PC-Gym, Bloor et al., arXiv 2410.22093: process-control environments (CSTR,
  first-order, multistage extraction, crystallisation, four-tank) compared
  against a perfectly estimated NMPC oracle. https://arxiv.org/abs/2410.22093v1
- BOPTEST `bestest_hydronic_heat_pump`: a single-zone dwelling heated by an
  air-to-water heat pump through floor heating, with peak and typical heating
  periods and constant, dynamic and highly dynamic prices.
  https://ibpsa.github.io/project1-boptest/testcases/ibpsa/testcases_ibpsa_bestest_hydronic_heat_pump
- Nuclear RL work found is per-paper, not an open benchmark: microreactor drum
  control with xenon feedback (arXiv 2504.00156,
  https://arxiv.org/abs/2504.00156v1) and Korean Nuclear Society papers on
  load following with RAST-K and on NuScale-class cores.
- Anti-surge RL work found is a thesis and a supervisory study, not a
  benchmark: https://www.politesi.polimi.it/handle/10589/188922 and
  https://psecommunity.org/wp-content/plugins/wpor/includes/file/2606/LAPSE-2026.0520-1v1.pdf
- The Industrial Benchmark, Hein et al., arXiv 1709.09480 (synthetic, not
  JAX), https://arxiv.org/pdf/1709.09480, and PowerZooJax for power systems in
  JAX, https://arxiv.org/pdf/2609.36052.
- Also compared from their public descriptions: safe-control-gym, controlgym,
  Sinergym, CityLearn, Grid2Op, gymnax and Brax.
