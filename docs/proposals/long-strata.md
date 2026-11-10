# Long strata: the tasks as continuing plants

Draft 3, 2026-11-10. All open questions decided with the owner; nothing is built.

## Why

The tasks are continuing plants, and J4 is already a cost per step. But today each scored
run is one finite window. It starts at the design point and stops at N. The start and the
end both enter the score, and an oracle with foresight spends the plant's stock before the
end.

The owner's calls of 2026-11-10:

- The design has two parts: a stream for learning, and scoring runs.
- Oracles charge their tail.
- The observation carries no window clock. The phase of each real cycle stays observed.
- J4's strata and weights stay.
- Each scored seed is one long run of its scenario.
- Slow plants burn in. Burn-in is preferred to pricing the boundary.

The setting has to survive a hostile reviewer. Each rule below says what it defends
against.

## Part A. The stream

**S1. No fixed length.** A stream yields steps for as long as it is asked.

- **Start and trips.** It starts once, in established operation, and trips restart the plant
  as `base.failure_kernel` does now.
- **Scenario switching** (Q7 decided 2026-11-10). Scenarios switch as a Markov chain whose
  long-run shares are the J4 weights.
  - A scenario lasts a geometric (memoryless) time, with mean
    `D_s = max(B + S, the duration the source gives)`. Sourced durations include seasons,
    weather and a grid's operating mode.
  - On leaving, the chain enters scenario s with probability proportional to `w_s / D_s`,
    which makes its long-run share exactly `w_s`.
  - An event in progress at a switch runs to its return.
  - The scenario label is never observed.
  - The minimum of `B + S` means every scoring run is a stretch the stream really produces.
- **Events recur as a Poisson process** (Q2 decided 2026-11-10). Inside a scenario, its
  events arrive at the scenario's rate, one per N steps on average.
  - The hazard is constant, so the gap since the last event says nothing about the next.
  - Events may overlap.
- **Events are excursions that end.** Each event's onset, size, duration and return are drawn
  from the family's ranges, keyed on the event's index. A level shift (catalyst, cooling,
  buffer) comes back, as a catalyst change or a repair does, so recurrence never compounds
  outside the family.
- **Cycles keep running.** Real cycles (daily, price period, season) keep running, and
  their phase is observed.
- **Nothing marks time.** The observation has no time since start, no scenario label and no
  marker of any kind.

*Defends against:* artificial episode boundaries, which are the original objection.

**S1b. Aircraft** (Q6 decided 2026-11-10). Aircraft tasks follow the same continuing rules.
Their paths and programs never run out: patrol and formation's lead cycles its route, and
circle, racetrack and figure8 are closed shapes.

- **Mass.** Mass cannot change in flight. It is drawn once per stream and per scoring seed
  from the family's range, then held.
  - plane's `heavy_high` and `light_low` stop being regimes that switch.
  - A mass band can still be a scoring stratum, because mass is constant over a run. In
    training, mass is a per-stream draw.
- **Flight level.** Flight level does change in flight: ATC step climbs, a stack moving down
  as it clears, deviations around storms. A level change is an event under S1's Poisson
  rule, so the job is to capture the new level and hold it.
  - Each new level is drawn from the band the drawn mass allows, so a heavy aircraft high
    up, with thin margins, still occurs.
  - The rates come from sources: step-climb spacing, how often a holding stack steps down,
    weather deviations.
- **Wind and turbulence** stay scenarios, since weather changes in flight.

*Defends against:* a scenario that changes something physically impossible mid-flight.

**S2. Common random numbers.** The outside world depends only on the seed and the time,
never on what the policy does. The outside world here means:

- events, walks, weather, prices and load programs;
- the scenario sequence;
- sensor noise.

A test pins it: two different policies on the same seed see bit-identical disturbance
sequences. This makes J4's pairing exact.

*Defends against:* "the agent was luckier with its events".

**S3. Training distribution.**

- **Headline runs** train on the natural-rate stream.
- **Variants.** Oversampling rare scenarios, curricula and domain randomisation beyond the
  family's ranges are allowed, but only as declared variants, reported apart from the
  headline.
- **Rare events are part of the task.** A rare event is not meant to be memorised. Each
  occurrence is redrawn, so a policy has to master the plant's structure.
- **They are visible in the report.** A failure on them shows in J4's per-scenario saving
  and worst scenario, and their weight bounds their effect on the headline.

*Defends against:* "you tuned the training mix" and "the agent only memorised the events".

**S4. Discount** (Q3 decided 2026-11-10). The task's objective is J4's average cost per step,
whatever the algorithm; `gamma` belongs to a learner, not to the task. The rule
`gamma = 1 - 1/N` is retired, since a stream has no N. The settings below are for the
baselines we publish.

- **Discount rule.** SAC and PPO, the headline, use `gamma = 1 - 1/(c T)` with `c = 10`.
  `T = max(t_s, the longest preview in the observation)`. Previews include prices ahead, a
  pull notice and a known schedule.
- **Not searched.** `gamma` is never in the hyperparameter search.
- **Sensitivity.** A check at `c = 5` and `c = 20` is reported.
- **Average-reward agent.** One average-reward agent (APO, built on Ajax's PPO) runs in the
  same table, with the same budget and scoring. Which agents form the headline is fixed
  now, before any result.
- **Reward scaling.** With `gamma` this close to 1, value estimates are roughly J divided by
  `(1 - gamma)`. The protocol states how rewards are scaled or centred, the same way for
  every agent.

*Defends against:* "you tuned `gamma`", "the wrong `gamma` sank RL", and "the headline was
chosen after the results".

## Part B. Scoring runs

**R1. Slow mode, by coupling** (Q1 decided 2026-11-10). `t_s` is how long the plant takes
to forget where it started, under the reference. A script measures it once per task
version; the result is kept in the task's numbers file and covered by the fingerprint.

- **Pairs of starts.** Each pair is the design point and a state the plant really visits.
  That second state is the end of another seed's long run under the reference, so hidden
  states (bricks, xenon, slab, stored charge) are at realistic values.
- **Same seed.** The reference runs from both starts on the same seed, with identical
  disturbances (S2).
- **The fit.** `t_s` is the decay time of the gap between the two runs, in cost per step and
  in the full state, fitted on about 10 pairs.
- **Outside processes need no burn-in.** Walks, weather and demand start at their stationary
  distribution, as compressor_surge's demand already does. Burn-in therefore covers only
  the plant's own states.
- **Agent diagnostic.** The evaluator runs the same coupling test on the agent over the
  scoring seeds. It reports the gap left when scoring starts, and flags a large one.
  - The diagnostic is reported only.
  - It never changes the burn-in.

*Defends against:*

- a burn-in or run length chosen by hand;
- a measure that misses hidden states;
- integrating and unstable plants, which have no open-loop settling time;
- an agent setting its own burn-in.

**R2. A scoring run** is the stream pinned to one scenario, cut into two parts.

- **Burn-in `B`.** Every task burns in; there is no trigger (Q5 decided 2026-11-10).
  - `B` is drawn per seed, uniformly between `5 t_s` and `10 t_s`, from a key that is never
    observed.
  - The start bias left after burn-in, about `e^-5` of the gap measured by coupling, is
    reported for each task.
  - The agent and the reference get the same draw.
  - The burn-in is charged in training like any step, but evaluation does not read it.
- **Scored part `S`.** `S` is `K` windows of the current N, with
  `K = max(1, ceil(20 t_s / N))`.
- **Events in a scoring run.** Each seed gets exactly `K` of the scenario's events, at times
  drawn independently and uniformly over `S`. That is a Poisson process conditioned on its
  count.
  - It shows the same timing as the stream, with no pattern to learn.
  - Fixing the count reduces the spread between seeds, and is unbiased when event costs
    add up.
  - The burn-in runs the scenario's normal background, with no scored event.

*Defends against:* start-up bias, unconverged observers, and integrators or recurrent
memory still filling. Nothing tells a policy when scoring starts, so it cannot bank a
stock during burn-in and spend it while scored. With `S` of at least `20 t_s`, a one-off
drawdown is worth little per step.

**R3. Seeds per scenario** (Q4 decided 2026-11-10). The seed counts are set when a task
version is frozen, before any agent runs, and published with the numbers they come from.

- **Spread.** Each scenario's seed-to-seed spread is that of the paired log cost ratio,
  reference against oracle, on the same seeds.
  - The oracle runs that room already needs are reused, so there is no extra run.
  - The pair brackets the widest comparison an agent can make. Its spread is likely larger
    than an agent's, so the seed count errs on the safe side. The first measurement checks
    this against the reference's own spread.
  - Every task must have a converged oracle, since its seed counts depend on it.
- **Seeds per scenario.** The largest of three numbers:
  - the scenario's share, weight times spread, of what a 1-point headline 95 % half-width
    needs;
  - what a 3-point half-width on the scenario's own saving needs;
  - 10, because a percentile bootstrap over seeds is unstable below that.
- **Events per scenario** are then seeds times `K`.
- **Over budget.** If a scenario's count exceeds the compute budget (J6), its interval is
  published as wider, with the reason. A scenario is never dropped.

*Defends against:*

- a rare scenario's saving resting on two events;
- seed counts picked after seeing results;
- an interval too wide to read.

**R4. Headline.** J4 keeps its form.

- **Cost per seed.** A scenario's cost on one seed is the mean cost per step over `S`.
- **Unchanged parts.** The weights, the bootstrap within scenarios, the agent and reference
  pairing, and the trip rule stay as they are.

**R5. No boundary pricing.** A stock is priced only where the source's own economics price
it as part of the job, as the battery's stored energy is in the market model. That is part
of the task's cost, not a fix for the run's boundaries.

*Defends against:* "you chose the shadow price".

**R6. Oracles.**

- **What is charged.** An oracle's objective charges the burn-in, the scored part and a tail
  past the end, at the same limits. The score reads only `S`.
- **Receding horizon.** Over a long run, it plans one window plus the tail, applies one
  segment, and replans. Convergence is shown on each segment.
- **Room.** Room is re-measured. It may shrink against today's foresight over one window.

*Defends against:* the end effect (glass's 0.6 points), and gaming the burn-in with
foresight.

**R7. Seeds.**

- Scoring seeds are derived from a hash of the tagged commit, as J2's are.
- They never overlap training seeds, which are reported for every run.

## Estimates (to be replaced by R1 measurements)

The rare-event column counts how many times the rarest scenario with a positive weight
recurs in 1M training steps of the natural-rate stream, at one event per N steps:
`1e6 w_min / N`. That number does not depend on how many environments run in parallel.

| task | N | t_s (steps) | B (steps) | K | rarest scenario | events in 1M steps |
| --- | --- | --- | --- | --- | --- | --- |
| first_order | 150 | 10 | 50 to 100 | 2 | single scenario | n/a |
| four_tank | 3300 | 38 | 190 to 380 | 1 | single scenario | n/a |
| cstr | 960 | 5 | 25 to 50 | 1 | feed_buildup, cooling_loss (0.05) | 52 |
| unstable_cstr | 1800 | 14 | 70 to 140 | 1 | initiator, throughput (0.05) | 28 |
| ph_neutralization | 1440 | 14 | 70 to 140 | 1 | buffer_loss, bad_day (0.05) | 35 |
| distillation | 400 | 194 | 970 to 1940 | 10 | any step (0.1) | 250 |
| compressor_surge | 1200 | 49 | 245 to 490 | 1 | fast_approach (0.05) | 42 |
| cement_kiln | 700 | 58 | 290 to 580 | 2 | single scenario | n/a |
| glass_furnace | 5040 | 1368+ | 6840 to 13680 | 6 | pull_up, pull_down (0.05) | 10 (1M steps is 3.8 years) |
| reactor | 2880 (int-a3) | xenon, to measure | to measure | to measure | each load program (0.022) | 8 |
| hvac | 1344 | slab, to measure | to measure | to measure | low_gains (0.038) | 28 |
| battery | 14400 | charge, to measure | to measure | 1 | long deviation (0.009) | 0.6 |
| wind_turbine | 2400 | to measure | to measure | to measure | near_cut_out (0.036) | 15 |
| boiler_drum | 1800 | to measure | to measure | to measure | industrial_swing (0.1) | 55 |
| plane | 1800 | 23 | 115 to 230 | 1 | worst_zone, long_zone (0.05) | 28 |
| plane3d_racetrack | 1200 | 14 | 70 to 140 | 1 | design_wind_low (0.019) | 16 |

Four scenarios carry weight 0 (extreme_gust, id_fan_runback, feed_pump_trip, and long
deviation with wrong estimate). They never occur in the stream and never enter J. They are
reported only.

## Work, in order

1. **Measure `t_s`** for every task, on bob, from a script (R1).
2. **The stream** (S1, S2): a stream wrapper, scenarios as recurring regimes, cycling
   schedules, and the common-random-numbers test.
   - The per-task edits need a version bump and a stamp each. Land them as one PR per
     family.
3. **Scoring runs** (R2 to R4, R7): in `scenarios.run`, `scenarios.costs`, `eval` and the
   records format.
4. **Oracles** (R6): glass first, then every oracle with an end effect.
5. **RL protocol rewrite** (S3, S4) of sections 5 and 6.
6. **Re-record and release.** Re-record every task, then update the docs and the CHANGELOG,
   with upgrade notes. This changes the benchmark definition, so it is a minor version bump.

All of it lands after `int/amendment3` merges.

## Decisions (formerly open questions)

- **Q1. Slow-mode definition.** Decided: coupling under the reference (R1).
- **Q2. Recurrence inside a regime.** Decided: Poisson in the stream; in scoring, Poisson
  with the count fixed (S1, R2).
- **Q3. Discount.** Decided: a rule-based `gamma` for SAC and PPO, plus APO in the same
  table (S4).
- **Q4. Events per stratum.** Decided: seed counts come from the reference-against-oracle
  spread, to precision targets (R3).
- **Q5. Burn-in trigger.** Decided: always burn in, with no trigger (R2).
- **Q6. Aircraft.** Decided: continuing; mass drawn per stream; flight level changes are
  events (S1b).
- **Q7. Regime dwell in the stream.** Decided: geometric, with mean
  `max(B + S, sourced)` (S1).
- **Q8. Battery's long deviation.** Decided: accept the real rate of about 0.6 per 1M
  steps, because 1M one-second steps are 11.6 days of operation. The weight is not raised;
  oversampling is allowed only as a declared variant (S3). Two follow-ups:
  - Replace our 20 a year (family range 5 to 50) with a sourced ENTSO-E count of long
    frequency deviations.
  - The RL protocol reports each task's training budget in plant time beside steps.
