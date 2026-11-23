# Task status

Where each task stands under the [task charter](task-charter.md). This page changes as the
work moves, while the charter holds the rules. Rule names (B1, J2, ...) are the charter's.

## Status on 2026-11-23 (draft)

One review session per task ran on 2026-11-23, and the owner piloted their decisions.
Scores are each session's own, out of 10 as a benchmark task, judged on realism and
sources, room over the reference, and the oracle's integrity. "With its open work" depends
on pull requests that are not merged. No record has been redone: every figure is a
measurement from a session, not a charter record. The sections below this one date from
2026-11-10 and are kept as they were.

| Task | On the branch | With its open work | Where it stands |
|---|---|---|---|
| `reactor` | 8 | 8+ | Rebuilt task merged (#205), room 4.46. The day's end is not priced; five baseload days where the planner loses to its expert are being explained. |
| `hvac` | 6 | 8 | Heat source at 38 W/m2 merged (#229). Room 1.36 (1.35 to 1.38) on 50 seeds once the reference is placed by the no-exit rule (#238). |
| `cstr` | 6 | 7.5 | Tracking task repaired (#215, #221). Production prototype, feed as a second handle: room 1.34, acceptance checks under way (#237). |
| `compressor_surge` | 5 | 7 | Reset in established operation and ceiling on the surge line merged (#223, #231). Reading lag and a hidden line shift in #243: margin 0.09, room 1.54 (1.45 to 1.65). The shift is being made local to the line. |
| `ph_neutralization` | 5 | 7 | Room of 14 was the set point. Placed by the no-exit rule at pH 6.5, room 1.16, with the perfect hold as ceiling (#232). |
| `distillation` | 5 | 7 | Bottoms rest where the score is cheapest (#219), room 1.02. Product priced in #228: room 1.056 (1.054 to 1.059). Oracle rebuild pending. |
| `plane3d_racetrack` | 6 | 7 | Scored on the fuel burned in the hold (#222). No oracle; waits for the corrected A320. |
| `wake_surf` | 6 | 7, then 8 | Station and reference fixed (#217). Next version in #240: drifting wake, predicted station, crossing as a trip. No oracle yet. |
| `glass_furnace` | 3 | 6 | Stays in the benchmark (owner). Room 1.048 (1.041 to 1.056) on `study/glass-room`; four directions chosen, worked in order. |
| `plane` | 5 | 7.5 | Excess climb margin traced to the induced drag and the engine. A320-233 on Airbus's chart with two mountain-wave scenarios in #239. No oracle. |
| `plane3d_heading` | 5 | 5 | A clean solved task (#222). |
| `wind_turbine` | 3.5 | not scored again | Recorded room of 136 not credible. Rebuild in #235: published controller as reference, power held at the set point, rotor-filtered wind, hub anemometer. Room 4.4 to 7.8 on one seed. |
| `boiler_drum` | 4 | 4 | Solved as scored (J5). Fuel loop prototyped (room 1.0004) and the turbine valve lead searched; both dropped (#218, #227). |
| `plane_energy` | 3.5 | Extra | Its cost was a climb-rate limit missing from its reference. Goes to Extra as solved once the reference has its rates (#239). |

Decided the same day, for every task: a reference rests at the lowest set point it never
leaves its zone from ([amendment 7, draft](charter-amendment-7-draft.md)); a task with
little room is kept when the room is beyond seed noise and worth something at the
plant's scale. `battery` and `unstable_cstr` moved to Extra on 2026-11-15, so they are not in
this table, though the older one below still lists them.

Not done for any task: the charter record on the plant as it now stands, and for
`plane`, the 3D hold and `wake_surf`, an oracle.

## Applying the charter

- **Scope.** Every benchmark candidate, the solved ones included, and `glass_furnace` and
  `reactor` again. Extra tasks are exempt. ExpertBench's task list follows the outcome.
- **Order.** Each task gets one fix session. It starts from the task's `COLD_REVIEW.md` and
  works in this order: the property, then the limits, then the observation, then the
  scenarios, then the score. Each change cites `REALITY.md`, and the task's behaviour tests
  from its audit then decide B6. After the five family
  sessions of 2026-11-08 report, each kept task's `REALITY.md` gains its property card
  (B3), and its `ours` numbers and ranges go into its package's table for the sweep (B7).
- **Sources.** Some sites block automated readers (NRC, EDF's archive), so the owner may
  need to open some documents by hand.

## The tasks

Candidates only: nothing has passed acceptance yet. The decisions are the owner's picks of
2026-11-08, being applied by the family sessions. Measurements quoted here come from those
sessions, on few seeds, and from stale oracle records where noted.

| Task | Named property | Decision |
|---|---|---|
| `ph_neutralization` | Titration curve | A finite diversion basin. Staging and residence time in the family range. Influent flow swings. |
| `distillation` | Feed steps, late analysers | Specs set a give-away margin from the products, so scenarios start on spec. D2's realistic drums. |
| `cstr` | Jacket dynamics, fouling | Back from Extra, rebuilt as a PARSEX-like jacketed reactor that runs away on loss of cooling. |
| `unstable_cstr` | Open-loop unstable reactor | Back from Extra, rebuilt as an LDPE-like adiabatic autoclave at its documented unstable point, with initiator flow as the action. |
| `compressor_surge` | Surge trip | Back from Extra, rebuilt with a counted-surge trip, valve lag, and a recycle-power score inside a header-pressure zone. |
| `boiler_drum` | Shrink and swell | Runback and feed-pump-trip scenarios. Cost in money at the imbalance price. |
| `hvac` | Slow concrete-slab heating | Supply capped at 49 C. A heat-pump COP that falls with supply temperature. A day-night tariff. Comfort priced from a family productivity source. |
| `battery` | Gate-closure delay of 60 to 90 min on the baseline | The BMS drift stays as realism. Baseline energy priced at a family wholesale price. |
| `wind_turbine` | Tower motion | Tower damage-equivalent load and pitch travel priced at family LCOE figures. Lost production measured against available power. |
| `reactor` | Axial offset, boron delay | A rod-gain behaviour test first. A reference with xenon feedforward and a bumpless override. The offset band becomes a penalty clock (1 h in 24 h at every power; an overrun is a forced derate). A test of the load-cut xenon swing size; the scram test retired. A family-shape rod insertion limit. Trips priced as lost production. A price-sensitivity report. |
| `glass_furnace` | Being decided in its own session | `REALITY.md` and a cold review first. Crown wear priced from refractory sources over a campaign's rebuild cost; if none, measured as is. A delayed quality reading. A fixed fining criterion. Planned pull changes as scenarios. |
| `plane`, `plane3d_heading` | Gusts through a fast height loop; height and path through turns | The reference cost 0 on seeds 100 to 104 in the aircraft session: search for a cost operators pay (J5). |
| `plane_energy` | Speed against height | A heavy family member at FL330 to 370. The reference gets AIM rates and a capture lead. Extra (B1) if residual climb stays well above 500 fpm. |
| `plane3d_racetrack` | Height and path through turns | The reference reads airspeed and floors at 1.2 times stall. A 30 degree protected bank limit for agent and reference alike, as on every aircraft task (J1). Steady-wind scenarios. |

**Tests.** The process session's test removals are approved where the subject is gone; the
others are rewritten against the new job. Distillation's three re-read tolerances are
restated as family ranges and cold-reviewed. The reactor's scram xenon test is retired.
Battery's test 11 and the reactor's two-moves scorer test await the owner.

## Extra

| Task | Reason |
|---|---|
| `patrol`, `patrol_range_bearing`, `formation` | Scope: several agents |
| `cement_kiln` | Named property absent: delay over time constant is about 0.03 |
| `four_tank` | A lab rig with no industrial job |
| `plane_sine`, `plane3d_circle`, `plane3d_figure8` | No airliner duty |

`first_order` is a test fixture, outside the count.

`four_tank` (owner, 2026-11-10): the table null beats the decentralized PI expert, as Johansson's non-minimum-phase rig predicts (settling about 10 times slower, arXiv 2304.04763 section 5.2). The expert stays as shipped, and room is measured from the better of the expert and the table: 4.11 on seeds 0 to 9 (ours).

## What the charter costs

- **Kept:** null policies, scoring code, expert tools, the oracles built so far.
- **Dropped:** protocols as rules, exceptions, xfails, per-task tuning scripts, `S` and `H`
  as headlines.
- **Redone:** the controllers, by one script per family; the redesigned plants; every
  oracle row on the tagged plants, since most records are stale; the recorded numbers;
  `CONTRIBUTING.md`; PHYSICS pages cut to model, sources, limits and deviations;
  ExpertBench's contract and task list.
- **Not yet known:** whether anything passes, whether redesigns show their property,
  whether PPO learns these tasks (it fails on `plane` and `battery` today), and whether
  oracles can run on every plant at a tolerable cost.
