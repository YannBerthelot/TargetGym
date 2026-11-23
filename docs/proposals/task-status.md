# Task status

Where each task stands under the [task charter](task-charter.md). This page changes as the
work moves, while the charter holds the rules. Rule names (B1, J2, ...) are the charter's.

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
