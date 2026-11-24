# Glass furnace: why there is no room, and what would give it some

A study of `glass_furnace-v6` as it stands on `rec/charter-records` (4be0ab9), done on
2026-11-15 on branch `study/glass-room`. Nothing in the task was changed. Every number
below was measured on the registered plant with the scripts in `scripts/glass_room/`;
their outputs are in `study/glass_room/out/`. Costs are EUR per 120 s step.

**Since then (2026-11-15 and 16)** the plant on this branch has one more loss: defects
released by the tank's blocks when their temperature rises. The sections from "In the
plant" on rerun everything on it. The reference costs 49.23, the library's planner
without foresight 48.25 and the ceiling 46.96: the room is 1.048, and 43 % of it is
reached without foresight. The sections before describe the task as it was.

## The short answer

The furnace, as modelled, is easy to run, and the recorded room of 1.14 % overstates what
is there. Four facts carry it:

1. **Fuel is the only thing to save, and the furnace offers 4.5 % of it.** With the valve
   held, glass is lost below 0.468 kg/s and the crown override caps the valve at
   0.4895 kg/s. Every sensible policy lives in that window. Firing flat out all week costs
   48.4; the reference 46.3; the ceiling 45.8.
2. **A fixed crown temperature already solves the normal day.** Holding the crown couple at
   one setpoint, with no quality feedback at all, costs 46.45 against the reference's
   46.31. The fining loop, the cold-end reading and the feedforward add 0.3 %.
3. **The reference's distance to the limit is an accident.** Its back-off was measured on
   the plain PI rung (spread 0.026) and applied to the cascade, whose own spread is 0.008.
   At two of its own sigmas the cascade loses glass on half the seeds and costs 17 % more.
4. **The ceiling's plans are not ones a furnace could run.** They slam the valve between
   its floor and the override every 12 to 20 h, swinging the crown between 1430 and
   1615 C, and they end further from the limit (0.523 on average) than a steady plan
   needs to be. None converged.

With one consistent rule on both sides (a reference that keeps the smallest margin losing
no glass, a ceiling that sits on the limit), the room on a stationary normal day is about
0.6 %. I then tried six ways to open it up without touching the physics. None gave more
than about 2 % against a well-formed reference. The reasons are physical, and they point
at what the model leaves out.

## The task as it is

- **Aim of the library.** Tasks where a good classical controller still leaves room, each
  standing for a plant family, scored on the operator's job in the source's units
  (charter, "The claim").
- **Plant.** A 500 t/d float furnace on 20 columns by 8 layers: zone radiation, a silica
  crown, averaged regenerators, a batch blanket, foam, buoyant long-tank flow, and a
  fining share carried with the glass, which rises only where the glass is between 1420
  and 1500 C.
- **Job.** One valve. Cost is fuel at 60.5 EUR/MWh plus the whole step's pull at
  600 EUR/t whenever the glass leaving is under a fining share of 0.5 or the canal under
  1100 C. The crown is held under 1615 C by the plant's own override, and the canal by
  the plant's own cooling loop.
- **Reference.** The ladder's cascade: a crown PI on the valve, under a fining PI on the
  crown setpoint, around a balance feedforward on the valve; setpoint 0.5523.
- **Ceiling.** SLSQP on 42 four-hour knots through the env's own step, fining and canal
  as limits, a charged 24 h tail. Room 1.0114, 0 of 25 solves converged.

## What was measured

| Measurement | Result |
|---|---|
| Valve held 30 days, walks off | 0.46 kg/s: fining 0.41, glass lost. 0.47: 0.516, cost 46.19. 0.48: 0.611, 47.17. 0.49 and above: override at 1615 C, fining 0.677, cost 48.11 |
| Design state | Hottest glass 1491 C, under the band's top. Only about the top 15 cm of the melt is above 1420 C. Glass leaving is 0.586 fined |
| Reference, 5 scenarios, seeds 100 to 104 | Reproduces the ladder (45.98 on normal seed 100). Fining 0.557 on a normal day, never under 0.51 in any scenario, spread 0.008 in the second half of a week. No glass lost anywhere |
| Recorded ceiling plans, replayed | Valve 0.25 to 0.60 kg/s, crown 1431 to 1615 C (spread 49 K), fining 0.523 on average. Normal day 45.81 in the window; a steady valve on the pass mark costs 46.03 from a settled start, walks off |
| Crown held, no quality feedback | Normal 46.45, gas rich 46.03, gas lean 46.95: no glass lost. Pull +10 %: 24.9 h lost |
| Steps at a held crown, fining after 24 h and 7 days | Gas +6 %: under 0.002. Pull +10 %: -0.024, -0.073. Cullet -10 points: -0.011, -0.033. Sulfate +50 %: -0.035, -0.105; -50 %: +0.091, +0.235. Fining band +10 K: -0.017, -0.043. Blanket +20 %: -0.003, -0.009 |
| Where the delay sits | The scored reading reaches half its answer in 30 to 37 h. The top 8 cm of the leaving glass in 12 to 16 h. The hot zone's surface in 1 to 8 h |
| Stationary normal day, 10 seeds, settled start | Setpoint 0.5523: 46.54. Lowest setpoint losing no glass (0.53): 46.32. Two sigma of the cascade's own reading (0.52): 54.50, 3.3 h lost on 5 seeds. Each seed's best constant: 46.22. Crown at its cap: 48.39 |

## Why there is no room

**The window is narrow and its floor is cheap to stand on.** Inside the 4.5 % window the
cost is linear in fuel, so the best place is just above the pass mark, and what keeps a
controller off it is how far the fining share wanders. Under the cascade it wanders by
0.008, worth 0.2 % of fuel.

**It wanders so little for four reasons, all of them the model's.**

- The glass leaving is the average of a pool with about 55 h of memory. Every disturbance,
  even a direct change of the fining chemistry, reaches the scored reading as a slow ramp.
  A slow integral loop follows it.
- Fining happens in about the top 15 cm, right under the crown, so the crown couple pins it.
- Gas, the only large disturbance, shows on the crown couple within minutes. A 6 % swing
  moves the fining by less than 0.002 once the crown is held.
- The cold-end reading is the scored variable itself, 18 min late, with noise of 0.005.

**Margin is cheap and both limits are walls.** Nothing is priced between the pass mark and
the override, so carrying a margin costs at most the window, and under 1.5 % of fuel in
every case I tried. That is the most any foresight can save.

## Mistakes and bugs

1. **The back-off is the PI rung's.** `scripts/reference_ladder.py:368` measures a zone's
   spread under `Rung("pi", ...)` and gives the setpoint to every rung. The log shows
   sigma 0.026 before the cascade is tuned; the cascade's own reading spreads 0.007 to
   0.011. This touches every task with a zone, not only glass.
2. **Two sigma does not fit a limit priced as lost production.** A lost step costs nine
   steps of fuel. At two of its own sigmas the cascade loses 3.3 h a week on 5 of 10
   seeds (54.50). The recorded setpoint sits near the cascade's economic best only
   because of bug 1.
3. **The feedforward does nothing under the cascade.** The balance biases the valve, the
   crown's rest is a constant, so the crown loop integrates the bias away
   (`scripts/ladder_rows/glass_furnace.py:134-152`). A planned pull rise of 30 % then
   loses 13 h of glass (78.3), and one of 42 % loses 48 h (170.5). With the crown
   setpoint following the pull charged, both lose nothing (44.9 and 44.8).
4. **The ceiling is not a ceiling one can believe.** No solve converged (KKT 1.0 on all
   25). The kept plans cycle the crown by 185 K, which nothing in the task prices or
   forbids, and they gain nothing over sitting on the limit. An agent can find the same
   freedom.
5. **The window flatters everyone.** Episodes start from a state fined to 0.586 that no
   policy holds. The reference coasts down to 0.552 and the ceiling to 0.50. From a
   settled start the reference costs 46.54 on the ten seeds, against 46.43 in the window.
6. **The pass mark is a category slip.** 0.5 is called the middle of Beerkens' 1420 to
   1500 C band, but it is a share: glass with half its fining undone passes. The furnace
   can only reach 0.68 under its crown cap, and its hottest glass never reaches the
   band's top (cold review item 5 said as much).
7. **The canal takes every depth alike** (`physical.py:688`), so the product is the
   pool's average. The forward stream answers two to three times faster. How a float
   canal draws is not in the record: a question, with a source to find.
8. **The furnace has a quarter of its capacity to spare.** Its two limits meet at about
   1.23 times the task's pull. At 1.2 times, the best policy holds the crown at its cap
   all week and still loses 12 h.
9. **The batch line is observed and does not matter.** The blanket ends 5 m into a
   21.5 m melter, 9 m short of the spring zone. A blanket 20 % thicker moves the fining
   by 0.009. On the furnace the geometry comes from, batch islands cover 85 % of the
   island zone [S10].
10. **Document drift.** `PHYSICS.md` says v5 at its head and v6 below, and gives the
    design fining share as 0.603 where the packaged state has 0.586.

## What I tried, to open the room

All against the cascade with its crown setpoint following the pull (bug 3 fixed), which
is what operators do: they keep tables of the furnace's readings for each configuration
and reproduce them (US6796147B2, seen once).

| Change | Reference | Best plan found with foresight | Gap |
|---|---|---|---|
| Normal day, stationary | 46.32 at its lowest safe setpoint | 46.22 with each seed's best constant; about 46.0 on the limit | 0.2 to 0.6 % |
| Planned pull rise of 30 %, shown 8 to 96 h ahead | 44.68 | 44.54 | 0.3 % |
| Planned pull rise of 42 %, a container plant's pair [S3] | 44.62 | 44.47 | 0.3 % |
| Planned pull cut of 30 % | 42.37 (the crown must follow slowly, or 2.2 h are lost) | 41.81 | 1.3 % |
| Job at 97 % of capacity for 3 to 6 days | 49.09, carrying a margin of 0.06 beforehand | 48.80 | 0.6 % |
| Steady work at 97 % of capacity | 89.1, the crown at its cap, 12 h lost | 89.0 | 0.1 %: the loss is the furnace's, no controller's |
| Fining 3 to 60 times faster, read as surviving seeds | The share leaving unfined falls from 0.41 to 0.008; its logarithm answers 1 % of fuel by 0.1 to 0.3 at every speed | | No steep tail: the pool's mixing sets it |

Reading the plan earlier changed nothing: the 55 h pool cannot be filled in the notice,
and a small margin carried all the time does the job for 0.5 % of fuel.

## Inside a transient

Plants lose glass in transients, so this is where the model should bite. It does not. A
planned pull change at hour 48 over its 2 h ramp, walks off, under the cascade with its
crown setpoint following the pull (probe 9):

| | Rise 30 % | Rise 42 % | Cut 30 % |
|---|---|---|---|
| Crown hot face, largest move | +63 K, 2.2 h after | +76 K, 3.9 h after | -86 K, 2.3 h after |
| Largest crown move in one hour | 34 K | 45 K | 45 K |
| Fined glass past the last port, top 16 cm | +11.7 K in 3.1 h | +13.9 K in 3.8 h | -17.0 K in 3.6 h |
| Canal | +0.8 K | +1.0 K | -0.9 K |
| Fining share leaving | no dip | no dip | 0.528 at worst, from 0.552 |
| Cost of the 48 h after, over the settled cost | 240 EUR | 300 EUR | none |

A week's fuel is about 232,000 EUR. Under the reference as recorded, the same cut
reheats the fined glass past the last port by 22 K over 12 h, also for free.

**Why the transient is free.** The score sees one quality number, which only improves
with heat and averages 55 h of history. The right answer to any production change is
then to jump the crown at once, and nothing charges for the jump: reboil from reheating
fined glass is left out (`PHYSICS.md`, section 1, against [S45]), the crown's rate of
change has no limit or price, and the forward stream's fresher glass is averaged into
the pool. The furnace has the inertia operators fear, and no reason to fear moving fast.
A model in which a fast temperature move is itself harmful would turn every transient
into a trade between moving in time and moving gently, which is where anticipation and
foresight earn glass.

**Not covered.** Only two kinds of transient ran scored: planned pull changes and the
suite's gas swings, one a week, held to the week's end. Repeated changes and returns
(the stream), unannounced pull changes, batch and cullet steps as scored events and
colour changes did not run. The foresight plan in each row is a clairvoyant setpoint
schedule on the same cascade, a lower bound on what an oracle would find.

## What a temperature move costs in a real furnace

Sourced on 2026-11-15 and written into `REALITY.md`, section 10, each figure read twice
unless marked there. What the sources give:

| Mechanism | What is sourced | Source |
|---|---|---|
| Tank blocks in contact with the glass | A temperature rise pushes bubbles, knots and stones out of fused-cast AZS. Held at 1550 C a block gives off 1.4 wt % of glassy phase and stops; five swings of 50 to 100 K give 1.8 to 2.1 wt %. A dip of 50 K and the return is enough. The advice is a constant process temperature | Dunkl 1989; Chmelar et al. 2000 |
| Reboil | Float glass keeps about 0.2 wt % SO3, and a melt holding 0.3 wt % gives it off from about 1400 C when heated. Firing the refiner's front too hard raises reboil bubbles. A melt near saturation makes new bubbles when its temperature changes | US5922097; US6673730; Hubert, slide 36 |
| Second fining on cooling | Bubbles under 0.1 mm only dissolve during controlled cooling through about 1350 to 1275 C | Hubert, slides 49 to 51 |
| Crown | Cycling of 25 to 30 K is given as the cause of corrosion about twice as fast | Balandis 2011 |
| Fining chemistry | The fining range moves with the batch's redox; organic contamination makes it fluctuate | Hubert, slides 54 to 56 |
| Foam | Up to 60 % less radiation into the melt; more foam at higher pull, which caps the pull | Pilon 2012 |
| What defects are worth | Yield up to about 8 % better and product-change losses up to halved under supervisory control; 50 % to over 90 % yield on one float furnace when bubble density fell | Chmelar et al. 2000; US6715319 |

Not found: how many kelvin of reheating start reboil in a float refiner and how fast; a
limit on the crown's rate of change; ramp rates, lead times and settling times for a pull
change; hours or tonnes lost after a change; defects per tonne per kelvin of anything.
The directions and the thresholds are sourced. The rates are not.

**How far the model's furnace already moves** (probe 10, the cascade with its crown
setpoint following the pull, one week):

| | Crown, span | Crown, largest rise in 1 h | Glass in the fining zone, span | Glass past the last port, rise in 12 h | Bottom glass under the hot spot, span |
|---|---|---|---|---|---|
| Normal day | 26 K | 8 K | 23 K | 13 K | 12 K |
| Gas swing of 6 % | 23 K | 8 K | 21 K | 10 K | 10 K |
| Pull rise of 10 % | 38 K | 15 K | 25 K | 13 K | 12 K |
| Pull rise of 30 % | 67 K | 34 K | 28 K | 13 K | 10 K |
| Pull cut of 30 % | 97 K | 10 K | 34 K | 13 K | 11 K |

Three readings:

1. **The reference already swings the furnace on a normal day.** It chases the fining
   share with the crown: 26 K of crown a week, the span the crown paper ties to doubled
   corrosion, and 23 K in the fining zone's glass. The glass just past the last port sits
   at 1401 C, where float glass begins to give off its SO3 on heating, and climbs 13 K
   within 12 h every week.
2. **A transient multiplies the crown's move by 2.5 to 4 and hardly changes the bath's.**
   The bath's inertia is real. What a pull change does is make the controller throw the
   crown.
3. **So a price on temperature movement would charge the reference's ordinary behaviour,
   and transients most.** Holding still would be worth something, moving in time would be
   worth more, and moving gently would have to be traded against both. That is the
   tendency the owner expects and the model lacks.

## The prototype: a temperature move that costs glass

Built on this branch in `scripts/glass_room/proto.py`, on top of the registered plant,
which is not touched. Two defect tracers are carried with the glass exactly as the plant
carries its fining share:

- **Contact.** Each cell's block follows its glass over 2 h (ours, the dwell of Dunkl's
  cycles) and gives off defects in proportion to its temperature rise and to the wall and
  floor area the cell touches. Nothing removes them on the way out.
- **Reboil.** The plant's own fining reaction, where it still runs past the last port.

The rates have no source, so they enter only when a cost is read: `q10`, the share of the
tank's glass that a 10 K rise of every block would reject, and `p1`, the chance that glass
doing 1 % of its fining past the last port is rejected. Three checks pass
(`proto_check.py`): the wrapper's plant is the env's to the last bit, its transport lands
on the plant's own within 2e-7 when fining is off, and each tracer's books close to 4e-5.

**A normal week** (five seeds, days 3 to 10, `q10` = 1 %):

| Controller | Fuel | Crown, span | Blocks' rise | Glass lost to block defects | Total |
|---|---|---|---|---|---|
| Cascade as tuned | 46.53 | 35 K | 20.3 K | 0.76 % | 49.70 |
| Its outer loop at a tenth of its gain | 46.58 | 20 K | 18.6 K | 0.69 % | 49.47 |
| Crown held, no quality loop | 46.63 | 19 K | 18.4 K | 0.69 % | 49.50 |
| A loop on the floor couple under bay 3, by SIMC | 46.64 | 45 K | 19.7 K | 0.73 % | 49.68 |
| A loop on the bath's true mean temperature, by SIMC | 46.63 | 56 K | 7.6 K | 0.25 % | 47.67 |

What moves the bath at a held crown, as the blocks' rise in a week: every walk 16.9 K;
sulfate alone 14.7; cullet alone 7.8; blanket 4.6; gas 3.1; pull 3.0; no walk, sensor
noise only, 0.2.

**A pull rise of 30 %** (three seeds, same week, `q10` = 1 %). The settled change from
the low pull to the high one is 3.7 K at the blocks and 42 K at the crown.

| Controller | Blocks' rise | Glass lost to the fining mark | Glass lost to block defects | Total |
|---|---|---|---|---|
| Crown held, its setpoint following the pull | 22.4 K | none | 0.82 % | 50.49 |
| The same, the crown ramped at 1 to 20 K/h | 23.7 to 25.1 K | none | 0.83 to 0.90 % | 50.4 to 50.7 |
| The same, read on the plan 8 or 24 h ahead | 24 to 34 K | none | 0.86 to 1.22 % | 50.6 to 52.1 |
| A loop on the floor couple | 24.0 K | none | 0.92 % | 50.86 |
| A loop on the bath's true mean, its target held | 10.1 K | 88.5 h | 0.31 % | 275 |

**Reboil.** At `p1` = 10 % it rejects 0.04 % of the glass in normal work, 0.36 % with the
crown 10 K hotter and 1.9 % with it 20 K hotter. Running hot stops being free.

Five readings:

1. **Every controller built from the plant's instruments loses the same glass.** Slowing
   the loop, holding the crown, or closing a loop on the floor couple moves the loss by
   less than a tenth. The reference now has an issue, and no setting of it helps.
2. **Knowing the bath helps 2.8 times.** A plain loop on the bath's true mean, which no
   instrument reads, cuts the loss from 0.69 to 0.25 % for the same fuel. If only blocks
   above 1400 C release, the factor is 2.1 and the floor loop still does not help.
3. **The hidden walks move the bath, foam above all.** The controller's own chasing adds a
   tenth.
4. **The pull change is a trade, and both wrong answers are expensive.** Following the
   table moves the blocks six times further than the settled change needs. Ramping does
   not help, and acting early is worse, since the bath then rises before the load cools
   it. Holding the bath still halves the travel and starves the fining for 88 h.
5. **The room is real at any rate, and its size is the rate.** On the normal week it is
   at least 49.50 over 47.67, 3.8 %, at `q10` = 1 %, from a plain loop and no oracle. It
   scales roughly with `q10`.

What is still ours, and carries the result:

- The two rates. Nothing sourced converts a kelvin of rise into defects.
- Which blocks release. Dunkl's tests ran at 1450 to 1550 C, and most of the contact
  area is the floor at 1100 to 1310 C. With only blocks above 1400 C releasing, the same
  rate loses 36 times less glass.
- The sulfate walk, 17 % on a 12 h memory, which makes most of the bath's movement.
- Release in proportion to the rise, the 2 h block lag, and defects that never clear.

What it does not fix: the glass leaving is still a 55 h average, so being late on the
fining side stays cheap until the mark is crossed.

## In the plant (2026-11-15, the owner's go)

The two mechanisms are now part of `glass_furnace-v6`, re-stamped in place since no
release has shipped it.

**What changed.**

- `physical.py` (+36 lines): two new fields, the blocks' temperature and the defects per
  plate, the latter carried by the same transport as the fining share; four parameters
  (`block_lag`, `block_release_T`, `block_release`, `reboil_plate`).
- `env_jax.py`: the reward's lost-glass term is now a share of the pull: all of it under
  a mark, else the plates rejected, `1 - exp(-defects)`. The observation gains a
  thirteenth channel, the defects per plate read at the cold end, 0.3 h late with the
  fining reading's scatter (the owner's go, 2026-11-15).
- `oracle.py`: its objective is fuel plus rejected plates, the env's own bill at the
  marks, so it still plans the task's reward.
- `design_state.npz`: the settled blocks and zero defects added; the other arrays are
  untouched.
- `tests/glass_furnace/test_glass_transients.py`: nine tests written from the contract
  before the code. One existing line changed, with the owner's go for the reading: the
  glass furnace's row in `tests/test_industrial_readings.py` gains channel 12 at 0.005.
- `PHYSICS.md`, `REALITY.md` section 10, `ours.json`, the version stamp.

**Checks.**

- Every earlier field is identical to the last bit over two days under a moving valve:
  temperatures, fining, crown, regenerators, blanket, foam, cooling, observation, fuel.
- The eight new tests were red before the code and are green after. Five deliberate
  breaks each turned some red: release on any change, no release temperature, reboil in
  the fired zone, defects not carried, defects costing nothing.
- All 58 fast glass tests pass, and the 9 slower oracle tests pass unmodified. Across the
  repository, every test that selects the glass furnace passes, with the stamp, the
  generated facts, the doc-drift check, ruff and black.
- The step is about 10 % slower at batch 256 (67,500 against 74,400 steps a second,
  timed together), which puts it near 3.2 times the median against J6's 3.

**The numbers it ships with, all `ours`.**

- `block_release` 270: set so that the furnace held on its tables loses 1 % of its glass
  to block defects on the normal day, the low end of the family's bracket (yield up to
  about 8 % better under supervisory control).
- `block_release_T` 1450 C and `block_lag` 2 h: the coolest temperature and the dwell of
  Dunkl's cycles.
- `reboil_plate` 10.5: glass doing 1 % of its fining past the last port loses one plate
  in ten.

**On the plant as it ships** (`plant_runs.py`; normal week, five seeds; pull rise of
30 %, three seeds):

| Controller | Normal week: glass lost | Total | Pull rise: glass lost | Total |
|---|---|---|---|---|
| Crown held on its tables | 1.03 % | 50.93 | 1.01 % | 51.30 |
| The ladder's cascade | 1.06 % | 50.94 | 1.01 % | 51.17 |
| A loop on the floor couple | 1.08 % | 51.16 | 1.06 % | 51.45 |
| A loop on the bath's true mean | 0.48 % | 48.62 | 0.44 %, and 88 h under the fining mark | 275.6 |

The room is at least 4.7 % on the normal week at this rate, from a plain loop on a
quantity no instrument reads. With hot blocks only, a pull rise costs the table
reference no more than a normal week; the trade in the transient is that a controller
holding the bath must move its target correctly or lose the fining.

**Still open.**

- The rate, for a practitioner to replace, and whether blocks under 1450 C release.
- The defect reading's scatter is the fining reading's, for want of its own; a real
  count is noisier the fewer defects there are.
- `scripts/generate_env_pages.py` stops on a `cstr` mismatch that predates this work,
  so the glass furnace's docs page was patched by hand for the new channel.
- The reference: rerun on this plant (next section); none of its rungs can hold the
  bath.
- The ceiling: rerun on this plant (the section after next). The MPC line: rerun too
  (the section after that), and the shipped planner without foresight fails on this
  plant. A planner without foresight that keeps its marks is in the section after
  (the fair line). Every recorded row for the task is stale.
- How much foam varies: no figure found; the sulfate walk stays `ours`.

## The reference ladder on the new plant (2026-11-15)

`scripts/reference_ladder.py --tasks glass_furnace --nulls`, as it stands, on the plant
with its defects: ten seeds, the five scenarios, a week from the reset. Median cost per
seed, weighted over the scenarios:

| | On the plant of 2026-11-13 | On the new plant |
|---|---|---|
| PI on the fining share | 73.15 | 76.35 |
| With the balance feedforward | 62.66 | 65.43 |
| Cascade through the crown, kept | 46.49 | 49.43 |
| Null: valve at mid range | 55.12 | 70.09 |
| Null: valve wide open, the crown on its override | 48.50 | 73.48 |
| Null: valve held where the reset put it | 78.63 | 77.72 |

The kept rung by scenario: normal 49.52, pull up 52.18, pull down 47.15, gas rich 49.09,
gas lean 50.04. It loses 0.74 % of its glass to defects and none to a mark.

- **Same tuning, same rung.** Every fit and the placed setpoint (0.5523) come out to
  the digit, since the old fields are untouched. No rung trips. No null beats the kept
  rung on any seed (J2).
- **Firing hard is no longer nearly as good as the reference.** The best null was 4 %
  above the kept rung and is now 42 % above it. Running hot costs plates.
- **A lower bound on the room, on the ladder's own suite and seeds: 1.7 %.** A loop on
  the bath's true mean temperature costs 48.59 (`ladder_suite_bath.py`). It holds the
  reset state's bath, which is hotter than the reference needs, so it burns 0.4 more in
  fuel than it has to. From a settled start the same loop gave 4.7 %.
- **The window still flatters the reference.** The week starts with a tank free of
  defects, and the reference spends its first days cooling from the design state to its
  setpoint, which sheds nothing.
- **Fixing the row's feedforward changes nothing here.** With the crown setpoint
  following the pull the cascade costs 49.49 (`ladder_suite.py`): the suite's pull
  changes are 10 %.
- **What stays wrong with the reference:** its back-off is still the PI rung's spread,
  and no rung of the ladder can hold the bath.

## The ceiling on the new plant (2026-11-15)

`scripts/glass_slow_oracle.py`: the oracle of `glass_furnace/oracle.py` plans each
episode with its walks and events foreseen, on the five scenarios and seeds 100 to 104,
from the design valve and then again from that plan, 300 iterations at most each. Every
plan is replayed through the benchmark's runner at sensor noise 1, as the ladder scores
its rungs. That is 51 solves (one seed had a third start), 21 hours of solving on bob in
all, 1.1 to 3.1 GB each.

Two changes came first, neither in the env's stamp:

- The script counted a replayed step as lost whenever its lost-glass term was above
  zero. On this plant nearly every step rejects a few plates, so every plan would have
  been thrown out. It now counts the steps under the canal or the fining mark.
- The oracle scaled its objective by a fixed hundredth, which suited the fuel alone. The
  rejected plates make the gradient 88 EUR per step per kg/s on its largest knot against
  2.4 (`oracle_scale.py`), and the solver's first steps threw the valve between 0.25 and
  0.54 kg/s. The scale is now set at each solve's start so that the first step moves no
  knot by more than 0.025 kg/s (`FIRST_STEP`), which is the old hundredth on fuel alone.
  The nine oracle tests pass unmodified.

Mean cost per step over the five seeds, and what the plans save of each term:

| Scenario | Weight | Reference | Ceiling | Room | Fuel saved | Glass saved |
|---|---|---|---|---|---|---|
| Normal | 0.70 | 49.21 | 46.95 | 1.048 | 0.51 | 1.75 |
| Pull up | 0.05 | 51.89 | 49.68 | 1.045 | 0.39 | 1.82 |
| Pull down | 0.05 | 46.72 | 44.32 | 1.054 | 0.62 | 1.78 |
| Gas rich | 0.10 | 48.83 | 46.55 | 1.049 | 0.50 | 1.79 |
| Gas lean | 0.10 | 49.66 | 47.45 | 1.047 | 0.52 | 1.70 |
| Weighted (J) | | 49.23 | 46.96 | 1.048 | | |

The room recorded on the plant without defects was 1.0114.

- **Every plan beats the reference, by 3.8 to 7.1 % on its seed.** No replayed plan has
  a step under a mark or a trip, and a plan's own cost and its replayed cost differ by
  0.0006 at most.
- **Three quarters of the room is glass, one quarter fuel.** On the normal week the
  plans lose 0.28 % of their glass to defects against the reference's 0.70 %, and burn
  1.1 % less fuel (`reference_split.py`, `plan_traces.py`).
- **The fuel is the margin.** The plans rest on the fining limit (0.5002 at their
  lowest, 0.522 on average). On the normal week the reference averages 0.557 and its
  lowest is 0.530 (0.514 on a pull rise): its back-off is still the PI rung's spread,
  as on the old plant (bug 1).
- **The glass is the blocks, held by foresight and by a quick answer** (the fair line
  holds half of it with no foresight). Reboil costs the plans 0.01 EUR per
  step and the reference nothing (`plan_mechanisms.py`). About 0.3 % of the block area
  sits at or over the release temperature, and over the week that area rises 60 K in
  all under the reference and 27 K under the plans. The plans' hot spot moves more, not
  less (standard deviation 12 K against 5.5 K) and runs 9 K cooler: they let it fall and
  keep the hot blocks from rising.
- **The reference's loss is not its own valve movement.** Reading without noise, the
  cascade loses the same glass (2.91 against 2.90 EUR per step) and its hot blocks rise
  the same 60 K. The walks move the blocks under a held crown, and no rung reads them.
- **These are plans a furnace could run.** The valve stays between 0.40 and 0.54 kg/s
  and moves 0.30 to 0.46 kg/s in all over the week, in 4 h holds. On the old plant the
  plans slammed it between its floor and the override every 12 to 20 h.
- **The week starts clean, which understates the gap.** Lost glass by day on the normal
  week, EUR per step: the reference 0.45, 1.61, 2.96, 3.49, 3.87, 3.82, 4.11; the plans
  0.08, 0.34, 0.88, 1.43, 1.68, 1.79, 1.87. The gap is 2.2 on the last day against 1.75
  over the week, and the plans' loss is still rising when the week ends.
- **Not converged by the certificate, so the room is a lower bound, but the plans are
  close.** None of the 25 kept plans meets the certificate's 1e-3, and it reads 1 on 24
  of them. That 1 is mechanical: the certificate counts a limit as active within 1e-6
  of its edge (scaled), under one float32 step of the canal and headroom rows, so it
  finds no active limit and nothing to set against the gradient. Read again with a band
  of 1e-4 (`plan_certificate.py`), the normal plans' stationarity is 0.003 to 0.04 of
  their largest gradient. The solver's log agrees: 46 of 51 solves stop on its own
  tolerance, 14 of 25 restarts after one iteration, and 23 restarts move the cost by
  0.003 or less. Two found 0.48 and 1.86 (gas rich, seeds 104 and 101), so a poor stop
  does happen, and a third start per seed would be the cheap insurance.
- **What the ceiling knows and the reference does not:** the week's walks and events
  ahead, and the true state. How much the true state alone gives is what the MPC line
  (the oracle without a seed) was run for, in the next section; as shipped it does not
  answer it. The fair line, in the section after, does: 44 % of the gap, half of the
  glass the plans keep. The loop on the bath's true mean gave 1.7 %.

## The MPC line on the new plant (2026-11-16)

`scripts/glass_room/mpc_line.py`: the oracle as the library ships it without a seed
(`make_oracle("glass_furnace")`, the runners' MPC row). It reads the true state, plans
the rest of the week on the walks' mean path with the events known, and plans again
every 24 h, starting from the valve it is holding. It ran through the benchmark's runner
on the ceiling's five scenarios and seeds, 300 iterations at most per plan: 175 plans,
48 hours of solving on bob, 1.3 to 3.2 GB an episode.

Mean cost per step over the five seeds:

| Scenario | Weight | Reference | MPC line | Ceiling | Episodes with a step under a mark | Steps under a mark |
|---|---|---|---|---|---|---|
| Normal | 0.70 | 49.21 | 99.45 | 46.95 | 3 of 5 | 12.4 % |
| Pull up | 0.05 | 51.89 | 116.83 | 49.68 | 3 of 5 | 14.4 % |
| Pull down | 0.05 | 46.72 | 91.97 | 44.32 | 1 of 5 | 12.8 % |
| Gas rich | 0.10 | 48.83 | 74.61 | 46.55 | 2 of 5 | 6.1 % |
| Gas lean | 0.10 | 49.66 | 58.66 | 47.45 | 2 of 5 | 1.9 % |
| Weighted (J) | | 49.23 | 93.38 | 46.96 | 11 of 25 | 10.8 % |

- **It costs 1.9 times the reference.** Of its 93.38, the fuel is 45.90 and 44.88 is
  whole steps lost under the fining mark, on 11 of the 25 episodes (39 to 3215 of an
  episode's 5040 steps). Nothing trips.
- **The cause is the missing margin.** The plan rests 2e-4 of fining share over the
  mark, which covers float32 rounding, on a forecast that leaves out the walks'
  innovations. Over the day a plan is played the fining comes out under its forecast by
  up to 0.023. The spread of that error grows through the day: 0.001 after 4 to 8 h,
  0.005 after 12 to 16 h, 0.012 after 20 to 24 h. On 36 of the 175 days the fining ends
  more than 0.01 under its forecast.
- **It cannot recover.** Once the state is bound to cross the mark inside the fining's
  dead time (11.8 h), no valve meets the limits. The solver stops on incompatible limits
  or a failed line search (17 plans), and what it returns is played. On pull down,
  seed 102, the fuel bill falls to 36 EUR a step and 64 % of the week is under the mark.
- **Each day starts cold.** A plan starts from the valve held, not from the last plan's
  remainder, and 33 of the 175 plans stop at the cap.
- **Where it keeps the marks it is the reference, not the ceiling.** On the 12 episodes
  with no step under a mark and no plan outside its limits it costs 48.45, against the
  reference's 48.81 and the ceiling's 46.62 on the same episodes: 16 % of the gap. Fuel
  is 45.84 against 46.04 and 45.47, lost glass 2.61 against 2.76 and 1.15. These are the
  episodes the walks spared, run without the margin the planner needs, so the fuel
  figure flatters it.
- **What it says about the room.** With a plan a day, the true state does not keep the
  glass: the plates lost to the blocks are the reference's, a tenth of that gap closed.
  That part of the room is foresight of the walks, or a faster answer to them than a
  plan a day. The next section shows it is partly the second. How much a competent
  planner without foresight reaches is not measured by this line, because this one is
  not competent.
- **The library ships this line.** `make_oracle("glass_furnace")` and the runners build
  the oracle without a seed, so the task's oracle row would be this one.

## The fair line on the new plant (2026-11-16)

`scripts/glass_room/fair_line.py`: the oracle without a seed, as in the last section
(the true state, the walks' mean path, the events known), with three changes, none of
which shows it the future:

- **A plan every knot (4 h), started from the last plan's remainder.** The lead it
  cannot correct is then the dead time plus one knot, 16 h, and not 36 h.
- **No fining limit: the risk is priced.** Each planned step is charged the glass it
  would lose (its whole pull at the lost-glass price, about 415 EUR) times the chance
  that the fining comes out under its mark, the forecast's error taken as normal with
  the spread measured at that lead. The spread is the shipped line's own, read from
  the last section's run on the 154 days whose plan started inside its limits: nothing
  before 4 h, then 0.0016 of share at 8 h, 0.0039 at 12 h, 0.0070 at 16 h. Past 16 h
  it is held at 0.0070 (ours), since a later plan can still answer.
- **When the returned plan breaks the canal or the headroom limit, the last plan goes
  on.**

I had announced a margin of two sigma and softened limits. Two sigma loses about 2 % of
the steps by construction, on a limit priced at nine times a step's fuel, which is the
ladder's bug 2 again, and softening a limit needs a constant with no source. Pricing
the chance puts the margin where one more step toward the mark saves as much fuel as
the glass it risks.

It ran through the benchmark's runner on the ceiling's five scenarios and seeds, 100
iterations at most per plan: 1050 plans, 82 hours of solving on bob, 2.8 to 3.6 hours
and 1.3 to 1.5 GB an episode. Mean cost per step over the five seeds:

| Scenario | Weight | Reference | Fair line | Ceiling | Share of the gap closed | Steps under a mark |
|---|---|---|---|---|---|---|
| Normal | 0.70 | 49.21 | 48.23 | 46.95 | 43 % | 0 |
| Pull up | 0.05 | 51.89 | 51.14 | 49.68 | 34 % | 0 |
| Pull down | 0.05 | 46.72 | 45.53 | 44.32 | 50 % | 0 |
| Gas rich | 0.10 | 48.83 | 47.76 | 46.55 | 47 % | 0 |
| Gas lean | 0.10 | 49.66 | 48.68 | 47.45 | 44 % | 0 |
| Weighted (J) | | 49.23 | 48.24 | 46.96 | 44 % | 0 |

- **It keeps every mark and beats the reference on all 25 episodes,** by 0.4 to 4.1 %
  on its episode. No step is under a mark and nothing trips. The reference costs 1.0205
  times the fair line, and the fair line 1.0271 times the ceiling.
- **It closes 44 % of the gap between the reference and the ceiling:** 0.18 of the 0.51
  EUR per step in fuel and 0.81 of the 1.75 in glass.
- **The margin that came out is 0.025 of fining share.** The lowest fining of an
  episode is 0.522 to 0.531, which is 3.2 to 4.4 times the spread at 16 h, and after
  the first two days the fining averages 0.538. On the normal week the reference
  averages 0.557 with a lowest of 0.530, and the foresight plans 0.522 and 0.5002.
- **It keeps half the glass the foresight plans keep, without foresight.** On the
  normal week the hot blocks rise 45 K in all, against 60 K under the reference and
  27 K under the plans, and it loses 0.50 % of its glass to defects against 0.70 % and
  0.28 %. Reboil costs it 0.001 EUR per step. So the blocks are not held by foresight
  alone: a plan a day was too slow to answer the walks, and a plan every 4 h from the
  true state answers about half of what they do to the blocks.
- **The valve is calm.** It stays between 0.41 and 0.54 kg/s over the 25 episodes and
  moves 0.30 to 0.41 kg/s in all over the week (a scenario's mean), in 4 h holds, as
  the foresight plans do. The reference moves 2.8.
- **From a warm start the solver has an easy job.** Of the 1050 plans, 1044 stop on the
  solver's own tolerance, after 25 iterations at the median. Six stop at the cap. Five
  of them are an episode's first plan, the only cold start, and were played inside
  their limits. The sixth (pull up, seed 103, day 6) was outside a limit by 6e-4,
  scaled, and the last plan was kept.
- **The week still starts clean.** Lost glass by day on the normal week, EUR per step:
  0.37, 1.28, 1.95, 2.39, 2.71, 2.94, 3.10, beside the reference's 0.45 to 4.11 and
  the plans' 0.08 to 1.87 (the ceiling's section). Its loss is still rising when the
  week ends.
- **What it says about the room.** Of the 4.8 %, at least 2.0 % is within reach of a
  controller with no foresight that reads the true state and knows the plant. The other
  2.7 % is foresight of the walks, or a better planner than this one: it plans on the
  walks' mean path, and one planning on sampled paths could hedge the blocks as well as
  the fining. An agent on the sensors has more to do than this line: it must also work
  out the state (the blocks, the bath, where each walk stands), which this line is
  given.

**What this does not show.**

- The spread was measured on the 25 episodes the line is scored on. It is one curve of
  17 numbers from 154 days, and its halves agree (0.0071 and 0.0069 at 16 h).
- **On seeds the spread was not measured on, the line holds.** The same planner on
  seeds 105 to 109 of the five scenarios, where the ladder has the reference's costs
  and there is no ceiling: no step under a mark, no trip, and it is under the
  reference on 25 of 25 episodes, by 0.7 to 6.6 %. Weighted, it costs 48.46 against
  the reference's 49.86 on those seeds (1.029). Its margin was thinnest on pull up,
  seed 107: on day 5 the solver stopped at its cap outside a limit three plans
  running, the plan in force went on for 12 h, and the fining came down to 0.515.
  Elsewhere an episode's lowest was 0.518 to 0.531.
- The spread is the shipped line's (a plan a day from a cold start), not this planner's
  own. Measured since (the next section), its own is a fifth wider.
- The 16 h hold, the spread's floor (2e-4, the oracle's own back-off) and the cap of
  100 iterations are ours. The canal and headroom limits stay hard; the canal never
  came nearer than 0.7 K to its target.
- It was a study script, which swapped the oracle's scan and back-off around each
  solve. The next section takes it into the library.

## The planner in the library (2026-11-16)

At the owner's ask the fair planner is now the library's oracle without a seed
(`glass_furnace/oracle.py`), which is what `make_oracle("glass_furnace")` and the
runners build. The oracle with a seed, the ceiling's, is unchanged.

- **What moved.** `solve` takes the fining forecast's spread per step of lead. Given
  one, it drops the fining limit and charges each step what losing it would add, by
  the env's own `costs` (the whole pull less the plates), times its chance of ending
  under the mark. The oracle plans each knot from the rest of the plan in force, and
  that plan goes on when no valve meets the canal's limit or the override's cap.
- **The spread comes from the model, not from a table.** At each episode's first plan
  the oracle samples 128 paths of the walks from the state it is shown, the valve held,
  on keys of its own, and takes the root mean square of the fining against the mean
  path, over the dead time and a knot (15.8 h). No scored seed enters it.
- **That spread is wider than the table the fair line ran on.** From five states (two
  resets, three mid-week states of the fair line) it is 0.0083 to 0.0096 at 16 h,
  against the table's 0.0070. The table was read on the shipped line's states, which
  sat nearer the mark: its first days, starting at a fining of 0.586, give 0.0082, and
  its days 4 to 7, starting at 0.548, give 0.0068. So the fair line ran on a spread a
  fifth narrower than its own forecasts had, and its lowest fining was 2.5 to 3.5 of
  its own sigmas over the mark.
- **Tests first.** Seven new tests in `tests/glass_furnace/test_glass_oracle.py`,
  written from the contract at the file's head and failing on the old oracle: the
  chance against the normal table; the spread nil without walks and halving with
  them; a priced plan staying further over the mark the wider the spread, at least
  two spreads, at no less than the limit plan's bill; a plan each knot; the plan in
  force kept from a bath made 20 K colder; and the spread read from the model at the
  first plan. Each fails when its behaviour is broken (seven breaks, one at a time).
  The nine earlier tests pass unmodified. The file takes about 8 minutes on a loaded
  bob, the new tests up to 80 s each.
- **The port is the study's planner.** Handed the table (`fair_line.py` as it now
  stands), a two-day episode costs 46.390 against the study's 46.400, with the valve
  within 0.0004 kg/s at every knot. Reading the model's spread it costs 46.526: the
  wider spread is paid in fuel.
- **Stamps.** The env's stamp is unchanged (1495ad0c77e3469b). The oracle's
  fingerprint moves, and its recorded rows were stale already.

The library's planner as it ships (`mpc_line.py`, 400 iterations at most per plan) on
the ceiling's five scenarios and seeds, mean cost per step over the five seeds:

| Scenario | Weight | Reference | Library planner | Ceiling | Share of the gap closed | Steps under a mark |
|---|---|---|---|---|---|---|
| Normal | 0.70 | 49.21 | 48.25 | 46.95 | 42 % | 0 |
| Pull up | 0.05 | 51.89 | 51.16 | 49.68 | 33 % | 0 |
| Pull down | 0.05 | 46.72 | 45.56 | 44.32 | 49 % | 0 |
| Gas rich | 0.10 | 48.83 | 47.76 | 46.55 | 47 % | 0 |
| Gas lean | 0.10 | 49.66 | 48.68 | 47.45 | 44 % | 0 |
| Weighted (J) | | 49.23 | 48.25 | 46.96 | 43 % | 0 |

- **It keeps every mark and beats the reference on all 25 episodes,** by 0.3 to 4.0 %.
  Nothing trips. The reference costs 1.0201 times the planner and the planner 1.0275
  times the ceiling: 0.17 of the 0.51 EUR per step in fuel and 0.81 of the 1.75 in
  glass.
- **The wider spread cost 0.02 EUR per step** against the fair line's 48.24, and the
  lowest fining of an episode is 0.525 to 0.532 against 0.522 to 0.531.
- **The solver.** Of 1050 plans, 1049 stop on the solver's own tolerance, after 25
  iterations at the median and 221 at most, none at the cap. One (pull up, seed 100)
  stopped on limits it could not meet together, and the plan in force went on. It took
  67 hours of solving on bob, 2.4 to 3.1 hours and 1.3 to 1.6 GB an episode.

**Not done.**

- Memory in the library's own runners. The study's runner clears JAX's caches after
  each plan; a runner that does not would hold 42 compiled horizons an episode.
- A week costs about 2.7 hours of one core an episode, so the task's oracle row is
  about 70 core-hours for 25 episodes.
- The registry's note on the task and the changelog do not mention the planner yet.

## As a nearly saturated task (2026-11-23)

The owner ruled on 2026-11-23 that the glass furnace stays in the benchmark: it is the
library's motivating example, a task with little room can be a kind of its own if the
room is beyond noise, and room that is missing is to be read as model that is missing.
This section answers the four questions put to this session. It made no new run: it
reads REALITY.md, the results above and four sources looked up that day ([N1] to [N4]).

### 1. Where real furnaces lose money and energy that the model does not carry

In the order of what the sources say each is worth:

| Loss | What the sources give | In the model | What it takes from physics | Not found |
|---|---|---|---|---|
| The operators' margin | Setpoints set by hand, with a safety factor [S4; TR13]. Under supervisory control the saving is the lower setpoint, 2 to 4 % of fuel [S2; S3] | The reference is an automatic cascade, not an operator. The margin a good loop needs is 0.5 % of fuel | What forces a margin: numbers a plant does not know about itself, drawn per stream (emissivity, foam tendency, couple offset, fining band), and readings as poor as real ones | Any spread for those numbers; the couples' delay and accuracy |
| Product changes | Time and production lost in changes cut by up to half [S7], losses cut by 20 to 50 % [S2]; float lines no longer run steady [TR14]; loops cycle for over 8 h after a pull change [S9] | Pull events of 10 %. No change of colour, thickness or glass | A composition carried with the glass, as the fining share is, that moves the melt's radiative conductivity and the fining range; the ribbon off its colour priced as lost glass until the tank has flushed (mean residence 68 h [S11]) | Sizes, frequency, notice, ramps and losses of a float change |
| Yield beyond the blocks | Yield up to about 8 % better [S7]; 50 to over 90 % on one furnace once its foam was dispersed [TR7]. Fining moves with redox, small bubbles need a second fining on cooling [TR1], foam leaves bubbles and sand [S32] | The blocks and reboil only, at rates of ours. The reference loses 0.70 % of its glass | Batch redox as a walk and as steps, setting the fining onset (1430 to 1480 C oxidised, 1140 to 1350 C reduced); a dissolved-gas balance for the second fining (1350 to 1275 C); sand dissolution as a second time-temperature integral | Every rate: defects per kelvin, reboil kinetics |
| Crown and campaign life | Corrosion doubled by cycling of 25 to 30 K [TR5]; one rebuild cost 35 MEUR for 18 years [S37], 1.9 MEUR a year of life | The override protects the crown. Wear has no price | A wear integral on the hot face's temperature and its cycling, priced by the rebuild | The rate against temperature; an activation energy |
| Ageing | Regenerator fouling raises a furnace's energy use by 1 to more than 3 % a year [N1]; flat glass furnaces burn 5 to 8.5 GJ/t [N2], and a float furnace's use depends strongly on its age [N4] | Clean, fixed regenerators. The furnace burns 3.97 GJ/t | A fouling factor on the checkers' heat transfer and flow, drawn per stream from the furnace's age | Nothing more than [N1] for a first range |
| The firing profile | The split between ports sets the hot spot [S15]; boosting where fitted [S5] | One valve, a fixed split, 10 % excess air (no source) | The split as moves: the bays and their gas temperatures are in the plant already. An air ratio needs a flue loss and an emissions limit | Per-port shares on a float furnace; excess-air practice |

Two of these say the same thing about the reference. The family's 2 to 4 % was measured
against operators, on furnaces that had aged and whose readings drift. The benchmark's
reference is an automatic loop on a young, clean furnace that knows its own numbers.
Little room over that reference is what the sources predict; what the model lacks is
less a mechanism than the reasons a real plant keeps its margin.

### 2. Is the room beyond seed noise, and what is it worth

Paired by seed, on the plant with its defects: for each of the five seeds the
scenario-weighted cost of each controller, then their difference, with Student's t on
four degrees of freedom (nine for the ten-seed row).

| | Difference, EUR per step | 95 % interval | Ratio | 95 % interval | Episodes won |
|---|---|---|---|---|---|
| Reference over ceiling | 2.26 | 1.91 to 2.62 | 1.0482 | 1.041 to 1.056 | 25 of 25 |
| Reference over the library's planner | 0.97 | 0.58 to 1.36 | 1.0201 | 1.012 to 1.028 | 25 of 25 |
| The library's planner over ceiling | 1.29 | 1.10 to 1.48 | 1.0275 | 1.024 to 1.032 | 25 of 25 |
| Reference over the study planner, ten seeds | 1.20 | 0.83 to 1.57 | 1.0247 | 1.017 to 1.032 | 50 of 50 |

- **The room is beyond seed noise.** One controller's J moves by 0.24 to 0.33 from seed
  to seed. The room is eight times that, and the paired difference itself varies by
  0.29.
- **Its limits.** Five seeds. The reference was tuned on them, which favours it. The
  interval is the seeds' noise around a lower bound (part 3).
- **The plant as recorded** (rec/charter-records, room 1.0114, 1.008 to 1.020 by
  scenario). The record holds the reference by scenario only, so no paired interval
  can be read from it; its notes say no seed's plan is dearer than the reference. That
  room was taken apart at the head of this report: about half is the reference's
  accidental back-off, and its plans swing the crown between 1430 and 1615 C.

What a furnace-year is worth, on the model's furnace of 500 t/d, at 60.5 EUR/MWh
[S21], 56.1 t of CO2 per TJ of gas [N3] and lost glass at 600 EUR/t (ours):

| | Fuel | Gas | CO2 | Glass | Money |
|---|---|---|---|---|---|
| Reference to ceiling | 1.10 % | 2.2 GWh | 450 t | 770 t, 0.42 % of the pull | 595 kEUR (501 to 688) |
| Reference to the planner without foresight | 0.36 % | 0.7 GWh | 145 t | 350 t, 0.19 % of the pull | 255 kEUR (152 to 358) |
| The recorded plant's room | 0.6 to 1.1 % | 1.2 to 2.2 GWh | 240 to 450 t | none | 73 to 134 kEUR |
| The family's saving against operators [S2; S3; S7] | 2 to 4 % | 4.0 to 8.0 GWh | 810 to 1630 t | up to 8 % of yield | 243 to 487 kEUR in fuel |

- **At a real furnace's fuel use the fuel shares are worth more.** The model's furnace
  burns 3.97 GJ/t, under the 5 to 8.5 GJ/t of flat glass furnaces [N2]. At those, the
  room to the ceiling is 2.8 to 4.7 GWh, 170 to 290 kEUR and 560 to 960 t of CO2 a
  furnace-year in fuel alone, and the planner's share 0.9 to 1.5 GWh and 180 to 310 t.
- **What rests on what.** The fuel figures rest on sourced prices and factors. The
  glass figures rest on two numbers of ours, the blocks' release rate and the price of
  lost glass, and scale with both.
- **Not counted.** Glass that is lost is melted again, and that fuel is not in these
  figures. Neither is any price on the CO2.

### 3. What the ceiling's non-convergence means

- **For part 2, the room and its worth are lower bounds.** The ceiling's figure is the
  cost of a plan replayed through the env: some valve path achieves it, so the true
  ceiling is at or under it. Whatever the solver missed adds to the room. The interval
  in part 2 is seed noise around that bound, not the solver's gap.
- **How much it may have missed.** 23 of 25 restarts moved the cost by 0.003 EUR per
  step or less; two found 0.48 and 1.86. So a first stop can fall short by as much as
  the room itself, about one time in twelve, and after the restart nothing bounds it:
  a third start was not run. The plans are also held on 4 h knots.
- **"0 of 25" overstates it.** The certificate counts a limit as active within 1e-6,
  under a float32 step, and so reads 1 on 24 plans. Read at 1e-4 the stationarity is
  0.003 to 0.04, against a tolerance of 1e-3.
- **For part 1, the finding does not rest on the ceiling.** That the recorded plant
  lacks room was shown without it: a usable fuel window of 4.5 %, a fixed crown within
  0.3 % of the reference, six attempts that opened under 1.3 % each. A better solver
  would not find the family's 2 to 4 % there.
- **The planner's share is a bound of the same kind.** It is a policy that ran, so its
  2.0 % is the least that is reachable without foresight from the true state. The
  split between state and foresight (43 % and 57 %) is the least certain figure here,
  since both ends are bounds.

### 4. Issues, options and the recommendation

Issues, ranked:

1. **The task on record is still the plant without defects**, with a room of 1.011.
   The plant with its defects and its planner sit on an unmerged, unpushed branch, and
   every glass record is stale.
2. **Three quarters of the new room is glass priced on numbers of ours**: the blocks'
   release rate, its 1450 C threshold and the price of lost glass. The room's glass
   part scales roughly with the rate.
3. **No controller has earned room from the sensors.** The planner is handed the true
   state; a loop on the floor couple gained nothing.
4. **The reference's margin is the PI rung's**, a quarter of the room. (Measured the
   same day, in the next section: that margin is 0.005 over the lowest a fixed
   setpoint can be, so this issue falls.)
5. **The plant is a young, clean furnace that knows itself**, run by an automatic
   loop (part 1).
6. **The ceiling is a bound, not a certified optimum** (part 3).
7. **Cost.** The step is 3.2 times the median; the oracle row is 70 core-hours and the
   ceiling 21 solve-hours.

Options:

- **I. Take the plant with its defects as the task and settle it**, with no new
  physics. It needs the owner's word to merge, a ruling on the reference's margin
  (decisions 1, 2 and 6), the planner run on an estimated state (the rung an agent
  competes with), the room read at a third and three times the release rate, and a
  third start for the ceiling. Evidence: room 1.048 (1.041 to 1.056), 2.0 % without
  foresight, held-out seeds.
- **II. A furnace that does not know itself.** Its age and its hidden numbers drawn
  per stream. [N1] gives the first sourced range, and it is what makes a margin real
  (part 1, first and fifth rows).
- **III. Product changes as the transients.** Hand control is weakest in changes of
  tonnage, glass or colour [TR13], and the sources put the most money there. Sizes
  and frequency would be ours until someone publishes them.
- **IV. Per-port firing** [S15]. The physics is in the plant; expect percent-level
  room.
- **V. Redox and the second fining.** More defect physics, at more rates of ours.

Recommendation: I now, since it settles what the present task is worth and changes no
physics. Then II, the one addition with a sourced range, aimed at why the reference
is too good. Then III, which matches the owner's reading of where control is hard but
needs sizes nobody has published. IV and V after.

Readiness: about 6 out of 10 as a benchmark task on `study/glass-room`, and about 3
out of 10 as recorded.

## Settling the plant with its defects (2026-11-23)

The same day the owner picked option I, then II, III and IV, in that order. Two cheap
measurements came first, each a few minutes on one core of bob.

### The room at other release rates and temperatures

`rate_replay.py`: the three valve paths in hand (the kept cascade's, the library
planner's and the ceiling's plans) replayed on plants whose blocks release at another
rate, or from another temperature. Defects feed nothing back into the furnace, so every
path keeps its marks. The planner's and the ceiling's paths are no longer the best for
the other plant, so each room is a lower bound. Weighted cost over seeds 100 to 104:

| Plant | Reference | Planner | Ceiling's plans | Room at least | Planner's share at least | Glass the reference loses |
|---|---|---|---|---|---|---|
| Registered: rate 270, from 1450 C | 49.23 | 48.26 | 46.96 | 1.048 | 1.020 | 2.91 |
| A third of the rate | 47.29 | 46.85 | 46.20 | 1.024 | 1.009 | 0.98 |
| Three times the rate | 54.97 | 52.43 | 49.24 | 1.116 | 1.049 | 8.66 |
| Release from 50 K lower | 53.04 | 51.05 | 48.62 | 1.091 | 1.039 | 6.72 |
| Release from 25 K lower | 51.14 | 49.66 | 47.42 | 1.078 | 1.030 | 4.82 |
| Release from 25 K higher | 47.35 | 46.85 | 46.07 | 1.028 | 1.011 | 1.04 |
| Release from 50 K higher | 46.32 | 46.15 | 45.82 | 1.011 | 1.004 | 0.01 |

- **The room exists at every setting.** On each plant the reference is dearer than the
  planner and than the ceiling's plans on each of the five seeds.
- **The glass part goes with the rate,** nearly in proportion.
- **The release temperature matters more than the rate.** From 50 K higher no block
  releases and the room is the fuel alone, 1.011, the recorded plant's figure. From
  50 K lower the reference loses 2.3 times the glass. Only 0.3 % of the block area is
  at or over 1450 C, so the glass part of the room lives in the hottest few blocks.
- **1450 C stays ours.** It is the lowest temperature in the one study sourced (swings
  between 1450 and 1550 C [TR4]). An onset temperature for the exudation of the blocks
  was searched for again on 2026-11-23 and not found; one patent places exudation
  trouble at about 1500 C, in the superstructure (US7642211, seen once).
- **So the room's existence is settled and its size is not.** It is 1.011 to 1.12 over
  these settings, and two numbers of ours set where it falls.

### The reference at other margins

`margin_scan.py`: the kept cascade at lower fining setpoints, on the suite and the
ladder's ten seeds.

| Setpoint | Cost | Fuel | Glass | Steps under a mark | Where |
|---|---|---|---|---|---|
| 0.5523, as placed | 49.55 | 46.44 | 3.10 | 0 | |
| 0.5500 | 49.49 | 46.41 | 3.08 | 0 | |
| 0.5475 | 49.43 | 46.38 | 3.05 | 0 | pull up's lowest fining is 0.501 |
| 0.5450 | 49.42 | 46.35 | 3.08 | 115 | pull up, 1 of 10 seeds |
| 0.5400 | 49.46 | 46.28 | 3.18 | 455 | pull up, 2 of 10 seeds |
| 0.5300 | 51.25 | 46.15 | 5.10 | 4939 | pull up, 7 of 10 seeds |
| 0.5200 | 61.22 | 46.02 | 15.20 | 15034 | every scenario but pull down |

- **The placed setpoint is 0.005 over the lowest a fixed setpoint can be.** At 0.5475
  no step is lost on any of the 50 episodes, and the cost is 0.12 EUR per step lower
  (0.24 %). Against a reference placed there the room would be 1.046, not 1.048.
- **The pull rise sets it.** On those weeks the fining comes down to 0.501 at 0.5475.
  The cascade's own spread of the fining after the first two days is 0.012 on a normal
  week and 0.016 on a pull rise; two of its own sigmas would put it near 0.525, where
  it loses glass.
- **This corrects issue 4 above.** The fuel part of the room is not a sloppy
  reference. It is a margin held every week for a transient that comes on one week in
  twenty. On a normal week alone the cascade could run at 0.53 without losing a step
  and save 0.51 EUR per step, 0.29 of it fuel and 0.22 glass, since a cooler furnace
  also sheds fewer plates.
- **For decisions 1, 2 and 6:** the evidence is for leaving the placed setpoint as it
  is on this plant.

### What the readings know of the state

The library's planner is handed the true state. `state_filter.py` puts a particle
filter in its place: the plant's own model on many sampled paths of the walks, each
weighed by how well its readings match the agent's, and resampled. It starts from the
task's reset law on keys of its own and is shown the readings and the valve only. It
ran along the library planner's valve path on three normal weeks (seeds 100 to 102),
70 s of one core a week at 512 particles. At each knot, the fining forecast over the
next 16 h under the valve then fired, against what came out:

| Lead | From the true state | From the readings, 128 particles | 512 particles |
|---|---|---|---|
| 4 h | 0.0001 | 0.0063 | 0.0027 |
| 8 h | 0.0018 | 0.0074 | 0.0047 |
| 12 h | 0.0042 | 0.0092 | 0.0070 |
| 16 h | 0.0071 | 0.0117 | 0.0097 |

- **The readings place the furnace to within 1 to 2 K.** Past the first day and at
  512 particles: the blocks within 50 K of their release temperature are missed by 1.4
  to 1.6 K, all blocks by 1.0 K, the bath's mean by 1.2 K, the fining now at the canal
  by 0.001 to 0.003.
- **They halve what is unknown of the hidden walks.** Gas is missed by 0.008 against
  its spread of 0.02, the blanket by 0.06 against 0.10, the sulfate by 0.09 against
  0.17. Pull and cullet are read exactly.
- **128 particles are too few.** One seed locks onto a wrong state (0.010 off at 4 h),
  and the particles' own spread is a third of their error. At 512 that seed is right.
- **The forecast is 1.4 times wider than from the true state at 16 h,** and where the
  true state forecasts 4 h to 0.0001, the readings give 0.003.
- **One averaged state forecasts as well as the particles together** (0.0096 at 16 h
  against 0.0097), so a planner can plan from it.
- **The spread such a planner would price is about right where the valve still acts.**
  Sixteen particles on sampled paths, against the averaged state's forecast: 0.0094 at
  16 h for an error of 0.0096; the error is 0.8 to 1.7 of it by seed at 12 and 16 h.
  At 4 h it is two to five times too narrow, but the valve no longer reaches the
  fining there.
- **Ours, all of it a filter's tuning:** the particle count, the couples weighed at
  1 K (three times their scatter), the batch line at 0.5 m, resampling under half the
  effective count. A better estimator would do better: these figures bound what the
  readings give from below.
- **Not covered:** the scenarios with events. The pull change is announced in the
  readings; the gas swing is not, and the filter does not infer it yet.

### The planner on the readings alone

`readings_line.py`: the library's planner handed the filter's averaged state in place
of the true one, pricing at each plan the spread of the filter's own particles on
sampled paths (64 of 512). The benchmark runs it as it runs any agent: the runner
never passes it the state. Normal scenario, seeds 100 to 104, 400 iterations at most
per plan; 3.0 to 3.4 hours and 2 GB an episode.

| | Cost | Fuel | Glass | Lowest fining of an episode |
|---|---|---|---|---|
| Reference | 49.21 | 46.31 | 2.90 | 0.526 to 0.536 |
| Planner on the readings | 48.80 | 46.27 | 2.53 | 0.531 to 0.542 |
| Planner on the true state | 48.25 | 46.14 | 2.11 | 0.525 to 0.528 |
| Ceiling | 46.95 | 45.47 | 1.15 | 0.5002 |

- **It keeps every mark.** No step under a mark, no trip, no plan kept; all 210 plans
  stop on the solver's tolerance.
- **It earns 0.41 EUR per step over the reference,** 0.8 %, on four seeds of five
  (per seed 0.04, -0.04, 0.89, 0.46, 0.69). Paired over five seeds the interval is
  -0.09 to 0.91: not yet beyond seed noise.
- **That is 42 % of what the true state earns** (0.96) and 18 % of the room to the
  ceiling. Not knowing the state costs 0.55 EUR per step (0.25 to 0.86).
- **Its saving is glass.** It burns the reference's fuel: the spread it prices (0.009
  at its last lead against 0.007 from the true state) keeps its fining 0.007 higher
  than the true-state planner's.
- **Its first day is blind.** No reading tells the hidden walks at reset, so its first
  plan holds the reset's valve (the two-day smoke run cost 47.40 against the true
  state's 46.53).
- **So a controller on the readings can earn some of the room,** with a plain filter
  whose tuning is ours. It bounds from below what an agent with memory can reach on a
  normal week; the event scenarios are not covered.

Still to do for option I: a third start for the ceiling (deferred: of little value now
that the room's size hangs on the release numbers).

## A furnace that does not know itself: first measurements (2026-11-23)

Option II. Before any design, cheap tests of where a number the furnace does not know
about itself could open room. Normal scenario, the ladder's ten seeds, two weeks, the
second week read (the first lets a controller settle on a member, since the reset state
is the registered plant's). Every size is ours; only the directions are sourced.

### The cascade absorbs a hidden constant

`hidden_members.py`: the kept cascade, as tuned on the registered plant, on members
that differ in one number it is not told.

| Member | Fuel | Against the registered plant | Glass | Fining, mean | Lowest | Spread |
|---|---|---|---|---|---|---|
| Registered | 46.39 | | 5.03 | 0.554 | 0.526 | 0.013 |
| Regenerators fouled, 0.80 to 0.72 a node | 47.89 | +3.2 % | 5.02 | 0.554 | 0.526 | 0.013 |
| Regenerators fouled, 0.80 to 0.64 a node | 49.68 | +7.1 % | 5.00 | 0.554 | 0.526 | 0.013 |
| Surface emissivity 0.7 | 51.83 | +11.7 % | 5.18 | 0.554 | 0.523 | 0.014 |
| Surface emissivity 0.9 | 42.03 | -9.4 % | 5.02 | 0.555 | 0.528 | 0.013 |
| Fining band 20 K higher | 48.50 | +4.5 % | 9.95 | 0.554 | 0.524 | 0.013 |
| Fining band 20 K lower | 44.71 | -3.6 % | 6.17 | 0.556 | 0.524 | 0.015 |
| Fining a quarter slower | 47.08 | +1.5 % | 9.79 | 0.554 | 0.521 | 0.014 |
| Crown loss up a quarter | 46.58 | +0.4 % | 5.03 | 0.555 | 0.526 | 0.013 |

- **The fining sits where the cascade puts it on every member,** with the same spread.
  Its outer loop integrates any constant away: it burns what the member needs and
  holds no margin for not knowing which member it is on.
- **A member that needs a hotter furnace loses twice the glass** through its blocks
  (the band 20 K higher, the slower fining), whoever fires it.
- **The canal is a member's own limit.** Three members have steps under the canal's
  target (13, 378 and 51, on one or two seeds each): the refiner's cooling loop, not
  the cascade, holds the canal.
- **So a hidden constant opens no room against this reference.** The second week of
  the registered plant also costs more than the first (51.42 against 49.52): the
  lost glass keeps rising after the week the task scores.

### A reference without quality feedback pays for the family

Plants read defects, which show nothing while the glass is above its pass mark, so
their crown setpoints come from tables with a safety factor [TR13]. `table_margin.py`:
the cascade's crown loop alone, its setpoint held some kelvin over the crown at which
the registered plant's fining sits at the cascade's setpoint (1544.8 C), on the
registered plant and on members whose fining band the batch's redox has moved by 20 K,
inside the 50 K the sources give for the fining onset (1430 to 1480 C [TR1]).

| Table setpoint | Registered | Band 20 K higher | Band 20 K lower |
|---|---|---|---|
| 10 K under | 258.1, glass lost on 9 seeds | 463.6, on all 10 | 49.96 |
| At the cascade's crown | 64.92, on 2 seeds | 429.8, on all 10 | 55.83 |
| 5 K over | 52.35 | 337.6, on all 10 | 60.45 |
| 10 K over | 54.48 | 196.7, on 7 seeds | 66.53 |
| 15 K over | 58.10 | 79.97, on 2 seeds | 74.58 |
| 20 K over | 64.13 | 60.32 | 84.75 |
| 25 K over | 72.31 | 67.52 | 97.34 |
| 30 K over | 83.62 | 77.76 | 113.43 |
| The cascade, for comparison | 51.42 | 58.45 | 50.88 |

- **A table needs a margin even on the plant it was made for.** With none, the walks
  take the fining under its mark on two seeds in ten.
- **To keep the mark on every member, the table must sit 20 K over.** At 10 K the
  member with the band 20 K higher loses glass on seven seeds in ten.
- **That margin is dear, and mostly in glass.** At 20 K over, the three members cost
  60.3, 64.1 and 84.7, against 58.5, 51.4 and 50.9 under the cascade: 3, 25 and 67 %
  more. The fuel is 0.4 to 3.6 % more; the rest is the plates a hotter furnace sheds,
  4 to 9 % of the pull against the cascade's 1.2 to 2.4 %.
- **These are the family's figures.** The sources give supervisory control 2 to 3 %
  of energy and up to about 8 % of yield against operators [S7]. The model shows
  gains of that size once its reference is what the plants had: a table and a safety
  factor, in place of a loop on a fining share no plant reads.
- **Why the task shows little room.** Its reference reads the fining share at the cold
  end as a clean number, above the mark as below (COLD_REVIEW.md, item 2). With that
  reading, the cascade is already the supervisory control the sources sell.

- **A table that knows its member is as good as the cascade.** Each member's best
  table that loses no glass costs 49.96 (10 K under, on the band 20 K lower), 52.35
  (5 K over, registered) and 60.32 (20 K over, on the band 20 K higher): 1.8 % under
  the cascade, 1.8 % over and 3.2 % over. What the cascade's fining loop buys is
  knowing the member.
- **Not knowing it costs 29 %.** The table safe for all three costs 69.7 on their
  plain average, against 54.2 for each member's own best (three members of equal
  weight, ours).

### What tells the member

`state_filter.py` again, 512 particles, with the fining reading left out of the
weights, and then the defects' reading too. Forecast error of the fining at 4, 8, 12
and 16 h:

| Readings weighed | 4 h | 8 h | 12 h | 16 h |
|---|---|---|---|---|
| All | 0.0027 | 0.0047 | 0.0070 | 0.0097 |
| Without the fining share | 0.0021 | 0.0045 | 0.0069 | 0.0095 |
| Without the fining share and the defects | 0.0025 | 0.0042 | 0.0065 | 0.0092 |

- **On a plant whose numbers are known, the couples place the fining.** The fining
  share's own reading adds nothing to the filter: fining is a time-temperature
  integral, and the temperatures are read well.
- **What the couples cannot tell is the member.** A moved fining band changes no
  temperature. Only the glass's quality shows it, and plants read that as defects,
  which appear near and under the mark.

### What this asks of a design

Three changes, each to what the task is, so each the owner's to rule:

- **The quality reading as plants have it:** a count of defects that says nothing
  while the glass is good, in place of a fining share read above the mark as below.
- **A reference from a table with a written safety rule,** since no loop can then hold
  a fining setpoint.
- **The numbers a furnace does not know drawn per stream,** inside sourced ranges: the
  fining onset has one (50 K); the others searched so far have none.

One thing stands in the way of an agent earning that room: a step under the mark
loses the whole pull, nine times its fuel. Probing toward the mark is then ruinous,
where a plant's quality falls off by degrees. The pass mark (0.5) and that cliff are
ours.

## The family run from a table: prototype (2026-11-24)

The owner's go of 2026-11-24: prototype, in study scripts with the plant untouched, a
quality reading that shows nothing while the glass is good, a table reference with a
written safety rule, and the fining band drawn per stream; then measure reference,
planner and ceiling.

**The prototype's terms** (`family.py`), each ours and for the owner to rule before any
of it enters the plant:

- **The family:** the fining band moved by -20, -10, 0, 10 and 20 K, equal weights,
  inside the 50 K the sources give for the fining onset [TR1].
- **The reference:** the kept cascade's crown loop alone, on a table setpoint. It reads
  the crown couple and nothing of the glass's quality.
- **The safety rule:** the lowest table setpoint, on a 5 K grid, at which no member
  loses a step on the ladder's ten seeds.
- **The start:** each member as its operators run it, a week on the walks' mean path
  under the family's table from the registered design state. The scored week starts
  there, on the episode's own walks and keys.
- **The reading an agent would get:** the defects per plate it has today, and in
  place of the fining share only how far the glass is under its pass mark, nothing
  above. No controller measured here reads it: the table reads the crown, and the
  ceiling and the planner read the state.

**The table.** Cost over the scored week, ten seeds:

| Table setpoint | Band -20 K | -10 K | 0 | +10 K | +20 K | Family |
|---|---|---|---|---|---|---|
| 15 K over | 60.10 | 54.46 | 52.37 | 51.93 | 63.88, glass lost on 1 seed | 56.55 |
| 20 K over | 66.33 | 58.27 | 54.63 | 53.73 | 54.25 | 57.44 |
| 25 K over | 78.82 | 64.06 | 58.32 | 56.40 | 57.91 | 63.10 |
| Each member's own best table | 47.42 | 47.54 | 49.23 | 50.84 | 54.25 | 49.86 |

- **The rule picks 20 K over.** At 15 K the member with the band 20 K higher loses
  1419 steps on one seed.
- **The margin is paid by the members that do not need it.** Under the family's table
  the member with the band 20 K lower sheds 4.8 % of its pull as plates (19.96 EUR per
  step) and the one it was sized for 1.3 %.
- **Not knowing the member costs 15 %.** The family's table costs 57.44 on the plain
  average, each member's own best table 49.86 (the two coolest members' best sits at
  the grid's low end, so theirs may be lower still).
- **Costs are lower than the second-week figures of 2026-11-23** because a start
  settled without walks has clean blocks, as the registered task's start has.

**The kept cascade cannot take over from the table.** Started from the table's state
with its crown setpoint where the table had it, the cascade loses glass on four
members of five: 36989, 34765, 23299 and 1393 steps of 50400, on all ten seeds for the
first three. Its fining starts 0.04 to 0.13 over its setpoint, its outer loop
integrates that through the 12 h dead time, and the fining then falls through the
mark. From the registered design state (0.03 over) it does not. So the reference as
tuned is safe near its setpoint only: a large change of where it should sit, as a
product change would be, needs a ramp or a limit on that integral.

Still running: the ceiling and the library's planner on the members at -20, 0 and
+20 K, three seeds each, from the same start (`family_line.py`; 18 runs, about 40
core-hours).

## What else the furnace lacks

Asked by the owner on 2026-11-15, in the order I would add them:

1. **Per-port firing.** What plants steer [S15]. Now that defects come from the hot
   blocks, where the heat goes matters.
2. **Batch redox.** It moves the fining range and the foam [TR1]. The batch varies today
   as smooth 12 h drifts of cullet, blanket, sulfate and pull, with sizes that are the
   owner's picks or ours; real changes come as steps, and redox is absent. The sulfate
   walk stands in for it and makes most of the bath's movement.
3. **The reversal as a real cycle.** The model averages the two half-cycles and the 40 s
   gap. An earlier measurement gave a 2 K ripple on the crown and under 0.02 K at the
   floor, so it matters to the fast loop and little to the glass.
4. **The cold side.** The glass leaving is still a 55 h average, so being late on the
   fining stays cheap until the mark.
5. **Outside air.** Fixed at 25 C. Measured at a held crown: 10 K warmer saves 0.32 % of
   fuel and moves the fining by 0.001 and the floor by 0.2 K. A daily cycle would be a
   small, honest disturbance; it is last on the list.

## What would make it a relevant task

The family's property is stated in a patent: operators keep a safety factor from the
optimum so as not to risk the glass (US6796147B2), and the saving is the lower setpoint,
2 to 4 % of fuel [S2; S3]. In this model the safety factor a good loop needs is 0.5 % of fuel. Something
that makes real margins large is missing. In order of evidence:

- **A. Fix and record it as it is.** Bugs 1 to 5, one margin rule on a lost-production
  limit, a ceiling that cannot cycle the crown. The honest room is then under 1 %, and
  by J5 the task is close to solved: core pool, off the flagship. A few days.
- **B. Hide the margin, as plants do.** Three sourced changes: the cold end shows defects
  when they occur and nothing above the pass mark (cold review item 2); the reference
  holds crown setpoints from tables with a safety factor set by a written commissioning
  rule; each stream draws the numbers a plant does not know about itself (foam tendency,
  fining band, couple offset, emissivity) from family ranges. Room becomes the safety
  factor: 2 to 4 % if the spreads are what plants' margins imply, and each spread needs
  a source or stays `ours`. An agent can earn it by inference: at a held crown, halving
  the sulfate moved the valve by 7 % within a day, well before the cold end had shown half of it. One to two
  weeks, mostly in the env and the ladder row.
- **C. Make moving cost what it costs.** In order of sourcing: defects released where
  the glass touches the tank blocks when the temperature there rises (Dunkl; Chmelar),
  carried with the glass as the fining share is; reboil where fined glass above about
  1400 C is heated; the crown's cycling charged against its life instead of a free
  override; then the second fining on cooling and the fining range moving with redox.
  Each direction and threshold has a source. Each rate would be `ours`, bracketed by what
  defects are worth (up to 8 % of yield). This is the route that gives the furnace a
  reason to move slowly, so that its inertia becomes the difficulty. It is unproven until
  a prototype shows the reference forced to slow down. The blanket's reach and the canal's
  draw stay open questions with nothing found.
- **D. More moves than one valve.** Per-port firing is what plants steer [S15]. Expect
  percent-level room: per-port crown loops are the classical answer.

My recommendation, after the prototype: A now, since the recorded 1.0114 should not be
quoted; then C, starting from the contact mechanism, which gave the furnace the
difficulty the owner describes. Before it goes into the plant it needs a rate from a
practitioner or a bracket from the yield figures, a source for how much foam varies,
and a ruling on which blocks release. The carried scalar will cost step time, and the
step is at 2.9 times the median (J6). B remains the fallback that keeps the plant as
it is.

## Decisions for the owner

1. The margin rule on a limit priced as lost production: two sigma of the kept rung (the
   reference then loses glass on half the seeds), or the smallest margin losing none on
   the nominal seeds.
2. Whether the zone spread should be the kept rung's on every task (bug 1).
3. What stops a ceiling from cycling the crown: a firing-rate limit from a source, a
   priced wear, or a tie-break toward the smoothest plan.
4. A, B, C or D.
5. On the new plant's ceiling: whether the certificate's active band goes from 1e-6 to
   1e-4, so that it reads the plans instead of returning 1; and whether the ceiling
   takes a third start per seed.
6. The reference's back-off again (decisions 1 and 2): a quarter of the new room is the
   distance the reference keeps from the fining mark.
7. The fair planner is in the library since 2026-11-16 (the owner's go), its spread
   read from the model. Left to rule: the normal error it assumes, the hold of the
   spread past the dead time and a knot, and 128 sampled paths, all ours.
8. After the ruling of 2026-11-23: which of I to V in "As a nearly saturated task",
   and whether the plant with its defects becomes the task on record. The owner picked
   I, II, III and IV that day, in that order of work. The merge still waits for his
   word.

## Sources seen today

- US6796147B2, Stein Heurtey, filed 1999 (seen once): operators' tables and safety
  factor; manual control is said to be most limited in transient phases, the changes in
  tonnage. https://patents.google.com/patent/US6796147
- Muller and Chmelar, Glass Worldwide, Sept-Oct 2012, pp. 52-53, as summarised by British
  Glass (seen once): float lines no longer run at stable pull rates, similar thicknesses
  and one colour or batch; customers ask for flexibility and just-in-time delivery.
  https://britglass.org.uk/knowledge-base/digital-library-and-information-services/advances-float-glass-furnace-control

## Sources seen on 2026-11-23

- [N1] Beerkens, Muysenberg, Barklage-Hilgefort, "Modeling of the aging of glass
  furnace regenerators", Glastech. Ber. 65 (1992) no. 9, 246-255, abstract (read
  twice): a model, said to agree with industrial furnaces.
  https://oa.tib.eu/renate/items/7dfcb0c7-cc75-43fa-aea6-e4dcec5b0d22/full
- [N2] IPCC AR4, WG3, section 7.4.5.3, citing Beerkens and van Limpt (2001): 4 to 10
  GJ/t of container glass, 5 to 8.5 GJ/t of flat glass (returned by two searches; the
  page refused a direct fetch).
  https://archive.ipcc.ch/publications_and_data/ar4/wg3/en/ch7s7-4-5-3.html
- [N3] 56.1 t of CO2 per TJ of natural gas, the IPCC 2006 default as reproduced in
  Commission Regulation (EU) 2018/2066, Annex VI (seen once, from a search result).
  https://www.legislation.gov.uk/eur/2018/2066/annex/VI/data.pdf
- [N4] Beerkens, van Limpt, Jacobs, "Energy efficiency benchmarking of glass furnaces",
  Glass Sci. Technol. 77 (2004) 47-57, abstract (read twice): a float furnace's energy
  use depends strongly on its size, pull and age; no float figure in the abstract.
  https://oa.tib.eu/renate/items/e9ec3ace-03db-4cf0-b650-3890c1731c23/full

## Reproducing

`PYTHONPATH=src python scripts/glass_room/probeN.py <out>`, each under a few minutes on
one server core set except probe 7 (about 25 min). `lib.py` rebuilds the kept cascade from
the tuning the ladder logged on 2026-11-13. Probe 1: steady map and design state. 2:
reference and ceiling traces. 3: crown held, steps and walks. 4: faster fining. 5: higher
pulls. 7: job changes, margins and plan reading. 8: where the delay sits. 9: inside one
pull-change transient. 10: how far the furnace moves against the sourced marks. `proto.py` and its
runs (`proto_check`, `proto_run`, `proto_bath`, `proto_bath_rise`): the prototype.

On the plant with its defects: `plant_runs.py`, `ladder_suite.py` and
`ladder_suite_bath.py` (the controllers of the prototype on the registered plant and on
the ladder's suite); `oracle_scale.py` (the gradient behind the oracle's `FIRST_STEP`);
and, from the ceiling run's output directory, `reference_split.py`, `plan_traces.py`,
`plan_mechanisms.py` and `plan_certificate.py` (the plans beside the cascade, one to
six minutes each on one server core). The ceiling itself is 25 chains of
`scripts/glass_slow_oracle.py solve`, each a design start and a restart, up to
100 minutes and 3 GB a solve. The MPC line is 25 runs of `mpc_line.py`, one per scenario
and seed, 0.8 to 3.7 hours and up to 3.2 GB each (the script at 7143497, when the
library's oracle without a seed was that planner). The fair line is 25 runs of
`fair_line.py` at 5108011, 2.8 to 3.6 hours and 1.5 GB each. Since then `mpc_line.py`
runs the library's planner, and `fair_line.py` hands it the measured table.
`rate_replay.py` (from the ceiling's and the library planner's output directories)
and `margin_scan.py` take two to three minutes each on one core.

