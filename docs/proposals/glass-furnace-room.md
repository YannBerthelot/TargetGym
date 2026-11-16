# Glass furnace: why there is no room, and what would give it some

A study of `glass_furnace-v6` as it stands on `rec/charter-records` (4be0ab9), done on
2026-11-15 on branch `study/glass-room`. Nothing in the task was changed. Every number
below was measured on the registered plant with the scripts in `scripts/glass_room/`;
their outputs are in `study/glass_room/out/`. Costs are EUR per 120 s step.

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
  plant. Every recorded row for the task is stale.
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
- **The glass is the blocks, held by foresight.** Reboil costs the plans 0.01 EUR per
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
  answer it. The loop on the bath's true mean gave 1.7 %.

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
- **What it says about the room.** The true state alone does not keep the glass: the
  plates lost to the blocks are the reference's, a tenth of that gap closed. That part
  of the room is foresight of the walks, or a faster answer to them than a plan a day.
  It agrees with the loop on the bath's true mean (1.7 %). How much a competent planner
  without foresight reaches is still not measured, because this one is not competent.
- **The library ships this line.** `make_oracle("glass_furnace")` and the runners build
  the oracle without a seed, so the task's oracle row would be this one.
- **A fair line, which I would build next.** The same planner with four changes, none
  of which gives it the future: each plan starts from the last plan's remainder; a new
  plan every knot (4 h), so the lead it cannot correct is 16 h and not 36 h; a margin on
  the fining limit of two sigma of the forecast error at that lead, read from this run
  (0.01 to 0.02), which is the charter's own rule for a zone; and the limits softened
  when no valve can meet them.

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
7. The MPC line: whether to build the fair planner without foresight (a start from the
   last plan, a plan every 4 h, a two-sigma margin from the measured forecast error,
   soft limits when none can be met), and whether the library's oracle without a seed
   becomes that planner for this task.

## Sources seen today

- US6796147B2, Stein Heurtey, filed 1999 (seen once): operators' tables and safety
  factor; manual control is said to be most limited in transient phases, the changes in
  tonnage. https://patents.google.com/patent/US6796147
- Muller and Chmelar, Glass Worldwide, Sept-Oct 2012, pp. 52-53, as summarised by British
  Glass (seen once): float lines no longer run at stable pull rates, similar thicknesses
  and one colour or batch; customers ask for flexibility and just-in-time delivery.
  https://britglass.org.uk/knowledge-base/digital-library-and-information-services/advances-float-glass-furnace-control

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
and seed, 0.8 to 3.7 hours and up to 3.2 GB each.

