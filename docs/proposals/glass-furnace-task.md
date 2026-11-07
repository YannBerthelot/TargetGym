# Glass furnace task

Proposal for the owner, 2026-11-07. The table of properties was measured on the shipped plant
at `a9c8ea8`. The rebuilt plant now exists beside it, unregistered, in
`src/target_gym/glass_furnace/rebuilt.py`. Nothing registered changed, and nothing was
re-recorded.

**Result, plainly.** The task has room of 1.61 on the foresight ceiling, and that room is not
in the bath. Half the gap is the reversal, which the oracle times from a clock the agent also
reads, and the bath factors can all be switched off without shrinking it. See
[Results of the judging](#results-of-the-judging).

**Latest.** Attempt 3 starts from what makes a furnace hard to control, and scores the floor reading
instead of the crown. See [Attempt 3](#attempt-3-what-makes-a-furnace-hard-to-control).

## The outcome

The shipped plant has little room, and rescoring it cannot create more. The owner chose to
rebuild the furnace's bath and roof first. This page names the control properties a real
furnace has, measures them on the plant, and says which ones physics can supply.

The problem is the bath. It has no behaviour of its own. With fuel held the furnace does
take about 32 h to settle, as `PHYSICS.md` says, but with the crown held the melt follows in
about 0.7 h and the working end in about 11 h. On a real furnace something always happens in
the bath, and that is what makes the crown hard.

## Control properties

Measured on the shipped plant with no controller: fuel held, noise off, comparing like phases
of the reversal cycle. Each row is a number to re-measure on the rebuilt plant.

| Property | How it is measured | Shipped plant | Real furnace | Can physics supply it? |
|---|---|---|---|---|
| Delay against lag | Dead time over t63 of a fuel step | 2 to 3 min over 32 h, about 0.001 | Not found | Probably stays small. A slow roof reading raises the delay, not the slow mode |
| Gain spread | Fuel step at each operating point | 2 % over crown 1570 to 1605, 4 % with pull +10 %. The pull -10 % run was invalid | Not found | No. Radiation near 1860 K varies by about this much, so it is not a hard feature here |
| Slow mode | t63 against the crown's first response | 32 h against about 15 min | Glass residence about a day | Yes, already there |
| Hidden state seen in the reading | Kick a hidden state, read the crown against 0.36 K noise | Melt +5 K shows 3.3 K at 1 h, working end +5 K shows 0.8 K, batch +10 % shows 0.43 K | Glass cannot be read at all | Partly. See below |
| Lookalike upsets | Compare the crown's path after each upset | Gas +1 % gives 1.1, 3.3, 7.9, 13.5 K at 1, 6, 24, 96 h. Pull -2 % looks the same, reversed from +2 %: 0.8, 3.8, 9.4, 16.1 K | Gas and bath demand both unmeasured | Yes. See below |
| Upset speed | Disturbance rate times loop delay, against the 1 K floor | 1 K/h times 3 min, 0.05 K | Crown spread 2.2 K under operators | Not at these numbers. This is the main gap |
| Change during an episode | Drift of gain or heat demand within the episode | 0: every parameter is constant | Batch and cullet change, blanket moves | Yes, from sourced balances |

**Lookalike upsets are the real hidden-state problem.** Pull is measured, but the same shape
would come from a bath that takes more heat. From the crown and fuel alone, a burner running
on weaker gas and a bath taking more heat look alike. Both need more fuel, but the glass is
fine in one case and colder in the other.

**Upset speed is the gap physics may not close.** A loop that answers in minutes rejects
anything that takes hours. Bath events take hours. If the rebuilt plant keeps that ratio, the
crown stays easy for a PI at any fidelity, and the page must then say so plainly.

## Assumptions the ladder needs

The reference ladder is PI, steady-state feedforward, cascade on an available measurement,
then a decoupler or override. Each rung assumes something.

| Rung | Assumes | Real furnace | Evidence |
|---|---|---|---|
| PI on the crown | The upset is slow next to the loop | Holds in the shipped plant | Fixed target, 0.97 K |
| PI on the crown | Holding the crown holds the glass | Broken: the glass is the product and cannot be read | Melt moves 2.7 to 6 K with the crown held |
| Feedforward | The upset is measured | Broken for gas, batch and blanket, true for pull | Feedforward takes pull out, leaving 0.09 K |
| Cascade | A reading of the outer variable exists | Broken: only proxies exist | A bottom reading took the working end from 0.09 K to 1.60 K |
| Fixed gains | The plant stays the same | Broken by batch and cullet changes | Not measurable yet: constant here |

**Switch-off test.** One real factor removed at a time, shipped PI against the shipped oracle,
13.3 h, seed 100, default target walk, scored with the 1 K floor. One seed, so small
differences mean nothing.

| Removed | PI | Oracle | Gap, decades above the floor |
|---|---|---|---|
| Nothing | 1.35 K | 0.68 K | 0.26 |
| Firing gap at reversal | 1.08 K | 0.39 K | 0.07 |
| Pull noise | 1.17 K | 0.64 K | 0.13 |
| Fuel slew limit | 1.23 K | 0.67 K | 0.18 |
| Charger pulses | 1.29 K | 0.66 K | 0.22 |
| Reading lag | 1.30 K | 0.68 K | 0.23 |
| Sensor noise | 1.33 K | 0.68 K | 0.25 |
| Gas drift | 1.34 K | 0.66 K | 0.25 |

The firing gap and pull make the shipped gap. The slew limit may add a little. Charger pulses,
reading lag, sensor noise and gas drift don't change it, so they stay out of any claim about
what makes this task hard. Delay in a 2 minute reading is not a hard feature, and neither is
gas, in spite of the table above: the gas only matters once a hidden bath shares its shape.

The oracle knows the reversal clock and the coming targets, so this is a ceiling, not a claim
about what an agent can reach. Real non-stationarity can't be switched off here,
because the plant has none. That test waits for the rebuilt plant, and any bath factor that
fails it leaves the claim.

## Measured evidence

Shipped crown PI, sensor noise on, 12 seeds (100 to 111), 3 days, first hour dropped. Fixed
crown target. Pull steps come on one day in two with a random sign. "Bath" means hidden random
walks on the melting heat and the open melt area. Pull step and bath sizes are ours.

| Question | Crown PI | Fair rung | Perfect reading (no such sensor) |
|---|---|---|---|
| Working-end glass, 5 % pull steps | 0.72 K | 0.09 K, pull feedforward | 0.09 K |
| Same, 10 % pull steps | 1.24 K | 0.13 K | 0.11 K |
| Same, 5 % steps, bath 10 % and 5 %, 12 h | 0.67 K | 0.31 K | 0.13 K |
| Crown, quiet | 0.97 K | | |
| Crown, 20 min reading, bath 10 % and 5 %, 2 h | 2.14 K | 1.29 K, lead on the reading | 1.07 K |
| Melt margin, 5 % steps, bath, 12 h, in fuel | 0.67 % | 0.60 %, pull feedforward | 0.48 % |

- **The fair cascade leaves almost nothing.** Feedforward alone takes the working end far
  below the 1 K floor with no glass reading, and the crown already sits at the floor on a
  fixed target. So my first idea, scoring the glass over 2 to 3 days, fails on this plant.
- **A slow reading alone creates no room.** A simple lead closes most of a 20 minute lag.
- **Bath events alone create no room here.** A 10 % slow change in melting heat barely moves
  the crown, because the integral takes it.
- **No oracle ran on a glass objective.** A perfect-reading controller served as the ceiling.

**Refuted: a blanket that runs away.** An earlier draft of this page leaned on it. The balance
says otherwise: melting grows with the area covered, so a longer blanket melts faster and
shrinks back. The plant's blanket already is this balance at fixed thickness. A real runaway
would need a term no source gives, so it is not proposed.

Probe scripts: `glass-crown_probe.py`, `glass-room_probe.py` and `glass-props.py` in the group
notes folder, with the sourced tables in `glass-sources.md`.

## What real furnaces show

Sources are vendor decks and patents. Each figure was checked on a second pass unless marked.

| Fact | Source |
|---|---|
| Crown spread on one float furnace: 2.21 under operators, 1.2 under model predictive control; canal 0.92 and 0.32. Units not stated, presumably K | [Glassman 2019](https://glassmanevents.com/content-images/main/01-Gabriel-Noboa-Glassman-Sao-Paulo-05122019.pdf), slides 13 and 15 |
| Fuel saving about 2 to 3 %, from lowering the setpoint once the spread falls | [Schill](https://www.ksr.tul.cz/glassman/download/09_05_16-10-Schill.pdf), p. 7; Glassman 2019, slide 18 |
| Melter bottom thermocouples exist, on a 650 t/day float furnace. Pull wanders by +-2 % | [US11261117B2](https://patents.google.com/patent/US11261117B2/en) |
| Crown and bottom setpoints are tuned by hand, over weeks to months | [US6454562B1](https://patents.google.com/patent/US6454562B1/en) |
| A canal pyrometer views the glass surface and steers the tweel | [US5055122A](https://patents.google.com/patent/US5055122A/en) |
| Thermocouples embedded in refractory give temperature trends | [US11312648](https://patents.google.com/patent/US11312648) |
| Blanket drift makes the furnace's heat asymmetric, on a flat-glass furnace | [US4319904](https://patents.google.com/patent/US4319904) |
| Silica crown 450 mm thick, one vendor deck, seen once | [ICG TC11](https://icglass.org/pdf/technical-committees/tc11/2%20-%20Crowns%20design%20%26%20material%20choices.pdf) |
| Tank currents scaled in metres per hour, 250 t/day container tank 1.5 m deep, seen once | [Glass tank geometries paper](https://oa.tib.eu/renate/bitstreams/e16035ca-4bd6-48fe-b737-a7994cd2bddd/download) |

**Glass temperature cannot be read.** A sensor in the glass would be destroyed, and the sources
agree. Bottom couples read the tank floor and a pyrometer reads the surface, and no source gives
their accuracy, delay or cadence. The reference has no glass outer loop, and this is a
hidden-state task, as a real furnace is.

**Not found:** thermocouple depth in the crown, the reading's delay, the size and timing of float
pull changes, settling time after one, any K/h limit at operating temperature, or sized bath
events. The shipped PI's 0.97 K beats the 1.2 K of real model predictive control, which is the
plainest sign the model is too easy.

## The task

Only these levers move. Plant equations come from physics, with sourced numbers. No invented
term or unsourced gain goes in to reach a property.

| Lever | Proposal |
|---|---|
| Scored | The crown's hot face against its target, plus fuel, on the existing reward form, with the 1 K floor (owner's choice). The batch charge first proposed here was removed by the owner, since neither its limit nor its price could be sized |
| Observed | Roof reading, air preheat, fuel, reversal phase, target, pull. No glass or bath state. The batch-line reading went with the charge |
| The agent's role | The controller. With the batch reading gone, there is no operator trim to do |
| Operating range | As shipped until a reason appears |
| Disturbances | Cullet moves of +-10 points and blanket moves of +-20 %, each over 1 h (sizes the owner's), gas drift, pull at the sourced +-2 % wander, reversals, charger pulses |
| Episode | 3 days, 8640 steps of 30 s, so slow bath heat can return. 5.4 times today's episode cost |
| Reference controller | The ladder: PI on the reading, pull feedforward, a lead on the reading. No cascade rung, because no reading of the bath exists. No search |

### The rebuild

The owner chose this over the charter's slim rule. Every number gets a citation or the label
`ours`, and any term without a sourced equation stays out.

1. **Roof.** Heat conduction through the 450 mm brick in a few nodes, with the thermocouple at a
   depth. The depth is not sourced, so it is `ours` and tested at delays of 5, 10 and 20 minutes.
2. **Bath.** Energy balances on three zones, blanket, hot spot and throat side, each with a
   surface and a deep layer, linked by the pull and the return current. The return flow needs a
   sourced relation before it goes in. Without one, this item shrinks.
3. **Blanket.** The mass and heat balance as it stands, with coverage moved by the sourced
   drift. Thickness and drift size need sources or the owner.
4. **Batch.** Melting heat walks with the cullet ratio, about 2.5 % less energy per 10 % of
   cullet, already cited in `PHYSICS.md`, deviation D4.

## How it is judged, once

Sizes come from sources and the owner before any controller runs. Then, one attempt:

1. **Properties.** Re-measure the table above with no controller. Report each one, whether it
   moved or not.
2. **Realism.** Run the plain PI. It should land near the real 2.2 K crown spread. If it does
   not, report that. Do not tune the bath to reach it.
3. **Room.** Reference cost over the cost of a converged oracle given the true model, the true
   state and foresight, on the whole episode, as in charter draft 3. It is reported, not gated.
4. **Switch-off.** Remove each named factor from the rebuilt plant. A factor that does not shrink
   the gap leaves the claim.
5. **Cost.** Charter rule 7, no task over 3 times the median step.

The owner's reason: the foresight ceiling is a sound number, and claiming what an agent could
reach is not. So there is no threshold. Whether the task joins the flagship is chosen by reading
the room table, with a written reason. There is still no second try and no tuning on results.

**The risk, plainly.** My expectation is that upset speed stays the sticking point: at real
numbers bath events are slow next to the loop, so a PI may reject them all on a rich plant too.
Then the rebuild buys realism and little room, and the ceiling's gap would be mostly the reversal
and foresight. That number is still reported as it stands.

## Results of the judging

One attempt, on the rebuilt plant, in the page's order. Numbers and sources are in `PHYSICS.md`
under "Rebuilt plant". Scripts are `scripts/glass_rebuilt_*.py`.

**1. Properties.** Measured with no controller, noise off, like phases compared.

| Property | Shipped | Rebuilt | Moved? |
|---|---|---|---|
| Delay against lag | About 0.001 | 189 s over 40 h on the reading, about 0.001 | No |
| Gain spread | 2 % over crown, 4 % with pull +10 % | 2 % over crown 1570 to 1605, 4 % across pull -10 to +10 % | No |
| Slow mode | 32 h against about 15 min | 37 h on the hot face, 40 h on the reading, with t10 at 0.6 h and 1.5 h. 9 % of the settling comes in 30 min | A little, and now the start is slow too |
| Hidden state seen in the reading, at 1 h | Melt +5 K shows 3.3 K, working end 0.8 K, batch +10 % 0.43 K | Hot spot surface 0.40 K, whole bath 2.3 K, working end 0.36 K, batch -1.75 K | Yes. Single zones now hide below the 0.36 K noise |
| Lookalike upsets, at 1, 6, 24, 96 h | Gas +1 %: 1.1, 3.3, 7.9, 13.5 K. Pull -2 %: 0.8, 3.8, 9.4, 16.1 K | Gas +1 %: 1.6, 4.0, 9.0, 16.9 K. Pull -2 %: 1.4, 4.8, 11.3, 21.1 K. Cullet +5 points: 1.0, 3.4, 8.2, 15.5 K | Yes. Cullet is a third lookalike, and it is not measured |
| Upset speed | 1 K/h times 3 min, 0.05 K | Gas 1.4, pull 1.1, cullet 0.8, blanket 3.7 K/h. Blanket times 3 min is 0.19 K | Four times faster, and still far under the 1 K floor |
| Change during an episode | 0 | Melting heat and blanket thickness move hidden in the episode. How far the gain moves within one was not measured separately | Yes |

The blanket's move fades instead of holding: its reading goes 3.5, 2.0, 1.1, 0.3 K at 1, 6, 24,
96 h, so a slowly thinning blanket doesn't move the crown at steady state. One move of 10 points
of cullet shifts the batch line only about 0.1 m. Reversal swing is 38 K peak to peak on the hot
face and 2.2 K on the reading.

**2. Realism.** Plain PI by SIMC, bath untuned, 12 seeds: reading spread 1.56 K against the real
2.2 K. The ladder ranges 1.44 to 1.57 K, no trips. It lands near, and a little under.

| Rung | Cost | Hot face rms | Reading sd |
|---|---|---|---|
| PI | 50.22 | 7.02 K | 1.56 K |
| + pull feedforward | 50.20 | 7.01 K | 1.56 K |
| + lead on the reading, 10 min | 47.98 | 6.85 K | 1.44 K |

**3. Room.** Best rung over the foresight oracle, whole episode, same draws on both sides.

| Seed | Ladder | Oracle | Room |
|---|---|---|---|
| 100 | 47.47 | 29.41 | 1.61 |
| 101 | 48.13 | 29.77 | 1.62 |
| 102 | 46.93 | 29.04 | 1.62 |
| 103 | 46.58 | 29.05 | 1.60 |

**Room is 1.61**, spread 1.60 to 1.62. The oracle is converged: taking knots from 150 s to 60 s
to 30 s moves its cost 0.5 % and then 0.1 %, and a plan from a flat valve reaches 29.4376 at
150 s against 29.4367 from the ladder. Fuel is 1.0 of either cost, so nearly all of this is
tracking. The oracle's own hot face still has 5.3 K rms, and the ladder's 6.8 K, against a 1 K
floor. So most of the score is a swing no controller removes, and in decades above the floor the
gap is 0.11, smaller than the shipped plant's 0.26.

**4. Switch-off.** Seed 100, 150 s knots, gains fixed at the full plant's recipe, first 2 h
dropped on both sides. One seed, so small differences mean nothing. The full plant's room on
this window is 1.62.

| Removed | Ladder | Oracle | Gap in cost | Room | Claim |
|---|---|---|---|---|---|
| Nothing | 47.55 | 29.44 | 18.1 | 1.62 | |
| Firing gap at reversal | 11.30 | 2.31 | 9.0 | 4.90 | Stays. It halves the gap |
| Roof reading at 5 min | 42.50 | 29.44 | 13.1 | 1.44 | Stays |
| Sheath lag | 43.26 | 29.44 | 13.8 | 1.47 | Stays |
| Charger pulses | 47.80 | 31.95 | 15.9 | 1.50 | Unclear. The gap shrinks only because the oracle gets worse, which I can't explain |
| Roof reading at 20 min | 46.56 | 29.44 | 17.1 | 1.58 | Inside one seed's noise; the delay isn't monotonic here |
| Cullet moves | 46.94 | 29.12 | 17.8 | 1.61 | Leaves |
| Blanket moves | 47.35 | 29.41 | 17.9 | 1.61 | Leaves |
| Gas drift | 47.54 | 29.53 | 18.0 | 1.61 | Leaves |
| Pull wander | 47.71 | 29.55 | 18.2 | 1.61 | Leaves |
| Bath currents, well mixed | 48.25 | 29.47 | 18.8 | 1.64 | Leaves |
| Sensor noise | 49.66 | 29.44 | 20.2 | 1.69 | Leaves. The ladder gets worse without noise, unexplained |
| Fuel slew limit | 55.06 | 23.74 | 31.3 | 2.32 | Leaves. It keeps the oracle back more than the ladder |

Every bath factor leaves the claim. The firing gap and the reading's lag carry it. Without the
firing gap, room is larger but small in cost: the ladder is left at 3.2 K and the oracle at
1.1 K.

**How much is the clock.** A note, not a rung and not a claim about any agent. Adding one fixed
reversal pattern to the ladder, averaged from the oracle's valve on seed 100 and played on seeds
101 to 103, takes the cost from about 47.3 to 38.2, first 2 h dropped. That needs no foresight,
because the reversal phase is observed. So about half of the gap is timing a known clock, and the
room left after it is about 1.3. A phase-locked feedforward isn't on the ladder, so it doesn't
change the reference.

**5. Cost.** 729,529 steps/s at batch 256 against a median of 858,616, so 1.18 times the median,
under rule 7's 3. The shipped plant runs 1,739,554. The episode is 5.4 times longer as well.

**Does the task have room?** Yes on the charter's number, 1.61, and no in the sense this rebuild
was after. The risk named above came true: bath events are slow next to the loop, so the PI takes
them, and the gap is the reversal plus foresight. The rebuild did make the plant more real, with
slow starts, a third lookalike and hidden moves, but none of that reaches the score.

Three limits on these numbers:

- **The hot face's swing rests on a choice of mine.** The 38 K sits in a 15 mm cell, a thinner
  cell would swing more, and no source gives the figure. The owner's scored variable is the one
  most exposed to this. Room's size follows it.
- **Four seeds for room, one for switch-off,** and gains that weren't retuned per row, as the
  page asked.
- **A learner's `measured_reward` scores the reading,** while the true score is the hot face, so
  training and scoring differ by the reading's lag and offset.

## Open for the owner

- **Join the flagship?** By the charter this is read from the table. The table says the room is
  the reversal.
- **Sensor to score.** If the score moved from the hot face to the reading, the real 2.2 K would
  be measured on the same footing. That is a new attempt, not done here.
- **Batch.** The dimension was removed. The owner said to re-discuss it if the properties suffer.
  They didn't, but it was the one term that would have scored the bath directly.
- **Charter.** The rebuild passes rule 7. It isn't registered, so the task table is unchanged.

## Attempt 2: from the equations

The owner, after attempt 1: "Please re-visit the model from scratch, again, in order to achieve
the desired properties of the system, using only physical equations." Every term is a conservation
law or a sourced relation. There are no calibration factors, and a property physics doesn't
produce is reported as not produced. New sources are in `glass-sources-3a.md` and
`glass-sources-3b.md` in the session notes.

### Desired properties, set before any code

Each is measured with no controller, fuel held unless the row moves it. "Attempt 1" is the rebuilt
plant as judged above. The desired numbers come from the sources where one exists, and otherwise
from what makes the property matter against the 1 K floor and a 3 minute loop. These are aims, and
none of them sets a parameter.

| Property | Measure | Attempt 1 | Desired | Equation expected to supply it |
|---|---|---|---|---|
| Bath event to the reading | Time for the reading to reach half its peak after a surface upset, and the peak | Peak 0.4 K, about 1 h | Half peak within 15 min, peak over 1 K | Radiating skin: Beer's law in 2 to 23 mm of melt under the zone exchange |
| Upset speed | Largest reading rate from a physical event, against 20 K/h, the rate at which a 3 min loop leaves 1 K | 3.7 K/h, blanket | Over 20 K/h | Batch tip and foam moving over the skin, at their own speeds, no imposed ramp |
| Delay against lag | Time to the event's first visible move over its settling time | 0.001 for fuel | Over 0.05 for a bath event | Transport along the surface current, 7 to 25 m/h in the sources |
| Lookalike upsets | Worst gap between paths at 1, 6, 24, 96 h, each scaled to its 24 h value | Gas, pull and cullet within about 0.1 of each other | Under 0.2 for a pair of which one harms the glass | Mass and energy balances on the blanket and the bulk |
| Hidden state in the reading | Change in the melt at the throat against what the crown reading shows at 1 h | Whole bath +5 K shows 2.3 K | A glass change over 5 K that shows under 0.36 K | Hot glass under a resistive foam, and the deep return current |
| Gain change in an episode | Spread of fuel-to-reading gain across states one episode reaches | Not measured, 4 % across operating points | Over 20 % | Foam and blanket cover change what the crown sees |
| Wrong way | After a foam change at held fuel, sign of the reading against the heat entering the glass | Not possible, no foam | Reading up 1 K or more while heat into the glass falls | Foam's radiative resistance, from Fedorov and Pilon, gas from the sulfate balance |

### Published figures to check

Surface temperature range along the melter 170 to 205 K, vertical 300 to 340 K, crown profile in
W2000b (1710, 1850, 1890, 1915, 1910, 1805 K at 4 to 20.5 m), surface current 7 to 25 m/h, first
loop 600 % of pull, foam a few centimetres over about a third of the surface, foam's flux cut 50
to 60 %, gas emissivity about 0.2, specific energy 4 to 6 GJ/t, batch line 3 to 6.5 m, forming
temperature around 1100 C.

### Design

Each row is a conservation law or a sourced relation. Worked out from the equations, not from
`rebuilt.py`.

| Part | Equations | Source |
|---|---|---|
| Geometry | Melter 21.5 m, refiner 20.5 m, 11 m wide, six ports | W2000b |
| Radiation | Zones along the tank: crown, gas, load. Gray gas k = 0.08 /m. Direct exchange areas, reflections by the net radiation method, computed once | Hoogendoorn; Feischl eqs 5 to 11 |
| Gas | Energy balance per firing zone; each cross-fired port goes straight to its regenerator | Conservation |
| Crown | 1D conduction per bay, implicit, graded cells, the cell size set by a convergence check | Fourier, W2000a |
| Melt | Along-tank and depth finite volumes, graded to the skin. Implicit in depth, upwind | Conservation |
| Melt conduction | k = kc + kr, Rosseland, bands 0 to 2.8 and 2.8 to 5 um | Liu 2016 eqs 5 to 7 |
| Melt radiation | Net surface flux deposited by Beer's law in those bands, Planck band fractions | Liu 2016 |
| Flow | Long-tank momentum: hydrostatic in depth, d/dz (mu du/dz) = dp/dx with free top and no-slip floor, the pressure gradient set by the pull. No scale factor | Stokes at aspect ratio about 10 |
| Viscosity | VFT | DGG X, already in `rebuilt.py` |
| Batch | Advected at the charging speed, melted by radiation on top and the melt below; the melt's face sits at the 1450 K clear-melt point | UV1986, Pilon II |
| Foam | Thickness from Pilon's correlation on gas from the sulfate balance, opposing flux through a radiative resistance from the 20 % transmittance at 2 cm | JCIS2001, JNCS2002, Rudolph |

Kept from attempt 1 on their merits: the regenerator stacks, the reversal and charger clocks, the
viscosity table. Everything else is new.

**Known limits, stated now.** Long-tank flow has no vertical instability, so the roll cells under
the batch are absent (Pilon II: up to 20 % of the melt-to-batch heat). The sulfate rate has no
source, so the gas is the mass balance's amount over the band 1420 to 1500 C. No surface
emissivities were found, and they are labelled `ours`, as are firing shares, and refiner cooling
has no sourced duty, so the forming temperature may not be reached.

**Size budget.** 800 lines of plant, ceiling 900. Rule 7 is measured as soon as the melt runs.

### Results before any oracle

Measured on `physical.py`, unregistered. **Physics produced** the gain change and the wrong way.
**Partly:** bath event to the reading, upset speed, lookalikes. **Not produced:** delay against
lag, and hidden state as the row defined it. The valve stayed the action. No environment wrapper
is built yet, so nothing here has a reward or an observation, and the sensors are functions.

| Property | Desired | Attempt 1 | Attempt 2 | Verdict |
|---|---|---|---|---|
| Bath event to the reading | Half peak within 15 min, peak over 1 K | Peak 0.4 K, about 1 h | Whole bath +5 K: 12 min, 1.6 K. Blanket -20 %: 28 min, 3.2 K. Sulfate +50 %: 33 min, 17 K | Partly. Only an imposed kick makes 15 min; real events take about 30 min, with far larger peaks |
| Upset speed | Over 20 K/h | 3.7 K/h | Sulfate +50 %: 23 K/h. Blanket -20 %: 4.6. Cullet +10 points: 2.2. Gas +1 %: 2.1. Pull: 1.0 | Partly. It needs foam, and the sulfate size is mine |
| Delay against lag | Over 0.05 for a bath event | 0.001 for fuel | Fuel 3 min over 29 h, 0.002. Blanket first seen 12 min, then a long tail, about 0.03 | Not produced |
| Lookalike upsets | Under 0.2 for a pair where one harms the glass | About 0.1 | Pull -2 % against cullet +5 points: 0.14 at worst, and the cullet path has an inverse start. Gas +1 % against cullet +5: 0.29 to 24 h, 0.47 at 96 h | Partly. The pair under 0.2 includes pull, which is measured |
| Hidden state in the reading | A glass change over 5 K showing under 0.36 K | Whole bath +5 K shows 2.3 K | Cullet +5 points crosses zero on the reading at 2.6 h with glass at 0.5 K; skin +5 K shows 0.15 K. Larger: glass -4.5 K while the reading says +17 K | Not as defined. The foam case is stronger, and of the opposite sign |
| Gain change in an episode | Over 20 % | Not measured, 4 % across operating points | Settled gain 2219 to 4078 K per kg/s, 1.8 times. At 24 h, 1290 to 2286. At 1 h, 476 to 587. Driven by blanket length | Produced |
| Wrong way | Reading up 1 K while heat into the glass falls | Not possible | Sulfate +50 %, foam 1.05 to 1.49 cm: reading +4.7 K at 15 min, +10.6 K at 1 h, +17 K. Heat into the tank -5 % at 6 min, -2.4 % at 15 min, back to level by 1 h. Glass -1.4 K at 1 h, -4.5 K at 24 h | Produced |

**Inverse response, not asked for.** More cullet needs less melting heat, and the reading first
falls, -1.4 K at 1 h for +10 points, then rises past zero and on to +13 K at 24 h. A controller
reading the first hour adds fuel, and the glass is already heading up. The reason is in the
equations: the charger's batch keeps arriving at the old rate of surface cover until the blanket
shortens.

**The bottom and canal readings tell the cases apart.** After sulfate +50 % the crown reads +17 K
while the bottom reads -2.5 K and the canal -5.4 K at 24 h. After gas +1 % all three rise. They
respond in hours, and their swing at reversal is under 0.02 K. So a bottom or canal reading now
carries real information about the case the crown gets wrong.

**Rule 7.** 2.5 times slower than attempt 1's whole environment step, both timed together on a
loaded machine. Scaled to the 729,529 steps/s of the judged run, that is about 290,000 steps/s or
2.9 times the median, under the 3. An environment wrapper is not in that number, so the margin
is thin. If it fails, depth from 10 to 8 layers is the first thing to try, and grid checks below
say what that would cost. The full 2D flow survived, because the long-tank form needs no linear
solve.

**Published figures.**

| Figure | Source | Attempt 1 | Attempt 2 |
|---|---|---|---|
| Surface current | 7 to 25 m/h | Not resolved | 7 to 24 m/h, no calibration |
| First loop, back over pull | About 6 | 6.6, by `current_scale` | 7.2, from the equations |
| Conditioning return over pull | 2.6 | 2.6, set | 3.3 |
| Surface difference along the melter | 170 to 205 K | 40 K | 144 K from the blanket's edge, 312 K from the batch's foot |
| Surface to bottom | 300 to 340 K in one model, over 100 K in another | 40 to 75 K | 185 K |
| Hot spot | 13 m in a 21.5 m melter | Zone edge | 15.7 m |
| Crown profile, K | 1710, 1850, 1890, 1915, 1910, 1805 | One crown | 1618, 1715, 1799, 1845, 1868, 1880. Same rise, 35 to 70 K low, no fall at the end, since firing is uniform |
| Blanket's share of net heat | 80 to 90 % | 87 % | 82 % |
| Batch line | 3 to 6.5 m | 3.5 m | 3.1 m |
| Foam | A few cm, a third of the surface | None | 0.7 to 1.6 cm, 37 % of the surface |
| Specific energy | 4 to 6 GJ/t | 5.13 | 3.89, just under |
| Air, stack | 1200 to 1400, 400 to 600 C | 1229, 522 | 1246, 407 |
| Forming end | About 1100 C | 1475 C | 1296 C. Not reached: no cooling duty was found |
| Energy balance | Closed | 1.2 % | Under 0.5 % over 100 steps |

**Grid convergence, at crown 1590 C.**

| Doubled | Firing | Bottom | Canal | Foam peak | Hot face swing | Reading swing |
|---|---|---|---|---|---|---|
| Base | 0.4508 | 1306 | 1296 | 1.64 cm | 21.3 K | 2.00 K |
| Depth layers | 0.4487 | 1293 | 1288 | 1.85 cm | 21.4 K | 2.05 K |
| Columns | 0.4530 | 1316 | 1304 | 1.67 cm | 21.8 K | 2.14 K |
| Roof cells | 0.4508 | 1306 | 1296 | 1.64 cm | 21.8 K | 1.81 K |
| Step 15 s | 0.4510 | 1306 | 1296 | 1.64 cm | 33.4 K | 2.49 K |
| Step 7.5 s | 0.4509 | 1306 | 1296 | 1.63 cm | 45.7 K | 2.83 K |

Firing moves 0.5 %, and temperatures by 13 K at most. One thing doesn't converge: **the hot face's
swing grows as the step shrinks, 21, 33, 46 K.** The firing gap lasts 40 s, against a 30 s step.
The reading's swing is far steadier. So the hot face, as a score, depends on the step, and that
weighs on the owner's choice of score.

**Numerics I had to add, all stated.** The current's drive is filtered over three faces, because
a long-tank current driven by one face feeds a two-cell wiggle; this was a real instability, seen
as a saw-tooth in the current, and it went at this filter with the long waves unchanged. A
viscosity floor of 3 Pa s keeps a run-away finite until it trips. A cold start on the doubled
column grid still fails; that grid was run from the settled orbit.

**What is still `ours`.** Surface emissivities 0.8, no source found. Firing even across ports. Fining
gas formed in the top 10 cm. Bubble radius 8.3 mm, and foam goes as its power -2.6. No foam onset
velocity. Refiner crown taken as uninsulated. Glass leaves uniformly over depth. Couple depth 36 mm.
Combustion height 2.4 m from a side-fired chamber, not a float. Sulfate moves have no sourced size.
The roll cells under the batch are absent, and the sulfate rate is a band, not a rate.

**Gone from attempt 1:** `current_scale`, `gas_view`, the imposed 1 h ramp, the 15 mm cell's hold on
the swing. Kept on merit: the regenerator stacks' equations, the clocks, the viscosity table,
batch figures. 726 lines of plant against the 800 budget.

**Decision for the owner, before any oracle.** Whether to spend on room, and which score: the hot
face as decided, or crown plus a bottom or canal reading.

## Attempt 3: what makes a furnace hard to control

The owner, after attempt 2: "We want to model what makes a glass furnace hard to control in the
first place." Three things were named: partial observability that has an impact; delay, inertia and
momentum that make operators move slowly for fear of overshooting; and non-stationarity, where the
system moves in a stable phase with no action change, yet reaches a stable state under a constant
action if nothing perturbs it. This section defines those as tests. Nothing has been built for it.

**The main finding.** Attempt 2's plant already has nearly all of this. Its report said delay was
not produced, and that was the wrong conclusion: I measured delay at the crown, which radiation
reaches in seconds. No physics can put a delay there. At the floor of the tank, the same plant has
a dead time of about 5 h.

### The scored reading

A delay only exists between the valve and whatever is scored, so the score decides whether the task
can be hard at all. Measured on `physical.py` with `scripts/glass_floor_props.py`:

| Reading | Fuel +1 %, first move (1 %) | Fuel +1 %, t63 | Crown held, fitted dead time | Crown held, t63 |
|---|---|---|---|---|
| Crown couple | 0.1 h | 35 h | Held | Held |
| Canal end | 1.0 h | 42 h | 0 | 5.2 h |
| Floor, bay 3 | 2.6 h | 45 h | 2.0 h | 6.7 h |
| Floor, end of the melter | 3.7 h | 46 h | 3.7 h | 10.6 h |

**Proposal: score the floor thermocouple at the end of the melter, against its target, plus fuel.**
It is a real instrument, not the true glass temperature the owner ruled out, and the sources put
bottom couples along the tank. The crown stays observed, as a fast and sometimes misleading hint.
The canal is not a good choice: in this plant it has no dead time, because the surface current
carries heat straight to it, and its level is 200 K too hot.

### The properties

Each row is one test. "Attempt 2" means the floor reading above on the plant as it stands. Aims come
from the sources in `glass-sources-4.md` in the session notes. They are aims, and none of them sets
a parameter. The step and pulse were measured before the aims were written, so B1 to B3 are not
blind tests; the rest were set before their runs. The walks in C2, C3 and A3 have standard
deviations of 1 % on gas over 24 h, and 5 points on cullet, 10 % on blanket and 17 % on sulfate over
12 h. These sizes are `ours`, inside the owner's +-10 points and +-20 %.

**A. Partial observability has an impact**

| Property | Test | Aim | Attempt 2 | Met |
|---|---|---|---|---|
| A1. Readings can mislead | Two real upsets where the fast readings agree in sign and the right fuel move does not | One such pair | Foam, and gas +6.2 %: couple +12.2 and +12.1 K at 2 h, air +4.5 and +7.0, stack +0.5 and +1.5. Floor at 24 h -4.4 and +22 K | Yes. The 2.5 K gap on air is a clue a good estimator could use |
| A2. The truth arrives late | Time until the floor shows which way to move, past 0.36 K | Over 2 h | 2.9 h after foam, 3.1 h after gas | Yes |
| A3. Seeing would pay | Same PI on the same walks, told the hidden upsets through settled gains only | Error at least 2 times smaller | 2.99 K to 0.99 K rms, 3.0 times. If cullet counts as known from the recipe, 2.05 K to 0.99 K, 2.1 times | Yes, at sizes of ours |

**B. Delay, inertia, momentum**

| Property | Test | Aim | Attempt 2 | Met |
|---|---|---|---|---|
| B1. Dead time | Fuel step, time before the floor moves | Over 2 h. Sources say 1 to 16 h, and about 5 h | 3.7 h to the first 1 %, 4.9 h fitted | Yes |
| B2. Slow to finish | Same step, t63 | Over 10 h. Source: "ten-odd hours" | 46 h with the valve held, 10.6 h with the crown held | Yes |
| B3. Keeps going after you stop | Fuel +4 % for 4 h, then back | Moves on for over 2 h, and ends over 2 times what it showed at the stop. Aim is `ours` | Peaks 8.8 h after the fuel came back, at 3.2 K, against 0.5 K at that moment | Yes |
| B4. Haste is punished | Target +5 K with a PI from the step test, then at 3 and 6 times its gain | Under 10 % overshoot, then over 30 %. A real PID cycled for over 8 h after a pull change | 6 %, then 61 %, then 312 % with no settling in 14 days | Yes |
| B5. Patience costs too | Time for the safe PI to stay within +-1 K | Over 12 h. Sources: 4 to 6 h in a forehearth, and a melter is slower | 14.1 h | Yes |

**C. Non-stationarity**

| Property | Test | Aim | Attempt 2 | Met |
|---|---|---|---|---|
| C1. Stable if left alone | Valve held, nothing else moves, two starts 40 K apart | Same steady state, within 0.1 K | Same to 0.001 K in double precision | Yes, with two notes below |
| C2. Moves on its own | Valve held, walks on | Leaves +-3 K within 48 h on most seeds | 5 of 6 seeds by 42 h. 8.2 K rms, worst 26 K | Yes, at sizes of ours. The +-3 K is from one vendor slide, seen only as a snippet |
| C3. Feedback can't clear it | Safe PI on the same walks | Worst error still outside +-3 K | 2.99 K rms, worst 7.0 K. A cascade through the crown gives 2.89 K rms, worst 9.1 K | Yes |
| C4. The plant itself shifts | Fuel-to-floor gain and dead time across operating states | Over 1.5 times. Source: production rate changes "gain and lag time" | Gain 1789 to 2679 K per kg/s, 1.5 times. Dead time 4.2 to 7.1 h | Just. Nearly all from pull +-5 %, beyond the sourced +-2 % wander. Cullet and blanket hardly move it |

**What these rows found.**

- **The crown cascade barely helps.** It is the rung real plants would reach for, and it gives 2.89 K
  against 2.99 K. So a fast reading does not stand in for the bath.
- **Gas and cullet carry the cost of hiding, not foam.** Told cullet alone, 2.05 K. Told gas alone,
  2.30 K. Told sulfate alone, no change. Foam is the striking case in A1, but at these sizes it is
  not what hurts.
- **None of this is room.** The told controller is not an oracle, and no oracle has run.

### What is still not right

1. **The drifts are pushed in by hand.** They are not part of a plant yet, and their sizes are mine.
   No source gives a typical gas swing in percent. Gas changes "weekly and sometimes even daily".
2. **The floor reading is the last melt cell.** A real couple sits in the floor block, so conduction
   through it would add a real lag that this plant lacks.
3. **C1, note 1: single precision stalls.** In float32 the same two starts end 0.36 K apart with no
   drift, since a 30 s step's change is smaller than the rounding. A longer step or double precision
   fixes it.
4. **C1, note 2: the reversal is not a stable state.** Under a constant valve it leaves a 2 K ripple on
   the couple and under 0.02 K on the floor. The owner's rule may need it averaged out.
5. **A 30 s step spends its budget on what the score can't see.** A seven-day episode is 20,160 steps.
6. **C4 is the weakest row,** and redox may be the missing physics: iron changes how far heat gets
   into the melt. The sources disagree on the direction of the effect on the floor, and none gives a
   size, so it stays out unless sources are found.

### What attempt 3 would be

The melt, flow, foam, blanket and radiation equations produced the rows above, so they stay on
merit. Starting from scratch would mean rewriting the part that works. What changes is the task
and the time scale.

| Change | Why | Rows |
|---|---|---|
| Score the floor at the melter's end | Puts the delay in the task | B1 to B5 |
| Step of about 5 min, regenerators averaged over a reversal | True steady state, no float32 stall, far fewer steps | C1 |
| Floor block conduction in the reading | The real instrument is slower than the melt | B1, A2 |
| Drifts as part of the plant: gas, cullet, sulfate, pull | Non-stationarity from named sources, with sourced sizes where they exist | C2, C3 |
| Environment: crown, floors, air, stack, fuel, pull observed. No upsets | Keeps A1 to A3 true | A1 to A3 |

Then re-measure this table, run the switch-off test, and stop before any oracle.

### The score, adopted

The owner chose this on 2026-11-07: each reading does the job it does in a plant, instead of a
weighted sum of three errors. Rewards are money per step, as HVAC's already are.

| Reading | Job | In the reward |
|---|---|---|
| Canal | The product | Glass made while the canal is below its target is lost to cullet, at its price. Nothing above the target: the refiner cools it |
| Fuel | The cost | Its price per kg. Overheating is paid for here, so it isn't charged twice |
| Crown | The roof's limit | A trip above the silica crown's limit |
| Floor | Early warning | Observed, not scored, unless the canal turns out too fast |

The refiner's cooling is an automatic local loop, as in plants, holding the canal 1 K above its
target (`ours`, a loop's accuracy). The canal falls below target only once the cooling has nothing
left to take out. **Margin** is what the cooling takes out, in kelvin of the glass leaving: zero
means the next shortfall is lost glass. The agent sees the cooling effort, as an operator would.

The twelve tests stay as written, read on the margin instead of the floor. A3 and C3 are also
reported in money: the fuel a controller spends on margin because it can't see ahead.

**Needs sources before it is built:** the refiner's cooling capacity, the price of glass lost to
cullet, the crown's limit, a gas price, the size of gas quality swings, and the floor block's
thickness and conductivity. Anything not found is labelled `ours` or left to the owner.

### Build plan and budget

| Part | Change | Budget |
|---|---|---|
| Time step | 180 s. Regenerators averaged over the reversal: each node takes half its flue duty and half its air duty, the first-order average of the same equations. The firing gap goes in as its mean | Plant ceiling 850 lines, from 726 |
| Refiner | A cooling sink on the refiner's glass, set by the local loop, up to its capacity | In the plant |
| Canal | The glass leaving, flow-weighted, through the canal's own short lag | In the plant |
| Floor | Refractory cells under the melt in the same depth solve, the couple at a depth in them | In the plant |
| Drifts | Gas quality, cullet, blanket, sulfate and pull as random walks fed to the plant | Environment, 250 lines |
| Environment | Reset, step, reward terms, observation. Not registered | Environment |
| Tests | From the equations, before the code, red first | 350 lines |

The step is set by the surface current: 24 m/h over a 1.8 m column allows about 270 s. Stops: if
energy doesn't close within 0.5 %, if the step cost breaks rule 7, and before any oracle.

### Build so far, before any property table

Built and tested: the averaged reversal, the refiner's cooling and its loop, the fining share,
the environment and its money reward. Nine deliberate breaks of the new code each turned the new
tests red. Owner's picks on 2026-11-07: fining carried to the canal guards quality, since the
canal alone would pay for unrefined glass (at 16 % less fuel the canal still reached 1100 C while
the melt peaked about 100 K under the fining band); lost glass at 600 EUR/t, `ours`.

Changed on the way, each for a stated reason:

- **Step 120 s, not 180.** At 180 s the surface current's Courant number reached 1 near the batch
  tip when firing hard. Day-mean currents stay at 16 to 22 m/h, inside the published 7 to 25.
- **No floor block.** No conductivity or heat capacity for AZS paving was found, and bottom couples
  sit in pockets with platinum thimbles touching the glass. The floor reading stays the glass at
  the floor.
- **Cooling capacity `ours`.** No source; set to twice the design's need.

**The design point:** fuel 0.561 kg/s with 6.05 MW of refiner cooling holds the crown's face at
1589 C in bay 3 and the canal at 1103 C, drifting 0.001 K over 5 days. The refiner loop, by its
recipe: 4.45e5 W/K, integral 2.1 h. Specific energy 4.7 GJ/t, inside 4 to 6.

**What it shows, plainly:**

| Finding | Number | Published |
|---|---|---|
| Cooling to hold the canal | 6.05 MW, 22 % of the firing | Not found |
| Forward flow into the refiner over pull | 4.3, and 3.05 with no cooling | 1.2 to 2.5, US4798616 |
| Hottest crown face at the design point | 1633 C, with firing even across ports (`ours`) | Silica degrades above 1620 C. W2000b's own modelled profile peaks at 1642 C |
| Fining share of the glass leaving | 0.52 | No pass mark found |
| Fuel +1 %: margin's dead time and t63 | 0.6 h and 5.5 h, little coast | The melter's 1 to 16 h |
| Fuel +1 %: fining's t63, and its size | 24 h, +0.026 | |

The refiner has no waist, so most of its cooling falls on glass going back to the melter, which
is why the cooling is so large and why the canal answers the cooling within minutes. The delay
then sits in the fining, not in the canal. Measuring the twelve tests on this refiner would
measure the missing waist.

### Waist and firing taper (owner's pick)

**Waist.** From a 1300 t/d float furnace (Materials 17 (2024) 4989, Table 1): waist 7.5 m long and
4.5 m wide on a 14 m melter, working end 10 m wide. Scaled to this 11 m melter: 3.5 m waist,
7.9 m working end. In the long-tank flow a narrower face carries the same exchange per metre of
width, so less in total. No side-wall drag: at 3.5 m wide and 1.15 m deep it would add about a
tenth.

**Firing taper, `ours`.** No per-port split was found. Ports 1 to 4 even, then 0.7 and 0.4: the
least taper of five tried that puts the crown's peak in bay 3, W2000b's 13 m hot spot.

**The design point:** fuel 0.514 kg/s, refiner cooling 5.25 MW, crown face 1590 C in bay 3, canal
1101 C, specific energy 4.33 GJ/t. Refiner loop by its recipe: 3.13e5 W/K, integral 2.1 h.

| Figure | Before the waist | With it | Published |
|---|---|---|---|
| Forward flow into the refiner, over pull | 4.3 | 3.3 | 1.2 to 2.5, US4798616 |
| Refiner cooling, share of firing | 22 % | 21 % | Not found |
| Hottest crown face | 1633 C | 1594 C | Silica degrades above 1620 C |
| Fastest day-mean current | 20 m/h | 41 m/h, in the waist | 7 to 25 m/h in melters |
| Fining share leaving | 0.52 | 0.47 | No pass mark found |

**Fuel +1 %, refiner loop working, no walks:**

| Reading | Settled | First move | t63 | Coast after a 4 h pulse |
|---|---|---|---|---|
| Margin (the cooling's work) | +22 K | 0.2 h, fitted dead time 1.1 h | 8 h | Peaks 0.9 h after |
| Fining share | +0.021 | 1.1 h | 20 h | Peaks 1.4 h after |
| Floor, melter's end | +3.0 K | 0.4 h, fitted 0.7 h | 7 h | Peaks 0.9 h after |
| Crown couple | +4.9 K | 0.1 h | 5.5 h | None |

**Reading this.** The waist cut the return flow by a quarter but not the cooling: the refiner's
cooling itself steepens the gradient that drives the exchange. With 5.25 MW of margin the canal
can't fall below target unless the fuel is cut by far more than any walk moves it, so in the score
the canal stays in place and the fining pass mark is what binds. The fining is slow, 20 h, but
moves little with fuel, and the delays measured on this plant sit near 1 h, under the 2 h aim.
