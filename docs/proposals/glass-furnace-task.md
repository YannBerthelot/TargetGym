# Glass furnace task

Proposal for the owner, 2026-11-07. The table of properties was measured on the shipped plant
at `a9c8ea8`. The rebuilt plant now exists beside it, unregistered, in
`src/target_gym/glass_furnace/rebuilt.py`. Nothing registered changed, and nothing was
re-recorded.

**Result, plainly.** The task has room of 1.61 on the foresight ceiling, and that room is not
in the bath. Half the gap is the reversal, which the oracle times from a clock the agent also
reads, and the bath factors can all be switched off without shrinking it. See
[Results of the judging](#results-of-the-judging).

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
