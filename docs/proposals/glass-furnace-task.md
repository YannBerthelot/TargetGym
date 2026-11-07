# Glass furnace task

Proposal for the owner, 2026-11-07. Measured at `a9c8ea8`. Nothing here is built, and no
plant, reward or expert changed.

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

The oracle knows the reversal clock and the coming targets, so part of this 0.26 is preview,
which does not count as reachable room. Real non-stationarity can't be switched off here,
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
| Scored | Crown against its target, plus fuel, on the existing reward form, with the 1 K floor. Plus a charge for unmelted batch past a limit (owner's decision), the one term that scores the bath directly. Vendors tie excess batch at the throat to defects, seen once. The limit and its price have no source yet, so they are `ours` until sized |
| Observed | Roof reading, air preheat, fuel, reversal phase, target, pull. A slow, coarse batch-line reading, as an operator has. No glass or bath state |
| The agent's role | Operator plus controller: operators watch the batch line and trim the controller |
| Operating range | As shipped until a reason appears |
| Disturbances | Batch and cullet changes and blanket moves (the owner's two main upsets), gas drift, pull at the sourced +-2 % wander, reversals, charger pulses |
| Episode | 3 days, 8640 steps of 30 s, so slow bath heat can return. 5.4 times today's episode cost |
| Reference controller | The ladder: PI on the reading, pull feedforward, a lead on the reading, and a slow operator-style crown trim from the batch-line reading, since the agent has it. No glass outer loop, no search |

The batch-line reading's pace has no source, so it is labelled `ours` and tested at about hourly
and each shift. It may let the reference close the room. That risk comes with giving the agent
what an operator has.

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
3. **Room.** Compare the best ladder rung with a controller given the true state and no preview.
   Room counts only if the gap is at least 0.3 decades above the 1 K floor.
4. **Switch-off.** Remove each named factor from the rebuilt plant. A factor that does not shrink
   the gap leaves the claim.
5. **Cost.** Charter rule 7, no task over 3 times the median step.

If it fails on room, the task stays as a solved task outside the flagship, with no second try.

**The risk, plainly.** My expectation is that upset speed stays the sticking point: at real
numbers bath events are slow next to the loop, so a PI may reject them all on a rich plant too.
Then the rebuild buys realism and not room, and the honest reachable room is the reversal and the
hidden-state problem of keeping the glass right with no reading.

## Open for the owner

- **Sizes.** How far and how fast do blanket moves and batch changes go? No source gives them.
- **Batch limit.** Where is the limit, and what does crossing it cost? Charging it is decided,
  but neither number has a source.
- **Charter.** The rebuild needs its row in the task table changed, and a plant of this size must
  still pass rule 7.
