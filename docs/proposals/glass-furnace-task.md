# Glass furnace task

Proposal for the owner, 2026-11-07. Measured at `a9c8ea8`. Nothing here is built, and no
plant, reward or expert changed.

## The outcome

The shipped plant has no real room, and rescoring it cannot create any. The owner chose to
rebuild the furnace's bath and roof first. This page says what the rebuilt task should be
and how to tell, once, whether it has room.

The bath is why. With the crown held, the melt follows it in about 0.7 h, not the 30 h
`PHYSICS.md` states, and the working end follows in about 11 h. Nothing happens in the bath
on its own. On a real furnace something always does, and that is what makes the crown hard.

## Measured evidence

Shipped crown PI, sensor noise on, 12 seeds (100 to 111), 3 days, first hour dropped. Pull
steps come on one day in two with a random sign. "Bath" means hidden random walks on the
melting heat and the open melt area. Pull step and bath sizes are ours.

| Question | Crown PI | Fair rung | Perfect reading (no such sensor) |
|---|---|---|---|
| Working-end glass, 5 % pull steps | 0.72 K | 0.09 K, pull feedforward | 0.09 K |
| Same, 10 % pull steps | 1.24 K | 0.13 K | 0.11 K |
| Same, 5 % steps, bath 10 % and 5 %, 12 h | 0.67 K | 0.31 K | 0.13 K |
| Crown, quiet | 0.97 K | | |
| Crown, 20 min reading, bath 10 % and 5 %, 2 h | 2.14 K | 1.29 K, lead on the reading | 1.07 K |
| Melt margin, 5 % steps, bath, 12 h, in fuel | 0.67 % | 0.60 %, pull feedforward | 0.48 % |

- **The fair cascade leaves almost nothing.** Feedforward alone takes the working end far
  below the 1 K floor, with no glass reading at all. The crown already sits at the floor.
- **The outer loop has nothing to do.** It was tuned by SIMC on an 11 h, 1.1 K/K fit. Pull
  acts on the working end with the same 11 h, so a static feedforward fits it. The bath
  barely reaches the working end.
- **A slow reading alone creates no room.** Without a lead, a 20 minute lag looks like
  room. A simple lead closes most of it.
- **Bath events alone create no room.** A 10 % slow change in melting heat barely moves
  the crown, because the PI's integral takes it. The blanket also cancels changes in its own
  shielding, within about an hour.
- **A proxy can hurt.** Steering by the melt, as a bottom reading would, took the working end
  from 0.09 K to 1.60 K. Pull moves the melt and the working end in opposite directions.

No oracle was run on a glass objective, since the numbers above already leave nothing to
bound. The recorded crown oracle, about 1.8x over the old expert on 3 seeds, plans to the
episode's end, so that gap is partly clairvoyance. The crown's own room sits under the floor:
0.74 K of the 0.98 K is the reversal gap, in a 13 h run on seeds 100 to 105.

Probe scripts: `glass-crown_probe.py` and `glass-room_probe.py` in the group notes folder, with the sourced tables in `glass-sources.md`.

## What real furnaces show

Sources are vendor decks and patents, and each figure was checked on a second pass.

| Fact | Source |
|---|---|
| Crown spread on one float furnace: 2.21 under operators, 1.2 under model predictive control; canal 0.92 and 0.32. Units not stated, presumably K | [Glassman 2019](https://glassmanevents.com/content-images/main/01-Gabriel-Noboa-Glassman-Sao-Paulo-05122019.pdf), slides 13 and 15 |
| Fuel saving about 2 to 3 %, from lowering the setpoint once the spread falls | [Schill](https://www.ksr.tul.cz/glassman/download/09_05_16-10-Schill.pdf), p. 7; Glassman 2019, slide 18 |
| Melter bottom thermocouples exist, on a 650 t/day float furnace. Pull wanders by +-2 % | [US11261117B2](https://patents.google.com/patent/US11261117B2/en) |
| Crown and bottom setpoints are tuned by hand, over weeks to months | [US6454562B1](https://patents.google.com/patent/US6454562B1/en) |
| A canal pyrometer views the glass surface and steers the tweel | [US5055122A](https://patents.google.com/patent/US5055122A/en) |
| Thermocouples embedded in refractory give temperature trends | [US11312648](https://patents.google.com/patent/US11312648) |
| Blanket drift makes the furnace's heat asymmetric, on a flat-glass furnace | [US4319904](https://patents.google.com/patent/US4319904) |

**Glass temperature cannot be read.** The owner is clear that a sensor in the glass would be
destroyed, and the sources agree. Bottom couples read the tank floor, a pyrometer reads the
surface, and no source gives their accuracy, delay or cadence. So the reference has no glass
outer loop, and this is a hidden-state task, as a real furnace is.

**Not found:** thermocouple depth in the crown, the reading's delay, the size and timing of
float pull changes, settling time after one, any K/h limit at operating temperature, or
sized bath events. The shipped PI's 0.97 K beats the 1.2 K of real model predictive control,
which is the plainest sign that the model is too easy.

## The task

| Item | Proposal |
|---|---|
| Scored | Crown temperature against its target, plus fuel, on the existing reward form, with the 1 K floor. Plus a charge when unmelted batch passes a blanket limit, the real cost of a blanket that runs away |
| Observed | The roof reading, air preheat, fuel, reversal phase, target, pull. No glass, blanket or bath state. A camera reading of the batch line is an open choice, below |
| Disturbances | Blanket moves, batch and cullet changes (the owner's two main upsets), gas drift, pull at the sourced +-2 % wander, charger pulses, reversals |
| Episode | 3 days, 8640 steps of 30 s: long enough for slow bath heat to come back. 5.4 times today's episode cost |
| Reference controller | The charter's ladder: PI on the reading, pull feedforward, a lead on the reading, override on the blanket limit. No glass outer loop, no search |

### The rebuild

This goes against the charter's slim rule, and the owner has chosen it. It should stay as
small as the problem allows, and every number gets a citation or the label `ours`.

1. **Roof.** Three nodes through the brick, with the thermocouple at a depth. The delay is
   not sourced, so it is `ours` and is tested at 5, 10 and 20 minutes.
2. **Bath.** Three zones along the tank, blanket, hot spot and throat side, each with a
   surface and a deep layer. Heat moves between zones by a convection flow that grows with
   their temperature difference. This is the part with a life of its own.
3. **Blanket.** A length along the tank, in place of one lump of mass. A colder melt lengthens
   it, and a longer blanket shields more glass, so it can run away. This is a real instability
   and the plant has none today.
4. **Batch.** Melting heat and moisture walk with the cullet ratio, about 2.5 % less energy
   per 10 % of cullet, already cited in `PHYSICS.md`, deviation D4.

## How it is judged, once

Sizes come from sources and the owner, before any controller runs. Then, one attempt:

1. **Realism.** Run the plain PI. It should land near the real 2.2 K crown spread. If it does
   not, report that. Do not tune the bath to reach it.
2. **Room.** Compare the best ladder rung with a controller given the true state and no
   preview. Room counts only if that gap is at least 0.3 decades above the 1 K floor. Preview
   does not count.
3. **Cost.** Charter rule 7, no task over 3 times the median step.

If it fails on room, the task stays as a solved task outside the flagship, with no second try.

**The risk, plainly.** The crown loop answers in minutes and bath events take hours, so a PI
may still reject them all, on a rich plant too. Then the whole rebuild buys realism but no
room. The runaway blanket is the best hope, because it is a limit a PI on the crown cannot
see coming.

## Open for the owner

- **Batch-line camera.** Should the blanket's position be observed? Vendors sell these
  cameras, and it would make the blanket limit a control problem, not a hidden one. It may
  also let a plain cascade close the room.
- **Sizes.** How far and how fast do blanket moves and batch changes go? No source gives them.
- **Charter.** The rebuild needs its row in the task table changed, and a plant of this size
  must still pass rule 7.
