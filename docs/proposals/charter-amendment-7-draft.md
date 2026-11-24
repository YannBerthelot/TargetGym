# Charter amendment 7: where a reference rests, and tasks with little room

Status: adopted by the owner on 2026-11-24 ("Adopt and merge"). The rules are in
[task-charter.md](task-charter.md) (the claim, J1, J3, J5); this file keeps the reasons and
the cases. Its decisions were taken by the owner on 2026-11-23 and 2026-11-24, by picker,
while piloting the per-task review sessions. J1 in the charter carries the pH session's
wording of the zone rule, which says the same as the text below in one paragraph.

It was first numbered 6. The charter's Amendment 6 is now the actuator's window, adopted
the same day with the wind turbine's oracle ([charter-history.md](charter-history.md)).

## Why

Twelve review sessions, one per task, each looked for where their task's room came from.
Five found most of it in where the reference rests, not in control:

| Task | What J1's zone rule gave | What was found |
|---|---|---|
| `ph_neutralization` | pH 6.107, 2 sigma above the permit's edge | Sigma was read once, at pH 7; near 6.1 the spread is 1.5 to 4 times larger. Room 14.05, of which the set point is all but 1.13 to 1.16. Time outside costs 8.33 L a step, 0.1 pH of back-off 0.0017 L: 2 sigma cannot be right at any sigma. |
| `hvac` | 0.94 K inside the comfort edge | Sigma was pooled over all hours (0.59 K, the room's daily swing); what can cross the edge, the room at 08:00, spreads by 0.076 K. Over half of the room of 1.45 was the set point. |
| `distillation` | Bottoms 2 sigma inside their spec | Since v8 the score prices the light key the bottoms carry away, so that edge is the dear one. 73 % of the recorded room was this one target. |
| `compressor_surge` | A 10 % surge margin copied from practice | On the plant as built the reference surges on no seed down to 7 %. |
| `boiler_drum` (fuel loop prototype) | Oxygen 2 sigma above the limit | Its room of 1.0004 is the margin for the analyser's noise. |

The second half is the glass furnace. Its room is 1.048 (1.041 to 1.056), and the first
readiness table marked it for the owner's call. His answer changes how small room is read.

## The changes

### 1. J1, zones: the reference rests at the lowest set point it never leaves from

Replace the *Zones* bullet of J1 by:

> - *Zones:* where the score prices each step outside a zone as a violation, the
>   set point is the first candidate, counting from the edge the economics push toward,
>   at which the reference leaves the zone on none of the ladder's seeds, in any
>   scenario, as the ladder keeps the best rung that trips none. The candidates are that
>   edge and ten equal steps to the zone's centre, fixed before any room is seen. Each
>   candidate is run from episodes established on it, and the task's episodes start on
>   the kept one, so an agent and the reference run the same episodes.
> - *Seeds that can be held decide.* A seed the reference leaves the zone on from every
>   candidate does not constrain the placement. It is listed in the record, and the
>   reference pays for its exits there. If no seed can be held at all, the set point is
>   the centre and the record says so.
> - *Each rung on its own.* Every rung of the ladder gets its own set point by this
>   rule. The ladder then keeps the cheapest rung that leaves the zone on none of the
>   seeds it can hold.
> - Where leaving a zone is priced as give-away, not as a violation, the set point stays
>   k = 2 standard deviations inside the cheap edge, never past the centre, with the
>   deviation measured on what can cross the edge, at the moment it can.

Cases: pH (6.4 to 6.5 in place of 6.107, room 1.16), hvac (22.00 C on the PI rung, room
1.36 on 50 seeds, to be run again rung by rung), distillation (distillate 0.002143 inside
its spec, room 1.056, two seeds of `both_steps` left out), compressor_surge (margin 0.09,
room 1.54 on held-out seeds).

Decided as: "Lowest with no exit", then "Seeds that can be held decide", then "Each rung
on its own". On the last bullet the owner answered "Keep 2 sigma there" on 2026-11-24. The
charter keeps the 2 sigma rule's earlier wording for those zones: this bullet's clause on
where the deviation is measured was not put to him and is not adopted.

### 2. The claim and J5: a task with little room is a kind of task

Add to *The claim*, after the sentence on room:

> Room need not be large. A task whose reference is close to its ceiling is kept when
> the room is beyond seed noise and worth something at the plant's real scale: on a
> furnace that burns fuel all year, one percent is a large sum.

Add to J5, after the paragraph on solved tasks:

> *Nearly saturated tasks.* A task whose room is small is neither solved nor dropped for
> that. Its page states the room with an interval on paired seeds, and what the room is
> worth in money and in energy per plant and year, from sources. Where the interval
> includes 1, the task is treated as solved. `glass_furnace` stays in the benchmark
> whatever its room: no room on it is a finding about the model, to be looked into, not
> a verdict on the task.

Add to J3's lines:

> A fifth line where the room is under 1.2: what it is worth per plant and year.

Cases: glass_furnace (EUR 255k to 595k a furnace-year on the model's furnace), hvac's
room after the set point (EUR 2,900 and 17 MWh a year on a 14,000 m2 office),
distillation (1.056), the reactor's baseload days (1.12 to 1.20), the hold's fuel in
`plane3d_racetrack` (0.4 to 0.7 % as a floor).

Decided as, in the owner's words: "Glass-furnace needs to stay as it is the motivating
example for this whole library. If it does not show room it means that we have not
modeled it properly", and "almost saturated tasks (little room, but it DOES exist beyond
noise) can be a specific type of task. The important part being that even if the room is
little, improving score can have huge economic and energetic benefits." The threshold and
the solved rule were this draft's, and he confirmed both on 2026-11-24: "Room under 1.2",
and "Yes, solved". A solved task goes to Extra with that reason, as `plane_energy` and
`boiler_drum` did, where J5 had kept it in the core pool.

## What the sessions found that this amendment does not settle

- **A target that is a schedule: settled.** Distillation's bottoms follow the metered
  feed, and the shared code placed one set point per zoned loop. The owner chose "Hook
  for a scheduled target" on 2026-11-24: a candidate shifts the whole schedule (J1).
- **Seeds the rule was not placed on.** The compressor's reference, placed at 0.09 on ten
  machines, surged once on an eleventh, held-out one. Where a plant draws a hidden
  number per episode, "none of the ladder's seeds" protects only those draws.
- **A reference tuned on the scoring seeds.** The glass session notes its reference was
  tuned on the five seeds its interval uses, which favours it.
- **Ceilings that are bounds.** On pH a hold with no foresight beat the recorded
  ceiling, and several ceilings stop on an iteration cap. J3 says "the foresight
  ceiling"; it does not say what a record states when the solve did not converge.

## What it costs

Every task with a zone gets its reference placed again, so its reference line, its room
and its J2 verdict are redone: pH, hvac, distillation, compressor_surge, cstr's
production prototype, and whichever of battery and glass_furnace use a zone. The shared
placement lives in `experts/ladder.py` (pull request 232). Scoring one candidate per rung
multiplies the ladder's runs by the number of rungs times up to eleven candidates, less
those past the first exit-free one.
