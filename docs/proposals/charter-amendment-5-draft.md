# Charter amendment 5 (draft): one member, both sides, every stop

Status: draft for the owner, 2026-11-10. Nothing here is adopted. It proposes edits to
[task-charter.md](task-charter.md) (version 2), which lives on main; this file only
gives the text. Each change names the rule it edits, the wording to add or replace, and
the reactor episode that showed the gap.

## Why

The reactor went through Amendment 3, then v7 and v8 in one day, and an independent
physics review (`reactor-physics-review.md`) still found two real faults. The charter's
cold review (B8) caught them, as it should. The gaps were earlier, in four places:

1. **One-sided pricing.** B5 says the score is "the least economic cost the source
   names", and we read "lost production" as "only what loses output is charged". A hot
   Tavg loses no output, so it went free up to the DNB value, 583 K, at every power. Both
   procedures the task cites keep a two-sided band, and at part power the real stop is
   the steam side, which no rule asked us to look at.
2. **Numbers from different family members.** The moderator coefficient was `ours`
   inside -10 to -50 pcm/K, a range that spans the whole fuel cycle, while every other
   number described the start of a cycle. Each value was in its range; together they
   were a core that never existed. B4 and B7 let that through, because ranges are per
   number and the sweep draws them independently.
3. **Scenarios and procedures from different members.** Once the coefficient matched
   the start of a cycle, the reference (Krsko's procedure: rods on the offset, boron on
   Tavg) tripped on the 5 %/min fast_return day. Krsko's procedure is a slow-ramp one;
   the 5 %/min figure comes from plants that carry fast ramps on rods. The old
   coefficient had hidden the mismatch.
4. **Conditions dropped from sources.** BEAVRS's boron worth is measured at hot zero
   power and 975 to 1405 ppm; the task runs at full power and 1690 ppm. A number "from
   the right cycle point" can still be from the wrong conditions. And an inherited
   envelope (620 K, "the saturation margin") sat above saturation, because nobody
   re-checked inherited limits when the plant was rebuilt.

One process gap ran through all of them: the family sweep (B7) has never been run on
the reactor, so moving an `ours` number was never tested across its range before results
were read.

## The changes

### 1. Procedure bands are two-sided (B5, *Score*)

Add after "Precision inside a band earns nothing more unless a source says it pays":

> A band the plant keeps by procedure is priced on both sides unless a source says one
> side is free. "Lost production" and the other costs set the price of leaving a band,
> not which side counts. A one-sided price names its source.

*Reactor:* Tavg is charged past 1.6 K on either side of its program (v8).

### 2. A task describes one family member (B2, B4, B5, J1, B7)

**B2**, add to the list of what the record states:

> - the member's operating state the task represents (for example a point in the fuel
>   cycle, a season, a product grade), and for each sourced number the conditions it was
>   measured at.

**B4**, replace "An `ours` number lies in the family's range and is labelled with that
range in `PHYSICS.md`." with:

> An `ours` number lies in the family's range at the member's operating state (B2), not
> merely somewhere in the family, and is labelled with that range in `PHYSICS.md`. A
> sourced number measured at other conditions than the task's is `ours`, with the source
> as one end of its range. Numbers that move together across the family (a reactor's
> moderator coefficient with its boron, a boiler's gain with its load) are given as one
> joint choice, not as separate ranges.

**B5**, *Scenarios*, add:

> A scenario's sources describe plants that fit the member's operating state and
> equipment. A figure from plants run another way (another control mode, actuator or
> point in the cycle) is not a scenario for this member.

**J1**, *Published*, add:

> A published procedure is a candidate only on the scenarios it is published for. Where
> the family runs a different procedure for some scenarios (slow ramps against fast
> ramps, say), the reference switches between them by a rule written before any run.

**B7**, replace "drawn jointly across their family ranges" with:

> drawn jointly across their ranges at the member's operating state, with coupled
> numbers moved together.

And add:

> The sweep runs whenever an `ours` number changes, before any result from the change is
> reported.

*Reactor:* the start-of-cycle MTC with its boron; fast_return served by the rods-on-Tavg
mode; BEAVRS's boron worth kept as one end of a range; the sweep still owed.

### 3. Every state the agent can push has its stop (B2, B4)

**B2**, add to the list:

> - for every state an agent can drive (a temperature, a pressure, a level), what stops
>   it on the real plant, from any subsystem: a relief valve, a trip, a protection, a
>   procedure limit.

**B4**, add:

> Each stop the record names is modelled, or listed in `PHYSICS.md` as a known deviation
> with what keeps the task from exploiting its absence. A limit carried over from an
> earlier version is re-checked against its source or a physical bound (a saturation
> temperature, a design pressure) whenever the plant is rebuilt.

*Reactor:* the steam side bounds Tavg at part power; it is D10 in PHYSICS.md, with the
two-sided band as what keeps an agent from exploiting it. The 620 K envelope above
saturation is now 610 K.

### 4. When the reference and the cost disagree, read the record first (J1)

Add to J1:

> When a reference copied from practice acts on something the score leaves free, or
> ignores something it charges, the disagreement is a finding about the score (B5) before
> it is one about the reference. The record decides which is wrong; matching one to the
> other without it is not allowed.

*Reactor:* when the null won baseload, the reference held Tavg on both sides as
operators do, and the cost charged only the cold side. We changed the controller to fit
the cost; the record said the cost was wrong.

## What it costs

- **Every task:** its `REALITY.md` gains a line naming the member's operating state and,
  for each driveable state, its stop. Each sourced number's conditions are noted where
  the record has them.
- **Tasks with coupled `ours` numbers:** their range table in `family_sweep.py` names the
  coupling. The sweep then draws those numbers together.
- **The reactor:** the sweep (B7) runs on v8 before acceptance; the Krsko-day validation
  failure is read against the start-of-cycle member before any fix.
- **charter-history.md** records the amendment and the reactor episode as its reason.

## Not proposed

- A rule that bans a controller from acting on unpriced terms. Some unpriced states
  predict priced ones (an offset drifting toward its band before a ramp), so the rule
  would be wrong; change 4 covers the case that went wrong here.
- A rule on test setups. Five reactor tests needed new setups after v8, each with its
  reason in a comment; TESTING.md already covers that.
