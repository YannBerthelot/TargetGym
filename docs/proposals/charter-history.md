# Charter history

How the [task charter](task-charter.md) came to be, and where each old rule went. It
carries no rule of its own.

- **Draft 3 (2026-11-07):** the ten original rules, adopted after three independent
  reviews and a re-check of every task.
- **Amendment 1, match the real plant (2026-11-08).** Every task had been designed in a
  closed loop, each version checked against the one before it and against our own
  controllers, so early deviations became the baseline and grew. The glass furnace's
  fault was found only by looking at an actual furnace, and the reactor's five
  mismatches only by reading how real plants are operated. It added the plant record,
  the behaviour tests, the ban on closed-loop references, behaviour before structure,
  and the cold review.
- **Amendment 2, the real job (2026-11-08).** Three family audits found the same three
  faults on nearly every task: limits modelled as trips, setpoint scores where the job is
  keeping a band at least cost, and named properties missing from the model. It added
  limits by kind, the operator's score and observation, real episodes, and the
  modelled-property rule.
- **Amendment 3, represent the family (2026-11-08).** Figures from one plant were sending
  good tasks to Extra. The owner ruled: "We are most likely never going to source all
  numbers, especially not from a single plant. The goal is to follow the spirit and
  properties of the task, not to have it be a specific realistic plant. The goal is to
  represent the plant family, not a specific instance." It added family ranges, scenario
  suites, the zone back-off and the search for a cost before a task is called solved.
- **Amendment 4, the signature, the sweep and the claim (2026-11-08).** Design had no
  stopping rule, and reviewers object to claims they can't trace. It added the claim, the
  property card, the stopping rule, the family sweep and the page that shows legitimacy.
- **Version 2 (2026-11-08).** The draft and its amendments were consolidated into one
  charter in building order. The owner settled these points at the same time:
  - **Rebuilds:** before the tag, sources drive design changes, and a redesign from
    scratch needs the owner.
  - **Headline:** a frequency-weighted saving over the reference replaces Amendment 3's
    equal-weight gap closed. Gap closed was anchored on non-converged oracles and blew up
    on low-room tasks, and equal weights let a rare upset outweigh a year of normal days.
    The oracle and an MPC became reported context lines.
  - **Ladder tuning:** each loop is tuned at its most sensitive operating point.
  - **Aircraft reference:** it flies at the same protected limits as the agent, with no
    extra margin; the old "80 % of the trip" went with bank and load factor becoming
    protected limits (rule 14).
  - **Acceptance seeds:** 30, or 5 per scenario if that is more, at least 5 in every
    scenario and the rest by weight, with at most 1 trip in every 30.
  - **Cost:** `J` includes unavoidable cost, so a saving is a share of the real
    operating cost, as industry reports it. Rule 21's "unavoidable cost cancels out of
    the gap" went with gap closed.
  - **Zone back-off:** from the edge the economics push toward, and never past the
    zone's centre.

  Rule 8's decades headline and rule 21's gap closed are retired.
- **Amendment 5, one member, both sides, every stop (2026-11-10).** An independent
  physics review of the reactor found Tavg free on its hot side up to the DNB value at
  every power, and a late-cycle moderator coefficient on a start-of-cycle core. Fixing
  the coefficient then broke the fast-ramp scenario for the adopted procedure, a
  slow-ramp one. Each number had been in its range; together they described no real
  plant. It added two-sided procedure bands (B5), the member's operating state and each
  source's conditions (B2, B4), joint choices for coupled numbers and a sweep after any
  `ours` change (B4, B7), scenarios and procedures fitted to the member (B5, J1), a real
  stop for every driveable state (B2, B4), and the record as judge when the reference
  and the score disagree (J1). Draft and reasons: `charter-amendment-5-draft.md` on
  branch `claude/reactor-axial-dd4617`.

- **Amendment 6, the actuator's window (2026-11-23).** The wind turbine's receding
  oracle rode its overspeed level on a forecast of decaying turbulence and tripped on 35
  of 60 MPC-line seeds. The owner chose "Actuator window, about 2 s": a limit's back-off
  covers a 2-sigma disturbance for the time the correcting actuator's rate limit needs to
  take it back (J3; oracle smoothing protocol, Amendment 6, K11a). The formula gave 1.5 s,
  and the owner kept it. It is general: wind is the first task it applies to.

- **Amendment 7, where a reference rests, and tasks with little room (2026-11-24).** One
  review session per task looked for where its room came from, and five found most of it
  in where the reference rests: pH's set point 2 sigma above the permit's edge made all
  of its room of 14 but 1.13 to 1.16. The owner chose "Lowest with no exit", then "Seeds
  that can be held decide", "Each rung on its own" and "Hook for a scheduled target",
  and kept 2 sigma for zones priced as give-away (J1). On small room he said: "almost
  saturated tasks (little room, but it DOES exist beyond noise) can be a specific type of
  task", and that glass_furnace stays, as "the motivating example for this whole
  library". So a task with room under 1.2 states what the room is worth per plant and
  year (J3), an interval of room that includes 1 means solved, and a solved task goes to
  Extra with that reason where J5 had kept it in the core pool, as plane_energy and
  boiler_drum did (the claim, J5). Draft and cases:
  [charter-amendment-7-draft.md](charter-amendment-7-draft.md).

**Old numbers.** Design step 0 became B2; step 1, B1; step 2, B4; step 3, B3 and B6;
step 4, B1 and D2.

| Old | Now | Old | Now | Old | Now | Old | Now |
|---|---|---|---|---|---|---|---|
| 1 | S3, B4, B6 | 8 | J4 (retired form) | 15 | B5 score | 22 | J1 zones |
| 2 | B5 score, D3 | 9 | J3, J5 | 16 | B5 observation | 23 | J5 |
| 3 | J1 | 10 | J7 | 17 | B5 scenarios | 24 | The claim |
| 4 | J2 | 11 | Building, intro | 18 | B1 | 25 | B3 |
| 5 | D2, D4, D5 | 12 | B4 | 19 | Words, B4, B6, D2 | 26 | B4 |
| 6 | D4 | 13 | B8 | 20 | B5 scenarios, J4 | 27 | B7, J2 |
| 7 | J6 | 14 | B5 limits | 21 | J4, Words (retired form) | 28 | S1, S2 |

Other old sections:
- the Goal became The claim; Disclosure became D1; the Definitions became Words;
- "Scoring constants already set from oracles are frozen at the tag" became D3;
- "New tasks" became D6;
- the amendments' Scope, Cost and How-it-is-applied notes moved to
  [task-status.md](task-status.md);
- rule 5's "no ... amendments" meant no amendment for one task; D5 keeps it as "no rule
  of its own". The charter itself changes only by the owner's decision.
