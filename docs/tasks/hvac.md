# Building HVAC

The task card for `hvac` ([its page](../environments/hvac.md), cite as `hvac-v6`).

## What the task is

Heat an office through a concrete slab by choosing the supply water temperature, so the room stays in its 21 to 26 C comfort band during occupied hours at the least electricity cost. The slab takes hours to respond, so the policy has to act ahead: it reads the room, slab and water temperatures, the outdoor temperature with its last-24 h mean and next-24 h forecast, the clock, and the tariff with the hours to its next change. Each step is charged, in euros, the heat pump's electricity at the tariff, each occupied kelvin-hour of room outside the band or of floor above its limit. The evaluation also charges, once, the heat the building's store lost across the scored days, at the cheapest price heat can be stored at. Nothing trips.

## Scenarios and weights

| Scenario | Share of operating time | What it is |
|---|---|---|
| `normal` | 0.50 | a heating fortnight, 5 C mean |
| `cold_snap` | 0.08 | a cold fortnight, -5 C mean, near the -10 C design point |
| `shoulder_season` | 0.42 | a mild fortnight, 12 C mean, around the heating limit |

## Its lines

The reference is the `feedforward` rung, frozen in `data/reference/hvac.json` (`tg.make_reference("hvac")`). It holds T_air on supply at 22. On 30 episodes of seeds it was never placed on (J2's), it leaves its zone on none.

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies | not current | |
| Reference | the controller to beat | not current | |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | not current | |
| Ceiling | the best known plan with the true model, the true state and the future: an achieved cost, not a bound | not current | |

Recorded on an earlier version of this task, so not shown: the best null policy, reference, MPC and ceiling lines, which show again once re-recorded. The room shows again with the reference and the ceiling.

The ceiling reads what the policy does not observe: the outdoor temperature ahead, hour by hour, past its 24 h mean. Part of the room may be out of reach of a policy that reads only the observations.

Score an agent with `tg.score("hvac", policy)`: [Comparing an agent with the reference](../comparing.md).

## What it does not model

From the physics contract's known deviations, section 7:

- Cooling. The plant only heats, where slab plants run both ways, so a room
  above the band can only stop being heated.
- Forecast error. The 24 h forecast the policy reads is the weather model's own
  mean, not a real day-ahead forecast with its error.
- A heat pump whose capacity moves with its supply and brine temperatures. The
  source has one size whatever water it makes.
- The radiant share of the slab's exchange. All its heat goes to the room's air.
- A day's practice of setting the curve once and spreading it over hours. The
  supply can move every 15 min.
