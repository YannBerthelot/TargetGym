# pH neutralisation task card

`ph_neutralization`, cited as `ph_neutralization-v6`. Its spaces, reward and arguments are on [its page](../environments/ph_neutralization.md).

## What the task is

Keep the discharge of a two-stage acid neutralisation inside its permit, pH 6 to 9, with a reagent valve on stage 1 and a trim valve on stage 2, without overdosing. The policy reads both stages' pH probes, its two valves, the influent flow, the time spent outside the permit and the diversion basin's level. The buffer flow and the acid strength are hidden. Each step is charged, in litres of reagent, the reagent used, the time outside the permit, and the effluent diverted while the discharge is below pH 5. One excursion longer than 60 min, or an overrun basin, is a violation and is charged as a restart. Most of the room between the reference and the ceiling comes from where the set point sits, not from riding the disturbances. The reference holds pH 6.45, the lowest set point at which it leaves the permit on none of its seeds. The ceiling holds the permit's edge, pH 6.00, because it knows the hidden buffer flow and acid strength. Placing a set point safely without that knowledge is a real control skill, and it is most of what this task scores (owner, 2026-11-25).

## Scenarios and weights

| Scenario | Share of operating time | What it is |
|---|---|---|
| `normal` | 0.80 | waste flows in, buffering drifts, reagent holds the pH |
| `bad_day` | 0.15 | calm to rocking and reeling: acid and flow both move |
| `buffer_loss` | 0.05 | buffering falls away and the curve steepens |

## Its lines

The reference is the `feedforward` rung, frozen in `data/reference/ph_neutralization.json` (`tg.make_reference("ph_neutralization")`). It holds pH on trim at 6.45. On 30 episodes of seeds it was never placed on (J2's), it leaves its zone on none.

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies | not current | |
| Reference | the controller to beat | not current | |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | not current | |
| Ceiling | the best known plan with the true model, the true state and the future: an achieved cost, not a bound | not current | |

Recorded on an earlier version of this task, so not shown: the best null policy, reference, MPC and ceiling lines, which show again once re-recorded. The room shows again with the reference and the ceiling.

The ceiling reads what the policy does not observe: the acid strength; the buffer flow; when the influent's acid and flow upsets come and how large. Part of the room may be out of reach of a policy that reads only the observations.

Score an agent with `tg.score("ph_neutralization", policy)`: [Comparing an agent with the reference](../comparing.md).

## What it does not model

From the physics contract's known deviations, section 9:

- Electrode drift and fouling (D1). The probes stay as fast as new ones, where
  coated probes turn sluggish and drive overdosing.
- Buffers other than carbonate (D2), so the titration curve is smoother than a
  real effluent's.
- Valve resolution, stick-slip and on-off or timer dosing (D3).
- An industrial stream. The streams are a bench rig's in tanks sized to the
  family's retention (D4).
- Running on after a permit violation, which restarts the plant here (D5).
- Upsets large enough for the basin, the diversion and the 60 min rule to act
  on a working controller (D7).
