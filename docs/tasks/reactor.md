# Reactor

The task card for `reactor` ([its page](../environments/reactor.md), cite as `reactor-v10`).

## What the task is

Run a day of a pressurized water reactor somewhere in the first 90 % of its fuel cycle, with control rods, soluble boron and the turbine's loading rate, knowing the day's load plan: take as much of the plan as the core can carry, hold the average coolant temperature within 1.6 K of its program on either side and the flux difference within 5 % of its target, keep the rods above their insertion limit, and dilute for the fuel's burnup, at the least cost in lost output, rod wear and effluent. Six scenarios weigh baseload days against five kinds of load-follow day (REALITY.md sections 6 and 7).

## Scenarios and weights

| Scenario | Share of operating time | What it is |
|---|---|---|
| `edf_daily` | 0.04 | 100-50-100 at 2 %/min, held 3 to 8 h |
| `krsko_slow` | 0.04 | 100-50-100 at about 8 %/h and straight back |
| `fast_return` | 0.02 | 100-50-100 at 5 %/min, the EUR's fastest, held 3 to 8 h |
| `baseload` | 0.89 | full load all day with the frequency-control wobble |

Each scenario is scored on its own 7-day stream. Twins share a weight: `edf_short_hold` into `edf_daily`, `krsko_shallow` into `krsko_slow`.

## Its lines

The reference is the shipped expert, the published procedure (`tg.make_reference("reactor")`).

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies | not current | |
| Reference | the controller to beat | not current | |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | not current | |
| Ceiling | the best known plan with the true model, the true state and the future: an achieved cost, not a bound | not current | |

Recorded on an earlier version of this task, so not shown: the best null policy, reference, MPC and ceiling lines, which show again once re-recorded. The room shows again with the reference and the ceiling.

The ceiling reads what the policy does not observe: the core's xenon and iodine, top and bottom; the boron in the core now, where the policy reads it 20 min late; the load's frequency-control wobble ahead. Part of the room may be out of reach of a policy that reads only the observations.

Score an agent with `tg.score("reactor", policy)`: [Comparing an agent with the reference](../comparing.md).

## What it does not model

From the physics contract's known deviations, section 5:

- The secondary side (D1): the turbine's governor, the steam's moisture and the
  steam generator's level. The load goes straight to heat.
- Grey rods for deep load following, EDF's Mode G (D8). The reference runs the
  Westinghouse procedure.
- Rods from both ends (D3). A bank reaches only the top.
- Local peaking (D4). The axial offset is scored on its band and clocks, with
  no link to local power or a trip.
- A derate after an overrun (D6). It is priced, and the load does not drop.
- Boronmeter noise (D7) and decay heat after a scram (D5).
- The axial mode's change through the fuel cycle (D11).
