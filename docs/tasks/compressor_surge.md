# Compressor Surge

The task card for `compressor_surge` ([its page](../environments/compressor_surge.md), cite as `compressor_surge-v5`).

## What the task is

Hold the header pressure of a variable-speed compressor on a ramped setpoint while the consumers' demand ramps between levels, with the drive speed and an anti-surge recycle valve. Flow below the surge line trips the plant, and neither the reward nor any observation reads the surge margin directly, so the policy has to learn to keep it from the flows it does observe.

## Scenarios and weights

| Scenario | Share of operating time | What it is |
|---|---|---|
| `normal` | 0.79 | demand high, the point right of the control line, the recycle shut or nearly |
| `slow_approach` | 0.16 | demand falls slowly toward the control line |
| `fast_approach` | 0.05 | demand falls fast; rate action or the backup line |

## Its lines

The reference is the `feedforward` rung, frozen in `data/reference/compressor_surge.json` (`tg.make_reference("compressor_surge")`). It holds dp on speed at 24.

What an agent is measured against: the expected cost per step over the task's weighted scenarios, on seeds 100 to 109, lower is better. An agent's result is its saving over the reference, `1 - cost_agent / cost_reference`.

| Line | What it is | Cost per step | Trips |
|---|---|---|---|
| Best null policy | the cheapest of the simple policies | not current | |
| Reference | the controller to beat | not current | |
| MPC | plans with the true model and the schedules operators know, without the disturbances ahead | not current | |
| Ceiling | the best known plan with the true model, the true state and the future: an achieved cost, not a bound | not current | |

Recorded on an earlier version of this task, so not shown: the best null policy, reference, MPC and ceiling lines, which show again once re-recorded. The room shows again with the reference and the ceiling.

The ceiling reads what the policy does not observe: the machine's true surge line, up to 5 % off its map; the consumers' demand levels ahead and their random part. Part of the room may be out of reach of a policy that reads only the observations.

Score an agent with `tg.score("compressor_surge", policy)`: [Comparing an agent with the reference](../comparing.md).

## What it does not model

From the physics contract's known deviations, section 9:

- Rotating stall (D1). Left of the peak the characteristic has one branch.
- Compressibility in the duct (D2). Its density is the suction's, up to 10.6 %
  off at the largest flows.
- The plenum's warmer gas (D3). Its speed of sound is the suction's, which puts
  the Helmholtz frequency about 4 % low.
- A tight check valve (D5). It passes a little flow at zero pressure drop.
