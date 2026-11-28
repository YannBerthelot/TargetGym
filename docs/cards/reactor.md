# Reactor: the task card

A v1 candidate, not frozen yet; its development stamp is reactor-v10.

## The job

A pressurized water reactor operator follows the day's load plan with the control rods, the soluble boron and the turbine's loading rate, taking as much of the plan as the core can carry while the average coolant temperature, the flux difference and the rod insertion stay inside their limits, at the least lost output and effluent.

**The decision.** Every 30 s the agent sets the rod demand, the boron flow (a dilution or a boration, which reaches the core 10 min later) and the turbine's loading rate.

## The score

The cost is in USD, averaged over a 7-day run of 20160 steps. The score is the saving over the expert, 1 - J / J_expert, where J weights each scenario's mean cost by its share of operating time, on the scoring seeds 200 to 209 (`tg.score`).

| Scenario | Weight | What it is |
|---|---|---|
| `baseload` | 0.892 | full load all day with the frequency-control wobble |
| `edf_daily` | 0.043 | 100-50-100 at 2 %/min, held 3 to 8 h |
| `krsko_slow` | 0.043 | 100-50-100 at about 8 %/h and straight back |
| `fast_return` | 0.022 | 100-50-100 at 5 %/min, the EUR's fastest, held 3 to 8 h |

## Baselines

The expert is the task's shipped controller, frozen as its reference.

Not recorded yet: `scripts/record_v1.py` records them.

The room, the best line's saving over the expert, is an upper bound on what a deployed controller would gain: the MPC plans on the task's own model, estimating only the state, and no plant's controller has its model exact.

## Prices

| Price | Value | Unit | Trades | Source |
|---|---|---|---|---|
| `derate_price_per_MWh` | 80 | USD/MWh | an MWh of the plan the turbine does not take, or that the offset's limits take off | PJM's 2022 real-time load-weighted average, $80.14/MWh (IMM report to the Members Committee, January 2023, slide 12) |
| `spot_price_per_MWh` | 80 | USD/MWh | the output Tavg's steam conditions lose past its 1.6 K free band, 0.23 % of it a kelvin (imbalance_multiple, the Carnot slope at 547 K steam and 305 K condenser) | PJM's 2022 real-time load-weighted average, $80.14/MWh (IMM report to the Members Committee, January 2023, slide 12) |
| `effluent_price` | 0.0628 | USD/kg | a kg of water or acid injected, which displaces a kg of coolant to treat | EPRI 1011728 (2005), pp. 3-4 and 3-15: liquid radwaste processing at $0.15 to $0.60 a gallon, typically under $0.25, so $0.040 to $0.159 a kg |
| `restart_steps` | 5760 | 30 s steps | a trip's dead time before the restart, each 30 s step at a step of the plant's whole output (failure_cost) | 48 h, inside APR1400's 22 to 70 h from the scram to the rods' withdrawal for the restart (Castro Monteiro et al., KNS Spring 2020, 20A-065) |
| `standby_steps` | 720 | 30 s steps | a timed limit's overrun: hot standby before the return, at the trip's step price | 6 h, Doel-3's time down before its return (INIS 1ypr9-dk595) |

5 of 5 prices have a published source.

## What it does not model

- The wear of rod stepping: no published price for a rod step was found, so the rods move free (their travel is still counted).
- Grey-rod operation (EDF's Mode G): one black bank holds the offset and the boron holds Tavg.
- Axial damping that changes through the cycle.
- The xenon peak after a scram: 1,419 pcm, under the 1,800 to 3,200 pcm test band.
- Load-follow days drawn from real units' output: each day is its source's standard shape.
- Xenon in the end charge: the stock a run ends with counts Tavg's place in its band, the rods and the boron line.
