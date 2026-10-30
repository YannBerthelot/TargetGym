# Reward shaping

TargetGym asks one question: **can a learned policy hold a setpoint better than
a PID or an MPC?** That question lives in the reward, so the reward is the
measuring instrument, and this page says how it is built.

Every environment scores through the version-2 reward described here. Twenty
normalise the tracking error by an achievable floor; four (reactor, battery,
wind turbine, building) put tracking and consumption in the owner's currency,
where the floor enters as the reference cost `rho_floor` instead. Version 1 is
described briefly at the end.

## The reward, in one line

```
reward = -( tracking_cost + running_cost + failure_cost )
```

Three non-negative costs, added rather than multiplied, each in a unit the
plant's owner could defend. `target_gym.reward` holds the arithmetic; every
environment's `compute_reward_terms` returns the three terms and
`compute_reward` is minus their sum.

The best achievable per-step reward is about `-1` per tracked output on the
dimensionless plants (tracking at the floor, nothing avoidable consumed), and
minus the unavoidable bill on the priced ones. The reward is not capped, and
its per-step scale differs by orders of magnitude between plants. To compare
across plants, use the normalised expert advantage and the two costs reported
separately (`target_gym.eval`), not the raw return.

### Tracking: the error in units of what is achievable

```
tracking_cost = ( max(|e| - e_tol, 0) / e_floor ) ** p
```

- **`e_floor`** is the achievable floor: the smallest long-run mean |error| any
  controller can hold on this plant under its shipped reference and
  disturbances, but never finer than the instrument the plant cites can
  resolve (the simulator has no measurement noise, so it can hold finer than a
  real transmitter reads). An error at the floor costs 1 per step, so a
  tracking cost reads directly as "how many floor-widths off".
- **`e_tol`** is a specification tolerance where the plant has one (a comfort
  band, a purity spec, a permit limit). The cost is zero inside it, so inside
  the band only consumption is optimised. A band must be small against the
  errors controllers actually make, or it hides them; the aircraft's ±30 m band
  was dropped for that reason.
- **`p`** is 1 where the owner's cost is linear in the error (an energy
  imbalance settled per MWh) and 2 otherwise. Multi-output plants sum one term
  per output, each with its own floor.

The cost is convex in |e| by construction. Version 1's log-scaled tracking was
concave, and a concave cost prefers rare large excursions to frequent small
ones of the same mean: the wrong preference for a hold. The Target-MDP note
(Proposition 8) gives an example where that preference costs an arbitrary
amount.

### Running cost: only what could have been avoided

Consumption (fuel, energy, boilup, reagent, actuator travel) is charged above
`c_hold`, what the best shipped controller consumes per step while holding.
That is the avoidable part. Dimensionless form:

```
running_cost = w * max(c - c_hold, 0) / c_hold
```

`w` reads as "one floor-width of tracking error is worth `w` times the
hold-phase consumption", default 1, swept over {0.5, 1, 2} where the number
matters. Where the owner's tariff is known the term is priced instead
(`price * quantity`), and tracking is in the same currency, so there is no
weight to choose: the reactor's imbalance at three times spot, the building's
gas per kWh, the battery's imbalance per MWh and fade per kWh of lost capacity.
Wear and fatigue have no defensible tariff, so they are charged per unit of
avoidable activity times a documented weight.

### Failure

Leaving the operating envelope trips the plant. A trip does not end the
episode: the step that leaves the envelope is charged the **trip cost**,
`restart_steps * failure_cost`, with that step's tracking and running cost
zeroed, and the plant restarts at once as `reset_env` would
(`base.failure_kernel`, `reward.trip_cost`). `terminated` stays false, so the
restart is an ordinary transition with a cost, as continuing and
average-reward methods want it, and every window is the same length.

- **`restart_steps`** is the time a real restart takes, a provisional number a
  plant engineer would supply (a SCRAM's xenon-limited restart, a furnace
  heat-up). Each plant's `PHYSICS.md` gives its value and source.
- **`failure_cost`** is twice the largest tracking cost the plant can reach
  (its target range against its trip bounds), per step.
- **A plant that does not restart cold** (the building, whose thermal mass
  persists through a lockout) has `restart_in_place = True`: it keeps its
  state, and every step outside the envelope is charged until the controller
  brings it back.

The long-run cost then decomposes as
`rho_hold + p * B + lambda * (restart_steps * failure_cost + B_restart)`,
with `lambda` the trip rate, which the protocol reports separately. The trip is
reported as `info["tripped"]` and the charge as `info["trip_cost"]`. Two plants
never trip: the pH loop and the CSTR cannot reach their limits, which are
documentation only.

**Episodic use (optional).** With `continuing = False` a trip ends the episode
instead: `terminated` is raised on that step, and the reward and `info` are the
same. The trip cost then also stops a discounted learner from crashing on
purpose (a terminal state worth zero would otherwise be the cheapest state),
as long as `restart_steps >= 1 / (2 (1 - gamma))`: 50 steps at `gamma = 0.99`,
which every plant clears, and 500 at `gamma = 0.999`, which `first_order`,
`cstr` and `distillation` do not.

### Learning with trips

The trip cost prices real downtime, so it is large: from 8 to over 80,000 steps
of the worst tracking cost, depending on the plant. On the step it lands it can
be orders of magnitude above every other reward, and a learner that does not
normalise its targets sees little else. Remedies that leave the objective
unchanged:

- **Treat the trip cost as its own channel.** `info["trip_cost"]` is the lump
  the step charged (0 on every other step). Subtract it from the reward to
  learn the tracking value, learn or normalise the trip part separately, and
  add the two back.
- **Use scale-invariant value targets.** PopArt (van Hasselt et al., 2016) and
  similar adaptive normalisation keep the value targets well scaled without
  changing what is optimal.

Do not clip the reward. Clipping caps the trip at the size of an ordinary step,
so a policy that trips often can outscore a safe one.

## Why the floor: loop performance assessment, in cost units

The floor-normalised cost is the reinforcement-learning form of **control-loop
performance assessment**. Harris (1989) showed that a loop's minimum-variance
bound can be estimated from routine closed-loop data, and that the ratio of the
actual variance to that bound, the Harris index, tells an engineer how much of
the variance the controller is responsible for. Desborough and Harris (1992)
turned it into the measures industrial loop auditing uses, Huang and Shah
(1999) extended it to multivariable and LQG benchmarks, and Jelali (2006)
surveys its industrial uptake.

`e_floor` is that bound computed on the simulator rather than estimated from
plant data. The benchmark's score is a Harris-type index too, in decades of
cost rather than cost: how far a policy has moved from the expert toward
tracking within the instrument's resolution
([RL protocol, section 4](rl-protocol.md#4-metrics)). `rho_floor` is the
lowest per-seed hold cost the oracle demonstrated, in the reward's units; the
score does not use it. The two-cost report, tracking against consumption, is
Huang and Shah's LQG performance curve: the trade-off frontier a controller is
judged against.

- T. J. Harris (1989). Assessment of control loop performance. *The Canadian
  Journal of Chemical Engineering* 67(5), 856-861.
- L. Desborough and T. Harris (1992). Performance assessment measures for
  univariate feedback control. *The Canadian Journal of Chemical Engineering*
  70(6), 1186-1197.
- B. Huang and S. L. Shah (1999). *Performance Assessment of Control Loops:
  Theory and Applications*. Springer, Advances in Industrial Control.
- M. Jelali (2006). An overview of control performance assessment technology
  and industrial applications. *Control Engineering Practice* 14(5), 441-466.

## How each floor was obtained

`scripts/measure_hold.py` runs the shipped PID and MPC on every plant through a
burn-in of three cost-bearing time constants and then a hold window, under the
shipped reference and disturbances. It records the long-run mean |error| per
output, the consumption per step and the settling time after a target change
(`src/target_gym/data/hold_measurements/`). Three kinds of floor come out
of it:

- **Certified or closed-form**, where the hold problem has a computable
  optimum: the reactor (`scripts/floor_reactor_hold.py`, a dynamic programme on
  the error against the demand's random walk) and the battery (white dispatch
  noise drawn after the action, so `E|error| >= sd * sqrt(2/pi)`). The shipped
  MPC holds at these floors within sampling error.
- **Upper bounds**, where no reduced-model optimum exists yet: the best shipped
  controller's own long-run hold, labelled as such in the plant's `PHYSICS.md`.
  A learner that beats it scores a tracking cost below 1, which is allowed and
  informative.
- **Documented minima** on the plants with no disturbance at all (the CSTR,
  first-order plant and four-tank), where the achievable error is zero and a
  floor of zero would make the cost unbounded. The instrument resolution is
  kept as the scale there.

A test (`tests/test_reward_contract.py`) checks every measured floor: the
shipped MPC's long-run tracking cost must sit at or above the floor's cost,
within sampling error. A floor the MPC beats by more is wrong. Each plant's
`PHYSICS.md` gives its floor's derivation.

## Per-plant summary

| plant | tracking | `e_floor` (how) | `e_tol` | running cost | units |
| --- | --- | --- | --- | --- | --- |
| `reactor` | p=1 | 0.00451 of rated (certified DP) | 0 | rod demand beyond the rate limit, weight 1 (provisional) | $ per 10 s step, imbalance $100/MWh |
| `hvac` | p=2, dead-zone | overheating-bound; MPC reference | +-0.5 K occupied; night lower bound only (provisional) | gas EUR 0.10/kWh, in full | EUR per step; comfort EUR 0.03/K^2 h (provisional; restarts in place) |
| `battery` | p=1 | 1596 W (closed form) | 0 | fade above hold at $300/kWh of capacity | $ per step, imbalance $100/MWh |
| `wind_turbine` | p=1 | 1680 W (an earlier oracle's lowest per-seed hold, upper bound; the current one holds 1454 W) | 0 | pitch activity above hold, weight 1 (provisional) | $ per step, imbalance $100/MWh (provisional) |
| `glass_furnace` | p=2 | 1 K (thermocouple resolution; the MPC holds 0.0614) | 0 | fuel above hold, w=1 | dimensionless |
| `cement_kiln` | p=2 | 5e-4 (assay resolution; the MPC holds 4.9e-5) | 0 (provisional) | fuel above hold, w=1 | dimensionless |
| `boiler_drum` | p=2 x2 | 2.7 mm level (an earlier oracle's lowest per-seed hold; the current one holds 2.45 mm), 0.05 bar (transmitter resolution; the MPC holds 0.052) | 0 | fuel above hold, w=1 | dimensionless |
| `distillation` | p=2 x2 | 1e-4 / 1e-4 (analyser resolution; the MPC holds 7.6e-7 / 5.2e-7) | 0 (provisional) | boilup above hold, w=1 | dimensionless |
| `ph_neutralization` | p=2 | 0.01 pH (electrode resolution; the MPC holds 0.0013) | 0 (provisional) | reagent above hold, w=1 | dimensionless |
| `unstable_cstr` | p=2 | 1e-4 mol/L (analyser resolution; the MPC holds 6.29e-6, lowest per seed, so `rho* = (6.29e-6 / 1e-4)^2 = 0.00396`) | 0 (provisional) | none | dimensionless |
| `compressor_surge` | p=2 | 0.0275 kPa (pressure transmitter accuracy, 0.055 % (read) of a 0 to 50 kPa span (ours); the MPC holds 1.03e-4, lowest per seed, so `rho* = (1.03e-4 / 0.0275)^2 = 1.40e-5`) | 0 (provisional) | recycle power above hold, w=1 (`c_hold` 62 270 W, the MPC's measured hold consumption) | dimensionless |
| `cstr`, `first_order`, `four_tank` | p=2 | documented minima (no disturbance; `rho_floor = 0`) | 0 | none | dimensionless |
| `plane`, `plane_sine`, `plane_energy` | p=2 | 1 / 1.26 / 1.20 m: `plane`'s is the altimeter resolution (the MPC holds 0.46 m); `plane_sine`'s an earlier MPC's hold, kept within 1.5x of that resolution (the MPC holds 0.53 m); `plane_energy`'s its MPC's lowest per-seed hold between ladder steps, from 4.55 m in `-v3` (the MPC holds 1.18 m under it); lowest per-seed holds in the test turbulence | 0 (a +-30 m band made the hold vacuous) | airspeed deviation above hold, w=1 | dimensionless |
| `plane3d_*` | p=2 | altitude 1 m (altimeter resolution; the MPC holds 0.59 / 0.67 / 0.68 m on heading / circle / racetrack), heading 0.0087 rad (AHRS resolution; the MPC holds 1.4e-5), path 3 m (GPS resolution; the MPC holds 0.45 / 0.47 / 1.29 m on circle / racetrack / figure-8); lowest per-seed holds in turbulence | 0 | none | dimensionless |
| `patrol`, `patrol_bearing_only` | p=2 x2 | slot 3 m (relative-GPS resolution; the oracle holds 0.10 m, lowest per seed; 18.6 m, an earlier planner's hold, until version 3), heading 0.0087 rad (AHRS resolution; the oracle holds 1.6e-3 rad), so `rho* = (0.1021 / 3)^2 + (1.631e-3 / 0.0087)^2 = 0.0363` | 0 (provisional) | none | dimensionless |
| `formation` | p=2 x6 | the lead's altitude 1 m (altimeter resolution; the oracle holds 0.80 m) and heading 0.0087 rad (AHRS resolution; 1.0e-3 rad), each wingman's slot 3 m (relative-GPS resolution; 0.23 and 0.23 m) and alignment 0.0087 rad (1.1e-3 and 1.1e-3 rad); lowest per-seed holds, so `rho* = 0.704`, 0.646 of it the lead's altitude | 0 (provisional, the patrol's) | none | dimensionless |

"Provisional" marks a number the plant's owner would supply (a tolerance from
the quality system, permit or grid code; a price from a tariff; a wear weight
from a maintenance model) that ships here as a documented stand-in. Each
`PHYSICS.md` says where a plant engineer would get the real one.

Under this reward the shipped MPC is the ceiling on every plant: its gain is
below the PID's everywhere ([Baseline results](baseline-results.md)).
[How the baselines work](baselines.md#why-the-mpc-does-not-minimise-the-reward)
says how the planners treat it.

## Version 1

Version 1, the reward of the 0.6 release, was a log-scaled tracking term capped
to [0, 1] and multiplied by one minus a running cost. It is still
constructible with `reward_version=1` on any params, and its baselines are
kept in `src/target_gym/data/baseline_returns_v1.json`. Version 2 replaced it
for three reasons: its concave tracking preferred rare large excursions, its
product charged nothing for consumption exactly where tracking was worst, and
a trip ended the episode, so its price depended on when it happened. The
CHANGELOG records the change.
