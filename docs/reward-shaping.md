# Reward shaping

TargetGym exists to ask one question: **can a learned policy hold a setpoint
better than a PID or an MPC?** That question lives entirely in the reward, so
the reward is the measuring instrument, and this page records how it is built.
Version 2 of every environment (the `-v2` stamps; the reactor is `-v3`) scores
through it -- seventeen with the tracking term normalised by a floor, and four
(reactor, battery, wind turbine, building) with tracking and consumption in
the owner's currency, where the floor enters as the reference cost `rho_floor`
rather than as a divisor; version 1, the capped log-scaled reward, is kept
constructible (`reward_version=1` on any params) and described at the end of
this page.

## The reward, in one line

```
reward = -( tracking_cost + running_cost + failure_cost )
```

Three costs, each non-negative, added -- never multiplied -- and each in a unit
the plant's owner could defend. `target_gym.reward` holds the arithmetic;
every environment's `compute_reward_terms` returns the three terms and
`compute_reward` is minus their sum. The best achievable per-step reward is
about `-1` per tracked output on the dimensionless plants (tracking at the
floor, nothing avoidable consumed), and minus the unavoidable bill on the
priced ones. It is not capped at 1, and the per-step scale differs by orders of
magnitude between plants: comparability across plants comes from the
normalised expert advantage and from reporting the two costs separately
(`target_gym.eval`), not from a cap.

### Tracking: the error in units of what is achievable

```
tracking_cost = ( max(|e| - e_tol, 0) / e_floor ) ** p
```

`e_floor` is the **achievable floor**: the smallest long-run mean |error| any
controller can hold on this plant under its shipped reference and
disturbance processes -- floored at the resolution of the instrument the
plant's table cites. Measurement noise is not modelled, so a simulator can
hold finer than a real transmitter reads; a hold the instrument cannot see
is not a floor, and on six plants (glass, kiln, boiler pressure,
distillation, pH, the aircraft's altitude and heading) the resolution is
the scale and the finer measured hold is recorded beside it. An error at the floor costs 1 per step. The scale is
the plant's own irreducible error, not a sensor resolution and not the
operating envelope, so a controller's tracking cost reads directly as "how
many floor-widths off". `e_tol` is a specification tolerance where the plant
has one (a comfort band, a purity spec, a permit limit): the cost is zero
inside it -- a dead-zone relaxation, which for a linear cost forfeits at most
the tolerance in long-run cost (for p = 2 the forfeit at an error `E` is
`(2 E e_tol - e_tol^2) / e_floor^2`, so the band must be small against the
errors a controller actually makes; the aircraft's +-30 m band was not, and
was dropped) -- and inside the band only consumption is optimised. `p` is 1
where the owner's cost is linear in the error (an energy imbalance settled
per MWh) and 2 otherwise. Multi-output plants sum one such term per output,
each with its own floor.

The shape is convex in |e| by construction. Version 1's log-scaled tracking was
concave: it paid the same for every halving of the error, and a concave cost
prefers rare large excursions to frequent small ones of the same mean -- the
wrong preference for a hold, where a controller that mostly sits well and
occasionally loses the plant should not outscore one that sits slightly worse
and never does. The Target-MDP note (Proposition 8) gives the three-state
example where that preference costs an arbitrary amount.

### Running cost: only what could have been avoided

Consumption -- fuel, energy, boilup, reagent, actuator travel -- is charged
above `c_hold`, what the best shipped controller consumes per step while
holding: the avoidable part. Dimensionless form

```
running_cost = w * max(c - c_hold, 0) / c_hold
```

with `w` read as "one floor-width of tracking error is worth `w` times the
hold-phase consumption again", default 1, swept {0.5, 1, 2} where the number
matters. Where the owner's tariff is known the term is priced instead
(`price * quantity`), and the tracking term is then in the same currency so
there is no weight to choose: the reactor's imbalance at three times spot, the
building's gas per kWh, the battery's imbalance per MWh and fade per kWh of
lost capacity. Wear and fatigue have no defensible tariff; they are charged
per unit of avoidable activity at what tracking at the floor costs, times a
documented weight -- a stand-in for a maintenance model, which is why the
sweep is the honest form.

Version 1 multiplied: `tracking * (1 - w * running)`. A product charges
nothing for consumption exactly where tracking is worst, and a cost that
vanishes when it should bind is not a cost.

### Failure

Leaving the operating envelope trips the plant. A reach-and-hold task is
continuing, so a trip is not the end of an episode: the step that leaves the
envelope is charged the **trip cost**, `restart_steps * failure_cost` -- the
time a real restart takes, priced at a per-step failure cost of twice the
largest tracking cost the plant can *reach* (its target range against its
trip bounds, or its steady-state range; not a nominal span it never visits)
-- with that step's tracking and running cost zeroed, and the plant restarts
at once as `reset_env` would, on the same window clock
(`base.failure_kernel`, `reward.trip_cost`). A plant that does not restart
cold -- the building, whose thermal mass persists through a lockout --
carries `restart_in_place = True`: it keeps its state, and every step outside
the envelope is charged until the controller brings it back.
The long-run cost then decomposes as
`rho_hold + p * B + lambda * (restart_steps * failure_cost + B_restart)`,
with `lambda` the trip rate, which the protocol reports separately; the
absorbing failure of the Target-MDP note is the `restart_steps -> inf`
limit. Restart times are provisional numbers a plant engineer would supply
(a SCRAM's xenon-limited restart, a furnace heat-up schedule, a lost sortie);
each plant's `PHYSICS.md` gives its value and source.

Two earlier forms were tried and dropped. Charging the trip once per terminal
step, and then for the steps the episode had left, both priced a trip by
where in the episode it happened. Freezing the plant at `failure_cost` for
`restart_steps` steps -- downtime lived through -- paid the same total but as
a dead stretch of identical steps: no information for a learner, and inside a
test window shorter than the restart (every plant but the building) the
restart never came, so the price was again set by the window. The lump is
that total, paid where it is incurred.

No plant raises `terminated`. Raising it tells a discounted learner the
crashed state is worth zero -- the cheapest state in the plant, and the
bootstrap behind every agent that learns to crash -- and cuts the chain an
average-reward learner estimates its gain on. The trip is `info["tripped"]`,
read by the evaluator and by nothing an agent runs. Two plants never trip:
the pH loop's effluent is a convex mix of its inlet streams and cannot reach
the 2 / 12 limits, and the CSTR settles between 318 and 329 K under any
coolant setting; their envelope limits are documentation.

## Why the floor: loop performance assessment, in cost units

The floor-normalised cost is the reinforcement-learning form of **control-loop
performance assessment**. Harris (1989) showed that the minimum-variance bound
of a loop can be estimated from routine closed-loop data given the process
delay, and that the ratio of the actual output variance to that bound -- the
Harris index -- tells an engineer how much of the variance the controller is
responsible for; Desborough and Harris (1992) turned it into the assessment
measures industrial loop auditing uses, Huang and Shah (1999) extended it to
feedforward, multivariable and LQG benchmarks, and Jelali (2006) surveys the
technology and its industrial uptake.

Our `e_floor` is that bound computed exactly on the simulator rather than
estimated from plant data -- the one thing a simulator gives that a plant
cannot, and the reason the sanity test below is possible. The normalised
expert advantage `NEA = (PID - x) / (PID - rho*)`, with `rho*` the cost at the
floor, is a Harris-type index in cost units: 1 at the bound, 0 at PID parity,
negative below PID. `rho*` is the lowest per-seed hold cost the reference
controller demonstrated, in the reward's units: 1 per tracked term where the
floor is that hold, and less where the floor is clamped at the instrument
resolution (the glass furnace's MPC holds 0.175 K against a 1 K scale, so
`rho* = 0.03`). On the deterministic plants, where `e_floor` is a
resolution used as a scale and exact hold is achievable, `rho*` is 0 -- a
reference of 1 there (the cost at the resolution) is not a bound, and the
shipped MPCs sit below it. And the two-cost report -- tracking against consumption --
is Huang and Shah's LQG performance curve: the trade-off frontier a controller
is judged against, rather than one scalar that hides where on it a controller
sits.

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

`scripts/measure_hold.py` runs the shipped PID and MPC on every plant for
three cost-bearing time constants of burn-in and then a hold window, under the
shipped reference and disturbance processes, and records the long-run mean
|error| per output, the consumption per step, and the settling time after a
target change (`src/target_gym/data/hold_measurements.json`). Three kinds of
floor come out of it:

- **Certified or closed-form** where the hold problem reduces to one the
  optimum can be computed for: the reactor (`scripts/floor_reactor_hold.py`, a
  one-state dynamic programme on the error against the demand's within-period
  random walk) and the battery (white dispatch noise drawn after the action,
  so `E|error| >= sd * sqrt(2/pi)`). The shipped MPC must hold at or above
  these floors, and does.
- **Upper bounds** where no reduced-model optimum exists yet: the best shipped
  controller's own long-run hold error, labelled as such in the plant's
  PHYSICS.md. A learner that beats it scores a tracking cost below 1, which is
  allowed and informative.
- **Documented minima** on the plants whose test configuration has no
  disturbance at all -- the aircraft and patrol tasks fly with zero
  turbulence, the CSTR, first-order plant and four-tank have none -- where the
  achievable hold error is zero and a floor of zero would make the cost
  unbounded. There the version-1 resolution floor is kept as the scale, and
  the PHYSICS.md says so.

The sanity test every measured floor has to pass, and a slow test enforces
(`tests/test_reward_contract.py`): the shipped MPC's long-run tracking cost
after burn-in is at or above the floor's cost. A floor the MPC beats is wrong.

## Per-plant summary

| plant | tracking | `e_floor` (how) | `e_tol` | running cost | units |
| --- | --- | --- | --- | --- | --- |
| `reactor` | p=1 | 0.00451 of rated (certified DP) | 0 | rod demand beyond the rate limit, weight 1 (provisional) | $ per 10 s step, imbalance $100/MWh |
| `hvac` | p=2, dead-zone | overheating-bound; MPC reference | +-0.5 K occupied; night lower bound only (provisional) | gas EUR 0.10/kWh, in full | EUR per step; comfort EUR 0.03/K^2 h (provisional; restarts in place) |
| `battery` | p=1 | 1596 W (closed form) | 0 | fade above hold at $300/kWh of capacity | $ per step, imbalance $100/MWh |
| `wind_turbine` | p=1 | 1680 W (lowest per-seed MPC hold, upper bound) | 0 | pitch activity above hold, weight 1 (provisional) | $ per step, imbalance $100/MWh (provisional) |
| `glass_furnace` | p=2 | 1 K (thermocouple resolution; the MPC holds 0.175) | 0 | fuel above hold, w=1 | dimensionless |
| `cement_kiln` | p=2 | 5e-4 (assay resolution; the MPC holds 3.4e-4) | 0 (provisional) | fuel above hold, w=1 | dimensionless |
| `boiler_drum` | p=2 x2 | 2.7 mm level (lowest per-seed MPC hold), 0.05 bar (transmitter resolution; the MPC holds 0.028) | 0 | fuel above hold, w=1 | dimensionless |
| `distillation` | p=2 x2 | 1e-4 / 1e-4 (analyser resolution; the MPC holds 1.35e-5 / 3.3e-5) | 0 (provisional) | boilup above hold, w=1 | dimensionless |
| `ph_neutralization` | p=2 | 0.01 pH (electrode resolution; the MPC holds 0.0080) | 0 (provisional) | reagent above hold, w=1 | dimensionless |
| `cstr`, `first_order`, `four_tank` | p=2 | documented minima (no disturbance; `rho_floor = 0`) | 0 | none | dimensionless |
| `plane`, `plane_sine`, `plane_energy` | p=2 | 0.84 / 1.26 / 4.55 m (lowest per-seed MPC holds in the test turbulence, upper bounds) | 0 (a +-30 m band made the hold vacuous) | airspeed deviation above hold, w=1 | dimensionless |
| `plane3d_*` | p=2 | altitude 1.44 / 4.06 / 1.39 m, heading 1.0e-4 rad, path 8.1 / 6.2 / 14.6 m (lowest per-seed MPC holds in turbulence) | 0 | none | dimensionless |
| `patrol` | p=2 | 18.6 m / 1.6e-3 rad (lowest per-seed MPC holds in turbulence) | 0 (provisional) | none | dimensionless |

"Provisional" marks a number the plant's owner would supply -- a tolerance
from the quality system, permit or grid code; a price from a tariff; a wear
weight from a maintenance model -- and that ships here as a documented
stand-in. Each PHYSICS.md says where a plant engineer would get the real one.

## What the shipped controllers do under it

The MPC is presented as the benchmark's ceiling, so under this reward it has
to be one: on every plant its episode return and its hold cost are at or
below the PID's (`docs/baselines.md`). Getting there was not a matter of
re-tuning. Five things in the planners had been written against the
version-1 reward and stopped being ceilings under version 2, and each is
fixed in `experts/mpc.py` with the measurement that found it:

- **The surrogate objectives mirrored the version-1 minimiser.** The
  gradient and sampling planners (wind turbine, battery, aircraft, boiler
  drum, cement kiln) now descend the plant's own version-2 cost in floor
  units, keeping their differentiable barriers (weighted like the failure
  charge); the HVAC CasADi planner minimises the priced dead-zone comfort and
  the gas. Where the tracking cost is linear in the error (p = 1) the planner
  squares it -- same minimiser, and a gradient that vanishes at it, where a
  normalised-gradient step on a linear cost never stops chattering.
- **A 60-step open-loop tail dominated the aircraft objective.** Under a
  bounded reward the tail was harmless; under an unbounded quadratic cost it
  was 1e5 per solve against 26 per step realised, and the planner optimised
  what the held action did later -- parking at the edge of the altitude
  tolerance 55 m/s below cruise. Version 2 plans without it.
- **A normalised-gradient planner cannot travel far in one solve**, so from
  a constant plan it could not find the pitch schedule the turbine needed
  (it braked the rotor with the torque instead) or the coordinated
  thrust-and-elevator move the aircraft needed. The wind, 2D aircraft and
  patrol planners now start from, and at every step are compared against,
  the shipped PID's rollout plan under the planner's own objective -- so the
  plan is never worse than the PID's under its model. (The patrol planner
  had kept its version-1 surrogate, a bounded multiplicative shape; once the
  descent below was made monotone, a better solve of that surrogate was a
  worse version-2 return, 18x on two seeds. It descends the follower's own
  cost now.)
- **Re-planning creates actuator activity no open-loop plan can see.** The
  turbine's pitch activity ran 3.4x the PID's with every plan predicting
  less; a move-suppression term on the first action, priced like the
  fatigue term, closes the gap. The planner also keeps the rotor within 5%
  of rated speed with a mild soft box -- what a turbine's own supervisory
  logic does -- because recovering a slowed rotor pays off beyond its
  horizon and without the box it drifted to 0.85x rated with a 250 kW error.
- **The descent was not monotone, and the planner took its last iterate
  regardless.** A fixed step along a normalised gradient overshoots wherever
  the cost has an edge -- a tolerance band, a barrier -- and fifty of them
  can end far from where they started: on the 2D aircraft's seed 1 the
  shifted plan scored -0.003 and the descended plan -3.31, three steps later
  -0.99 against -19.7; the planner then reached for the PID's guide, dived
  33 m out of the tolerance band and, over the hold, paid twice the PID.
  The one-seed protocol run had not seen it. The gradient planner now
  returns the best iterate of the solve, warm start included, so under its
  own model it never leaves a solve with a worse plan than it entered with.

One measurement bug came out with it: `runners.baseline_policy` built the
MPC on the raw params rather than `plan_params`, so on the plants with
`noise_fields` (wind, battery) every hand-run MPC planned against one fixed
noise realisation -- a wrong forecast, and on seed 0 a perfect one.

---

## Version 1: the log-scaled reward (history)

What follows is the page as it stood for version 1, kept because the shape it
argues for is still the right *sensitivity* argument -- a reward has to keep
paying for precision all the way down -- and because the resolution floors it
tabulates are the documented minima the version-2 floors fall back on where a
plant has no disturbance.


TargetGym exists to ask a specific question: **can a learned policy hold a
setpoint better than a PID or an MPC?** That question lives entirely in the
reward. A reward that saturates once the error is "small enough" scores a
policy holding 1 m the same as one holding 10 m, and the comparison the
benchmark was built to make becomes invisible — not wrong, invisible.

So the reward is not a detail to be tuned afterwards. It is the measuring
instrument, and this page records how it was chosen.

## The constraint nobody escapes

A bounded per-step reward has a fixed amount of dynamic range to spend, and an
environment has a wide error range to cover. The aircraft's altitude envelope
is 12 km; the tracking that distinguishes a good controller from a great one
happens over metres. **Every reward shape is a decision about where to spend
that range.**

- **Linear in the error** spends it uniformly per metre. With 12 000 m to cover,
  the last 10 m receive 0.08 % of the range.
- **A band** (Gaussian or rational, `e/band`) spends nearly all of it within a
  few multiples of the band, and almost none inside it or far outside.
- **Logarithmic** spends it uniformly *per decade* — the same amount separating
  1000 m from 100 m as separating 1 m from 0.1 m.

Only the last is scale-free, and scale-free is what "closer is better" has to be
if it is to mean the same thing near the target and far from it.

## What was measured

One family of controllers — the shipped cascaded PID, detuned by scaling its
outer altitude gain — flown on the 2D aircraft over three seeds. The *same*
trajectories scored under four reward shapes, each normalised so that reward is
1 at the target and 0 at the edge of the envelope, which makes the columns
comparable.

| `Kp_alt` | settled \|err\| | original `(1-e/span)¹⁰` | pseudo-Huber | band `1/(1+(e/b)²)` | log-scaled |
|---|---|---|---|---|---|
| ×0.15 | 311.75 m | 0.7883 | 0.9775 | 0.2386 | 0.4491 |
| ×0.30 | 63.17 m | 0.9516 | 0.9966 | 0.6234 | 0.6512 |
| ×0.60 | 17.42 m | 0.9861 | 0.9994 | 0.8868 | 0.7633 |
| ×1.00 | 13.36 m | 0.9892 | 0.9997 | 0.9016 | 0.7920 |
| ×1.60 | 9.41 m | 0.9924 | 0.9998 | 0.9330 | 0.8297 |
| ×2.40 | 6.49 m | 0.9947 | 0.9999 | 0.9578 | 0.8613 |

All four rank the controllers correctly. Ranking is not the hard part. What
matters is **how much reward the last improvement is worth**, because that is
the signal an optimiser has to find and hold on to:

| shape | 312 m → 6.5 m | 9.4 m → 6.5 m | share earned in the last step |
|---|---|---|---|
| original `(1-e/span)¹⁰` | 0.2065 | 0.0024 | 1.1 % |
| pseudo-Huber | 0.0224 | 0.0001 | **0.4 %** |
| band `1/(1+(e/b)²)` | 0.7193 | 0.0249 | 3.5 % |
| **log-scaled** | 0.4122 | **0.0317** | **7.7 %** |

The log-scaled reward pays the most, in absolute terms, for the most precise
improvement — and it is the only one where refining an already-good controller
is worth a meaningful share of the total.

## Why pseudo-Huber did worst, which was not expected

Quadratic near zero and linear far away is the standard robust-regression
answer, and its *shape* is reasonable. It came last anyway, separating the
whole family by 0.022 out of 1.

The reason is the normalisation, not the shape. Made bounded over a 12 km
envelope, its near-linear far field consumes essentially the entire range, so
everything below the transition scale compresses into ≈ 1. It fails for the
same reason the original did: it spends its dynamic range on the far field.

This is worth stating plainly because it is the trap. Reasoning about a reward's
*local* shape — "quadratic near the target, so it is sensitive there" — says
nothing until you ask what fraction of the bounded range that region receives.

## The chosen shape

```python
# doc: skip -- the formula as it appears in plane/env.py, not a runnable snippet
reward = 1 - log1p(error / precision_floor) / log1p(span / precision_floor)
```

1 at the target, 0 at the edge of the envelope, and every halving of the error
worth the same: 0.0835 from 1600 m to 800 m, 0.0824 from 100 m to 50 m, 0.0678
from 6.25 m to 3.13 m.

Two parameters, and the distinction between them matters:

- **`span`** normalises the result into [0, 1]. Unlike the shapes above, it does
  *not* set the sensitivity — a logarithm has no scale, so widening the envelope
  rescales the reward without changing what it prefers.
- **`precision_floor`** is the error below which more precision stops *meaning*
  anything. It is a resolution, not a tolerance: the reward pays for every
  halving down to this point and only flattens beneath it. It should be a
  physical limit — a barometric altimeter reads to about a metre, so tracking
  tighter would reward chasing measurement noise. It is also what keeps the
  reward bounded, since an unfloored logarithm diverges at zero error.

A **band** is the thing to avoid here, and the distinction is easy to lose: a
band says "get inside this tolerance", a floor says "keep getting closer, until
closer stops being measurable".

## Status

**Every environment used this shape in version 1.** The aircraft family came first (the
2D plane, all three 3D tasks, and the lead term of the patrol formation); the
twelve process and energy plants followed. All of them route through a single
`log_scaled_reward` in `utils.py` rather than eighteen transcriptions of the
formula.

Converting the rest was not a sweep. Measuring first turned the phase into a
list: four environments — the glass furnace, boiler drum, reactor and HVAC —
scored *identically zero* across the first three halvings of their error, which
is no gradient at all for a controller far from its setpoint. That is the same
clipped plateau that had made two MPC baselines give up, still sitting in the
rewards themselves.

Each floor is an instrument resolution, never a tolerance:

| Environment | Floor | Source |
| --- | --- | --- |
| aircraft (altitude) | 1 m | barometric altimeter resolution |
| aircraft (heading) | 0.0087 rad (0.5°) | AHRS / compass resolution |
| aircraft (position) | 3 m | civil GPS horizontal accuracy |
| `cstr` | 1e-4 mol/L | composition analyser |
| `first_order` | 6e-3 | generic transmitter span |
| `four_tank` | 1e-3 m | level transmitter |
| `ph_neutralization` | 1e-2 pH | glass pH electrode |
| `distillation` | 1e-4 mole fraction | composition analyser |
| `glass_furnace` | 1.0 K | type-B/S thermocouple at 1700 K |
| `reactor` | 1e-4 | neutron flux instrumentation |
| `hvac` | 0.1 K | room temperature sensor |
| `cement_kiln` | 5e-4 | free-lime assay |
| `boiler_drum` | 1e-3 m / 0.05 bar | level transmitter / pressure transmitter |
| `wind_turbine` | 1e3 W | revenue-grade power metering |
| `battery` | 1e3 W | revenue-grade power metering |

Envelopes are the span at which the plant is lost, so the reward reaches zero
exactly where the episode would end — the boiler drum's level trip, the
reactor's flux limits.

### Costs multiply, they do not subtract

Tracking is not the only thing these plants are scored on: several also pay for
fuel, reagent, boil-up, pitch activity or cell degradation. Those terms used to
be *subtracted*, which let two environments score below zero — and a negative
step reward means ending the episode early beats tracking badly, which inverts
the entire point of a target MDP.

Adding the cost back as a credit fixes the sign and introduces something worse.
Measured: switching the HVAC heater off and abandoning the setpoint entirely
still banked **0.310 every step**, because a term paid independently of the
target rewards ignoring the target.

A cost is not a second objective competing with the setpoint. It is a
tiebreaker among ways of *holding* the setpoint, and multiplying says so:

    reward = (tracking terms, multiplied) * (1 - sum of weighted costs)

Bounded in `[0, 1]`; maximised only by holding the target; zero tracking earns
nothing however little is spent; two controllers that track equally are still
separated by what they burn. Non-negativity then comes free, so a short episode
can never beat a long one — which also made the aircraft's flat `-200` crash
penalty redundant, and it has been removed. Termination already costs every step
it forgoes.

### The gradient is not scale-free, only the value is

Worth knowing before reusing this shape for anything gradient-based. A
log-scaled reward is scale-free in *value* — each halving of the error is worth
the same increment, which is what makes it good to learn from. Its gradient is
not. Differentiating gives

    dr/de = -1 / ((floor + e) * log1p(envelope / floor))

which decays like `1/e`: the pull toward the setpoint is weakest exactly where
the controller is furthest from it. For a learner reading returns this is
fine. For a gradient planner descending the reward directly it is not — the
wind turbine's MPC scores 341.9 on a quadratic surrogate against 172.1 on the
reward itself. That is why two MPC baselines still carry surrogate objectives
sharing the reward's minimiser; see `docs/baselines.md`.

### What converting the 3D tasks turned up

The two path-following tasks scored proximity with a Gaussian a tenth of the
path radius wide — σ ≈ 1 km for a 10 km circle. Across the entire band those
controllers actually operate in, that reward is flat: from a 100 m cross-track
error down to 0.1 m it moves by 0.005. The log-scaled form moves by 0.43 over
the same range.

Fixing the shape then exposed a defect underneath it. The figure-8 finds its
cross-track error by `argmin` over 400 samples of a 44 km curve, so the number
is quantised by the sample spacing: an aircraft flying the commanded curve
*exactly* was reported up to **66 m** off it — more than the figure-8 expert's
own settled error, meaning the reward had been measuring its own discretisation.
Projecting onto the two adjacent chords brings the floor below a millimetre and
makes a known 1 m offset read as 1.000 m. A reward can only pay for precision
its error metric can see; that is now check 11 on the model review checklist.

Two things found while doing the aircraft, recorded so the phase starts from
them:

- **The existing bands are guesses.** Five of eleven are not mentioned in their
  own `PHYSICS.md` at all, against a convention that marks 235 parameter rows
  as sourced and 54 as `TUNED`. Where a real operational tolerance exists —
  ASHRAE comfort ranges, drum-level trip bands, free-lime quality specs — the
  floor can be sourced rather than guessed.
- **Four-tank's D1 already records this defect**: *"the reward band was three
  times the operating range … originally normalised by the full tank span."*
  The aircraft was an unfixed instance of a failure mode this repository had
  already diagnosed once, which is the argument for writing the convention down
  rather than rediscovering it a third time.
