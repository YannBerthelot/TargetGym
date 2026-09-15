# First-order system — model, provenance and validation

A **single first-order lag**. This is the only environment in the suite that is
not a physical plant, and the contract exists mostly to say so plainly.

Contract for `target_gym.pc_gym.first_order`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model and provenance

```
τ · dx/dt = K·u − x
```

**Provenance: adapted from [PC-gym](https://github.com/MaximilianB2/pc-gym).**

There is no physics to source here and nothing to derive. `K` and `τ` are not
measurements of anything — they are the two numbers that define a first-order
lag, and they were chosen for convenient dynamics rather than fitted to a
process. Every other `PHYSICS.md` in this repository validates a model against
published behaviour; this one validates that the analytic solution of a lag is
what the integrator produces, and is otherwise a statement of scope.

That is the honest framing, and it is worth stating because the environment
looks like the others in the registry, ships the same baselines, and appears in
the same gallery. It is a **conformance and sanity-check fixture**: if an
algorithm cannot solve this, the problem is the algorithm.

---

## 2. Validation targets

Everything here is checked against the closed-form solution
`x(t) = K·u·(1 − e^{−t/τ})`, not against a reference process.

| Quantity | Target | Model | |
|---|---|---|---|
| Step response reaches 63.2 % of final at t = τ | analytic | matches | ✅ |
| Settles to `K·u` | analytic | matches | ✅ |
| Time-constant resolution | ≥ 5 steps per τ | **10 steps** | ✅ |
| Episode covers settling | ≥ 4 τ | **100 steps = 10 τ** (benchmark; the class default of 200 is 20 τ) | ✅ |
| Every target reachable | required | `u = x/K` needs 0.5–1.5 of ±2.0 | ✅ |
| No overshoot from a step | first order cannot overshoot | none | ✅ |
| Monotone step response | required | yes | ✅ |

The reachability check is the one that earns its place: it is the same check
that the four-tank environment failed, and it costs nothing to assert here.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `K` | 1.0 | – | PC-gym; a convenient gain, not a measurement | ⚠️ |
| `tau` | 0.5 | s | PC-gym; likewise | ⚠️ |
| `u_min`, `u_max` | −2.0, 2.0 | – | THIS REPO | ✅ |
| `x_min`, `x_max` | −3.0, 3.0 | – | THIS REPO | ✅ |
| `target_x_range` | (0.5, 1.5) | – | THIS REPO; reachable with 25 % input margin | ✅ |
| `delta_t` | 0.05 | s | 10 steps per time constant | ✅ |

---

## 4. Task design

**Observation** `[x, target_x]`. **Reward** — `log_scaled_reward` against the
6.0 envelope with a floor at `precision_floor = 6e-3`, the same form the rest of
the suite uses. It replaced a squared normalised band, which spent nearly all of
its range on errors a controller had already closed.

There is no disturbance, nothing hidden, and no irrecoverable state. The
terminal guard on `|x| > 3` is unreachable: `x` is first-order toward `K·u`,
which the action bounds cap at ±2, from a start inside ±0.5. This environment
cannot end early, so its `mpc_terminated_early` is always 0.

The environment exists so that a new algorithm, wrapper or integration method
can be checked against something with a known answer before being pointed at a
kiln.

---

## 5. Known deviations

**❌ D1 — this is not a physical model.** No conservation law, no sourced
parameter, no regime of validity. Nothing about it should be read as evidence
that an algorithm will work on a real process.

**⚠️ D2 — nothing is hidden and nothing is stochastic.** Fully observed,
deterministic given the sampled setpoint, and linear. It shares none of the
properties — partial observability, irrecoverable states, non-minimum phase,
transport delay — that the rest of the suite exists to pose.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `first_order` | 100 | 0.05 | 5 s | 1 in [-1, 1] | 2 | 3 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 6e-3 | documented minimum: a thousandth of the span. The test configuration has no disturbance and a fixed target, so the achievable hold error is zero -- the shipped MPC holds 0.0 after settling (`scripts/measure_hold.py`, 300 hold steps after a 30-step burn-in; PID 2e-6). |
| `e_tol` | 0 | no specification band on a generic plant |
| `tracking_exponent` | 2 | quadratic |
| `failure_cost` | 2e6 | twice the span's cost, (6 / 6e-3)^2 |
| `restart_steps` | 100 (5 s) | restart time priced into a trip, `restart_steps x failure_cost` (a generic loop's reset; provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the tracking cost per step at the floor in the reward's
units (the NEA floor for tracking) and `rho_floor` the full floor including
consumption charged in full; `floor_is_documented_minimum` records whether
`e_floor` is a measured/certified floor or a resolution used as a scale, and
where it is a resolution (a deterministic plant) both references are 0, since
exact hold is achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` is never
raised; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
