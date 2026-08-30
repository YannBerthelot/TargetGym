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
| Episode covers settling | ≥ 4 τ | **200 steps = 20 τ** | ✅ |
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

**Observation** `[x, target_x]`. **Reward** — a squared normalised tracking
band. There is no disturbance, nothing hidden, and no irrecoverable state.

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
