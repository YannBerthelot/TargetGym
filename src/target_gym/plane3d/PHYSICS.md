# 3D aircraft — physics model, provenance and validation

The 3D aircraft extends the 2D longitudinal model with **roll**, and everything
that follows from it: banked turns, load factor, and heading as a derived
quantity rather than a state you command.

Contract for `target_gym.plane3d` **and `target_gym.patrol`**, which imports
the same dynamics and adds only task geometry. Longitudinal aerodynamics —
lift curve, drag polar, atmosphere, thrust, stall — are unchanged from
[`plane/PHYSICS.md`](../plane/PHYSICS.md) and are not restated here.

Method: `docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. What the third dimension adds

Three things, and nothing else:

1. **Lift tilts with bank.** The wing's force stays perpendicular to the wing,
   so banking splits it: `L·cos φ` holds the aircraft up and `L·sin φ` turns it.
2. **A roll moment**, from ailerons against roll damping:
   `M_roll = M_aileron + C_lp · (p·b / 2V) · q̄ · S · b`.
3. **Heading is derived**, not integrated as an input. It comes out of the
   horizontal velocity vector, so a turn is something the aircraft *does*, not
   something the model is told.

The consequence worth stating: **the aileron commands a roll rate, not a bank
angle.** Roll damping sets a steady rate for a steady deflection, exactly as on
a real aeroplane — hold the stick over and it keeps rolling. Holding a bank
requires returning the aileron to neutral, which is why every 3D controller
here is a cascade with an inner bank loop.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Yaw, rudder, sideslip | Turns are assumed coordinated; there is no sideslip state and no adverse-yaw compensation. |
| Spiral mode and Dutch roll | Both are lateral-directional modes that need yaw. |
| Roll–pitch inertial coupling | Pitch and roll moments are computed independently. |
| Aileron reversal, control-surface aeroelasticity | Out of scope at these speeds. |

**Regime of validity.** Bank angles up to about 45°, coordinated flight, in the
same envelope as the 2D model. Beyond that the missing yaw axis matters.

---

## 2. Validation targets

| Quantity | Target | Model | |
|---|---|---|---|
| Coordinated-turn rate ψ̇ = g·tan φ / V | analytic | **within 0.5 %** at 10°, 20°, 30° | ✅ |
| Load factor n = 1/cos φ | analytic | 1.015 / 1.064 / 1.155 at those banks | ✅ |
| Roll damping derivative C_lp | −0.4 to −0.5 (transport) | **−0.4** | ✅ |
| Wingspan | 35.8 m (A320neo, sharklets) | 35.8 | ✅ |
| Steady aileron → steady roll **rate** | required | yes, not a steady angle | ✅ |
| Turn radius at 30° bank, 230 m/s | kilometres | **9.3 km** | ✅ |
| Longitudinal behaviour vs the 2D model | identical at φ = 0 | shared code path | ✅ |

**The coordinated-turn check is the load-bearing one.** Nothing in the model
computes a turn rate: heading falls out of the horizontal velocity, which falls
out of the tilted lift vector. That it reproduces `ψ̇ = g·tan φ / V` to half a
percent across the usable bank range says the force decomposition and the
heading derivation agree with each other — the kind of agreement that a sign
error or a missing `cos φ` destroys immediately.

It also explains the circle task's scale: at 30° bank and cruise speed the
turn radius is **9.3 km**, which is why that environment's reference path is
kilometres across rather than metres.

---

## 3. Parameter table

Only the parameters the third dimension introduces. Everything else is in
[`plane/PHYSICS.md`](../plane/PHYSICS.md).

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `wingspan` | 35.8 | m | A320neo with sharklets | ✅ |
| `C_lp` | −0.4 | – | Transport-aircraft roll damping derivative | ✅ |
| `aileron_surface` | 6.0 | m² | Consistent with the wing area | ⚠️ |
| `moment_arm_aileron` | 14.0 | m | ~0.78 semi-span, plausible for an outboard aileron | ⚠️ |

---

## 4. Task design

Three tasks share these dynamics and differ only in the reference:

* **Heading** — hold an altitude and a commanded heading.
* **Circle** — hold an altitude while orbiting a fixed circular path.
* **Figure-8** — follow a twisted lemniscate, which forces the bank to reverse
  through wings-level at the crossover.

**Patrol** adds a second aircraft and a slot defined relative to it, but no new
physics.

---

## 5. Known deviations

**⚠️ D1 — no yaw axis, and it costs less than it appears to.** Turns are
coordinated by assumption rather than by a rudder: there is no sideslip state,
so adverse yaw and the fin's response to it are not modelled.

This was recorded as the single largest simplification and as the reason
validity stops near 45° of bank. Measurement does not support either claim.
Solving the steady balance a rudderless turn would reach — the fin's
weathercock moment against yaw damping, `C_nβ·β + C_nr·(r·b/2V) = 0`, with
transport derivatives `C_nβ = 0.12`, `C_nr = −0.15` — gives a sideslip of
0.06° at 15° of bank and **0.97° at 60° of bank and 150 m/s**, the most extreme
corner of the envelope. The drag that sideslip adds, taking the fuselage and fin
side-on as a flat plate, is at most **0.7 % of cruise drag**. Full-aileron
adverse yaw contributes a further 1.25°, transiently.

A large transport is strongly weathercock-stable and turns slowly, so the
coordinated-turn assumption is a good one for *this* aircraft. Adding the axis
would change the forces by well under a percent while adding a state, a control
and four unsourced derivatives.

What actually limits steep turns is load factor, and the model already captures
it: `n = 1/cos φ`, so 60° of bank needs `CL = 1.42` at 150 m/s against a CL_max
of 1.56, and 75° needs 2.74, which the wing cannot make. The aircraft is refused
the turn by the lift it cannot generate, not by a missing axis.

The deviation stays open because the axis genuinely is absent — a rudder input,
a sideslip-induced roll (dihedral effect) and engine-out asymmetry are all
outside the model. But it is a limit on *scope*, not an error in the regime the
model claims.

**⚠️ D2 — aileron authority is not calibrated.** `aileron_surface`,
`moment_arm_aileron` and `aileron_response_rate` are plausible rather than
sourced, so the achievable roll *rate* is not validated against a real
aircraft. The roll *damping* is, and the coordinated-turn relation is
independent of all three.

**⚠️ D3 — patrol formation quality.** The physics documented here is shared and
validated. Both patrol variants now ship a PID baseline, including
`patrol_bearing_only`, which had none when this deviation was written; what
remains is quality rather than absence. The follower settles roughly 139 m from
its slot against a 60 m tolerance, so six formation scenarios are `strict`
xfail. That is a controller gap, not a physics gap; it is tracked in
[the patrol contract](../patrol/PHYSICS.md).
