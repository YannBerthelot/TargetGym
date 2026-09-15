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

### Observation

Each task exposes what its own reference needs, on top of the longitudinal
state the 2D aircraft reports: bank angle and roll rate, heading, and the
task's own error terms -- commanded heading for the heading task, the offset to
the nearest point on the path for the circle, racetrack and figure-8.

What is hidden is the same in every case and is the same as in 2D: **the gust
field**. Turbulence is a disturbance, and an aircraft flies into it without
warning.

The honest framing is that these tasks are *not* deeply partially observed, and
should not be quoted as if they were. The aircraft knows where it is and where
the path is. Their difficulty is elsewhere: the aileron commands a roll rate
rather than a bank angle, so heading control is a cascade through two
integrators; the turn radius at cruise is kilometres, so the reference is large
compared with what the aircraft can do about it; and the lift the wing can make
bounds how tight a turn is available at all. For genuinely hidden references,
see `patrol_bearing_only`, where the lead's heading is not measured and must be
reconstructed.

### The tracking reward

Every task composes its objectives multiplicatively — both must be met, not one
traded against the other — and each factor is the same log-scaled shape the 2D
aircraft uses (`plane/PHYSICS.md` §5, `docs/reward-shaping.md`):

```
reward = 1 - log1p(error / floor) / log1p(envelope / floor)
```

| Task | Factors | Envelope | Floor |
| --- | --- | --- | --- |
| Heading | altitude × heading | 12 191 m, π rad | 1 m, 0.0087 rad |
| Circle | altitude × cross-track | 12 191 m, `target_radius` | 1 m, 3 m |
| Figure-8 | 3D distance to the curve | `target_radius` | 3 m |

Normalising the path terms by `target_radius` keeps the reward independent of
the size of the commanded path, and the aircraft starts *on* the path in both
path-following tasks, so the point where the reward reaches zero is only reached
by a controller that has already lost the shape entirely.

Each floor is an instrument resolution rather than a chosen tolerance: 1 m for a
barometric altimeter, 0.5° for an AHRS, 3 m for civil GPS. The reward pays for
every halving of the error down to that point and only flattens beneath it,
where further "improvement" would be chasing measurement noise.

Until the reward was converted, all three tasks divided the error by the
altitude envelope and raised it to the tenth power, and the two path tasks used
a Gaussian a tenth of the path radius wide. Both are near-flat over the range a
working controller occupies — the second scored a 0.1 m and a 100 m cross-track
error within 0.005 of each other — which makes the PID-versus-RL comparison the
suite exists for unmeasurable.

**Cross-track error on the figure-8 is searched, not solved.** There is no
closed form for the nearest point on the twisted lemniscate, so it is an
`argmin` over 400 samples of the curve refined by projection onto the two
adjacent chords. Without that refinement the result is quantised by the sample
spacing (~100 m), and an aircraft flying the curve exactly is reported up to
66 m off it; with it the residual is the curve's sagitta over one segment, below
a millimetre. The circle needs none of this — its distance is analytic.

---

### Turbulence

As on the 2D aircraft: the test configurations fly in light-to-moderate
turbulence, an Ornstein-Uhlenbeck gust with `turbulence_sigma = 1.2` m/s per
step and `turbulence_theta = 0.2` (2 m/s stationary gust std), without which
the altitude hold scored nothing. The planners plan on the mean wind.

### Integration order

``rk4_2`` -- two RK4 substeps per environment step -- rather than one. The
trajectory is converged there: over 150 steps of a fixed control input the
altitude moves 20.4 m between one substep and two, then 0.030 m between two and
four and 0.007 m between four and eight.

One substep was the original setting and it is not defensible for this
environment. The tracking reward has a ``precision_floor`` of 1 m and the shipped
controllers settle between 1 and 11 m, so a 20 m integration error is larger than
the quantity the benchmark scores -- the simulator could not resolve the
precision the reward was paying for. It is the same failure as check 11 of the
model review checklist, one level down: there the *metric* could not see what the
reward asked for, here the *integrator* could not.

It costs throughput: measured at batch 4096 when the change was made, the 3D
aircraft went from 1.88 to 1.05 M steps/s and the 2D aircraft from 2.71 to 1.61.
Those are the numbers that justified the trade; current figures on whatever
machine you are reading this from are in
[docs/performance.md](../../docs/performance.md). The 2D aircraft's own
error at one substep was 0.823 m, inside its 1 m floor rather than outside it,
but it is set the same way so the family stays consistent.

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

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `plane3d_circle` | 300 | 1 | 5 min | 3 in [-1, 1] | 17 | 26 |
| `plane3d_figure8` | 400 | 1 | 7 min | 3 in [-1, 1] | 19 | 26 |
| `plane3d_heading` | 200 | 1 | 3 min | 3 in [-1, 1] | 15 | 26 |
| `plane3d_racetrack` | 650 | 1 | 11 min | 3 in [-1, 1] | 21 | 26 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`, one quadratic term per tracked output, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_altitude` | 1.44 m (`plane3d_heading`), 4.06 (`plane3d_circle`), 1.39 (`plane3d_racetrack`) | lowest per-seed long-run mean |error| the shipped MPC held in the test turbulence (`scripts/measure_hold.py`, 2 seeds, hold steps after a 210-step burn-in); an upper bound on the achievable floor, and below the instrument resolution a real aircraft would have (1 m barometric, 0.5 deg heading, 3 m GPS): measurement noise is not modelled. PID: 44 / 7.2 / 5.2 m |
| `e_tol_altitude` | 0 | no dead zone; a +-30 m band made the altitude hold vacuous (see the 2D aircraft) |
| `e_floor_heading` | 0.0087 rad | the 0.5 deg AHRS resolution. The MPC holds below it in the test turbulence (1e-4 rad on two seeds, 6e-3 on three; `scripts/measure_hold.py`, `target_gym.eval`), so the instrument, not the simulator, sets the scale; the PID never captures the heading (0.96 rad) |
| `e_floor_path` | 8.12 m (`plane3d_circle`), 6.17 (`plane3d_racetrack`), 14.6 (`plane3d_figure8`); 3 m (GPS accuracy) where no path term is scored | lowest per-seed MPC path-distance holds in the test turbulence; upper bounds. PID: 53 / 408 / 1767 m |
| `tracking_exponent` | 2 | quadratic |
| `failure_cost` | 1.43e8 (`plane3d_heading`), 1.8e7 (`plane3d_circle`), 1.54e8 (`plane3d_racetrack`), 3.75e6 (`plane3d_figure8`) | twice the altitude envelope's cost, 2 x (12 192 / e_floor_altitude)^2; the figure-8 scores the path alone, 2 x (20 000 / 14.6)^2 |
| `restart_steps` | 3600 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (a crash loses the sortie: an hour of flight, provisional); where a plant engineer would get it: the plant's restart procedure |

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
