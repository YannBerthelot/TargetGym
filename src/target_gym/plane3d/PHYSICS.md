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
2. **A roll moment**, from ailerons against roll damping on the rate relative to
   the air: `M_roll = M_aileron + C_lp · ((p - p_g)·b / 2V) · q̄ · S · b`.
3. **Heading is derived**, not integrated as an input. It comes out of the
   horizontal velocity vector, so a turn is something the aircraft *does*, not
   something the model is told.

The consequence worth stating: **the aileron commands a roll rate, not a bank
angle.** Roll damping sets a steady rate for a steady deflection, exactly as on
a real aeroplane — hold the stick over and it keeps rolling. Holding a bank
requires returning the aileron to neutral, which is why the plant carries a bank
loop. **A policy does not touch the surfaces**: an action is `[throttle, pitch
demand, bank demand]`, held for the 1 s step, and full action is +-15 deg of pitch
and +-45 deg of bank. The fixed inner loop of
[`plane/PHYSICS.md`](../plane/PHYSICS.md) flies the elevator and the ailerons to
them, with a roll integral.

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
| `aileron_response_rate` | 1/0.06 | 1/s | As the elevator's: 0.06 s, borrowed from another aircraft | ⚠️ |
| `bank_limit` | 60 | deg | 14 CFR 91.307(c), a trip | ✅ |
| `turbulence_scale_length` | 533.4 | m | MIL-F-8785C above 2000 ft, 1750 ft | ✅ |

---

## 4. Task design

Four tasks share these dynamics and differ only in the reference. **Patrol**
adds a second aircraft and a slot defined relative to it, but no new physics.

### Task: `plane3d_heading`

Hold a target altitude and turn onto and hold a target heading with the 3D
airliner, using throttle, elevator and aileron. The aileron commands a roll
rate, not a bank angle, so heading is controlled through a cascade of two
integrators, and the turbulence is hidden.

| obs | meaning |
|---|---|
| 0 | `x_dot`, velocity along world x (m/s) |
| 1 | `y_dot`, velocity along world y (m/s) |
| 2 | `z`, altitude (m) |
| 3 | `z_dot`, vertical speed (m/s) |
| 4 | `theta`, pitch angle (rad) |
| 5 | `theta_dot`, pitch rate (rad/s) |
| 6 | `phi`, bank angle (rad) |
| 7 | `phi_dot`, roll rate (rad/s) |
| 8 | `gamma`, flight-path angle (rad) |
| 9 | `psi`, heading (rad) |
| 10 | target altitude (m) |
| 11 | target heading (rad) |
| 12 | throttle, 0 to 1 |
| 13 | elevator deflection (rad) |
| 14 | aileron deflection (rad) |
| 15 | assigned speed, as a true airspeed (m/s) |
| 16 | airspeed, against the wind and gust (m/s) |
| 17 | normal load factor (g) |
| 18 | heading, where the nose points (rad), the vector's quantity (Amendment 4) |

### Task: `plane3d_circle`

Hold altitude while orbiting a fixed circle kilometres across with the 3D
airliner. The orbit asks for a sustained, coordinated bank, and the circle is
large compared with what the aircraft can change in a few steps.

| obs | meaning |
|---|---|
| 0-9 | the aircraft's state, as in `plane3d_heading` |
| 10 | target altitude (m) |
| 11 | x offset from the circle's centre (m) |
| 12 | y offset from the circle's centre (m) |
| 13 | circle radius (m) |
| 14 | throttle, 0 to 1 |
| 15 | elevator deflection (rad) |
| 16 | aileron deflection (rad) |

### Task: `plane3d_racetrack`

Fly a holding pattern, two straight legs joined by half circles, at a target
altitude with the 3D airliner. The observation gives the signed cross-track
error, the tangent heading and the curvature at the nearest point of the
pattern, so the work is in the transitions between straight and turning
flight.

| obs | meaning |
|---|---|
| 0-9 | the aircraft's state, as in `plane3d_heading` |
| 10 | target altitude (m) |
| 11 | x offset from the pattern's centre (m) |
| 12 | y offset from the pattern's centre (m) |
| 13 | turn radius of the pattern (m) |
| 14 | direction of the legs (rad) |
| 15 | throttle, 0 to 1 |
| 16 | elevator deflection (rad) |
| 17 | aileron deflection (rad) |
| 18 | cross-track error to the nearest point, positive outside the pattern (m) |
| 19 | heading of the pattern at the nearest point (rad) |
| 20 | signed curvature at the nearest point, 0 on the legs (1/m) |
| 21 | the hold's half leg (m) |
| 22 | assigned speed, as a true airspeed (m/s) |
| 23 | airspeed, against the wind and gust (m/s) |
| 24 | normal load factor (g) |

### Task: `plane3d_figure8`

Follow a twisted figure-8 that climbs and descends, with the 3D airliner. The
bank has to reverse through wings level at every crossover, and the
observation gives the vector to the nearest point of the path and the path's
heading there, but not its curvature. The half-width is 34 to 36 km, so the tips,
of radius a / 3, take 25 deg of bank at 230 m/s and less at the speed flown. They
used to be 2.7 to 4 km, 46 to 57 deg at 200 m/s. A lap is 183 km, so 1000 steps
reach both tips at the slowest speed the schedule flies.

| obs | meaning |
|---|---|
| 0-9 | the aircraft's state, as in `plane3d_heading` |
| 10 | target altitude (m) |
| 11 | size of the figure-8 (m) |
| 12-14 | vector from the aircraft to the nearest point of the path, x, y and z (m) |
| 15 | heading of the path at the nearest point (rad) |
| 16 | throttle, 0 to 1 |
| 17 | elevator deflection (rad) |
| 18 | aileron deflection (rad) |
| 19 | distance to the path, the tracked output (m) |
| 20 | target distance to the path, always 0 (m) |

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
see `patrol_range_bearing`, where the lead's heading is not measured and must be
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
error within 0.005 of each other — which makes the expert-versus-RL comparison the
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

The gust is the same across the span, so on its own it never reached the bank:
with no aileron the bank stayed 0 and the heading moved 0.002 deg. So the wing also
meets a rolling gust `p_g` (`rolling_gust`). MIL-F-8785C's Dryden spectrum for it,
`sigma_w^2 / (V L) 0.8 (pi L / 4b)^(1/3) / (1 + (4 b w / pi V)^2)`, rolls off at
about 20 rad/s, far above the step, so a step holds a white draw, sized to keep the
spectrum's level at low frequency: 0.0136 rad/s at 230 m/s and 2 m/s of gust. It
adds to the roll rate in the damping moment. Left alone it walks the bank by 9 deg
in 2 minutes; the inner loop holds it to about 0.8 deg RMS. On the patrol tasks the
caller owns the air, so it is not there yet: that is PR 7b.

### Integration order

One RK4 step per inner-loop sub-step, four to the 1 s step (`rk4_1`, `INNER_STEPS`).
The old `rk4_2`, two RK4 steps of 0.5 s on held surfaces, moved the altitude by
0.030 m against four steps over 150 s. An integrator of 0.25 s is finer than that
on the same dynamics, and the check that matters is
`test_a_step_is_its_four_sub_steps`. Current throughput is in
[docs/performance.md](../../docs/performance.md).

### The expert

The PID slot holds what a fixed-wing autopilot flies (PX4, ArduPilot), from the
observation alone:

- **Height and speed: TECS** (Lambregts; `plane/experts_tecs.py`). Throttle
  closes the total energy error and pitch the balance between height and speed,
  weighted to hold height (`TECS_SPDWEIGHT` 0). Its pitch demand goes to the
  plant's inner loop. Its aircraft numbers come from a flight test of the model at
  230 m/s: full throttle climbs 7.9 to 10.2 m/s and idle sinks 23 to 27 m/s
  between 7500 and 3500 m.
- **Bank: a demand to the plant's inner loop.** The 45 deg limit is ArduPilot's
  default and the edge of this model's validity. Stall prevention, as ArduPilot's, keeps the speed above 1.3 times
  the clean stall (104 m/s equivalent airspeed) scaled for the full bank, and
  the bank within what the speed supports.
- **Guidance:** a course hold on the heading task; L1 guidance (Park, Deyst and
  How, 2004), the law ArduPilot flies paths and loiters with, on the circle,
  racetrack and figure-8, with the path's curvature as a coordinated-turn
  feedforward. A curved path is flown at the speed whose steady bank in its
  tightest turn leaves part of the limit for L1's corrections: 15% on the
  circle and racetrack, 25% on the figure-8. Otherwise it flies the 2D tasks'
  speed schedule, 290 kt CAS then M 0.78.
- **The figure-8's geometry.** Its observation gives the nearest point and
  its tangent but not the curvature, and the tangent is read off the 800
  sampled points, so it jumps by up to 1.4 deg. The expert knows the
  pattern it was given, as an autopilot knows its mission: Bernoulli's
  lemniscate of half-width a, its altitude z_mean + dz sin(tau). The nearest
  point's altitude gives sin(tau), and with it the curvature,
  3 cos(tau) / (a sqrt(1 + sin^2 tau)), which turns the way the path climbs,
  and the tangent, 3 arctan(sin tau) from its heading at the lobe tip. That
  heading is the pattern's orientation, constant over the flight, so the
  expert averages it from the observed tangent. The curvature is read one
  second ahead, the bank loop's lag.

Its guidance and TECS gains predate the inner loop, the 1.7 s engine and the
envelope, so the figures that used to stand here no longer hold. **TECS's
`time_const` of 2 pulls past 2.5 g on height captures**, stepping the pitch demand
by about 10 deg in a second, and PR 9 retunes it by the accepted procedure. On
training seeds, the procedure's starting value of 5 flew `plane` with no trip on
10 of 10 seeds, where 2 trips on 6 of them.

It replaced, in 2026-10, cascades tuned on the scored seeds: the heading and
circle ones fed roll rate back with the wrong sign and held 44 m and 1 rad
off, and the racetrack's chased an observed tangent that pointed backwards on
one cap (the `-v4` fix), drifting kilometres.

### The oracle

The MPC slot holds a gradient MPC (`make_plane3d_mpc` in `oracle.py`): it
differentiates `step_env` over a 30-step horizon and descends the reward, on
params with the turbulence zeroed, so it plans on the mean wind. Its
normalised step is also its resolution: it cannot place an action closer than
about the step to the optimum. At a fixed 0.05 for 50 iterations that left
steady offsets, 4.1 to 4.5 m of altitude and 14.3 to 15.5 m of path on the
circle in its last hold measurement. Since the oracle audit (2026-10) the step
decays geometrically from 0.05 to 0.002 over each solve, with 200 iterations
on every task (the settings table in `experts.py`). Under the earlier floors
the audit measured, on the three protocol seeds, the cost falling from 5.05 to
0.380 (heading), 9.84 to 0.123 (circle) and 2.10 to 0.031 (figure-8) at 200
iterations, and from 7.04 to 1.49 (racetrack) at 50, with zero trips. The
iterations matter. On the figure-8 at 50, the decaying step lost one seed to
the fixed step (3.9 against 1.18, with 40 m excursions at the lobe tips), and
at 200 that seed held 3.3 m. On the racetrack, measured under the `-v3`
floors on the same seeds, 200 iterations cost 1.07 (0.960 / 1.29 / 0.962)
against 4.58 (2.23 / 7.54 / 3.97) at 50, with zero trips either way, so the
racetrack moved to 200 as well.

The oracle then held below every instrument resolution, so the floors were
reset to them (the `-v3` tasks) and the oracle re-measured under them, since
the floors weight its objective. Its lowest per-seed holds are 0.67 / 0.67 /
0.68 m of altitude (heading, circle, racetrack), 2.1e-5 rad of heading and
0.45 / 0.47 / 1.29 m of path (circle, racetrack, figure-8;
`scripts/measure_hold.py`), and its protocol cost under the `-v3` reward is
0.726 (heading), 1.93 (circle), 1.07 (racetrack) and 0.770 (figure-8), with
zero trips (`scripts/evaluate_baselines.py`). Those are larger numbers than
the audit's mostly because the floors are smaller: the audit's own
trajectories, rescored at the new floors, cost 0.788 (heading), 1.60 (circle)
and 0.744 (figure-8), and 5.01 on the racetrack at 50 iterations, where the
oracle re-planned at 50 cost 4.58. The heading and the racetrack land below
their rescored costs. The figure-8 scores one output, so its new floor only
rescales the objective, and it lands 3% above (0.770 against 0.744) with its
holds about unchanged (3.11 / 1.29 m against 3.08 / 1.33 m); the likely cause
is the planner's gradient-norm clip, which a larger objective meets more
often, not a trade between terms. The circle is the one clear loss. Its
altitude term gained 16.5x and its path term 7.3x, and the re-planned oracle
trades path for altitude: its altitude holds improved (0.83 / 0.67 m against
0.92 / 0.71 m) and its seed-1 path hold grew from 2.2 to 3.8 m. The budget
costs compute: 0.9 to 1.3 s per step on the four tasks in the recorded
protocol rows (one core, on a shared machine), against 0.4 to 0.5 s for the
fixed step in the audit's timings.

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
`patrol_range_bearing`, which had none when this deviation was written; what
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
| `plane3d_circle` | 600 | 1 | 10 min | 3 in [-1, 1] | 17 | 42 |
| `plane3d_figure8` | 1000 | 1 | 17 min | 3 in [-1, 1] | 21 | 42 |
| `plane3d_heading` | 600 | 1 | 10 min | 3 in [-1, 1] | 19 | 42 |
| `plane3d_racetrack` | 1200 | 1 | 20 min | 3 in [-1, 1] | 25 | 42 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Amendment 2: `plane3d_heading` and `plane3d_racetrack` (2026-11-08)

Redesigned under charter Amendment 2 from [plane/REALITY.md](../plane/REALITY.md)
and its [COLD_REVIEW.md](../plane/COLD_REVIEW.md); tests in
`tests/plane3d/test_plane3d_reality.py`. The changes sit behind `amended`, which
these two tasks set and the Extra tasks (`plane3d_circle`, `plane3d_figure8`,
[plane/EXTRA.md](../plane/EXTRA.md)), patrol and formation do not.

**Limits (rule 14).** As the 2D aircraft (plane/PHYSICS.md): load factor and AoA
protected through the pitch demand, VMO/MMO a warning (`info["overspeed"]`), the
ground the one trip. Bank is no longer a trip (E1): normal law protects it
[BEA447]; the demand still stops at 45 deg and the load-factor protection bounds
the turn, 2.5 g being a 66 deg level turn.

**Readings (rule 16).** The assigned speed, airspeed and normal load factor are
appended (`env.obs_index`); the racetrack also gives its half leg.

**Episode (rule 17).** Established at the assigned level and speed, wings level, in
trim (E6, F2, H6). One Dryden turbulence zone per episode, as the 2D aircraft;
the rolling gust blows only in the zone. Heading: established on heading 0 and
cleared onto the target at once, since heading select's capture is the job (F1).
Racetrack: established on the inbound leg of a hold built from the holding rules
[IFH ch. 10] (H2, H3): turns at 3 deg/s or 30 deg of bank, whichever is less, at
the assigned speed; an inbound leg of 1 min at or below 14 000 ft, 1.5 min above;
right turns.

**Score (rule 15).** Squared excess past each band, in band widths, per second.
Altitude +-65 ft [RVSM]. Speed +-10 kt [AIM 4-4-12]: the cruise schedule on the
heading task, 10 kt under the holding cap on the racetrack, 230 KIAS to 14 000 ft
and 265 above [IFH ch. 10]. Heading +-10 deg [ACS IR.III.B.S4]: on the heading
task about a heading that turns toward the target at a holding turn's rate from
the clearance (the band travels with that turn), so a turn at that rate or faster
costs nothing (F3: the
steepest bank no longer wins); on the racetrack about the pattern's tangent, which
also tells right turns from left (H1). Racetrack path: 3/4 of the +-1 NM GPS
terminal CDI scale, 1 389 m [ACS IR.III.B.S4, AIM 1-1-17].

| Number | Value | Status |
|---|---|---|
| bands | 65 ft, 10 kt, 10 deg, 1 389 m | cited: RVSM A.4.1.4, AIM 4-4-12, ACS IR.III.B.S4, AIM 1-1-17 |
| CDI scale | terminal +-1 NM, not en route +-5 NM | ours: the tighter of the two the AIM gives |
| holding turn | 3 deg/s or 30 deg, the lesser | cited: IFH ch. 10 |
| inbound leg | 60 s to 14 000 ft, 90 s above | cited: IFH ch. 10 |
| holding speed | the cap less 10 kt; caps 230 and 265 KIAS | caps cited (IFH ch. 10); the 10 kt under is ours, so the band's edge is the cap |
| heading-change size | uniform in +-180 deg | ours: ATC vectors of any size; no source for a distribution |
| episodes | 600 s heading, 1 200 s racetrack | ours: the longest turn then a held heading; three laps of the slowest hold |
| `turbulence_sigma`, zone | 0.55 m/s, 3 to 30 min | as the 2D aircraft (derived; BEA447) |
| steady wind | none in the scored configuration | Amendment 3: wind comes in the scenarios (below) |

**What the reference does on it.** Heading: TECS and the course hold turn at their
45 deg limit, faster than the 30 deg the score asks, and cost 0 on seeds 100 to
104: on this score the reference solves it. Racetrack: in calm air the reference leaves the
+-65 ft band by 0.5 m in its first seconds (B1 fails, a finding), and on the holds
at or below 14 000 ft it cannot fly the holding speed: its minimum airspeed,
104 m/s EAS scaled for a 45 deg bank, is about 241 kt, above the assigned 220 KIAS,
so it costs 1 800 to 2 000 per episode on the speed band (seeds 102 and 104); on
the higher holds 0 to 1.4. That is the reference recipe's floor (rule 3), not the
plant's. The 3D oracle plans the earlier cost, so neither room is measured yet.

**Observations.** B4: a 90 deg change at 25 deg bank, rolling out with the IFH's
lead of half the bank, comes back to wings level within 15 % of 90 / rate, but the
roll loop then swings to about -3.5 deg and settles in about 20 s, and the
half-bank lead, a standard-rate rule, stops the turn about 11 deg short at this
slower rate. Turn rate and load factor follow the coordinated-turn kinematics
within 3 % and 0.02 g (A1, A2).

## Amendment 3: `plane3d_heading` and `plane3d_racetrack` (2026-11-08)

The owner's ruling: a task represents its family, the A320 family; a number not
sourced is allowed inside the family's documented range, labelled "ours, family
range". Tests: `tests/plane3d/test_plane3d_reality.py`,
`tests/plane/test_aircraft_scenarios.py`.

**Protected bank (rule 14).** The plant clips the bank demand at `protected_bank`,
30 deg, for agent and reference alike: an airliner autopilot's bank is a protected
limit of 25 to 30 deg (the IFH's 30 deg holding bank [IFH ch. 10]; ICAO Doc 8168's
and the A320 HDG mode's 25 deg, unconfirmed). Before, the agent could demand 45 deg
for free. The limit sits on the demand, as the autopilot's own does; the roll loop
then overshoots to about 33 deg and settles 1.2 deg high (the B4 swing again).

**The reference (rule 3).** TECS reads airspeed, the amended air data, not the
ground speed it read before, which any wind broke. Its slowest speed is 1.2 times
the model's clean 1 g stall, 96 m/s EAS, "20% higher than level flight stall speed"
(ARSPD_FBW_MIN [P409]), raised by the load factor of the bank it demands, as
STALL_PREVENTION's "roll limits based on aerodynamic load factor in turns" [P409],
not of the full limit on every leg. Its bank limit is the plant's 30 deg. Starting
established, it engages bumpless: on its first step TECS's integrators take up the
trim it finds. The patrol's lead keeps the earlier course hold. B1 now passes on
the racetrack (largest altitude error 4 m), and a low hold flies 220 KIAS inside
+-10 kt in still air and in a 30 kt wind.

**`plane3d_heading`: vectors (rule 17).** A new heading every
`heading_clearance_period`, 150 s, over the whole compass, so the 600 s episode is
four clearances, not one turn and a long straight hold. Each clearance's band
travels from the heading flown then, so a new clearance costs nothing when given.

**The hold's altitude band (owner, 2026-11-08).** The racetrack holds at FL100 to
FL260, below RVSM airspace, so its altitude band is the ACS holding tolerance,
+-100 ft [ACS IR.III.B.S4: holding at "altitude +-100 feet"], not RVSM's +-65 ft
(plane3d_racetrack-v8). The heading task flies the same levels on +-65 ft; the
owner's answer covered the hold only.

**Scenarios (rule 20)**, `plane3d/scenarios.py`. Both: normal (one light zone),
calm, the worst recorded zone (2.5 times the light sigma, 0.5 g [BEA447 Figure
24]), a steady 30 kt and a 50 kt wind [IFH ch. 10: tracks and holds are flown
correcting for wind]. The wind blows along x; the heading's clearances and the
hold's orientation are drawn per seed, so it meets each from its own side. Heading
adds a zone over the whole window. The racetrack adds a low hold in its design
wind, 67 kt at 10 000 ft (2h + 47 kt, PANS-OPS as quoted by secondary sources),
and splits the 50 kt wind into a low hold (1 min legs, 230 KIAS cap) and a high
hold (1.5 min legs, 265 KIAS cap) [IFH ch. 10].

| Number | Value | Status |
|---|---|---|
| `protected_bank` | 30 deg | ours, family range 25 to 30 deg (above) |
| reference `v_min_eas` | 96 m/s | cited: 1.2 x the model's 80 m/s stall [P409 ARSPD_FBW_MIN] |
| `heading_clearance_period` | 150 s | ours: vectors one after another; the longest turn, 180 deg at a holding turn's rate, takes under 110 s. No source gives their spacing |
| heading-change size | uniform over the compass | ours: vectors of any size |
| steady winds | 30, 50, 67 kt | ours, family range calm to the holding area's design wind 2h + 47 kt (secondary, REALITY.md) |
| worst zone | 1.375 m/s | derived: 2.5 x 0.55 m/s, the load linear in the gust |
| scenario weights | heading: normal 0.35, calm 0.30, worst 0.05, 30 kt 0.20, 50 kt 0.10, long zone 0.05; racetrack: normal 0.35, calm 0.30, worst 0.05, 30 kt 0.20, low hold 50 kt 0.07, design wind 0.02, high hold 50 kt 0.08 (shares of operating time, normalised) | ours, family ranges in `plane3d/scenarios.py`; no source gives how often a wind aloft is met |

**What the reference does on it**, whole-episode cost on seeds 0 to 4:

| Scenario | `plane3d_heading` | `plane3d_racetrack` |
|---|---|---|
| normal | 0 on all | 0 on all |
| calm | 0 | 0 |
| worst zone | 0 | 0 |
| 30 kt wind | 0 | 11 to 132 |
| 50 kt wind | 0 | |
| zone over the window | 0 | |
| low hold, 50 kt | | 400 to 834 |
| low hold, 67 kt | | 1 130 to 2 203 |
| high hold, 50 kt | | 166 to 665 |

The racetrack's figures are on v8 (+-100 ft); on v7's +-65 ft they were the same to
within 2 % (seed 104 of the low holds 576 and 1 113), since the reference's
altitude stays inside both bands and its cost is path, heading and speed in wind.
Weighted by the scenarios' shares, the racetrack reference costs about 123 per
episode: 30 kt 15.5, low hold 50 kt 44.6, design wind 29.7, high hold 33.4.

`plane3d_heading` costs 0 on every scenario: no family cost was found that the
reference pays (rule 23: fuel in a cruise hold gives about 1.02, as on `plane`),
so it is listed as solved and stays in the core pool. The racetrack's cost is all
wind: in still air the fixed reference is solved. Room is not measured: no oracle
ships for either task until the 3D oracle is rebuilt for the band score.

**Reference ladder row (for `scripts/reference_ladder.py`, not edited here).** Both
tasks need a row with three loops on the 3D plant: altitude (CV `z`, MV pitch
demand), airspeed (CV `airspeed`, MV throttle) and the lateral loop (heading: CV
`psi` against `target_heading`; racetrack: CV `cross_track`, with `curvature` as a
feedforward), bump-tested at the corners of `target_altitude_range` and, on the
racetrack, of the wind scenarios' speeds; the bank MV is bounded by
`protected_bank`; the zones for rule 22's back-off are the altitude band (+-65 ft heading, +-100 ft hold), +-10 kt, +-10
deg and 1 389 m bands. Until then TECS with course hold or L1 is the reference.

**Known deviation: the `amended` flag (rule 13), resolved in Amendment 4: every aircraft task now flies the amended plant.** The Extra 3D tasks
(`plane3d_circle`, `plane3d_figure8`), patrol, `patrol_range_bearing` and
formation fly the unamended plant, which trips at 60 deg of bank and at VMO, which
REALITY.md contradicts, and lets the bank demand reach 45 deg. Extra tasks are
exempt, so this breaks no rule, but it keeps two plants. The owner's decision
(2026-11-08): flip the flag on each of them the next time it is touched, and
re-record its baselines then.

## Amendment 4: the 2026-11-10 review

The plant changes of plane/PHYSICS.md's Amendment 4 (the swept wing, the tail and
its downwash, inertia, the engine at altitude, the autopilot's 0.2 g, the
retuned attitude loop) reach every 3D task through `longitudinal_airframe`. What is
the 3D aircraft's own:

**Heading is the nose, not the track.** ATC vectors are headings [IFH ch. 7]; the
task scored the ground track, which a crosswind turns away from it by the drift
angle (7.7 deg at 50 kt across 193 m/s). `plane3d_heading` now scores `heading_3d`,
the direction of the velocity through the air (with no sideslip in the model, where
the nose points), and observes it as an IRS does, a nineteenth channel (`heading`;
`psi` stays the track). The expert's heading hold turns on it at the coordinated rate
for its airspeed. Tests: `test_a_vector_is_scored_on_the_heading_flown_not_the_track`,
`test_the_irs_heading_reads_the_nose_and_psi_the_track`.

**The hold is flyable in its wind.** The racetrack's turns were sized for still air,
so in its 30 to 67 kt scenarios the downwind turn asked 36 to 43 deg of bank against
the 30 deg protected bank, and the reference's whole cost there was that. An RNAV hold
sizes its turns for the greatest ground speed at the AIM's 25 deg, so the downwind
turn takes 25 deg and the upwind one less (Garmin, RNAV holding:
<https://twincommander.com/newsletter/garmin-tips-and-tricks-rnav-holding>): the
radius is `(TAS + wind)^2 / (g tan 25 deg)` (`hold_design_bank`). Test:
`test_the_holds_downwind_turn_is_flyable_in_its_wind`.

**What the operator pays.** `plane3d_heading` charges `economy` as `plane` does (fuel
and time by the cost index per air distance, plane/PHYSICS.md); the racetrack does
not (a hold's time is ATC's).

**Scenarios.** The light sigma is re-derived at 5 500 m on the airplane's own lift
slope, 6.23 /rad at M 0.61: dn/dw = 0.0714 g per m/s, so 0.564 m/s (was 0.55), the
worst zone 1.41. `plane3d_heading` adds a severe zone, MIL-F-8785C's severe intensity
at 5 500 m, 21.5 ft/s (Figure 7, as JSBSim tabulates it), 6.54 m/s, at a 1e-4 share.

**The circle and the figure-8 fly the amended plant.** The owner's decision of
2026-11-08 was to flip `amended` on the Extra tasks when next touched; done on
2026-11-10. They gain the 30 deg protected bank, the autopilot's 0.2 g and the
protected load factor and incidence, light Dryden zones (0.564 m/s) in place of the
continuous 2 m/s gust, no bank or VMO trips, and an established start (at the level
and the cruise speed, in trim, on the path's tangent). Their scores carry no speed
term, so nothing else moves.

The bank a level orbit needs on the cruise schedule is `atan(V^2 / (g r))`: 8 km took
31.5 deg at 8 000 m (219 m/s), past the protection, though only at that corner (20.5
deg at 3 000 m). The circle's radius is now 10.5 to 15 km, where a 25 deg turn flies
the schedule at every level (`V^2 / (g tan 25 deg)` = 10.5 km at 219 m/s), and its
episode 600 s, a lap of the largest orbit at the slowest speed to two of the smallest
at the fastest. The figure-8's tips, of radius a / 3, needed 23.4 deg at 8 000 m on
the old half-width of 34 km: inside the protection but past the 22.5 deg its expert
flies with a quarter in hand, so the half-width is now 36 to 38 km (`3 V^2 / (g tan
22.5 deg)` = 35.5 km). Both experts fly the protected 30 deg and the amended speed
floor, 96 m/s EAS. On training seeds 100 and 101 the circle's expert holds 0.5 to
0.7 m mean path error (2.4 m worst) and the figure-8's 1.0 m (3.4 m worst), with no
trip; the roll loop overshoots the 30 deg demand to 34 deg, as on the hold.

### The F/A-18 (patrol and formation)

The owner's decision of 2026-11-10: a fighter's formation on a fighter. `patrol`,
`patrol_range_bearing` and `formation` fly an F/A-18 Hornet on this plant,
`plane3d/hornet.py`, the amended plant (limits protected, the ground and the
formation's own trips the only ones). Sources: [HARV] Chakraborty, Seiler and Balas,
AIAA GNC 2009 (Tables 1, 2, 7, 8); [AFF] Hanson et al., NASA/TM-2002-210729; [F404]
and [NATOPS] secondary.

| Number | Value | Status |
|---|---|---|
| weight, wing, span, MAC | 15.1 t (33 310 lb), 37.16 m^2, 11.41 m, 3.51 m | cited: [HARV] Table 1 |
| radii of gyration | 3.68 m pitch, 1.44 m roll | cited: [HARV] Iyy 151 293, Ixx 23 000 slug ft^2 |
| lift slope | 0.0831 /deg, CL0 -0.020 | derived: the secant of [HARV]'s cubic over 0 to 10 deg |
| `CL_max`, stall | 1.70 at 20.7 deg | cited peak [HARV]; where the secant reaches it (its 38 deg LEX lift: a known deviation) |
| drag | CD0 0.020, e 0.8 | ours |
| stabilator arm, area | 4.96 m, 11.4 m^2 | calibrated: [HARV] Cm_delta -1.061 and Cm_q -3.00 at 4 deg, its slope DATCOM's for A 2.4, 35 deg (ours) |
| downwash | 0.46 of the far field, de/da 0.4 | ours: a fighter's |
| wing-body centre | 0.26 m behind the centre of gravity | calibrated: [HARV] Cm_alpha -0.993 |
| stabilator setting | -5.33 deg | calibrated: zero deflection at the AFF point |
| roll | Cl_p -0.337, aileron calibrated to Cl_delta_a 0.143 | cited: [HARV] Table 7 |
| engines | 2 x 47.6 kN military, SFC 22.9 g/(kN s), 3.25 s idle to full | secondary: [F404]; afterburner not modelled (ours) |
| limits | +7.5, -3.0 g, no autopilot limit, bank unprotected to 67 deg | +7.5 secondary [NATOPS]; the rest ours |
| attitude loop | pitch 2.29 / 0.032 / 0.127, roll 0.637 / 0.032 / 0.344, scaler about 116 m/s EAS | the A320's procedure at the AFF point (`tune_inner_loop.py --airframe hornet`), roll D up to [HARV]'s roll damper, 0.154 rad per rad/s there (docs/proposals/expert-protocols-aircraft.md, amendment of 2026-11-10) |
| pitch per extra g in a turn | 0.101 rad | derived: W / (q S a) at the AFF point; the formation hold's `k_bank_comp` |

At the AFF point, M 0.56 and 25 000 ft: trim incidence 5.8 deg, airplane lift slope
5.63 /rad ([HARV]'s 5.68), static margin 18 % MAC ([HARV] gives 17.5 to 20 %), short
period 2.34 rad/s at a damping of 0.18, phugoid 75 s, a 25 m/s climb at military power.

The mission is AFF's: the lead at 250 KCAS between 20 000 and 32 000 ft, a slot in the
cruise position on a bearing line 3 to 6 plane widths out with a step down [CNATRA],
34 to 70 m back, 17 to 45 m to a side drawn at random (the formation's V from 22 m, so
its spawn noise cannot put two wingmen of a rank together), 0 to 10 m below; never
closer than one length, 56 ft, AFF's nose-to-tail limit (`min_separation`); the
station-keeping limit AFF's own, 10 ft one sigma, not the A320's span scaled; light
turbulence, as AFF flew ("no turbulence" to "light chop"). The lead turns at up to
0.0303 rad/s, the parade turn's 30 deg [CNATRA] at 25 000 ft (28 to 33 deg over the
band): at 0.0476 rad/s (40 deg) the band's top took 45.4 deg, past the 45 deg both the
lead's and the wingmen's autopilots fly, and the wingmen fell hundreds of metres behind.

Both aircraft now spawn in level trim (`trim_level`): spawned at `initial_theta` with
the elevator merely balanced there, the Hornet, which trims near 6 deg, started 6 deg
short of its lift and its lead lost 1.2 km of height.

### Wake surfing (`wake_surf`)

The owner's decision of 2026-11-10: wake-energy retrieval for the A320. Airbus's
fello'fly flew two A350s 3 km apart, the follower in the leader's upwash, for a fuel
saving its press release puts at over 5 % on long haul
(<https://www.airbus.com/en/newsroom/press-releases/2021-11-airbus-and-its-partners-demonstrate-how-sharing-the-skies-can-save>);
the C-17 $AVE trials held 3 000 to 6 000 ft in trail for up to 10 %
(<https://www.af.mil/News/Article-Display/Article/466878/inspired-by-nature-innovative-c-17-flight-tests-to-save-af-millions/>).
`wake_surf` is the patrol with an A320 on both seats (`WakeSurfParams`, the lead on the
airliner autopilot) and the follower at a station 2.8 to 3.7 km in trail.

The station is where the model's own wake gives the most upwash behind a level lead:
0.87 of a span outboard, 31 m, NASA's 13 % tip overlap, and 19 to 26 m below, the
vortex's sink over 12 to 16 s. There the upwash is 2.1 m/s and saves 6.7 kN of the
follower's drag, with a steady rolling gust of -0.16 rad/s to trim out; the vortex core
lies 15 m inboard. The score is a 30 m box about the station (ours; no public tolerance
was found) and the follower's `economy`: inside the box the cost falls only where the
follower finds the upwash, about 0.82 a second against 1.0 out of it. A bank past
60 deg is an upset, a trip (Hazardous [AMC 25.1329, A4.2]). The air memory is 20 s so
the follower 16 s behind still meets the air its lead left.

**The reference.** The patrol's formation hold on the A320's own autopilot (290
KCAS then M 0.78, TECS), its energy step retuned on `wake_surf` alone
(`scripts/tune_formation_hold.py --wake-surf`, training seeds 100 to 109): TECS's speed
weight 1.0, throttle damping 6, time constant 5 s and `k_ref` 1.0 bring the worst
seed's RMS distance from the station to 156 m, from 215 m, with no trip. That is far
outside the 30 m box: a weak baseline, documented as such. Its slot loops were set for
close formation, and 3 km in trail, with an engine 6 s slow at cruise, they porpoise.

**Known deviations.** The lead flies straight and level: the station is held in the
lead's frame, and in a turn the wake stays on the lead's past track, 100 m off that
frame at 0.3 deg/s. The vortex does not meander in the turbulence; it decays and sinks
as in the patrol (Holzapfel and Steen's median). No oracle ships yet
(`baselines_note`).

## Reward (version 2)

`compute_reward = -(tracking + failure)`, one quadratic term per tracked output, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_altitude` | 19.812 m (`plane3d_heading`, Amendment 2: +-65 ft [RVSM]); 30.48 m (`plane3d_racetrack`, Amendment 3: +-100 ft, the ACS holding tolerance); 1 m (`plane3d_circle`); `plane3d_figure8` scores no altitude and keeps the 1.44 m class default | the barometric altimeter's resolution. The oracle holds below it in the test turbulence: its lowest per-seed long-run mean \|error\| is 0.67 / 0.67 / 0.68 m (`scripts/measure_hold.py`, 2 seeds, hold steps after a 210-step burn-in, the second half of the episode on the 200-step heading task). Measurement noise is not modelled, so the resolution sets the scale: a hold the instrument cannot see is not a floor. Until the oracle audit (2026-10) the floors were an earlier oracle's holds, 1.44 / 4.06 / 1.39 m (the `-v2` tasks). Expert: 1.6 / 1.6 / 1.7 m on held-out seeds (the cascade it replaced: 44 / 7.2 / 5.2 m) |
| `e_tol_altitude` | 19.812 m on `plane3d_heading` (Amendment 2, +-65 ft [RVSM]), 30.48 m on `plane3d_racetrack` (Amendment 3, +-100 ft [ACS IR.III.B.S4]); else 0 | Amendment 2: precision inside the band earns nothing (rule 15). Else no dead zone; a +-30 m band made the altitude hold vacuous (see the 2D aircraft) |
| `e_floor_heading` | 0.1745 rad on `plane3d_heading` and `plane3d_racetrack` (Amendment 2: +-10 deg [ACS IR.III.B.S4]); else 0.0087 rad | Amendment 2: the instrument-rating ACS's heading tolerance. Else the 0.5 deg AHRS resolution. The oracle holds far below it in the test turbulence (1.4e-5 rad, lowest per seed over the 300-step episode; `scripts/measure_hold.py`), so the instrument, not the simulator, sets the scale; the expert holds about 1e-4 rad on held-out seeds |
| `e_tol_heading` | 0.1745 rad on `plane3d_heading` and `plane3d_racetrack`; else 0 | Amendment 2: +-10 deg [ACS IR.III.B.S4], about a heading the aircraft turns toward at a holding turn's rate on the heading task, about the pattern's on the racetrack |
| `e_tol_path` | 1 389 m on `plane3d_racetrack`; else 0 | Amendment 2: 3/4 of the +-1 NM GPS terminal CDI scale [ACS IR.III.B.S4, AIM 1-1-17]; terminal rather than en route (+-5 NM) is ours |
| `e_floor_path` | 1 389 m (`plane3d_racetrack`, Amendment 2); 3 m (`plane3d_circle`, `plane3d_figure8`; also the default where no path term is scored) | civil GPS horizontal accuracy. The oracle holds below it: its lowest per-seed path-distance holds are 0.45 / 0.47 / 1.29 m (`scripts/measure_hold.py`). Until the oracle audit (2026-10) the floors were 8.12 / 6.17 / 14.6 m: the racetrack's and figure-8's were an earlier oracle's holds, while the circle's 8.12 m was labelled one but equals 2 x 4.06 m, and that oracle's recorded hold there was 14.3 m. Expert: 0.96 / 2.3 / 1.4 m on held-out seeds, the figure-8's where its lobes are flyable (the cascades it replaced: 53 / 408 / 1767 m) |
| `tracking_exponent` | 2 | quadratic |
| `failure_cost` | 3.77e5 (`plane3d_heading`) and 1.59e5 (`plane3d_racetrack`): Amendments 2 and 3, ours, an hour at 40 000 ft off the level past the altitude band; 2.97e8 (`plane3d_circle`), 1.09e9 (`plane3d_figure8`) | twice the altitude envelope's cost, 2 x (12 192 / e_floor_altitude)^2; the figure-8 scores the path alone, 2 x (70 000 / e_floor_path)^2 |
| `restart_steps` | 3600 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (a crash loses the sortie: an hour of flight, provisional); where a plant engineer would get it: the plant's restart procedure |
| `rho_floor_tracking`, `rho_floor` | 0 (`plane3d_heading`, `plane3d_racetrack`: a hold inside the bands costs nothing), 0.466 (`plane3d_circle`), 0.184 (`plane3d_figure8`) | the oracle's lowest per-seed holds above in floor units, (hold / floor)^2 summed over the scored outputs (`scripts/measure_hold.py`): (0.5867 / 1)^2 + (1.4e-5 / 0.0087)^2 over the heading task's 300 steps (`-v4`), (0.6657 / 1)^2 + (0.4504 / 3)^2, (0.6803 / 1)^2 + (0.4660 / 3)^2 and (1.2871 / 3)^2 |

Each 3D task's values above are its registry parameters,
`spec.make_test_params()`. The `PlaneParams3D()` class defaults, which patrol
inherits, are still `plane3d_heading-v2`'s (a 1.44 m altitude floor, its
`failure_cost` and its `rho_floor`), so params built from the class alone
score the `-v2` reward.

`rho_floor_tracking` is the oracle's best hold for tracking -- the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution -- and `rho_floor` the same with consumption
charged in full; `floor_is_documented_minimum` records whether `e_floor` is a
measured/certified floor or a resolution used as a scale, and where it is a
resolution on a deterministic plant both references are 0, since exact hold is
achievable there and the resolution only sets the unit;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce, and `restart_steps` the time a restart would take,
so a trip costs `restart_steps x failure_cost` (`reward.trip_cost`). A trip never
ends the window (`base.failure_kernel`): the step that leaves the envelope is
charged the trip cost, with tracking and running cost zeroed, and the plant
restarts at once as `reset_env` would, on the same clock. `terminated` stays
false unless `continuing = False`, which ends the episode on that step instead; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).

## Readings

The observation reports the aircraft through its instruments, with every target
exact: altitude, heading, radius and the pattern's centre
(`docs/proposals/measurement-noise.md`). `sensor_noise_scale = 0` gives the
noise-free task, rounding included.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `z` | 0.2 m | 1 ft | Air data computer baro altitude | Provisional: a ground test set's figure, no airborne one found | [DPST-8000](https://www.avionteq.com/Document/DPST-8000-specification-sheet.pdf) |
| `x_dot` | 0.03 m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling: receiver accuracy, not scatter | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `y_dot` | 0.03 m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling: as `x_dot` | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `z_dot` | 0.03 m/s | 1 ft/min | IRS inertial vertical speed | No source: a placeholder, not a bound | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `theta` | none | 0.010986 deg | IRS pitch attitude | Corrected: rounding only | [HG9900](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `theta_dot` | 4.3e-6 rad/s | 0.015625 deg/s | IRS body pitch rate | Confirmed | [Honeywell Mk VI guide](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `phi` | none | 0.010986 deg | IRS roll attitude | Corrected: rounding only | [HG9900](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `phi_dot` | 4.3e-6 rad/s | 0.015625 deg/s | IRS body roll rate | Confirmed | [Honeywell Mk VI guide](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `gamma` | 1.3e-4 rad at 230 m/s, as 0.03 / V | from the velocities | IRS flight-path angle | Confirmed | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `psi` | 1.3e-4 rad at 230 m/s, as 0.03 / V | from the velocities | IRS true track angle | Confirmed | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `power` | none | 0.003 | Engine thrust indication via FADEC | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `stick` | none | 1.6e-3 rad | Elevator surface position sensor | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `aileron` | none | 2.6e-3 rad | Aileron surface position sensor | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `x`, `y` | 0.27 m | none | GNSS position | Corrected | [NASA AFF](https://ntrs.nasa.gov/api/citations/20030005804/downloads/20030005804.pdf) |

`gamma` and `psi` have no noise of their own: they are worked out from the
velocities read, as an IRS does, and `psi` is the ground track, wrapped after
rounding. `x` and `y` apply to the circle, racetrack and figure-8, where the path
errors are computed from them, so those channels inherit the noise. The heading
task observes no position.

The figure the tracked altitude rests on is weak. No air data computer publishes
its scatter, so 0.2 m comes from a ground test set, and the benchmark runs at
`sensor_noise_scale` 0.5 and 2 to show how far a result depends on it. The
ground velocity figure is a receiver's accuracy, so true scatter is smaller, and
the vertical speed figure is a placeholder.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `info["measured_reward"]` is the same cost on the readings,
with `psi` from the velocities read and any trip taken from the true state.

## The stream (docs/proposals/long-strata.md, S1, S1b and R8)

As a continuing plant (`target_gym.stream`) the 3D tasks fly a sequence of
sorties, as the 2D aircraft (plane/PHYSICS.md, "The stream"): a sortie lasts
1.90 h on average [AIRBUS-A320], starts established in cruise at a mass drawn over
60 to 78 t and held, and turbulence zones start at 0.45 an hour with their own
intensity. Sources and searches: plane/REALITY.md, "The stream". What differs:

- **`plane3d_heading`'s vectors** are events at 24 an hour (ours: the window's one
  per 150 s; ATC gives about two headings per aircraft per approach sector
  [PRINZO], but no source gives the time on frequency). A vector draws a heading
  over `target_heading_range`; its size is the turn from the heading flown. The
  window's periodic vectors are off in a stream (`heading_clearance_period` 0).
- **`plane3d_racetrack`'s holds** are its sorties: a hold lasts 7.5 min on average,
  Heathrow's 2016 average [NATS-HOLDS], and each new hold is a new aircraft at its
  own mass and level, on a fresh orientation. No level changes in the stack (owner,
  2026-11-11).
- **The steady wind** moves to a new scenario's as a first-order lag of
  `wind_tau`, 2 010 s, instead of jumping: the largest switch, 67 kt, then starts
  at 2 kt a minute, the horizontal shear AC 00-30C marks as moderate turbulence
  near a jet, 40 kt per 150 NM crossed at 450 kt [AC00-30C]. The state carries the
  wind the aircraft is in (`mean_wind_x`, `mean_wind_y`).
- **Restart** (R8; owner, 2026-11-11): the sortie's replacement, established at
  its level with its mass, on its own pattern: on the circle, the figure eight or
  the hold it flies (the hold's turns as sized when it began), or on its cleared
  heading.
- **Outside world and records.** The gusts and a vector's size (`heading_change`,
  the turn from the heading flown) are records, carried through a trip but the
  policy's.
- `plane3d_circle`, `plane3d_figure8` and the racetrack are closed shapes, so
  their paths never run out.

In a window nothing changes (PR description: bit-identical, traced and constant).
