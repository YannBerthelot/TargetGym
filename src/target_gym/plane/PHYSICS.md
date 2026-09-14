# Plane 2D — physics model, provenance and validation

Reference aircraft: **Airbus A320-200**, clean configuration, cruise regime.

This document is the contract for `target_gym.plane`. Every constant is either
derived here from geometry/first principles, cited, or explicitly flagged
`TUNED — not sourced`. The **Validation** table is not prose: each row is
asserted by a test in `tests/plane/test_plane_physics.py`, so the document
cannot silently drift from the code.

Status legend: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model scope

Longitudinal (pitch-plane) rigid-body flight with three degrees of freedom:
horizontal position `x`, altitude `z`, pitch `θ`. State derivatives come from
Newton's second law in the inertial frame; aerodynamics act on the
**air-relative** velocity so wind, shear and turbulence enter correctly.

Deliberately **not** modelled (and why):

| Omitted | Rationale |
|---|---|
| Lateral/directional dynamics (roll, yaw, sideslip) | 2D task; the 3D envs add roll. |
| ~~Fuel burn~~ | **Now modelled.** ``specific_fuel_consumption`` charges thrust against the tank each step and ``m`` is recomputed from what is left, so the aircraft lightens as it flies. This row said mass was held constant long after that stopped being true. |
| Flaps / slats / gear | Clean configuration only. Makes the low-speed regime unrepresentative — see §5. |
| Aeroelasticity, ground effect | Negligible at task altitudes. |
| Engine spool dynamics beyond first-order lag | `compute_next_power` is a 1st-order lag; adequate for setpoint tracking. |

---

## 2. Atmosphere — International Standard Atmosphere

Troposphere (0–11 km), ISO 2533 / ICAO Doc 7488:

```
T(h) = T0 - L·h            T0 = 288.15 K,  L = 0.0065 K/m
P(h) = P0·(T/T0)^(g/(R·L)) P0 = 101325 Pa, g = 9.80665 m/s², R = 287.05 J/(kg·K)
ρ(h) = P/(R·T)
a(h) = sqrt(γ·R·T)         γ = 1.4
```

Implemented by `compute_air_density_from_altitude` and
`compute_speed_of_sound_from_altitude`. This is a published standard, so it is
validated against tabulated values rather than merely self-consistent.

| h (m) | ρ ref (kg/m³) | ρ model | err | a ref (m/s) | a model | err | |
|---|---|---|---|---|---|---|---|
| 0 | 1.22500 | 1.22501 | +0.00 % | 340.294 | 340.263 | −0.01 % | ✅ |
| 5 000 | 0.73643 | 0.73612 | −0.04 % | 320.545 | 320.500 | −0.01 % | ✅ |
| 11 000 | 0.36392 | 0.36392 | −0.00 % | 295.070 | 295.042 | −0.01 % | ✅ |

The small `a` bias is `R = 287.0` in the speed-of-sound path vs `287.05` in the
density path. Harmless at 0.01 %, but the two should be unified.

---

## 3. Geometry and mass

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `wings_surface` S | 122.6 | m² | A320 published reference wing area | ✅ |
| span (implied) | 34.1 | m | A320 published wingspan (not a param; used to derive AR) | ✅ |
| aspect ratio AR | 9.48 | – | `span²/S = 34.1²/122.6` | ✅ |
| `initial_mass` m | 73 500 | kg | Typical A320 takeoff mass; MTOW 78 000, OEW ≈ 42 600 | ✅ |
| `initial_fuel_quantity` | 19 088 | kg | A *component* of `initial_mass`, not additional | ✅ |
| `thrust_output_at_sea_level` | 240 000 | N | 2 × CFM56-5B ≈ 2 × 120 kN static SL | ✅ |
| `I` | 9.0 × 10⁶ | kg·m² | TUNED — not sourced. Order-of-magnitude plausible for a 37 m fuselage. | ⚠️ |
| `moment_arm_stabilizer` | 15.0 | m | TUNED — plausible tail arm for A320 geometry | ⚠️ |
| `moment_arm_wings` | 1.5 | m | TUNED — wing AC to CG offset | ⚠️ |
| `power_response_rate` | 0.05 | 1/s | TUNED — first-order spool rate; an airliner reaches ~63 % of a throttle change in ~20 s | ⚠️ |
| `stick_response_rate` | 0.9 | 1/s | TUNED — control-surface lag, deliberately far faster than the engine | ⚠️ |

**Mass bookkeeping.** `compute_acceleration` integrates `params.initial_mass`.
`state.m` previously reported `initial_mass + fuel = 92 588 kg`, which both
disagreed with the integrated mass and exceeded the A320 MTOW of 78 000 kg.
`state.m` now equals `initial_mass`, with fuel a component of it.

---

## 4. Aerodynamics

Lift/drag coefficients (`aero_coefficients`), parabolic polar with a sigmoid
stall cutoff and Prandtl–Glauert compressibility:

```
CL_lin = cl0 + cl_alpha·α                       α in degrees
CL     = min(CL_lin / (1 + exp(1.5·(α - (aoa_stall + aoa_stall_width)))), CL_max) / β
CD     = cd0 + k·CL²  + k_drag·max(0, M - M_crit)²
β      = sqrt(1 - M²)
```

| Symbol | Value | Derivation / source | |
|---|---|---|---|
| `cd0` | 0.020 | Typical transport-category zero-lift drag (0.017–0.022) | ✅ |
| `k` | 0.045 | `1/(π·AR·e)` = 0.0395 at e = 0.85; 0.045 implies e ≈ 0.75. Defensible. | ✅ |
| `cl_alpha` | 0.08786 /deg | Prandtl finite-wing `a0/(1 + a0/(π·AR·e))`, AR 9.48, e 0.85. Derived, not tuned. | ✅ |
| `cl0` | 0.20 | TUNED — plausible cambered-wing zero-α lift | ⚠️ |
| `aoa_stall` | 15.0 deg | Typical swept-wing stall AoA; `cl0 + cl_alpha·aoa_stall` = 1.52 ≈ `CL_max` | ✅ |
| `aoa_stall_width` | 3.0 deg | Offset of the separation sigmoid past `aoa_stall`, so lift peaks *at* the stall angle | ⚠️ |
| `CL_max` | 1.5 | Attainable: peak CL 1.53 at α = 15.0°, so the clamp now binds. | ✅ |
| `M_crit` | 0.80 | Drag-divergence Mach. Was declared **twice** (0.78, 0.80); 0.80 always won. | ✅ |
| `k_drag` | 5.0 | TUNED — transonic drag-rise strength | ⚠️ |

### Post-stall — the flat-plate branch

Below the stall the wing is an aerofoil and the model above describes it. Well
past the stall it is a barn door, and a separate branch is blended in with the
same centre and steepness as the stall sigmoid, so lift is handed to the plate
exactly as separation takes it away:

```
CL -> sin 2a        CD -> cd0 + 2 sin^2 a
```

where `C_N_max = 1.11 + 0.018·AR` — Viterna & Corrigan (1981) for a finite
wing, so the magnitude comes from the aspect ratio the geometry already fixes
rather than being assumed. A 2D flat plate would reach 2.0; this wing, relieving
around its tips, reaches 1.28. The result is CD ≈ 1.30 with the wing side-on and
a lift curve that peaks at 1.56, breaks to 0.38 at 19° (24 % of peak) and
recovers to 0.64 at 45° — the shape a real wing shows.

The sharpness of the attached-to-separated transition is derived rather than
chosen. The separated branch develops far less lift, so any appreciable blend at
`aoa_stall` would eat CL_max and move the validated stall speed. Requiring the
wing to be 1 % separated there, given a sigmoid centred `aoa_stall_width` beyond
the stall angle, fixes it at `ln(99)/aoa_stall_width` = 1.53 — which is the 1.5
that was previously written as a literal, rounded. The resulting 10–90 % band is
2.9°, inside the 2–5° a clean transport wing shows.

Full Viterna was tried for the whole post-stall branch and rejected on evidence:
anchored at the stall point it is continuous by construction, but it declines
only to 69 % of peak, so it has no stall *break*. It is built for wind-turbine
blades, where the transition is gentle. A transport wing breaks sharply, and
that break is the behaviour that makes a stall dangerous, so only its
finite-wing magnitude is borrowed. Incidence is wrapped into ±180° first, because the
integrator does not wrap pitch and a departed aircraft can arrive with several
thousand degrees of it.

This is applied after the Mach corrections deliberately: separated flow is not
a compressibility effect, and Prandtl–Glauert has no business scaling it. Below
the stall the branch contributes under 2 %, so every figure of merit below is
unchanged except the clean stall speed, which moves 154 → 154.6 kt and stays
inside its reference band.

### Figures of merit

| Quantity | Model | Reference | |
|---|---|---|---|
| L/D max = `1/(2√(cd0·k))` | 16.67 | A320 ≈ 17 | ✅ |
| CL at L/D max = `√(cd0/k)` | 0.667 | — | ✅ |
| Cruise CL, FL350 M0.78 | 0.579 | Jet cruise typically 0.4–0.6 | ✅ |
| Cruise trim AoA | 1.85° | A320 ≈ 1.5–3° | ✅ |
| Clean stall speed (SL) | 154 kt | A320 ≈ 145–156 kt | ✅ |
| Pitch static stability | dM/dα < 0 | Required for longitudinal stability | ✅ |
| Thrust available / drag at FL350 | 1.31 | Must exceed 1 to sustain cruise | ✅ |

---

## 5. Task design — the tracking reward

Hold a sampled target altitude. Per step:

```
reward = 1 - log1p(|z - target| / precision_floor) / log1p(span / precision_floor)
       = -max_steps_in_episode            outside the altitude envelope
```

1 at the target, 0 at the far edge of the envelope, `precision_floor` = 1 m.

**Why logarithmic.** This benchmark exists to ask whether a learned policy can
hold a setpoint better than a PID. That question is only askable if the reward
keeps paying for precision all the way down. Two earlier shapes did not:

`((span - |e|)/span)**10` normalised by the whole 12 km envelope, so over any
realistic error it is `1 - 10 e/span` — effectively linear, worth 8e-4 per metre
whether the aircraft is 10 m or 400 m out. Closing the last 9 m gained 0.007
against 0.26 for halving a 1600 m error.

A band-scaled kernel, `1/(1 + (e/band)^2)`, fixes the mid-range but peaks *at*
the band and collapses inside it: each halving below the band is worth a
quarter of the one before, so it rewards reaching a tolerance and then stops
caring. That is the wrong incentive for a benchmark whose question is "how
precisely".

The logarithm makes every halving worth the same — 0.083 from 1600 m to 800 m,
0.082 from 100 m to 50 m, 0.068 from 6.25 m to 3.13 m — which is what "closer is
better" has to mean if it is to mean the same thing at every scale. It also
keeps a gradient arbitrarily far out, where a Gaussian on a band is identically
zero in floating point.

**The floor, not a band.** `precision_floor` is where precision stops being
meaningful, not where it stops being required: the reward flattens beneath it,
and it is set to a physical resolution (a barometric altimeter reads to about a
metre) rather than to a chosen tolerance. It also keeps the reward bounded,
since an unfloored logarithm diverges at zero error.

### Observation

`[x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick,
target_speed]`.

Nearly full state, and deliberately so: this is the suite's *transparent*
aircraft, where the difficulty is the plant rather than what can be seen of it.
An airliner does measure all of these. Airspeed, altitude and vertical speed
come from the air-data system, pitch and pitch rate from the inertial
reference, flight-path angle from the two together, and the aircraft knows its
own thrust and elevator commands.

Hidden: **the gusts** (`gust_x`, `gust_z`), which is the point -- turbulence is
a disturbance to be rejected, not a signal to be read, and no aircraft measures
the gust field it is flying into. Also hidden are **mass and fuel**, which move
slowly and which a controller tracking altitude has no need of, and `x`, which
nothing depends on.

Compare `plane3d`, where the same aircraft becomes partially observed once the
task is a path rather than a level, and `patrol`, where the reference itself
must be inferred.

### Task variants and initial conditions

The plant is shared; `target_pattern` selects what is commanded. `plane` holds a
level, `plane_sine` tracks a sinusoid as a frequency probe, and `plane_energy`
walks a ladder of levels *and* scores airspeed, which removes the spare degree of
freedom two actuators against one objective would otherwise leave. Setting
`speed_weight = 0` on `plane_energy` recovers the pure altitude staircase, which
used to be registered separately as `plane_steps`.

The ladder is eight levels drawn from `_STEP_LEVELS`, fractions of
`target_amplitude` chosen so that adjacent changes run 0.2 to 0.8 of it and
neither the level nor the direction repeats on a two-tread cycle. It replaced a
square wave alternating between two altitudes 2.4 km apart, which the aircraft
could fly but which is not what an altitude-hold task looks like.

Episodes start near the commanded level, within
`initial_altitude_offset_range`, rather than drawing the start independently of
the target over the same 5 km band. Independent draws gave a median start
1.7 km from the assigned level, so the episode opened with minutes of open-loop
climb before any tracking began.

## 5. Known deviations

**✅ D1 and D2 — resolved.** They were one defect, not two.

`cl_alpha` was 0.04 /deg, 54 % below the 0.0879 /deg that the wing's own
aspect ratio implies — the same geometry `k` already encoded correctly. With the
slope too shallow, the linear lift curve never approached `CL_max` before the
separation sigmoid (then centred *on* `aoa_stall`, halving lift exactly where it
should peak) cut it off at 0.70. Two symptoms, one cause.

The fix restores the derived slope and centres the sigmoid `aoa_stall_width`
degrees beyond the stall angle. The four lift constants then agree:
`cl0 + cl_alpha·aoa_stall = 1.52 ≈ CL_max = 1.5`.

| Quantity | Before | After | Reference |
|---|---|---|---|
| Peak CL (M 0.2) | 0.701 | 1.531 | `CL_max` = 1.5 |
| α at peak CL | 12.83° | 14.98° | `aoa_stall` = 15° |
| Clean stall speed (SL) | 228 kt | 154 kt | A320 ≈ 145–156 kt |
| Cruise trim AoA | 4.06° | 1.85° | A320 ≈ 1.5–3° |

Because the horizontal stabiliser and elevator use the same `aero_coefficients`,
this raises the destabilising wing moment *and* the stabilising tail moment.
`test_aircraft_is_statically_stable_in_pitch` confirms dM/dα < 0 still holds.

**✅ D3 — FIXED: compressibility no longer raises max lift.**
The stall clamp was applied *before* the Prandtl–Glauert `1/β` factor, so peak
CL kept rising with Mach — at M 0.9 the attainable CL reached the model's ±2
backstop instead of collapsing. Physically, shock-induced separation makes
CL_max *fall* past the critical Mach number; that is lift divergence, and it is
why transport aircraft have an overspeed limit at all.

The lift cap is now the Prandtl–Glauert-scaled stall limit up to `M_crit` and
falls beyond it (`k_shock_stall = 4.0`, floored at 25 %), so peak CL runs
1.53 → 2.00 → 1.50 → 1.00 across M 0.20 / 0.80 / 0.90 / 0.95. The two branches
agree at `M_crit`, making the change a **no-op everywhere the model was
validated** — cruise sits near M 0.7 — and correcting only the regime it
previously had backwards. Two tests pin it: the fall above `M_crit`, and the
Prandtl–Glauert rise below it that a naive clamp would have destroyed.

Worth noting what this did *not* fix. It was the leading suspect for the patrol
follower's departures, since every failed episode crossed `M_crit` 40–50 steps
beforehand. It made no difference to them: that failure is a lateral
bank oscillation, and the Mach excursion is a symptom of the thrashing rather
than its cause.

**✅ Post-stall — FIXED, and this note was stale long enough to mislead a
review.** It used to read "post-stall lift decays to zero ... would need a
Kirchhoff-style linear/flat-plate blend to fix". That blend is implemented, and
with a better constant than the note proposed: the separated branch uses
Viterna & Corrigan (1981), `CN_max = 1.11 + 0.018·AR`, which for this wing's
aspect ratio of 9.48 gives 1.28, so the peak separated normal force comes from
the wing's own geometry rather than the 2D flat plate's 2.0. Separation is
blended on `|α|` so both branches stall, and it is applied *after* the Mach
corrections, because separated flow is not a compressibility effect and
Prandtl-Glauert has no business scaling it.

**This is what makes the aircraft's operating envelope emerge instead of being
imposed.** A fully separated wing here makes at most `CL ≈ 0.64`. Holding 77° of
bank needs about 2.2, so an aircraft that asks for it is refused the lift,
descends under separated drag approaching 1.3, and fails through the ground
limit it already has. That is why `check_is_terminal` trips on altitude alone
and needs no stall, load-factor or attitude bound: the aerodynamics decline the
manoeuvre, and flying into the ground is the physical failure. Before the blend
a departed wing produced no force at all, fell at 300 m/s and tumbled without
damping, and nothing stopped a planner exploring there.

## 6. Validation method

Structural claims are tested from first principles; parameter claims against the
derivations in §3–4; behavioural claims against the reference points in §4.
Specifically, `tests/plane/test_plane_physics.py` asserts:

1. **Atmosphere** — ISA table values (§2), tight tolerance.
2. **Figures of merit** — L/D max, cruise CL, trim AoA, thrust margin (§4).
3. **Geometry consistency** — `k` against `1/(π·AR·e)` for a defensible `e`.
4. **Structural monotonicity** — drag ∝ V², lift ∝ ρ, CD rises past `M_crit`,
   CL falls past `aoa_stall`.
5. **Integrator convergence** — trajectories agree across `rk4_1` / `rk4_10` /
   `euler_100`, so the dynamics are not an artefact of step size.
6. **Trim existence** — a level-flight equilibrium exists at cruise.
7. **Lift curve** — measured dCL/dα equals `cl_alpha`; peak CL at `aoa_stall`;
   the four lift constants mutually consistent; stall speed and cruise trim.
8. **Pitch static stability** — dM/dα < 0, plus elevator sign and authority.
9. **Known deviations** — each asserted, so a fix flips the test rather than passing unnoticed (this is how D3 was closed).

Tests assert *emergent* behaviour, never a re-implementation of the formula
under test. A test that restates the implementation validates transcription, not
correctness, and fails in exactly the same way as a wrong formula.

**✅ D4 — RESOLVED: mass now follows the tanks.** Fuel burn is charged against
the thrust actually produced, `ṁ = c_T · T`, so the aircraft lightens as it
flies and `state.m` tracks `state.fuel` exactly. `specific_fuel_consumption` is
thrust-specific — 17.5 g per kilonewton-second, i.e. 1.75 × 10⁻⁵ kg/(N·s), the
right order for a high-bypass turbofan.

The figure is anchored rather than asserted. In level cruise thrust equals drag,
and drag is weight over the lift-to-drag ratio, so the fuel flow follows from
quantities validated in §4: 2 725 kg/h against an A320's 2 400–2 600. At full
thrust the model burns 5 007 kg/h, roughly double cruise, as expected.

Wiring it exposed a 20-tonne inconsistency that had been invisible. `reset_env`
set `state.m = initial_mass + initial_fuel_quantity` = 92 588 kg — above the
A320's MTOW of 78 000 — while the dynamics integrated `params.initial_mass`
directly. `state.m` was decorative, so nothing noticed; the moment mass became
load-bearing it would have stepped 20 tonnes between reset and the first
update. Fuel is a component of `initial_mass`, not an addition, and all three
sites (2D, 3D and patrol) now say so.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `plane` | 280 | 1 | 5 min | 2 in [-1, 1] | 10 | 16 |
| `plane_energy` | 1200 | 1 | 20 min | 2 in [-1, 1] | 10 | 16 |
| `plane_sine` | 480 | 1 | 8 min | 2 in [-1, 1] | 10 | 16 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + running + failure)`, shared by `plane`, `plane_sine` and `plane_energy`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1 m | documented minimum: barometric altimeter resolution. The test configurations fly with zero turbulence, so the achievable hold error is ~0 (the shipped PID holds 0.08 m on the altitude hold, `scripts/measure_hold.py`). |
| `e_tol` | 30 m | **provisional**: a defensible vertical-separation margin; an operator's tolerance would come from the flight rules |
| `tracking_exponent` | 2 | quadratic outside the tolerance |
| `c_hold` | 6.1 m/s (`plane`), 8.6 (`plane_sine`), 5.6 (`plane_energy`) | airspeed deviation from `target_speed` while holding, PID (`scripts/measure_hold.py`; the version-1 MPC ignored speed and sat 50-65 m/s off, so the better controller set the reference; the version-2 MPC holds cruise speed). Set per task in the registry. |
| `running_weight` | 1 | the speed term stands in for fuel; sweep 0.5 / 1 / 2 |
| `failure_cost` | 3e8 | twice the altitude envelope's cost, (12 192 / 1)^2 |
| `restart_steps` | none | steps the plant is down after a trip before it restarts (a crash: the aircraft stays down to the window's end); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the tracking cost per step at the floor in the reward's
units (the NEA floor for tracking) and `rho_floor` the full floor including
consumption charged in full; `floor_is_documented_minimum` records whether
`e_floor` is a measured/certified floor or a resolution used as a scale;
`failure_cost` is the per-step cost of a tripped plant, above the largest tracking
cost the envelope can produce. A trip never ends the window (`base.failure_kernel`):
the plant is frozen at that cost, with tracking and running cost zeroed, for
`restart_steps` steps and then restarts as `reset_env` would; a plant with no
restart stays down to the window's end. `terminated` is never raised;
`info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
