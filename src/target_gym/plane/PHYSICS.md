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
| Fuel burn | `specific_fuel_consumption` exists but mass is held constant; the altitude task is short relative to the fuel time constant. |
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

**⚠️ Post-stall lift decays to zero.** A fully separated wing still behaves
roughly like a flat plate (CL ≈ 2·sin α·cos α, so ~0.9 at 25°), whereas this
model sends CL → 0. Deep-stall recovery is therefore not represented. Acceptable
while the tasks stay inside the normal envelope; would need a Kirchhoff-style
linear/flat-plate blend to fix.

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

**⚠️ D4 — mass is constant.** `initial_mass` is used throughout; fuel burn is
not subtracted, although `initial_fuel_quantity` and
`specific_fuel_consumption` are carried in the parameters. Over the episode
lengths here the error is small — an A320 burns on the order of 0.7 % of its
mass in ten minutes — but it means the model cannot represent a long cruise, and
that the range implied by the fuel parameters is not something this model
computes. This was previously a `TODO` in the code; it is a documented
simplification rather than pending work.
