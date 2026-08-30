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

**❌ D3 — compressibility raises max lift instead of lowering it.**
`CL / β` is applied uniformly, with no Mach effect on stall onset, so peak CL
*rises* with Mach. Physically, buffet onset and CL_max *fall* with Mach on a
swept wing; the sign of the effect on maximum lift is inverted. Prandtl–Glauert
is also only valid below drag divergence, so applying it unmodified at M 0.78 is
already marginal. Out of scope for the D1/D2 fix; tracked by a `strict` xfail.

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
9. **Known deviations** — D3 as a `strict` xfail.

Tests assert *emergent* behaviour, never a re-implementation of the formula
under test. A test that restates the implementation validates transcription, not
correctness, and fails in exactly the same way as a wrong formula.
