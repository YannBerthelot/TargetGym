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
| Engine spool dynamics beyond first-order lag | `compute_next_power` is a 1st-order lag above idle; a real engine spools slowest near idle. |

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
| `power_response_rate` | 1/1.7 | 1/s | 14 CFR 33.73(b): idle to 95 % in 5 s. A first-order lag takes ln 19 of its time constant, so 1.7 s gives 5.006 s, the rule to two figures and 6 ms over it (1.698 s is exact). A sea-level static certification limit, flown here at every altitude | ✅ |
| `idle_power` | 0.05 | – | **Ours, for the owner.** The plan's flight idle, 5 % of full power. The lever's range maps onto idle to full | ⚠️ |
| `stick_response_rate` | 1/0.06 | 1/s | First-order 0.06 s, which matched an industry actuator model for the elevator of the VELA concept (ICAS 2006, paper 531). Another aircraft, so borrowed | ⚠️ |
| `surface_rate_limit` | 30 | deg/s | Ours, accepted by the owner on 2026-11-06 as a working value. Full elevator travel in a second. No A320 rate was found in an open source | ⚠️ |

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
CL_wing = min(CL, CL_max - buffet_slope·max(0, M - buffet_mach))
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
| `buffet_mach`, `buffet_slope` | 0.3, 1.4 | Ours, accepted by the owner on 2026-11-06 as a working value. Prandtl-Glauert let the wing reach CL 2.0 at M 0.7 and full stick pull 5.2 g at 3500 m (the review read 5.5). No open A320 chart was found. The line starts at `CL_max` and is anchored on the 30 % lift margin a cruise keeps to buffet (Dandois et al., ONERA AerospaceLab 6, 2013): 65 t at FL390 and M 0.78 has CL 0.632, so 0.82, and the line gives 0.83 there. The wing's only: the tail keeps its own limit | ⚠️ |

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

## 5. Task design

Hold a sampled target altitude. The reward is the floor-normalised cost
described in *Reward (version 2)* below; `docs/reward-shaping.md` explains why
it replaced the logarithmic reward of version 1.

### Task: `plane`

Cruise an A320-class airliner at its assigned level and speed for 30 min,
through one zone of light turbulence, with throttle and a pitch demand. Redesigned
on 2026-11-08 under charter Amendment 2 from [REALITY.md](REALITY.md) and
[COLD_REVIEW.md](COLD_REVIEW.md); the section *Amendment 2: `plane`* below gives
every change and the status of every number it brought in.

### Task: `plane_sine`

Track an altitude that moves as a sinusoid, with the same airliner, throttle
and pitch demand as `plane`. A moving reference tests how well a controller
follows a frequency, not only how well it holds a level.

### Task: `plane_energy`

Fly ATC's level and speed clearances with the same airliner as `plane`: TECS's
job, altitude changes with the speed held and speed changes at a constant level
[TECS]. Redesigned on 2026-11-08 under Amendment 2; see *Amendment 2:
`plane_energy`* below.

### Observation: `plane`, `plane_sine`, `plane_energy`

| obs | meaning |
|---|---|
| 0 | `x_dot`, horizontal speed (m/s) |
| 1 | `z`, altitude (m) |
| 2 | `z_dot`, vertical speed (m/s) |
| 3 | `theta`, pitch angle (rad) |
| 4 | `theta_dot`, pitch rate (rad/s) |
| 5 | `gamma`, flight-path angle (rad) |
| 6 | target altitude (m) |
| 7 | throttle, 0 to 1 |
| 8 | elevator deflection (rad) |
| 9 | commanded airspeed (m/s) |
| 10 | airspeed, true, against the wind and gust (m/s) |
| 11 | normal load factor at the end of the step (g) |

Channels 10 and 11 were added under Amendment 2, rule 16: the autopilot reads air
data and the IRS's accelerations [BEA447, TECS, P409]. `env.obs_index` names
every channel.

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

### Turbulence

*Superseded for the scored tasks by Amendment 2 (below): a Dryden gust in one zone
per episode. The impulse mode below is unchanged.* The test configurations fly in light-to-moderate turbulence: an
Ornstein-Uhlenbeck gust on both axes with `turbulence_sigma = 1.2` m/s per 1 s
step and `turbulence_theta = 0.2`, a stationary gust standard deviation of
2 m/s (MIL-F-8785C puts light turbulence near 1.5 m/s and moderate near 3 m/s
at low altitude; less at cruise). It is what makes the altitude hold a
problem: in still air both shipped controllers held within 0.1 m of the
commanded level and the tracking term scored nothing, so the benchmark was
deciding on airspeed alone. The planners plan on the mean wind
(`turbulence_sigma` is in the spec's `noise_fields`), as a real MPC would.

### The inner loop

**A policy does not touch the surfaces.** An action is `[throttle, pitch demand]`,
in [-1, 1], held for the 1 s step, and full action is +-15 deg of pitch. The plant
flies the elevator to the demand. Policies, experts and oracles all send the same
1 Hz demands, so they share one action space. The review's reason: a 1 s step sits
on the 2.5 s short period (2.5 rad/s, damping 0.36 at 5500 m), so a controller on
the surfaces at 1 Hz was fighting its own sample rate. The rejected alternative was
a 4 Hz step, at four times the steps and the oracle's cost.

The loop runs at 4 Hz inside the step (`attitude_axis`, `compute_next_state`), and
each sub-step moves the surface and the engine and integrates the dynamics with
one RK4 step, so the sixteen force evaluations a step costs are twice the eight of
the old `rk4_2`. One axis is `kp s err + integral - kd s^2 rate`, clipped to the
surface's travel, which is `AP_RollController` with its integral in surface units,
frozen while the surface is saturated. `s` is ArduPilot's airspeed scaler, tuning
EAS over EAS, clamped to [0.3, 2].

The gains are task constants, one tune for every task, from the attitude procedure
of `docs/proposals/expert-protocols-aircraft.md` (ArduPlane 4.0.9's manual P, D, I),
with its amendment of 2026-11-06, run once in still air by
`scripts/tune_inner_loop.py`. The throttle holds 230 m/s through both tests, by a PI
at 1 Hz whose gains follow from the plant's numbers by SIMC, 0.265 per m/s and 10.8 s.
That hold is ours: the guides give none. It keeps the speed within 224 to 234 m/s, so
the speed mode no longer reads as overshoot.

| axis | P (ArduPilot, ours) | D | I | how the recipe ended |
|---|---|---|---|---|
| pitch | 5.0, 6.366 | 0.1, 0.1273 | 0.2, 0.2546 | **P at its cap**, D at the guide's limit, I halved from the first test that overshot, 0.4 |
| roll | 0.7, 0.8913 | 0.1, 0.1273 | 0.025, 0.0318 | P walked down from 1.0, D at the guide's limit, I halved from 0.05, its first test |

**⚠️ For the owner.**

- **Pitch P is the cap's, not the recipe's choice.** No test overshoots at any P up
  to the recipe's cap of 5.0, so the number says it found no limit below that.
- **D stops at the guide's 0.1,** by the amendment. No test oscillated on either axis:
  past about 10 the roll loop only slows down.
- **Roll's P stands at 0.7 under the overshoot rule,** the owner's call on 2026-11-06.
  The defaults already overshoot and one step back still does, by 8 %, so P walks down
  to the last clean step.
- **Step 4 halves I and does not fly the result,** so the shipped roll loop overshoots
  a 45 deg step by 12 %, past the 5 % reading and inside the 22 % that reads as
  oscillation. It reaches the step in 4 s at 14.5 deg/s, near the 15 deg/s of the
  A320's normal law, and the integral takes until 20 s to bring it back inside 5 %.

Pitch reaches 63 % of a step in about 1 s, overshoots by 2.5 % at most, and is inside
5 % by 5 s.

A reset aircraft engages in trim (`trim_stick`): its elevator started 5 deg from
trim, so a 1.8 deg demand pulled 1.9 g in the first second.

### The envelope

*Superseded for the 2D aircraft by Amendment 2 (below): load factor and AoA are
protected, VMO/MMO is a warning, and only the ground trips. The table stays true
for the 3D aircraft until its own change.* The plant trips outside these limits (`check_is_terminal`, read on the true state):

| limit | value | source | |
|---|---|---|---|
| load factor | +2.5, -1 | 14 CFR 25.337(b) and (c) | ✅ |
| `vmo_eas` | 350 kt EAS | OpenAP's A320 data; the FAA sheet A28NM was not opened. VMO is in CAS, and EAS is within a few knots of it below M 0.6 | ✅ |
| `mmo` | 0.82 | as above | ✅ |
| bank (3D) | 60 deg | 14 CFR 91.307(c). The demand stops at 45 | ✅ |

The load factor read is the step's worst, not its last, so a pull inside a step
counts. Neither the altitude band nor these limits end an episode: `failure_kernel`
charges the trip and restarts the aircraft. The patrol tasks do not read them yet,
which is PR 7b.

### The speed schedule

The airspeed demand is 290 kt CAS, then M 0.78 (`scheduled_airspeed`), EUROCONTROL's
A320 climb figures, since a fixed 230 m/s was 385 KEAS at 3000 m, past VMO. It gives
171 m/s at 3000 m, 193 at 5500 and 219 at 8000, the CAS branch all the way: the
crossover is near 9600 m. Observation 9 and the airspeed term read it at the
aircraft's height.

### The expert

The PID slot holds TECS (`plane_tecs_step` in `experts.py`), what PX4 and ArduPilot
fly for height and speed and the 3D expert's longitudinal half, with the same
aircraft numbers and tuning (`plane3d/PHYSICS.md`,
`scripts/tune_aircraft_guidance.py`). Its pitch demand goes to the inner loop. Throttle closes
the total energy error and pitch holds height. On held-out seeds 200-209 it
holds altitude within 1.5 m on average on `plane`, 1.9 m on `plane_sine`, and
1.6 m on `plane_energy` once each ladder step is captured; the airliner climbs
those 360-720 m steps at its 8 m/s, which fills most of each 150 s rung. It
replaced, in 2026-10, a cascaded PID tuned on the scored seeds.

### The reference controller

The MPC slot holds the shared solver (`plane_model` in `oracle.py`,
`target_gym.oracles`): the episode reward on the noise-free plant, over the
horizon the horizon protocol measured, solved to convergence once every Cooper
sample time, 3 steps on all three tasks, which share one test. Its model is the plant's
step, inner loop included, restated in `oracle_model.py` with its kinks treated by
the smoothing protocol, and K8's replay runs each plan on the exact step. The
trips are hard limits on every visited state: the altitude band, the load factor
at each inner sub-step, VMO and MMO, each backed off by the planner's worst
one-step error on it. It reads no expert code. A surface's rate limit needs no
row: 30 deg/s for a 1 s step crosses the whole travel, so no demand lies past its
reach, and the corner inside the step is smoothed.

## Amendment 2: `plane` (2026-11-08)

Each change cites [REALITY.md](REALITY.md) (tags in square brackets are its
sources) and, where it fixes a finding, [COLD_REVIEW.md](COLD_REVIEW.md). Tests:
`tests/plane/test_plane_reality.py`, from [BEHAVIOUR_TESTS.md](BEHAVIOUR_TESTS.md).

**Limits (rule 14).** Load factor and AoA are protected, as normal law protects
them [CFR 25.337, NTSB 1.6.3, BEA447] (cold review A1, A3):
`protected_pitch_demand` clips the attitude demand to the incidence that lifts
`load_factor_max` (2.5) or `load_factor_min` (-1.0) times the weight, or reaches
the buffet line, whichever binds first. The inversion uses the wing's linear lift
curve and its Prandtl-Glauert factor and leaves the tail's share out. Probe from
trim, full demand, calm air: at most 2.08 g, at least -0.55 g, AoA at most 8.5 deg,
from 3 000 to 8 000 m. VMO/MMO is the TCDS's warning (`info["overspeed"]`), not a
trip (A2); no high-speed protection is modelled, since what the controls do above
VMO is not found. The ceiling is 39 100 ft [TCDS], an operating limit. The ground
is the one trip (A4, A5).

**Readings (rule 16).** Airspeed, gust included, and normal acceleration are
observed (A6, A7). Sensor latencies are not found [REALITY.md], so none.

**Episode (rule 17).** The hold starts established: at the assigned level, at the
scheduled airspeed against the mean wind, in trim (`trim_level`, Newton's method
on the model's accelerations) (A9, A10). The target is one level [RVSM]. One
turbulence zone per episode [BEA447 Figure 24] (A11); the gust is a Dryden
first-order gust with the 1 750 ft scale length [MADDEN] on every axis (A12).

**Score (rule 15).** `tracking`: zero inside +-65 ft of the level [RVSM]; past it,
`((|e| - 65 ft) / 65 ft)^2` per second, which is the shared `tracking_cost` with
`e_tol = e_floor = 65 ft`. `running`: the same on the assigned speed's band
(`speed_band_cost`), +-10 kt CAS below the schedule's crossover and +-0.02 M above
it [AIM 4-4-12]. `rho_floor` is 0: the best hold inside the bands costs nothing.
The 2D oracle plans the earlier cost and refuses `plane`'s, so `plane`'s room is
"not measured" (rule 9) until the oracle is rebuilt. Precision inside a band earns nothing (A16). The squared shape is the
owner's choice of 2026-11-08: real regimes (RVSM monitoring, RNP containment)
budget large deviations more tightly than small ones, and the square is the
smooth form of that.

| Number | Value | Status |
|---|---|---|
| `load_factor_max`, `load_factor_min` | 2.5, -1.0 g | cited: CFR 25.337(b), (c), the limits normal law protects |
| AoA limit | the buffet line's | ours: the model's own stall boundary; the A320's alpha max is not found |
| `vmo_eas`, `mmo` | 350 kt, 0.82 | cited: TCDS item 10 |
| `max_alt` | 39 100 ft | cited: TCDS item 11 |
| `altitude_band` | 65 ft | cited: RVSM A.4.1.4 |
| `speed_band`, `mach_band` | 10 kt, 0.02 | cited: AIM 4-4-12, read 2026-11-08 on faa.gov |
| band cost shape | squared excess in band widths | ours: the owner's choice, see above |
| `gust_scale_length` | 1 750 ft | cited: MADDEN, after MIL-F-8785C |
| `turbulence_sigma` | 0.55 m/s | derived: the sharp-edged gust load dn/dw = rho V S a / (2 m g) is 0.073 g per m/s at the schedule's 193.5 m/s at 5 500 m (Prandtl-Glauert slope); a minute holds 60 V / L = 22 scale lengths, whose expected range is 2 sqrt(2 ln 22) = 4.96 standard deviations; so 0.2 g peak to peak, the level BEA447 Figure 24 records most, needs 0.2 / (4.96 x 0.073) = 0.55 m/s. Checked by A5: median minute 0.155 g, worst 0.45 g |
| `zone_duration_range` | 180 to 1 800 s | cited: BEA447 Figure 24 |
| `zone_onset_range` | anywhere in the episode | ours: every episode holds one onset; real calm lasts over an hour [BEA447 Figure 24], longer than an episode |
| episode | 1 800 s | ours: one zone of up to 30 min, with calm around a shorter one |
| `failure_cost`, `restart_steps` | ((12 192 - 19.8) / 19.8)^2, 3 600 | ours: a crash is not priced in any source; an hour at the cost of 40 000 ft off the level |
| model gas constant | 287.0 J/(kg K) | known deviation: ISA's is 287.053, so the model's CAS differs from TAS at sea level by 6e-5; kept, since the patrol tasks share the schedule |
| `target_cas`, `target_mach` | 290 kt, M 0.78 | unsourced in REALITY.md (EUROCONTROL's A320 climb, cold review D4) |
| `target_altitude_range` | 3 000 to 8 000 m | unsourced: cruise levels are not in REALITY.md (cold review B1) |
| TECS `time_const` | 5 s | cited range 3 to 10 s [P409, TECS]; 5 is AP_TECS's default (C1) |

**What the reference does on it.** TECS, still air, 10 min: within 11.3 m (B1).
In a light zone for 30 min: within 12.3 m and 2.5 m/s, median minute 0.155 g. On
five scored seeds (100 to 104) the whole-episode cost is 0: the reference never
leaves a band. On this score `plane` is solved by the reference; whether it stays
a benchmark candidate is for the room table (rule 9).

**Known deviations (rule 13).** The gust is held over the step's four sub-steps,
so loads step each second; the vertical Dryden gust is second order, ours first
order with the right scale; no sensor latency; no high-speed protection; the
speed schedule and altitude range are unsourced. None changes the score while the
reference stays inside the bands.

## Amendment 2: `plane_energy` (2026-11-08)

The plant, the readings, the turbulence zone and the established start are
`plane`'s (above). Tests: `tests/plane/test_plane_energy_reality.py`.

**Mission (rule 17).** The cold review found the old ladder's level steps
instant and its score without a trade (D1, D3). Now, from established cruise,
every 300 s one clearance (`CLEARANCE_LEVELS_FT`, `CLEARANCE_SPEEDS_KT`): +2 000 ft
at 300 s, +10 kt at 600 s, -1 000 ft at 900 s, -10 kt at 1 200 s, -2 000 ft at
1 500 s. Levels move in whole 1 000 ft, the vertical separation minimum [RVSM];
speeds in 10 kt [AIM 4-4-12: adjustments "in 5 or 10 knot increments"].

**Score (rule 15).** The speed band, +-10 kt about the assigned speed, all
along [AIM 4-4-12]. The altitude band, +-65 ft [RVSM], about a level the
aircraft must close at no less than 500 fpm from each level clearance: "if at
anytime the pilot is unable to climb or descend at a rate of at least 500 feet
a minute, advise ATC" [AIM 4-4-10]; ahead of that profile costs nothing.
`level_off`: within 1 000 ft of the level, no more than 1 500 fpm [AIM 4-4-10:
"between 500 and 1,500 fpm until the assigned altitude is reached"]. Thrust is
finite, so a climb at the minimum rate with the speed held is the
speed-against-height trade TECS makes (B3: a pitch-only step moves total energy
under 20 % of what a throttle step does, true of the plant before and after).

| Number | Value | Status |
|---|---|---|
| level step | 1 000 ft | cited: RVSM's vertical separation minimum |
| speed step | 10 kt | cited: AIM 4-4-12 |
| minimum level-change rate | 500 fpm | cited: AIM 4-4-10, read 2026-11-08 on faa.gov |
| level-off rate | at most 1 500 fpm within 1 000 ft | cited: AIM 4-4-10 |
| level-off band width | 500 fpm | ours: half the AIM's 500 to 1 500 fpm range |
| `clearance_period` | 300 s | ours: a 2 000 ft change at 500 fpm takes 240 s, plus a minute held |
| clearance order and sizes | +2 000 ft, +10 kt, -1 000 ft, -10 kt, -2 000 ft | ours: both kinds, both directions; a published A320 profile is not found |
| `target_altitude_range` | 3 400 to 7 300 m | ours: keeps every cleared level inside 3 000 to 8 000 m |
| episode | 1 800 s | ours: six clearances |

**What the reference does on it.** TECS on seeds 100 to 104 never leaves the
altitude or speed band (at most 0.1 per episode) and never trips, but costs 159
to 180 per episode on `level_off`: it closes on a level at up to 4 300 fpm where
the AIM asks for 1 500. A controller that levels off by the rule scores less.
The 2D oracle plans the earlier cost, so the room is "not measured" (rule 9).

**Known deviations.** The speed schedule's CAS is still the unsourced 290 kt;
no published clearance profile was found, so the sequence is ours.

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
| `plane` | 280 | 1 | 5 min | 2 in [-1, 1] | 10 | 21 |
| `plane_energy` | 1600 | 1 | 27 min | 2 in [-1, 1] | 10 | 21 |
| `plane_sine` | 480 | 1 | 8 min | 2 in [-1, 1] | 10 | 21 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

*On `plane` and `plane_energy`, the terms below are replaced by the band costs of
Amendment 2 (below); `plane_sine` (Extra) keeps them.*

`compute_reward = -(tracking + running + failure)`, shared by `plane`, `plane_sine` and `plane_energy`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 19.812 m (`plane` and `plane_energy`, Amendment 2: the +-65 ft band [RVSM], below), 1.26 (`plane_sine`) | `plane` before Amendment 2 used 1 m, the resolution. The long-run mean \|error\| the shipped MPC holds in the test turbulence, lowest per seed (`scripts/measure_hold.py`, 2 seeds), floored at the 1 m barometric resolution: a hold the instrument cannot see is not a floor. `plane`: the resolution; the MPC holds 0.55 / 0.46 m (the cascaded PID 1.83 / 0.87; the expert 1.5 on held-out seeds). `plane_sine`: an earlier MPC's hold. The current one holds 0.53 / 0.53 m (the cascaded PID 45.4 / 44.7; the expert 1.9 on held-out seeds), below the resolution, and the resolution is within 1.5x of 1.26 m, so the floor stays (oracle audit, 2026-10). `plane_energy` (`-v3`): the MPC's 1.198 m between ladder steps, rounded; under that floor it holds 1.35 / 1.18 m (the cascaded PID 8.5 / 12.1). It was 4.55 m, an earlier MPC's hold taken before its moves toward the next target were separated from its hold. Upper bounds otherwise |
| `e_tol` | 0; 19.812 m on `plane` and `plane_energy` | `plane` (Amendment 2): +-65 ft [RVSM A.4.1.4], inside which precision earns nothing (rule 15). The others: no dead zone. A +-30 m band (the pilot's instrument tolerance; the autopilot's is +-20 m, RVSM +-65 ft) was tried and made the hold vacuous: both controllers held within 0.1 m in still air, and the tracking cost was identically zero |
| `tracking_exponent` | 2 | quadratic outside the tolerance |
| `c_hold` | 5.06 m/s (`plane`), 8.15 (`plane_sine`), 5.23 (`plane_energy`) | airspeed deviation from `target_speed` while holding, the cascaded PID's (`scripts/measure_hold.py`), the better controller on speed when it was the expert. The MPC holds 5.13 and 8.25 m/s on `plane` and `plane_sine`, where its throttle is the PID's own airspeed loop once it holds, and 19.6 on `plane_energy`, where it plans the throttle itself |
| `running_weight` | 1 | the speed term stands in for fuel; sweep 0.5 / 1 / 2 |
| `failure_cost` | 3e8 (`plane`), 1.9e8 (`plane_sine`), 2.1e8 (`plane_energy`) | twice the altitude envelope's cost, 2 x (12 192 / e_floor)^2 |
| `restart_steps` | 3600 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (a crash loses the sortie: an hour of flight, provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the oracle's best hold for tracking: the lowest per-seed
hold cost the reference controller demonstrated, in the reward's units, which
is 1 per term where the floor is that hold and less where the floor is clamped
at the instrument resolution. Here it is (0.4581 / 1)^2 = 0.210 on `plane`,
(0.5285 / 1.26)^2 = 0.176 on `plane_sine` and (1.1773 / 1.20)^2 = 0.963 on
`plane_energy`. `rho_floor` is the same with consumption
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

The observation reports the aircraft through its instruments, with the target
altitude and commanded airspeed exact (`docs/proposals/measurement-noise.md`).
`sensor_noise_scale = 0` gives the noise-free task, rounding included.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `z` | 0.2 m | 1 ft | Air data computer baro altitude | Provisional: a ground test set's figure, no airborne one found | [DPST-8000](https://www.avionteq.com/Document/DPST-8000-specification-sheet.pdf) |
| `x_dot` | 0.03 m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling: receiver accuracy, not scatter | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `z_dot` | 0.03 m/s | 1 ft/min | IRS inertial vertical speed | No source: a placeholder, not a bound | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `theta` | none | 0.010986 deg | IRS pitch attitude | Corrected: rounding only | [HG9900](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `theta_dot` | 4.3e-6 rad/s | 0.015625 deg/s | IRS body pitch rate | Confirmed | [Honeywell Mk VI guide](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `gamma` | 1.3e-4 rad at 230 m/s, as 0.03 / V | from the velocities | IRS flight-path angle | Confirmed | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `power` | none | 0.003 | Engine thrust indication via FADEC | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `stick` | none | 1.6e-3 rad | Elevator surface position sensor | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |

`gamma` has no noise of its own: it is worked out from the velocities read, as an
IRS does. With `observe_wind` the wind channels stay exact, since no scored task
reads them and no instrument was sourced.

The figure the tracked altitude rests on is weak. No air data computer publishes
its scatter, so 0.2 m comes from a ground test set, and the benchmark runs at
`sensor_noise_scale` 0.5 and 2 to show how far a result depends on it. The
ground velocity figure is a receiver's accuracy, so true scatter is smaller, and
the vertical speed figure is a placeholder.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, including its throttle loop, so it stays a full-state bound.
`reward`, trips and the score read the true state. `info["measured_reward"]` is
the same cost on the readings, with any trip taken from the true state, and a
learner can train on it.
