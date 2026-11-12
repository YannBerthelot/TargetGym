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
the total energy error and pitch holds height.

**Its pitch loop is ArduPilot master's, not 4.0.9's** (owner's decision of
2026-11-12, departing from decision D1 of `docs/proposals/expert-protocols-aircraft.md`,
which pinned 4.0.9). On long streams the 4.0.9 port fell into lasting height
cycles: with `TECS_SPDWEIGHT` 0, 4.0.9 weights potential energy by 2, which doubles
the height loop's gain and feed-forward, and against the autopilot's 0.2 g band
(`pitch_demand_bounds`) an upset of 20 to 50 m became a cycle of +-400 to 600 m that
only a new sortie ended (plane, plane3d_heading, plane3d_racetrack; plane_energy's
windows too, from its first clearance). ArduPilot fixed this upstream in
`AP_TECS::_update_pitch` [SRC-M]: neither weight above 1, "to prevent instability if
tuned at a speed weight of 1 and weighting is varied to end points in flight"; the
energy-balance rate a height error asks for, error over `TIME_CONST` plus the demand's
rate, clipped to `TECS_CLMB_MAX` and `TECS_SINK_MAX`; and the integral on the rate
error, held 5 deg past the pitch limits and faded while the demand saturates. The
throttle loop, the demand slews and the gains are unchanged; `TECS_VERT_ACC` stays
out, as in the 4.0.9 port, the plant's autopilot holding its 0.2 g. The guide's own
remedy, `TIME_CONST` up to its cap of 10, cured `plane` alone (from 6); the heading,
racetrack and energy tasks still cycled at 10. `tests/plane/test_tecs_upset.py` holds
the result: displaced 20 to 100 m off its level in calm air, the expert is back inside
G-TECS's 10 m within 5 min and stays there. The patrol family's scripted lead, part of
its plant, keeps the 4.0.9 port (`tecs_step_409`).

On held-out seeds 200-209 it
holds altitude within 1.5 m on average on `plane`, 1.9 m on `plane_sine`, and
1.6 m on `plane_energy` once each ladder step is captured; the airliner climbs
those 360-720 m steps at its 8 m/s, which fills most of each 150 s rung. It
replaced, in 2026-10, a cascaded PID tuned on the scored seeds. (These figures predate
the master pitch loop and the re-record pass; step 5 of
`docs/proposals/re-record-pass.md` re-measures them.)

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
| altitude band | 65 ft | cited: RVSM A.4.1.4 |
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
along [AIM 4-4-12]. The altitude band, +-65 ft [RVSM], about a level that
closes at 500 fpm from each level clearance, so the band travels with that profile
and a prompt start costs nothing: "if at
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

## Amendment 3: `plane` and `plane_energy` (2026-11-08)

The owner's ruling: a task represents its family, here the A320 family of airliners;
a number not sourced is allowed inside the family's documented range, labelled
"ours, family range". Tests: `tests/plane/test_plane_reality.py`,
`tests/plane/test_aircraft_scenarios.py`.

**`plane`: cruise levels.** FL290 to FL370, RVSM airspace where the +-65 ft band
applies [RVSM], clear of the 39 100 ft maximum altitude [TCDS] (the reviewer's
FL390 cost 294 at that edge). The light zone's sigma is re-derived at FL330 on
the M0.78 schedule: dn/dw = 0.4097 x 233.4 x 122.6 x 8.04 / (2 x 73 500 x 9.81) =
0.0654 g per m/s, a minute 26.3 scale lengths, expected range 5.11 sigma, so 0.2 g
peak to peak needs 0.598 m/s. A5 still holds at the new levels.

**`plane`: a cost operators pay (rule 23).** Operators pay fuel and time, traded by
the cost index; long-range cruise sits at 99 % of the best specific range, where
the curve is flat (secondary sources, REALITY.md, unconfirmed). Fuel is modelled in
kg. Measured on seeds 0 to 2: flying 8 kt slow, inside the band, burns 1.3 to 1.9 %
less fuel in the 30 min window (1 171 to 1 156 kg, 1 285 to 1 261, 1 266 to 1 244),
and 9.5 kt slow 1.8 to 2.4 % less while touching the band's edge (0.3 to 1.0). That
is a room of about 1.02 on fuel in a fixed-time window, and less on the operator's
real cost, fuel per distance plus time. No other cost is found: the reviewer's ride
and throttle terms were refuted (gusts set the ride; throttle travel has no wear
source). So `plane` is listed as solved (rule 23), and stays in the core pool.

**`plane`: scenarios (rule 20)**, `plane/scenarios.py`: normal (one light zone),
calm, the worst recorded zone (0.5 g peak to peak, 2.5 times the light sigma
[BEA447 Figure 24]), a 30 min zone over the whole window, heavy and high (78 t,
FL350 to 370 [TCDS]), light and low (60 t, FL290 to 310).

| Number | Value | Status |
|---|---|---|
| `target_altitude_range` | FL290 to FL370 | cited: RVSM's airspace, under TCDS's maximum altitude |
| `turbulence_sigma` | 0.598 m/s | derived at FL330, above |
| `target_cas`, `target_mach` | 290 kt, M 0.78 | ours, family range: under the A320's VMO 350 kt and MMO 0.82 [TCDS]; a cruise Mach range is not sourced in REALITY.md (cold review D4) |
| `initial_mass` | 73.5 t; 60 t and 78 t in scenarios | ours, family range 42.6 t (OEW) to 78 t (MTOW) |
| worst zone | 2.5 x the light sigma | derived: 0.5 g, BEA447 Figure 24's top, over 0.2 g, the load being linear in the gust |
| episode | 1 800 s | ours: one zone of up to 30 min [BEA447] |
| scenario weights | normal 0.40, calm 0.35, worst zone 0.05, long zone 0.05, heavy-high 0.075, light-low 0.075 | ours, family ranges in `plane/scenarios.py`; the turbulence shares bounded by BEA447 Figure 24 (a zone in force at most about a third of cruise time) |

**`plane_energy`: Extra (rule 18).** The owner's criterion: a heavy member, 78 t,
at FL330 to 370; Extra if residual climb stays well above 500 fpm. It is 909 to
1 163 fpm (V (T_max - D) / W), and 879 to 1 123 fpm in a full-thrust climb at held
airspeed in the plant, so thrust never runs out and the speed-against-height trade
never happens. Reasons and evidence in [EXTRA.md](EXTRA.md).

**Reference ladder row (for `scripts/reference_ladder.py`, not edited here).**
`plane` needs a row with two loops on the 2D plant: altitude (CV `z`, MV pitch
demand, action 1) and airspeed (CV `airspeed`, MV throttle, action 0), bump-tested
at the corners of `target_altitude_range` and of `initial_mass` (60 and 78 t, the
scenarios' ends); rung 2's balance is `trim_action`; the zone for rule 22's
back-off is the +-65 ft and +-10 kt bands; limits are the protected pitch demand
the plant already applies. Until then TECS is the reference.

## Amendment 4: the 2026-11-10 review

An outside review of the aircraft family (a pilot's and aeronautical engineer's
read, not of our earlier versions) found the model's longitudinal derivatives
wrong in ways that cancelled, an engine that answered as fast at FL370 as at sea
level, a policy free to command a negative load factor, and `plane` scoring zero
for the reference in every scenario. The owner's decisions of 2026-11-10 and what
changed. Tests: `tests/plane/test_plane_physics.py` (sections 5, 7 and 8),
`tests/plane/test_autopilot_limits.py`, `tests/plane/test_plane_reality.py`.

Sources opened for it, beside REALITY.md's:

- [Jenkinson] Jenkinson, Simpkin and Rhodes, *Civil Jet Aircraft Design*, aircraft
  data file, A320-200 column. <https://booksite.elsevier.com/9780340741528/appendices/data%2Da/table-1/table.htm>
- [JSBSim] JSBSim's A320 model (`aircraft/A320/A320.xml`, inertias) and its
  MIL-F-8785C Figure 7 table (`src/models/atmosphere/FGWinds.cpp`).
  <https://github.com/JSBSim-Team/jsbsim>
- [AMC 25.1329] EASA CS-25, AMC 25.1329, Flight Guidance System (online Easy Access
  Rules), altitude capture and Appendix A4.2.
  <https://www.easa.europa.eu/en/document-library/easy-access-rules/online-publications/easy-access-rules-large-aeroplanes-cs-25>
- [KFC 500] BendixKing KFC 500 performance specifications: altitude capture at
  +-0.1 g. <https://www.manualsdir.com/manuals/651155/bendixking-kfc-500.html?page=43>
- [TM 1-1510-225-10] An Army operator's manual's preselect altitude capture at
  +-0.2 g. <https://aviationandaccessories.tpub.com/TM-1-1510-225-10/css/TM-1-1510-225-10_481.htm>

**The swept wing.** The A320's wing is swept 25 deg at the quarter chord [Jenkinson],
and its lift slope is DATCOM's for a swept wing, `2 pi A / (2 + sqrt(A^2 (beta^2 +
tan^2 L) + 4))`, with `L` the half-chord sweep (21.9 deg, from 25 deg and a taper of
0.24): 4.80 /rad at Mach 0 (`cl_alpha` 0.0838 /deg) and 6.39 at M 0.78. Its growth
with Mach (`compressibility`) replaces Prandtl-Glauert's 1/beta, a 2D section's,
which overstated the slope at M 0.78 by a fifth (8.04 /rad). `aoa_stall` moves to
15.5 deg, where the swept slope's linear lift reaches `CL_max`.

**The tail and its downwash.** The tail was the wing's own section, cambered, at the
wing's slope, with a separate 10 m^2 all-moving "elevator" and no downwash: the
stick-fixed static margin came out at 56 % of the MAC. Now
(`longitudinal_airframe`, shared by the 2D and 3D aircraft and restated in
`oracle_model.py`):

- the horizontal tail is the A320's, 31 m^2, aspect ratio 5.0, 29 deg of sweep, 13.53 m
  of tail arm from the wing's quarter chord [Jenkinson], on its own DATCOM slope and a
  symmetric section;
- it meets the body's incidence less the wing's downwash, lifting-line's far field
  `2 CL / (pi A)` (`downwash`): it takes 0.32 of every change of incidence at low
  speed, 0.43 at M 0.78;
- the elevator turns the tail's incidence by half its deflection (`elevator_effectiveness`
  0.5, ours: a 30 % chord flap, 0.66 by thin-aerofoil theory reduced to what flaps
  reach), about a fixed stabiliser setting that holds the reference cruise at zero
  elevator (`stabilizer_setting`, calibrated, 1.38 deg);
- the wing-body's aerodynamic centre sits 0.822 m ahead of the centre of gravity: the
  wing's quarter chord moved forward 0.152 MAC by the fuselage (Torenbeek's 1.8 b h l /
  (a S c), the 3.95 x 4.14 m section [Jenkinson], 13 m of nose ahead of the wing root,
  ours) and the centre of gravity at 29 % MAC (ours).

The neutral point this predicts, `h_ac + eta V_H (a_t / a_w)(1 - de/da)`, is 0.41 MAC
at M 0.78 and 0.50 at low speed: a static margin of 12 % and 21 % at a 29 % centre of
gravity. The model's own moments give 11.3 % at FL330.

**Inertia.** `I` was 9e6 kg m^2, 2.4 times a radius-of-gyration estimate, which is
what kept the short period plausible against the overstated margin. Now the pitch
radius of gyration is JSBSim's A320 model's, 3.78e6 kg m^2 over its 50.3 t empty mass,
8.67 m, applied at the all-up mass (`pitch_inertia`: 5.5e6 at 73.5 t), and so the roll
one, 5.04 m (1.9e6 at 73.5 t, against 2.5e6 before).

| Bare airframe, FL330, M 0.78, 73.5 t | Before | After | Reference |
|---|---|---|---|
| Static margin, stick fixed | 56 % MAC | 11.3 % | transports, 5 to 25 % (ours) |
| Airplane lift slope | 10.5 /rad | 7.03 /rad | swept wing 6.39 plus tail 0.65 |
| Short period | 1.99 rad/s, zeta 0.30 | 0.98 rad/s, zeta 0.40 | Etkin and Reid's 747 at cruise, about 1 rad/s and 0.4 |
| Phugoid | 82 s | 74 s | Lanchester's 106 s, less the growth of lift with Mach |
| Cruise trim incidence, FL350 | 1.47 deg | 2.50 deg | A320 about 2 to 3 deg |
| Elevator across the cruise envelope | -3.1 to -0.4 deg | -0.04 to +0.55 deg | trimmed by the stabiliser |

**The attitude loop, retuned.** The new airframe gives 26 % more elevator power per
unit of inertia, and the shipped gains, P at the recipe's cap, ran the 4 Hz loop near
its edge: a 2 deg pitch step from trim climbed 177 m in a minute where the loop at
lower gain climbs 308, and the integrators disagreed by 6 m. The procedure
(`scripts/tune_inner_loop.py`, unchanged) was run again on the new airframe: pitch P
1.20, D 0.10, I 0.05 in ArduPilot's units (1.528, 0.127, 0.064 ours); roll P 1.0, D
0.1, I 0.025 (1.273, 0.127, 0.032). A 3 deg pitch step reaches 63 % in about 2 s.

**The engine at altitude.** The 1.7 s lag is the certification's sea-level static
rule [14 CFR 33.73(b)]. A gas generator's transients scale in corrected time,
`t delta / sqrt(theta)` (the torque that spins the spools goes with the inlet
pressure, the speed it must reach with the root of its temperature), so the lag is
`1.7 sqrt(theta) / delta`: 4.9 s at FL290, 6.3 s at FL350 (`engine_response_rate`,
derived). Secondary F404 figures agree: about 3 s idle to military at low level, 6 s
at 20 000 ft and 9 s at 30 000 ft as forum posts quote NATOPS, where the rule gives
6.1 and 9.0.

**The autopilot's manoeuvre limit.** A full pitch demand from cruise pulled -0.6 g
and +1.4 g, normal law stopping it only at -1 and 2.5 g, its structural limits. A load
factor below zero is a Hazardous effect [EASA AMC 25.1329, A4.2], and an altitude
capture uses "an acceptable acceleration limit with consideration for occupant
comfort" [AMC 25.1329]. Autopilots in the record bound their captures at 0.1 to 0.2 g
[BendixKing KFC 500; TM 1-1510-225-10]. `protected_pitch_demand` now also clips the
demand to `ap_load_increment`, 0.2 g (ours, the top of that range), either side of
the load the flight path needs, for every controller. Tests: a full demand stays
inside 0.75 to 1.25 g.

**What the operator pays inside the bands (rule 23).** With the bands alone the
reference cost 0 on every scenario, the worst zone over the whole window included
(its largest error 16.6 m against the 19.8 m band), so no policy could beat it.
`plane` now also charges `economy`: fuel plus time by the cost index per air distance,
`(fuel flow + CI / 60) V_assigned / TAS`, normalised so the reference cruise costs 1
a second (`economy_cost`; CI 30 kg/min, ours, family range 6, an operator's, to 40,
the A320 QRH's long-range cruise equivalent, both secondary). The time is spent
whatever the speed over a fixed window, so what a controller changes is the cost of
the distance it covers; air distance, since the wind is not the controller's. Inside
the bands the economy is all that counts.

**Scenarios.** The light zone's sigma is re-derived on the airplane's own lift slope,
7.03 /rad at M 0.78: dn/dw = 0.0572 g per m/s, so 0.684 m/s (was 0.598); the worst zone
is 1.71 m/s, MIL-F-8785C's moderate at FL330. A severe zone joins, MIL-F-8785C's
severe intensity at FL330 (Figure 7, the 1e-5 exceedance curve, as JSBSim tabulates
it), 5.12 m/s, at a share of 1e-4 (ours, between the curve's 1e-5 and the moderate
curve's 1e-3). The heavy corner stops at FL350: at 78 t past it the margin to the
buffet line falls under the 1.3 g the recommended maximum altitude keeps (1.23 g at
FL370 by the model's own line).

| Number | Value | Status |
|---|---|---|
| wing sweep, taper, MAC | 25 deg, 0.24, 4.29 m | cited: [Jenkinson] |
| tail area, aspect ratio, sweep, arm | 31.0 m^2, 5.0, 29 deg, 13.53 m | cited: [Jenkinson] |
| `cl_alpha`, `tail_cl_alpha` | 0.0838, 0.0702 /deg | derived: DATCOM's swept slope |
| downwash | 2 CL / (pi A) | derived: lifting-line far field |
| `elevator_effectiveness` | 0.5 | ours |
| `tail_efficiency` | 0.9 | ours: the textbook 0.85 to 0.95 |
| `moment_arm_wings` | 0.822 m | derived: Torenbeek; centre of gravity 29 % MAC and 13 m nose, ours |
| `stabilizer_setting` | 0.0242 rad | calibrated: zero elevator at FL330, M 0.78, 73.5 t |
| radii of gyration | 8.67 m pitch, 5.04 m roll | cited: JSBSim's A320 (secondary) |
| engine lag | 1.7 s sqrt(theta) / delta | derived: corrected time |
| `ap_load_increment` | 0.2 g | ours, family range 0.1 to 0.2 g [KFC 500, TM 1-1510-225-10] |
| cost index | 30 kg/min | ours, family range 6 to 40 (secondary) |
| `economy_scale` | 1.242 kg/s | derived: the model's reference cruise |
| `turbulence_sigma` | 0.684 m/s | derived, above |
| severe zone | 5.12 m/s, share 1e-4 | cited: MIL-F-8785C Figure 7; the share ours |
| heavy corner | FL340 to 350 | derived: 1.3 g to the buffet line |

**The 2D oracle.** The protection's clip binds on every capture at 0.2 g, where at
2.5 g it rarely did, and the planner met it as a kink: `plane_sine`'s cold solve
hit its iteration cap twice. The oracle model now smooths it as a site of its own
(`protection`, `oracle_model.protected`, the plant's `pitch_demand_bounds`), its
width measured by the protocol (K5) with the others on the new plant
(`scripts/measure_oracle_widths.py`, `data/oracle_widths`).

**Readings.** The commanded airspeed (channel 9) is a setpoint, ATC's assignment at
the aircraft's height: it is now worked out on the true state, not on the altitude
read, which moved it with the altimeter's noise against this section's "commanded
airspeed exact" (`Airplane2D.get_obs`).

**Known deviations (rule 13).** The trimmable stabiliser does not move: the attitude
loop's integral trims on the elevator, about the stabiliser's cruise setting. The
downwash reaches the tail at once: its lag, about a fifth of a transport's pitch
damping, is left out. The buffet line, the drag polar and the thrust lapse are
unchanged; the thrust at FL350, 57 kN at M 0.78, sits at the top of the CFM56-5B
family's installed maximum climb, 5 630 to 6 420 lb a engine as a brochure is indexed
(secondary; its table is paywalled), so the heavy member's residual climb at the
lower end would be near the AIM's 500 fpm, which `plane_energy`'s Extra verdict
(EXTRA.md) does not yet account for.

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
| `plane` | 1800 | 1 | 30 min | 2 in [-1, 1] | 12 | 27 |
| `plane_energy` | 1800 | 1 | 30 min | 2 in [-1, 1] | 12 | 27 |
| `plane_sine` | 480 | 1 | 8 min | 2 in [-1, 1] | 12 | 27 |

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

## The stream (docs/proposals/long-strata.md, S1, S1b and R8)

As a continuing plant (`target_gym.stream`) `plane`, `plane_energy` and
`plane_sine` fly a sequence of sorties. Each scenario's `stream_overrides` set
`stream_mode`, and the stream starts every event through `start_event`, which
picks a sortie, a turbulence zone or the task's own event by `event_mix`, each at
its own rate (`plane/stream_events.py`). Sources and searches: `REALITY.md`, "The
stream".

- **Sorties** (S1b; owner, 2026-11-11). A sortie lasts a memoryless time of mean
  1.90 h, the A320 fleet's average flight [AIRBUS-A320]. It starts established in
  cruise, level and in trim, with an all-up mass drawn uniform over
  `sortie_mass_range` (60 to 78 t, ours: the span of the scored corners) and a
  level drawn over the task's levels up to the ceiling that mass allows, linear
  between FL370 at 73.5 t and FL350 at 78 t (`ceiling_masses`, `ceiling_levels`,
  the two scored corners). In a stream the burn no longer lowers the mass, which
  is held for the sortie; the economy term still charges the fuel flow. The
  model's mass is its zero-fuel mass plus `fuel`, so a sortie's mass is set
  through `fuel`, which can then read above the tanks' figure.
- **Mass corners.** `heavy_high` and `light_low` stop moving the mass in flight:
  in a stream they keep `initial_mass` at 73.5 t, and a sortie that starts in
  them flies at their mass and levels. A pinned scoring run of either still
  starts at its corner.
- **Turbulence zones** start at 0.45 an hour in every scenario but `calm`, with
  the scenario's intensity and a length over `zone_duration_range` (3 to 30 min
  [BEA447]). Each carries its intensity (`zone_sigma`) and its end, so it runs to
  its end through a switch; one starting inside another merges with it. The rate
  is set so that over `plane`'s suite the stream spends 7.9 % of its time in a
  zone [LEE-SPRENGER], which leaves 2.2 h of calm between zones on average, "over
  an hour" [BEA447]. The window's own zone never starts in a stream.
- **Regimes** last 2 h on average (ours).
- **`plane_energy`'s clearances** are events at 12 an hour (ours: the window's
  one per 300 s). A clearance moves the level 1 000 or 2 000 ft, or the speed
  10 kt, with chance 0.58 of a level, from ATC's recorded instructions [PRINZO].
  A level stays inside `initial_altitude_range` and a speed within 10 kt of the
  schedule; a move that would leave them goes the other way. Each carries its
  size in the state (`cleared_speed`, `level_change_at`, `level_change_size`), and
  the band closes at 500 fpm from it [AIM 4-4-10].
- **`plane_sine`'s sinusoid** reads its clock wrapped on whole periods, as many as
  cover a window and one more, so it repeats exactly however long the plant runs.
- **Restart** (R8; owner, 2026-11-11). `REALITY.md` names no post-trip state: no
  source has a trip end a flight. A crash loses the sortie, and the plant
  restarts as its replacement, established in cruise at the sortie's level with
  its mass (`restart_plant`). Both are the outside world, so two policies keep
  meeting the same sorties; the hour of `restart_steps` prices the gap.
- **Outside world and records.** The sortie's mass (`m`, `fuel`), the levels, the
  zone and the clearances are the outside world. The gust (`gust_x`, `gust_z`) is
  a record: the turbulence's draws are the world's, but the Dryden filter runs on
  the aircraft's airspeed, so two policies meet different gusts.
- **Time.** Zones keep their times in seconds of float32, exact to 2^24 s (194
  days of 1 s steps).

In a window nothing changes: `stream_mode` is off, and the scored windows are
bit-identical to `ad6ece5`'s in observation, reward and trips, with params traced
and constant, on bob and on the Mac (PR description).
