# Wind turbine — physics model, provenance and validation

Reference machine: the **NREL 5 MW reference wind turbine** (Jonkman et al.,
NREL/TP-500-38060) — the standard open reference for wind-turbine control
research.

Contract for `target_gym.energy.wind_turbine`. Method:
`docs/PHYSICS_METHODOLOGY.md`.

Status: ✅ validated · ⚠️ defensible but approximate · ❌ known deviation

---

## 1. Model

Single-mass drivetrain driven by rotor aerodynamics:

```
J dω/dt  = τ_aero − N τ_gen
τ_aero   = ½ ρ A v³ Cp(λ, β) / ω,        λ = ωR / v
P_elec   = η N τ_gen ω
```

`J` is the drivetrain inertia referred to the rotor, `N` the gearbox ratio.
Pitch and generator torque follow first-order actuators; pitch is additionally
**rate-limited**, which is the binding constraint during gusts.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Blade/tower flexibility, drivetrain torsion | Single rigid mass. A real turbine has structural modes that pitch control must avoid exciting. |
| Individual pitch, yaw, wind shear/veer | Collective pitch, uniform inflow. |
| Blade-element aerodynamics | Cp surface from an analytic fit; see D1. |
| Region 2 operation, start-up, shutdown | Above-rated regulation only. |

**Regime of validity.** Above-rated wind (≥ ~13 m/s) with the rotor near rated
speed. Below rated the setpoint is not achievable and the task is ill-posed —
which is why the wind range starts at 13.5 m/s (see §4).

---

## 2. Validation targets

| Quantity | Published | Model | |
|---|---|---|---|
| Rated generator torque | 43 093.55 N·m | **43 093.55** | ✅ |
| Rotor radius / rated speed | 63 m / 12.1 rpm | 63 / 12.1 | ✅ |
| Cp_max | 0.482 at λ = 7.55 | **0.480 at λ = 8.11** | ⚠️ |
| Aero power at rated wind & speed, β = 0 | ≈ 5 MW after η | **4.82 MW** | ✅ |
| Drivetrain inertia (rotor-referred) | — | 4.379 × 10⁷ kg·m² | ✅ |
| Rotor time constant Jω²/P | — | 14.1 s | ✅ |
| Pitch increases monotonically with wind above rated | required | yes | ✅ |

**Rated generator torque matching to two decimals is the load-bearing check.**
It is a joint consequence of rated power, rated speed, gearbox ratio and
generator efficiency, so hitting 43 093.55 N·m pins all four at once. That
rated power *emerges* at rated wind (4.82 MW vs 5.0) rather than being imposed
is the second independent check.

---

## 3. Parameter table

| Symbol | Value | Unit | Source | |
|---|---|---|---|---|
| `R` | 63.0 | m | Rotor radius (126 m diameter) | ✅ |
| `P_rated`, `v_rated` | 5.0e6, 11.4 | W, m/s | Reference ratings | ✅ |
| `omega_rated_rpm` | 12.1 | rpm | Reference | ✅ |
| `J_rotor`, `J_gen` | 3.876e7, 534.116 | kg·m² | Reference | ✅ |
| `N_gear`, `eta_gen` | 97.0, 0.944 | –, – | Reference | ✅ |
| `pitch_rate_max` | 8.0 | deg/s | Reference actuator limit | ✅ |
| `cp_c1..c6` | 0.5176, 116, 0.4, 5, 21, 0.0068 | – | Standard analytic Cp fit | ⚠️ |
| `torque_max` | 47 400 | N·m | ~110 % of rated | ⚠️ |
| `turbulence_std` | 1.2 | m/s | TUNED — gust amplitude | ⚠️ |
| `delta_t` | 0.25 | s | ~56 steps per rotor time constant | ✅ |

---

## 4. Task design

**Curtailed power tracking in above-rated wind** — the problem a modern
turbine actually faces when the grid dispatches it below available power. The
turbine holds a setpoint while turbulence moves the available power around it,
without losing rotor-speed regulation.

**The rotor-effective wind is hidden.** A nacelle anemometer sits in the
rotor's own wake and is unreliable, so the controller must infer the wind from
rotor speed and power. This is a real estimation problem, not an artificial
restriction.

**Wind range starts at 13.5 m/s**, not 12. With 1.2 m/s turbulence a mean of 12
dips below the 11.4 m/s rated wind regularly, and there the setpoint is
physically unachievable: the torque demand exceeds what the wind can supply and
the rotor decelerates to a stall no matter what the controller does. That is
not a control problem, it is an ill-posed one.

---

## 5. Baselines

| controller | return | power error | episodes completed |
|---|---|---|---|
| PID | **392** | ~0.04 MW | 400/400 |
| constant actions | ≈ 0–4 | 1.5–2.8 MW | 23–306/400 |

The constants all trip on over- or under-speed, which is the point: rotor-speed
regulation is not optional, and the reward cannot buy back a trip.

**The torque loop needs rotor-speed protection.** Without it the power
feedforward keeps demanding a setpoint the wind cannot supply, dragging the
rotor to a stall — observed directly during development. Capping instead at the
classical Region 2 law `τ = K ω²` is *worse*: at rated speed that law sits well
below rated torque, so it under-brakes and the rotor runs away to an overspeed
trip. The fix that works is backing the torque demand off linearly between 0.6
and 0.9 of rated speed, which is inactive during normal Region 3 regulation.

---

## 6. Known deviations

**⚠️ D1 — analytic Cp surface.** The fit peaks at 0.480 at λ = 8.11 against the
reference's 0.482 at 7.55: the peak value is right to 0.4 % but its location is
~7 % high, and the pitch sensitivity differs from the real blade. Concretely,
holding rated power at 25 m/s needs ~33° here against ~23° for the real
machine, so pitch travel is overstated at high wind. A blade-element table
would fix this at the cost of a lookup.

**⚠️ D2 — rigid drivetrain.** No shaft torsion or tower motion, so the pitch
controller cannot excite the structural modes that constrain real gain
selection. Achievable bandwidth here is therefore optimistic.

**⚠️ D3 — uniform inflow.** One rotor-effective wind speed, no shear, veer,
tower shadow or turbulence structure across the disc, so there is no 1P/3P
loading and individual pitch control has nothing to act on.
