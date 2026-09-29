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

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `wind_turbine` | 400 | 0.25 | 100 s | 2 in [-1, 1] | 5 | 8 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per step

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1680 W | the lowest hold error a shipped controller demonstrated under the shipped OU turbulence: the MPC's lowest per-seed mean \|power error\| over the 300 hold steps of the test episode (seed 1 of five; mean 2.1 kW; `scripts/evaluate_baselines.py`, `target_gym.eval`). A per-seed minimum, so no run of the reference sits below it. Over a 5 min hold (`scripts/measure_hold.py`, 1200 hold steps after a 300-step burn-in, 3 seeds) the MPC holds 2.2 kW (best seed 1.9 kW) and the PID 4.6 kW. Upper bound; the 1680 W was demonstrated by the planner before its descent was made monotone (commit cc39157) and is kept as the tighter of the two demonstrated bounds. |
| `e_tol` | 0 | none |
| `tracking_exponent` | 1 | an imbalance is settled linearly in energy |
| `imbalance_price` | 100 $/MWh | **provisional**; the one imbalance price of the priced plants (reactor, battery); a wind farm's would come from its balancing tariff |
| `c_hold` | 0.0013 | pitch activity fraction \|cmd - achieved\| / pitch_max while holding, PID (`scripts/measure_hold.py`; 0.0526 deg of a 40 deg range) |
| `fatigue_weight` | 1 | **provisional.** Avoidable activity is charged per unit at what tracking at the floor costs per step; a maintenance model would give the price. Sweep 0.5 / 1 / 2. |
| `failure_cost` | 2 x the imbalance of the 7 MW the overspeed limit and `torque_max` allow, per step left | overspeed / underspeed trip |
| `restart_steps` | 2400 (10 min) | restart time priced into a trip, `restart_steps x failure_cost` (an overspeed trip's reset and re-synchronisation; provisional); where a plant engineer would get it: the plant's restart procedure |

`rho_floor_tracking` is the NEA reference for tracking -- the lowest per-seed
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
restarts at once as `reset_env` would, on the same clock. `terminated` is never
raised; `info["tripped"]` marks the event for the evaluator. `reward_version = 1` reconstructs the
capped log-scaled reward of the previous version (`precision_floor` and the
old weights are read only by it).
