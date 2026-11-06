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

### Task: `wind_turbine`

Hold the electrical power of the NREL 5 MW reference turbine at a curtailed
setpoint in turbulent, above-rated wind, with blade pitch and generator torque.
The rotor-effective wind is hidden and has to be inferred from how the rotor
behaves, while rotor speed stays regulated. Costs are in dollars.

| obs | meaning |
|---|---|
| 0 | rotor speed (rpm) |
| 1 | blade pitch (deg) |
| 2 | generator torque, percent of maximum |
| 3 | electrical power (MW) |
| 4 | target power (MW) |

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

**The expert: the NREL 5 MW baseline controller.** The controller published
with the reference turbine (Jonkman et al. 2009, NREL/TP-500-38060,
chapter 7), on the observation only (`experts.py`, `nrel_baseline_step`):

- **Speed filter.** A single pole at 0.25 Hz on generator speed. Both loops
  read the filtered speed.
- **Torque.** Jonkman's table on generator speed: none below 670 rpm, a line
  to the region-2 curve at 871 rpm, `0.0255764 N m/rpm^2 w^2`, a 10 % slip
  line ending at 1161.963 rpm, and region 3, constant power `P / w`, whenever
  the previous pitch command is at least 1 deg or the filtered speed at least
  1161.963 rpm. Limited to 47,402.91 N m and 15,000 N m/s.
- **Pitch.** A PI on the filtered speed error against 1173.7 rpm, saturated
  at the pitch limits, moving at most 8 deg/s, started bumpless from the
  measured pitch.
- **Ours, for the curtailed task.** `P` is the dispatched target over the
  generator efficiency (Jonkman's 5.296610 MW is 5 MW / 0.944), and the
  below-rated table is capped at region 3's torque, so a curtailed turbine
  never exceeds its target. The pitch limit is the plant's 40 deg.
- **Gains, fixed.** Jonkman's pole placement (`wn` 0.6 rad/s, `zeta` 0.7)
  on the best-fit `dP/dtheta` along the region-3 curve
  (`scripts/wind_turbine_gains.py`, wind 12 to 25 m/s in 0.5 m/s steps):
  Kp 0.01514 s and Ki 0.00649 on generator rad/s, at -31.7 MW/rad. The
  published 0.01882681 and 0.008068634 assume the real blade. On this
  plant's analytic Cp (D1) the sensitivity is not monotone in pitch: about
  -66 MW/rad near 1 deg, -4 at 9 deg and -58 at 33 deg. Its best-fit line is
  flat (theta_K 512 deg), so Jonkman's schedule `1 / (1 + theta / theta_K)`
  is 1. A schedule read off the exact sensitivity instead multiplies the
  gains where the surface is almost flat, and tripped the turbine 48 times
  over the ten training seeds.

It replaces a PID that held power by the same constant-power torque but on
the unfiltered speed, backed off linearly between 0.6 and 0.9 of rated speed,
under a grid-searched pitch PI (Kp 12 deg/rpm, Ki 0.6). The proposal's change,
Region 2's `k w^2` below rated, does not apply here: the task runs only above rated wind with a curtailed target, and
capping the torque at that law under-brakes at rated speed.

On the held-out seeds 200 to 209 the expert holds 1.10e-4 $ per step, the PID
it replaces 2.81e-5, against a ceiling of 6.94e-6: 1.20 decades of room,
against 0.61. Spec, a strict xfail in
`tests/energy/wind_turbine/test_wind_turbine_nrel_expert.py`: rotor speed
within 10 % of rated on 99 % of hold steps (met on every step of every
seed) and a mean power error under 1 % of the target. Measured 0.25 to
1.10 %, 1.10 % on seed 200, against 0.08 to 0.14 % for the PID: region 3
divides by the filtered speed, so the power lags the rotor's speed swings.

**The oracle plans on the shared solver.** Since the oracle rule (2026-11) the
MPC slot holds the planner of `oracle.py`, run by
`target_gym.oracles.controller.Oracle`: the episode's reward on the plant with
the turbulence at its mean, solved once a step over the 290 steps the horizon
protocol measured (`data/oracle_horizons/wind_turbine.json`). The imbalance
and the pitch activity are epigraph terms, the rotor speed a hard limit inside
the trips, the pitch command stays within the actuator's reach over the step,
and the two kinks left, Cp's floor and the corner of the pitch rate, are
smoothed at the widths in `data/oracle_widths/wind_turbine.json`.

The rotor is an energy store, so the reward alone leaves the plan loose: any
speed path inside the band that delivers the target costs the same. Left free,
the plan knows only the wind's mean, rides the speed down to about 0.82 of
rated in a lull, and pays there with torque dips of hundreds of kilowatts. So
the plan rests the speed at rated, as the reference turbine does in region 3,
with a term that costs at most the tie-break's budget over an episode (G12 in
`docs/proposals/oracle-solver.md`). The reward never sees that term.

That term only settles a tie, so it cannot hold the rotor against a real cost.
Seed 9 asks for 4.96 MW, close to what the torque limit can deliver, and a long
lull takes the speed to 0.91 of rated, where the torque reaches its ceiling and
the dips return: -0.064 against -0.0071 for the oracle this one replaced, and
-0.070 for the expert. The other nine seeds beat that oracle by about 7 %.

The oracle it replaces, a one-step Newton on the torque under the PID's pitch
PI (`WindTurbineNewtonPI`), went with it, along with the version-1 gradient
planner.

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
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2), in dollars per step

`compute_reward = -(tracking + running + failure)`, see docs/reward-shaping.md.

| parameter | value | source |
| --- | --- | --- |
| `e_floor` | 1680 W (the oracle holds 1454 W) | an earlier oracle's lowest hold error under the shipped OU turbulence: the gradient planner's lowest per-seed mean \|power error\| over the 300 hold steps of the test episode (seed 1 of five; `scripts/evaluate_baselines.py`, `target_gym.eval`), demonstrated before its descent was made monotone (commit cc39157). A per-seed minimum, so no run of the reference sat below it; an upper bound. The oracle that replaced the planner in the oracle audit (2026-10) holds 1454 / 1687 / 2106 W per seed (`scripts/measure_hold.py`, 400 hold steps after a 300-step burn-in, 3 seeds; PID 3702-5211 W, the planner 1907-2366 W). The floor is 1.16x its best seed, within the audit's 1.5x rule, so the floor, and the reward, stay. |
| `e_tol` | 0 | none |
| `tracking_exponent` | 1 | an imbalance is settled linearly in energy |
| `imbalance_price` | 100 $/MWh | **provisional**; the one imbalance price of the priced plants (reactor, battery); a wind farm's would come from its balancing tariff |
| `c_hold` | 0.0013 | pitch activity fraction \|cmd - achieved\| / pitch_max while holding, PID (`scripts/measure_hold.py`; 0.0526 deg of a 40 deg range; the oracle holds 0.00124) |
| `fatigue_weight` | 1 | **provisional.** Avoidable activity is charged per unit at what tracking at the floor costs per step; a maintenance model would give the price. Sweep 0.5 / 1 / 2. |
| `failure_cost` | 2 x the imbalance of the 7 MW the overspeed limit and `torque_max` allow, per step left | overspeed / underspeed trip |
| `restart_steps` | 2400 (10 min) | restart time priced into a trip, `restart_steps x failure_cost` (an overspeed trip's reset and re-synchronisation; provisional); where a plant engineer would get it: the plant's restart procedure |

On this plant `rho_floor` and `rho_floor_tracking` are the tracking cost of
the oracle's best per-seed hold, 1454 W: 1.01e-5 $ per step, 0.865 of the cost
at `e_floor`.

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
