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
Pitch and generator torque follow first-order actuators, and both are
**rate-limited**. The pitch rate binds in gusts, and the torque rate binds on a
dispatch move.

Deliberately **not** modelled:

| Omitted | Rationale |
|---|---|
| Blade/tower flexibility, drivetrain torsion | Single rigid mass. A real turbine has structural modes that pitch control must avoid exciting. |
| Individual pitch, yaw, wind shear/veer | Collective pitch, uniform inflow. |
| Blade-element aerodynamics, dynamic inflow | A steady Cp table; see D1. |
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
| Cp_max | 0.482 at λ = 7.55 | **0.466 at λ = 7.5** (the table's own peak) | ⚠️ |
| Aero power at rated wind & speed, β = 0 | ≈ 5 MW after η | **4.93 MW** | ✅ |
| Pitch that holds 5.297 MW at 13, 15, 20, 25 m/s (Jonkman Table 7-1) | 6.60, 10.45, 17.47, 23.47° | **6.50, 10.34, 17.35, 22.84°** | ✅ |
| dP/dθ at 15 m/s, 10.45° (Table 7-1, frozen wake) | −64.44 MW/rad | **−46 MW/rad** (steady surface, D1) | ⚠️ |
| Drivetrain inertia (rotor-referred) | — | 4.379 × 10⁷ kg·m² | ✅ |
| Rotor time constant Jω²/P | — | 14.1 s | ✅ |
| Pitch increases monotonically with wind above rated | required | yes | ✅ |

**Rated generator torque matching to two decimals is the load-bearing check.**
It is a joint consequence of rated power, rated speed, gearbox ratio and
generator efficiency, so hitting 43 093.55 N·m pins all four at once. That
rated power *emerges* at rated wind (4.93 MW vs 5.0) rather than being imposed
is the second independent check. The third is the pitch schedule: nothing in the
table was fitted to Table 7-1, and it lands within 0.7° of it from 12 to 25 m/s
(`test_the_surface_reproduces_the_published_region_3`).

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
| Cp table | 26 × 31 nodes | – | ROSCO's `Cp_Ct_Cq.NREL5MW.txt` (below) | ✅ |
| `torque_max` | 47 400 | N·m | ~110 % of rated | ⚠️ |
| `torque_rate_max` | 15 000 | N·m/s | Jonkman et al. 2009, section 7.2 and Table 7-2 | ✅ |
| `turbulence_std` | 1.2 | m/s | TUNED — gust amplitude | ⚠️ |
| `delta_t` | 0.25 | s | ~56 steps per rotor time constant | ✅ |

**The Cp table.** `cp_table.py` holds ROSCO's published surface, 0 to 30° of pitch
by 1° and tip-speed ratio 2 to 14.5 by 0.5, read bilinearly: commit
[b994e4de67](https://github.com/NatLabRockies/ROSCO/blob/b994e4de67c423ff598a4bb354657b62ff61a50d/Examples/Test_Cases/NREL-5MW/Cp_Ct_Cq.NREL5MW.txt),
ROSCO v2.9.0, copyright NREL, Apache License 2.0. The licence's text is
`LICENSE.ROSCO` beside the table, the columns below 0° are dropped, and the kept
values are unchanged. The GitHub organisation was renamed from NREL to
NatLabRockies, and the old URLs redirect. Past 30° the table holds its last column.
The plant's range stays 40°, against Jonkman's 90°, since Cp is already clipped to 0
there in every reachable state.

**Timing and actuators.** A status per figure, for the owner to accept the ones
marked ours. Nothing here is Provisional in the noise proposal's sense.

| figure | value | status |
| --- | --- | --- |
| Pitch rate | 8 deg/s | read: Jonkman et al. 2009 section 7.3 and Table 7-2, which calls it speculated from GE's long-blade programme |
| Generator torque rate | 15 000 N·m/s | read, twice: Jonkman section 7.2 and Table 7-2 |
| `pitch_tau` | 0.2 s | **ours.** Jonkman's FAST model has no pitch actuator and his ADAMS one sits at 30 Hz (section 7.4). Odgaard's model is second order, and its numbers are in equation images this repository could not read |
| `torque_tau` | 0.1 s | **ours.** Odgaard's converter is first order, with the same limit on reading its number |
| Control step | 0.25 s | **ours.** Odgaard's controller runs at 100 Hz (read), so this step is 25 times slower than either source's |
| Sensor dead time and sampling | none | **ours.** No source gives one for these signals, so readings arrive within the step |

## 4. Task design

### Task: `wind_turbine`

Hold the electrical power of the NREL 5 MW reference turbine at a curtailed
setpoint in turbulent, above-rated wind, with blade pitch and generator torque.
The rotor-effective wind is hidden and has to be inferred from how the rotor
behaves, while rotor speed stays regulated. The dispatch moves every 25 s. Costs
are in dollars.

| obs | meaning |
|---|---|
| 0 | rotor speed (rpm) |
| 1 | blade pitch (deg) |
| 2 | generator torque, percent of maximum |
| 3 | electrical power (MW) |
| 4 | target power (MW), the live dispatch level |

**Curtailed power tracking in above-rated wind** — the problem a modern
turbine actually faces when the grid dispatches it below available power. The
turbine holds a setpoint while turbulence moves the available power around it,
without losing rotor-speed regulation.

**The dispatch moves.** Four iid uniform levels in 3.5 to 5 MW, held 25 s each
(ours), so the target steps three times per episode. The levels ahead are hidden.
It steps rather than ramps because the turbine's own torque rate is the ramp: 15 kN
m/s at rated speed is 1.74 MW/s (derived), so the largest move, 1.5 MW, takes 0.86 s.
A trip's restart draws a fresh schedule from its first level. The scorer's fixed
settle is still 0, from when the target never moved, so a move's transient counts as
hold until PR 8 sets it. The expert settles about 36 steps after a move and the
oracle about 48, on training seeds 100 to 109, inside the rule of twice the settle
per 100-step cycle.

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

**This section predates the Cp table.** Its gains, holds and screens were measured
on the analytic surface, and PR 9 rewrites both controllers on the new plant. On the
held-out seeds, for this branch's plant with both controllers unchanged: the expert
never trips, its rotor stays within 0.97 to 1.03 x rated (1.07 to 1.09 at its peaks
before) and it holds 11.5 to 19.9 kW. The oracle never trips, peaks at 1.05 to 1.11 x
rated (1.12 to 1.16 before) and holds 2.8 to 3.7 kW. The sensitivity now grows with
pitch as Jonkman's does, so the fixed gains' flat-line argument below no longer holds.

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
the turbulence at its mean, solved once a step over the 266 steps the horizon
protocol measured (`data/oracle_horizons/wind_turbine.json`). The imbalance, the
pitch's travel and the speed above its band are epigraph terms, the rotor speed a
hard limit inside the trips, and both commands stay within their actuator's reach
over the step. The kinks left, Cp's floor, the table's nodes on either axis and
the corners of the two rate limits, are smoothed at the widths in
`data/oracle_widths/wind_turbine.json`.

The rotor is an energy store, so the reward alone leaves the plan loose: any
speed path inside the band that delivers the target costs the same. Left free,
the plan knows only the wind's mean, rides the speed down to about 0.82 of
rated in a lull, and pays there with torque dips of hundreds of kilowatts. So
the plan rests the speed at rated, as the reference turbine does in region 3,
with a term that costs at most the tie-break's budget over an episode (G12 in
`docs/proposals/oracle-solver.md`). The reward never sees that term.

That term only settles a tie, so it cannot hold the rotor against a real cost. On
the ten test seeds the oracle costs -0.0093 to -0.0187 against the expert's -0.050
to -0.084, about 5 times better, with no trip and no failed or capped solve. Its
replay stays within 7.5e-4 and 1.4 resolutions: the table's smoothing is not exact.

The oracle it replaces, a one-step Newton on the torque under the PID's pitch
PI (`WindTurbineNewtonPI`), went with it, along with the version-1 gradient
planner.

---

## 6. Known deviations

**⚠️ D1. A steady Cp table.** The surface is ROSCO's. It peaks at 0.466 at λ = 7.5
against Jonkman's 0.482 at 7.55, so rated power arrives slightly above 11.4 m/s, and
its pitch schedule lands within 0.7° of Table 7-1. A steady surface has no dynamic
inflow: Table 7-1 holds the wake frozen, so a real blade answers a pitch move at
about −64 MW/rad at 15 m/s and relaxes toward the surface's −46 as the wake catches
up. The plant has the relaxed figure at once. The analytic fit it replaced gave
−4.2 MW/rad there, which left the mid-band speed loop near open loop.

The figure a review quoted, −62.7 MW/rad at 15 m/s and 9.2°, is not a row of Table
7-1 or a value of this table (−42 there), so it was dropped.

The table is negative at high pitch, where a real rotor brakes, and Cp is clipped at
0 as the fit was, so feathering past that point stops the drive but adds no braking.
Bilinear reading has kinks at the nodes, which a smooth solver would have to handle.

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
| `wind_turbine` | 400 | 0.25 | 100 s | 2 in [-1, 1] | 5 | 17 |

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
| `c_hold` | 0.0023 | pitch travel per step / pitch_max while holding: the oracle's mean over training seeds 100 to 109 on this plant (the expert's is 0.0034). Measured on the branch; PR 8 re-measures it with the other constants |
| `fatigue_weight` | 1 | **provisional.** Avoidable travel is charged per unit at what tracking at the floor costs per step; a maintenance model would give the price. Sweep 0.5 / 1 / 2. |
| `speed_band` | 1.10 | **ours.** The speed above which every step pays, the 10 % the expert's spec allows. The trip is at 1.25 |
| `speed_weight` | 200 | **ours, provisional.** Floor-widths per step per unit of speed ratio, so 1.15 x rated costs ten, as the gradient planner's speed box did. Added to `running` |
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

## Readings

The observation reports speed, pitch, torque and electrical power through their
instruments (`docs/proposals/measurement-noise.md`), and `omega_rpm` and
`torque_pct` inherit their sensors' noise. The target stays exact.
`sensor_noise_scale = 0` gives the noise-free task.

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `omega` | 5.2e-4 rad/s | | Generator-shaft speed sensor | Confirmed | [Odgaard benchmark](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `pitch` | 0.2 deg | | Blade pitch position sensor | Confirmed | [Odgaard benchmark](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `torque` | 90 N m | | Generator torque from the converter | Confirmed | [Odgaard benchmark](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |
| `power` | 1 kW | | Generator power measurement | Confirmed | [Odgaard benchmark](https://vbn.aau.dk/ws/files/56213377/safeprocess_2009_1.pdf) |

**Weak figures.** Confirmed here means the numbers are on the page. All four are
assumed simulation inputs of a published 4.8 MW benchmark, not measurements. The
90 N m was sized for about 30 kN m, so scaled to this turbine it would be about
130 N m. `omega` divides the benchmark's generator-shaft figure by the gear ratio.
Power has an independent cross-check: the IEC class 0.1S bound over three gives
1.7 kW at 5 MW. The ceiling is built on that bound, not on the assumed 1 kW:
`precision_floor` is 1.7 kW, which puts it at 1.18e-5 $ per step. `e_floor`, 1.68 kW,
stays an earlier oracle's hold until PR 9 re-measures it.

Electrical power is its own instrument. Speed times torque through their own
sensors would scatter about 10.6 kW against the 1 kW meter and the 1.68 kW
`e_floor`. The proposal lists it as a channel reading, but the reward tracks it
and a channel reading stays out of the measured copy, so the task keeps `power` as
a state field, always `electrical_power(omega, torque)`, with a field reading.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `info["measured_reward"]` is the same cost on the readings,
with any trip taken from the true state. Its tracking term sees the 1 kW meter.
The fatigue term reads the plant's travel, not a reading, so the pitch sensor's noise
does not reach it.
