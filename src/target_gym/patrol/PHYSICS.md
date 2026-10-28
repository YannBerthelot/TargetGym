# Close patrol — physics model, provenance and validation

Two aircraft in formation. The **lead** flies a scripted patrol route; the
**follower** is the agent and must reach and hold a *slot*, a position fixed
in the lead's body frame, so many metres behind, to one side, and above or
below.

This is a **dynamic** target MDP. The target subset of states is "the follower
is in the slot", but the slot moves with a manoeuvring lead, so the set the
agent must stay inside is itself in motion.

## 1. Model and provenance

**The flight physics is not this module's.** Both aircraft are advanced by
`compute_next_state_3d`, reused verbatim from
[the 3D aircraft](../plane3d/PHYSICS.md) — the same aerodynamics,
Prandtl–Glauert compressibility correction, shock-stall lift limit and
coordinated-turn kinematics, with the same sources and the same validation
targets. Nothing about the aerodynamic model is re-derived here, and any
correction to it applies to this environment automatically.

What this module adds is composition and geometry:

**Slot geometry.** The commanded slot is expressed in the lead's body frame
(back, right, up) and rotated into the world by the lead's heading. The slot
error is the world-frame distance between the follower and that point. This is
the standard way formation position is specified, and it is what makes the
reference non-stationary: a constant slot in the lead's frame sweeps a circle
in the world when the lead turns.

**Heading alignment.** A wingman flies *parallel* to the lead, not merely at
the right coordinates. The reward charges the squared heading difference
beside the squared slot error, each in units of its floor, so occupying the
slot while pointing the wrong way still costs.

**Collision is a trip.** Separation below `min_separation` trips the plant, as
do either aircraft leaving the altitude envelope and the follower ending up
`max_slot_error` from its slot. The step is charged the trip cost and both
aircraft restart as `reset_env` would, while the episode clock keeps running
(with `continuing = False` the episode ends there instead). This is the
qualitative addition over the single-aircraft tasks: the follower is scored
for approaching something it must not hit, so the optimal policy closes in and
stops short.

**The lead's motion.** The lead flies a route of eight legs, cycled. Each leg
lasts `lead_leg_seconds` (60 s) at a turn rate drawn at reset from
`lead_turn_rate_range`, 0 being straight and level, and the tuned 3D heading
autopilot flies it. The schedule is part of the state, so a planner rolling
the environment forward sees the next turn coming, while a policy sees only
the current leg. It is scripted, not adversarial: the difficulty is tracking a
reference that changes its turn every minute, not fighting an opponent. A
single turn rate held for the whole episode was too easy, since feedforward of
a constant, observable rate cancels it.

### The bearing-only variant

`PlanePatrolBearingOnly` replaces the full relative state with what a passive
sensor gives: **range, azimuth and elevation** to the lead, measured in the
follower's body frame, plus the follower's own state and the commanded slot.

Range with azimuth and elevation is a *complete relative-position*
measurement, so the follower always knows where the lead is. The genuinely
unobservable quantity is the lead's **heading** — and the slot is defined in
the lead's frame, so the follower cannot place the slot without it. It is
recovered by differencing the estimated relative position over time and
filtering. This is what makes the variant a partial-observability task rather
than a noise-robustness one: a single missing scalar, but the one the target
definition depends on.

## 2. Validation targets

| Property | Target | Asserted by |
|---|---|---|
| Slot error is zero when the follower is at the slot | exact | `test_slot_error_zero_at_slot` |
| Commanded slot lies behind the lead | negative along-track offset | `test_desired_slot_offset_is_behind_lead` |
| The version-1 reward approaches 1 in the slot and aligned | ≈ 1 | `test_reward_near_one_in_slot` |
| Collision trips the plant | the version-2 reward is at most minus the trip cost | `test_collision_terminates_with_penalty` |
| Losing formation trips the plant | at `max_slot_error` | `test_lost_formation_terminates` |
| The shipped expert holds formation through a turn | across sampled turn rates | `test_expert_holds_formation` |
| The expert never collides | separation > `min_separation` | `test_expert_never_collides` |
| The expert holds its slot on held-out seeds, both variants | RMS slot error after the burn-in at most 8.5 m on seeds 200-209, no trip | `test_the_follower_holds_its_slot` |
| Slots vary across seeds | distinct | `test_slots_randomized_across_seeds` |

The environment also inherits every shared contract in
`tests/test_env_conformance.py`, including that its PID beats the best
constant action.

## 3. Parameter table

Flight parameters are inherited from `PlaneParams3D`; see the
[3D aircraft table](../plane3d/PHYSICS.md). Patrol adds:

| Parameter | Default | Meaning |
|---|---|---|
| `slot_back_range` | (150, 300) m | Along-track slot offset, sampled per episode |
| `slot_right_range` | (−150, 150) m | Cross-track offset |
| `slot_up_range` | (−60, 60) m | Vertical offset |
| `slot_tolerance` | 60 m | Version-1 reward only: σ of its slot term |
| `heading_tolerance` | 0.5236 rad (30°) | Version-1 reward only: σ of its heading-alignment factor |
| `min_separation` | 25 m | Collision distance, a trip |
| `max_slot_error` | 1500 m | Formation lost, a trip |
| `lead_turn_rate_range` | (−0.003, 0.003) rad/step | Range each route leg's turn rate is drawn from; 0 is straight and level |
| `lead_leg_seconds` | 60 s | Length of each leg of the lead's route |
| `follower_spawn_noise` | 40 m | Isotropic spawn offset, so the episode starts solvable but untrimmed |
| `delta_t` | 1.0 s | Step size |

The defaults place the follower in a trailing echelon roughly 200 m back and
120 m to one side. At `delta_t = 1 s`, the turn-rate bound of 0.003 rad/step is
about 0.17 °/s — a gentle orbit, not an aerobatic one.

## 4. Task design

### Task: `patrol`

Fly as a wingman: reach and hold a slot fixed in the lead aircraft's frame,
behind, beside and above or below it, while the lead flies a patrol route
that turns onto a new rate every minute. The slot moves whenever the lead
turns, so the reference is non-stationary, the heading difference to the lead
is charged too, and coming too close to the lead trips the plant.

| obs | meaning |
|---|---|
| 0-9 | the wingman's own state, as in `plane3d_heading` |
| 10-12 | offset from the slot in the lead's frame: back, right, up (m) |
| 13-15 | velocity relative to the lead in the lead's frame: forward, right, up (m/s) |
| 16-18 | the slot in the lead's frame: back, right, up (m) |
| 19 | the lead's heading relative to the wingman's (rad) |
| 20 | the lead's speed (m/s) |
| 21 | throttle, 0 to 1 |
| 22 | elevator deflection (rad) |
| 23 | aileron deflection (rad) |
| 24 | slot error, the distance to the slot (m) |
| 25 | target slot error, always 0 |

### Task: `patrol_bearing_only`

The `patrol` task flown from a passive sensor: the wingman sees range,
azimuth and elevation to the lead instead of the lead's relative state. The
lead's heading, which places the slot, is not measured and has to be inferred
from how the bearing changes. The tracked value is the measured range,
against the distance at which the slot sits from the lead.

| obs | meaning |
|---|---|
| 0-9 | the wingman's own state, as in `plane3d_heading` |
| 10 | range to the lead (m) |
| 11 | azimuth of the lead in the wingman's body frame (rad) |
| 12 | elevation of the lead (rad) |
| 13-15 | the slot in the lead's frame: back, right, up (m) |
| 16 | throttle, 0 to 1 |
| 17 | elevator deflection (rad) |
| 18 | aileron deflection (rad) |
| 19 | range to the lead (m), the tracked value |
| 20 | distance from the lead to the slot (m), its target |

### Scoring

The tracked scalar is the **slot error** for the full-observation variant and
the **measured range** for the bearing-only one, each against its commanded
value. The version-2 reward charges the squared slot error and the squared
heading misalignment with the lead, each in units of its floor, plus the trip
cost on a step that trips (see *Reward (version 2)* below).

## 5. Baselines

A **PID** ships for both variants: a formation hold over the follower's own
autopilot (`experts.py`). It is the practice of formation-flight research, not
of a deployed product: no wingman autopilot that holds a slot is sold or in
service. AFIT's formation-hold research and NASA's F/A-18 Autonomous Formation
Flight trials flew this law: the slot error, in the lead's frame, closes along
track on a speed demand, across it on a course and vertically on a height, each
a PI with the lead's speed, heading and turn rate fed forward. In a turn the
slot swings with the lead, so its velocity includes omega x slot offset. The
autopilot underneath is the 3D tasks' (`plane3d.experts`): TECS on height and
speed, a pitch loop, a damped bank loop limited to 45 deg and stall
prevention, as ArduPilot flies them. Two things this plant needs beyond the
textbook hold:

- The engine lags its lever by 20 s (`power_response_rate` 0.05), so the
  throttle leads the demanded power by its gap to the engine's state, as an
  airliner's autothrottle closes on N1 rather than on the lever.
- The height and speed demands go to TECS as they are, with only their rates
  filtered. Unfiltered, every step-to-step wobble of the lead's track reaches
  the pitch loop.

On `patrol_bearing_only` an estimator feeds the same hold. Range, azimuth and
elevation fix the relative position. The lead's velocity is the follower's own
over the last step plus the change in that position, and its heading is that
track less the crab the follower flies in the same air.

The gains were tuned on training seeds 100-109 and checked once on held-out
seeds 200-209 (`scripts/tune_formation_hold.py`,
`tests/patrol/test_formation_hold.py`). The limit is NASA's F/A-18 figure
scaled by span: 10 ft one-sigma on a 12.3 m span is 8.5 m on the A320's
34.1 m. Over the scored part of the episode the full-observation expert holds
an RMS slot error of 3.4 to 5.2 m (median 4.0) and the bearing-only one 3.8
to 5.7 m (median 4.4), with no trip. The pursuit-guidance PID it replaced
(2026-10) held 44 to 68 m and 40 to 57 m.

**The oracle** in the MPC slot is `PatrolTwinOracle` (`experts.py`), a
feedback law with a short planner on top. The law flies the lead's own heading
autopilot on the follower's state, aimed at the lead's commanded heading and
altitude plus the slot offset, and adds small corrections on the slot error
and a turn feedforward. In the slot with nothing to correct it commands
exactly what the lead commands, so the shared gust moves both aircraft alike
and drops out of the relative position the reward scores. A 30-step
receding-horizon planner then adds a residual to the law inside its rollout of
`step_env` (turbulence zeroed) and descends the reward itself. The lead is
scripted, so the rollout propagates it for free.

It replaced a 20-step `GradientMPC` started from the PID's rollout in the
oracle audit (2026-10). Under the earlier 18.6 m slot floor that planner cost
3.02 per step on the protocol seeds and held the slot to 29-32 m RMS, where
the law alone cost 0.109 (4-6 m RMS) and the law with the planner 0.0316
(0.4-1.1 m RMS). The gust is common-mode, so what the old planner lost was
solver weakness, not disturbance. Its own hold had set the slot floor, so the
floor became the 3 m relative-GPS resolution (version 3 of both tasks). Under
that floor the oracle costs 0.0371 per step on the protocol seeds
(0.0407 / 0.0459 / 0.0248), never trips, and holds a
mean slot error of 0.10 to 0.12 m and a heading error of 1.6e-3 to 1.7e-3 rad
(`scripts/measure_hold.py`, seeds 0-1). What is left is mostly heading: in a
turn a slot behind the lead moves sideways at omega x slot_back, so holding it
exactly needs a heading offset of about omega slot_back / V, 4.5e-3 rad at the
largest turn rate and slot (0.003 rad/s, 300 m, 200 m/s).

On `patrol_bearing_only` the same oracle reads the true state. It is a
full-state bound, not a bearing-only controller: on a task defined by what the
observation withholds, its score measures information and control together.

## 6. Known deviations

**D1: the expert is research practice, not a product.** No deployed
autopilot holds a wingman's slot; the hold reconstructs the laws flight
research published (AFIT, NASA's AFF, West Virginia University's
three-aircraft trials) on this plant, tuned here. Until 2026-10 the expert
was a pursuit-guidance PID that departed on some geometries; the hold
replaced it.

**D2 — the lead is scripted, not reactive.** It flies its pattern regardless of
the follower, including through a collision. From the untrimmed start its
heading autopilot climbs about 600 m at full power in the first 30 s; a
follower that idles its engine meanwhile cannot regain the energy on the same
airframe, which is why the expert's throttle closes on the engine's state. Formation flight against a lead
that reacts (a break turn, a station change) is a different and harder task,
and is not modelled.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `patrol` | 200 | 1 | 3 min | 3 in [-1, 1] | 26 | 15 |
| `patrol_bearing_only` | 200 | 1 | 3 min | 3 in [-1, 1] | 21 | 15 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`: slot position plus heading alignment with the lead, see docs/reward-shaping.md. `PatrolParams` inherits the 3D aircraft's reward parameters (`e_floor_altitude`, `e_tol_altitude`, `e_floor_heading`, `e_floor_path`, `tracking_exponent`, `failure_cost`, `rho_floor_tracking`, `rho_floor`, `floor_is_documented_minimum`); the heading term uses `e_floor_heading`, the others are unused here.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_slot` | 3 m | relative position from differenced GPS fixes (`slot_precision_floor`), an instrument resolution: the oracle holds 0.10 m (lowest per-seed hold in the test turbulence, `scripts/measure_hold.py`; the expert 3.4 to 5.2 m RMS on held-out seeds), so the instrument sets the scale. Until version 3 it was 18.6 m, an earlier planner's hold; the oracle audit (2026-10) found the current oracle holding 0.41 m under it |
| `e_tol_slot` | 0 | **provisional.** The formation's station-keeping radius is a procedural number to be supplied. |
| `e_floor_heading` | 0.0087 rad | inherited 0.5 deg AHRS resolution; the oracle holds alignment below it (1.6e-3 rad, lowest per-seed hold in the test turbulence), so the instrument sets the scale |
| `failure_cost` | 2 x ((1500 / 3)^2 + (pi / 0.0087)^2) = 7.6e5 per step | losing the formation, a collision or a crash: twice the slot-loss bound plus the heading term; overrides the inherited aircraft value |
| `restart_steps` | 3600 (1 h) | restart time priced into a trip, `restart_steps x failure_cost` (a crash, collision or lost formation loses the sortie: an hour of flight, provisional); where a plant engineer would get it: the plant's restart procedure |

Version 3 of both tasks (oracle audit, 2026-10) changed `e_floor_slot` from
18.6 m to 3 m, and `failure_cost` with it (from 2.7e5); the form of the reward
is unchanged. `rho_floor = rho_floor_tracking = (0.1021 / 3)^2 +
(1.631e-3 / 0.0087)^2 = 0.0363`, the oracle's lowest per-seed holds in floor
units.

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
