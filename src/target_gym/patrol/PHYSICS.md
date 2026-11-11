# Close patrol — physics model, provenance and validation

> **Amendment 4 (2026-11-10): the patrol flies F/A-18 Hornets.** Both aircraft are
> the F/A-18 of plane3d/PHYSICS.md ("The F/A-18", `plane3d/hornet.py`) on the
> amended plant, on NASA's AFF mission (NASA/TM-2002-210729): the lead at 250 KCAS
> between 20 000 and 32 000 ft, the slot in the cruise position 34 to 70 m back and
> 17 to 45 m to a random side, `min_separation` one Hornet length (56 ft, AFF's
> nose-to-tail limit), and the station-keeping limit AFF's own 10 ft one sigma,
> replacing the A320's 8.5 m "scaled by span". Where this document still speaks of
> the A320, the 40 m separation, the 290 KCAS lead or the 8.5 m limit, read the
> Hornet's. Both aircraft spawn in level trim; the air keeps 20 s. `wake_surf`
> (plane3d/PHYSICS.md, "Wake surfing") is this module with A320s 3 km apart.
>
> The lead turns at up to 0.0303 rad/s, the parade turn's 30 deg of bank [CNATRA] at
> 25 000 ft. The expert is NASA AFF's direct hold (figure 3 of the TM): on each axis a
> PID on the slot error straight to the demand, along track on the throttle at 0.3
> rad/s, across on the bank at 0.4, vertically on the flight path at 0.3, with the
> Hornet's own extra pitch per g in a turn, 0.101 rad, the lead's turn rate read
> unfiltered, and each integral held while its demand is at its limit. Under it the
> Hornet's attitude loop has its roll D raised to the F/A-18's roll damper
> (plane3d/PHYSICS.md, "The F/A-18"); on the guide's cap it could not follow a
> reversal of the bank demand. On `patrol_range_bearing` the wingman sees the lead
> and knows its range from vision (the owner, 2026-11-10), in place of TCAS's 15 m,
> which was set for an A320 slot 150 to 300 m out and left that follower 130 m off;
> its estimator takes the lead's velocity as the raw difference (`t_rate` 0) and
> filters the turn rate (`k_omega` 0.3), since its lead heading is an estimate.
>
> The expert is a reference, not a bound: no deployed autopilot holds a wingman's
> slot (D1), and it need not meet the 10 ft objective (the owner, 2026-11-10). It
> holds it in steady flight and steady turns (`TestExpert`), and every held-out seed
> without a trip (`tests/patrol/test_formation_hold.py`). On seeds 200-209
> (2026-11-10) the patrol follower holds a median 3.2 m and at worst 5.7 m, the
> range-and-bearing one 6.3 m and 9.6 m, and the formation's wingmen 3.9 m and 6.7 m,
> the misses coming at the lead's turn entries.

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
Prandtl–Glauert compressibility correction, shock-stall lift limit,
coordinated-turn kinematics and fixed inner loop, with the same sources and the
same validation targets. Nothing about the aerodynamic model is re-derived here,
and any correction to it applies to this environment automatically.

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
do either aircraft leaving the 3D tasks' envelope (altitude, 60 deg of bank, the
load limits, VMO and MMO) and the follower ending up `max_slot_error` from its
slot. `min_separation` is 40 m between centres, over the A320's 34.1 to 35.8 m
span: at the 25 m it replaced, two aircraft abeam had overlapping wings and no
trip. The step is charged the trip cost and both
aircraft restart as `reset_env` would, while the episode clock keeps running
(with `continuing = False` the episode ends there instead). This is the
qualitative addition over the single-aircraft tasks: the follower is scored
for approaching something it must not hit, so the optimal policy closes in and
stops short.

**The lead's motion.** The lead flies a route of eight legs, cycled. Each leg
lasts `lead_leg_seconds` (60 s) at a turn rate drawn at reset from
`lead_turn_rate_range`, 0 being straight and level. The 3D heading expert flies
it, on the 3D tasks' 290 kt CAS then M 0.78 schedule, from a spawn on that
schedule (`LEAD_AUTOPILOT`, task constants, so a retune of that expert does not
change this plant). The hardest leg, 0.03 rad/s, is 1.7 deg/s, under a
standard-rate turn's 3 deg/s, and takes 28 to 34 deg of bank at the flown 171 to
219 m/s. The 0.003 rad/s it replaced took 4 deg, which no wingman would notice.
The schedule is part of the state, so a planner rolling the environment forward
sees the next turn coming, while a policy sees only the current leg. It is
scripted, not adversarial: the difficulty is tracking a reference that changes
its turn every minute, not fighting an opponent. A single turn rate held for the
whole episode was too easy, since feedforward of a constant, observable rate
cancels it.

**The air.** The two aircraft are in the same air, not in the same gust, because
they are in different places in it (`advance_air`). Three things follow:

- **Delay.** The field is taken as frozen as it passes (Taylor's hypothesis), so a
  follower `slot_back` behind the lead meets the lead's gust `slot_back / V` later,
  0.7 to 1.8 s on these slots, read between step samples.
- **Decorrelation.** The follower's offset from the lead's path, `r =
  hypot(slot_right, slot_up)`, leaves `rho = exp(-r / L)` of that gust shared, the
  Dryden longitudinal correlation on MIL-F-8785C's `L` of 533.4 m (0.74 at the
  widest slot). The rest is the follower's own OU draw, weighted by
  `sqrt(1 - rho^2)`, so its gust keeps the lead's 2 m/s. Dryden's form is read;
  using it across the path and on all three axes is ours.
- **Rolling gust.** Its scale is the span, far under any slot, so each wing draws
  its own (`plane3d/PHYSICS.md`).

When the gust was one shared value it was common-mode, moved both aircraft alike
and dropped out of the relative position the reward scores, so turbulence cost a
follower nothing. Each aircraft now holds the gust it flies in, and the trips read
it. `key=None` keeps a steady wind.

**The wake.** The lead's lift leaves a vortex pair, and a follower flies in it
(`wake_air`). This is Blake and Multhopp's horseshoe vortex, the model close
formation work starts from, with Dogan's way of handing it to a flight model.

- **The pair.** Two legs `b0 = pi b / 4` apart at circulation `Gamma = L / (rho V
  b0)`, from the lead's lift, density and airspeed as it passed. Burnham's core,
  0.0582 of the span, keeps the centre finite and differentiable. These slots sit
  4.4 to 8.8 spans back, so the legs read as straight lines in the cross-plane.
- **Where it is.** Where the lead was `slot_back / V` ago, the gust's delay, tilted
  as it was banked, so in a turn it stays on the lead's past track. By then it has
  sunk along its lift at `w0 = Gamma / (2 pi b0)`, about 1.1 m/s here, and lost
  `exp(-0.2 t / t0)` of its strength, on `t0 = 2 pi b0^2 / Gamma`. At these ages
  `t / t0` stays under 0.15, so that is at most 3%, and the sink does the work:
  alone, it gives the fall with distance NASA measured.
- **How it reaches the follower.** Upwash and sidewash go in as a gust, and their
  spanwise gradient as a rolling gust, weighted by an elliptic lift across the
  follower's wing. That gradient is the plant's own `p_g`: a wing rolling at `p`
  sees `w = p y`, and `16 / b^2` of the first moment gives `p` back.
- **The sweet-spot drag saving is modelled, with no term of its own.** Upwash `w`
  rotates the lift forward by `w / V`, which is the saving, and the plant's lift
  already lies across the air-relative velocity. So a follower in upwash needs less
  thrust, and in downwash more.

It reproduces NASA's Autonomous Formation Flight data for two F/A-18s, read off the
charts: induced drag 40 to 52% lower near a 13% tip overlap, a rolling moment that
peaks near 0.01 away from the lead and reverses near a 27% overlap, and a saving
that falls by about a quarter from 2.75 to 9.5 spans aft. On this plant at 5000 m
and 190 m/s, 200 m back, the peak upwash is 1.5 m/s near that overlap, the downwash
dead astern 4.1 m/s, and the sweet spot saves 6 kN of a 52 kN drag, 12%. The peak
rolling moment, on a leg, is 0.012 against full aileron's 0.0168, so a slot in the
wake can be flown, as NASA's pilots flew it.

What it leaves out: the tail and fin (no yaw or pitch moment), the follower's
effect on the vortex, the roll-up itself, the gust carrying the pair, and a
wingman's own wake, which only the unscored ranks behind it would meet.
`wake_strength` 0 turns it off.

### The range-and-bearing variant

`PlanePatrolRangeBearing` replaces the full relative state with what the wingman
sees and a TCAS-style sensor gives: **range, azimuth and elevation** to the lead, measured in the
follower's body frame, plus the follower's own state and the commanded slot. It
was `patrol_bearing_only` until 0.8.0. A bearings-only tracker is passive and must
work out range from its own manoeuvres; this one measures range, by interrogating
the lead, so that name claimed a problem the task does not pose.

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
| Separation trips before the wingtips meet | span + 0.5 m trips, 41 m does not | `test_separation_trips_before_the_wingtips_meet` |
| No aircraft trips past its protected envelope | load, bank, MMO past the protection, on each seat, the slot out of reach | `test_no_aircraft_trips_past_its_protected_envelope` |
| Both aircraft spawn on the schedule, inside the envelope | the schedule's speed at the drawn height | `test_every_aircraft_spawns_on_the_schedule_inside_the_envelope` |
| The lead holds its level on the schedule at any mass | 2 m and 1 m/s in still air, 73.5 and 62 t | `test_the_scripted_lead_holds_its_level_on_the_schedule` |
| The hard turn is coordinated | 0.03 rad/s, `tan(bank) = V w / g` | `test_the_hard_turn_is_a_coordinated_30_deg_bank` |
| The follower meets the lead's gust a slot distance later | dead astern, the delayed gust | `test_the_follower_meets_the_leads_gust_a_slot_distance_later` |
| Offset air decorrelates and keeps its variance | correlation `exp(-r / L)`, 2 m/s, 4000 chains | `test_offset_air_decorrelates_and_keeps_its_variance` |
| Turbulence rolls a follower held wings level | a bank, 0 in still air | `test_turbulence_rolls_a_follower_held_wings_level` |
| Each wing has its own rolling gust | banks differ across seats | `test_each_wing_has_its_own_rolling_gust` |
| The induced drag saving peaks where NASA measured it | 40 to 52% at 8 to 20% tip overlap | `test_the_induced_drag_saving_peaks_where_nasa_measured_it` |
| The rolling moment has NASA's shape | peak, reversal, far field | `test_the_rolling_moment_has_nasas_shape` |
| The saving falls with distance as NASA measured | about 0.75 from 2.75 to 9.5 spans | `test_the_saving_falls_with_distance_as_nasa_measured` |
| Upwash saves the drag the rotated lift gives back | `L w / V`, within 2% | `test_upwash_saves_the_drag_the_rotated_lift_gives_back` |
| The aileron can hold the worst of the wake | under 0.0168 | `test_the_aileron_can_hold_the_worst_of_the_wake` |
| The expert flies a slot in the wake | a strict expected failure until PR 9, D3 | `test_the_expert_flies_a_slot_in_the_wake` |
| The shipped expert holds formation through a turn | across sampled turn rates | `test_expert_holds_formation` |
| The expert never collides | separation > `min_separation` | `test_expert_never_collides` |
| The expert flies every held-out seed, both variants | no trip on seeds 200-209; the slot against 10 ft is reported, not required of a reference (owner, 2026-11-10) | `test_the_expert_never_trips` |
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
| `min_separation` | 40 m | Collision distance between centres, a trip |
| `max_slot_error` | 1500 m | Formation lost, a trip |
| `lead_turn_rate_range` | (−0.03, 0.03) rad/step | Range each route leg's turn rate is drawn from; 0 is straight and level |
| `lead_leg_seconds` | 60 s | Length of each leg of the lead's route |
| `wake_strength` | 1 | Scales the lead's wake; 0 turns it off |
| `wake_core_ratio` | 0.0582 | Burnham core radius over the span |
| `wake_decay_rate` | 0.2 | The wake's decay per `t0` |
| `follower_spawn_noise` | 40 m | Isotropic spawn offset, so the episode starts solvable but untrimmed |
| `delta_t` | 1.0 s | Step size |

The defaults place the follower in a trailing echelon roughly 200 m back and
120 m to one side. `initial_x_dot` is not read here: both aircraft spawn on the
lead's schedule.

### What each figure rests on

| figure | value | status |
| --- | --- | --- |
| Turn rate | +-0.03 rad/s | the review's; 4 deg of bank before |
| Standard-rate turn | 3 deg/s | common knowledge, for scale only |
| Collision distance | 40 m | ours, over the A320's 34.1 m span and 35.8 m with sharklets |
| Gust delay | `slot_back / V` | ours: Taylor's frozen turbulence, as the plan's row asks |
| Correlation | `exp(-r / L)` | read: Dryden's longitudinal form ([Antonova et al. 2014](https://jrnl.kai.edu.ua/index.php/visnik/article/download/6790/7602/16811), equation 1). Across the path and on all axes: ours |
| Scale `L` | 533.4 m | MIL-F-8785C, as `plane3d/PHYSICS.md` |
| Horseshoe pair, Burnham's profile, `Gamma`, lift rotated by `w / V` | `b0 = pi b / 4`, `L / (rho V b0)` | read ([Blake and Multhopp, AIAA-98-4343](https://ocw.mit.edu/courses/16-886-air-transportation-systems-architecting-spring-2004/ab208a8f2d9631cd6acb5f2d8e29d713_01_blakemulthopp.pdf), appendix) |
| Core radius | 0.0582 of the span | read, for a 747 ([NASA review](https://ntrs.nasa.gov/api/citations/20140003974/downloads/20140003974.pdf), table). On this airframe: ours |
| Effective wind and gradient | Dogan's method | abstract read (J. Aircraft 42(4), 2005); the full paper was not opened |
| Elliptic weight, `p_g` as `16 / b^2` of the first moment | | ours |
| Sink and time scale | `w0 = Gamma / (2 pi b0)`, `t0 = 2 pi b0^2 / Gamma` | read ([Holzapfel and Steen, AIAA J. 2007](https://elib.dlr.de/47183/1/AIAA-23917-368.pdf)) |
| Decay | `exp(-0.2 t / t0)` | ours, fitted by eye to that paper's figure 3 lidar median |
| Drag, roll and distance targets | as above | read off NASA's charts ([AFF overview](https://ntrs.nasa.gov/api/citations/20080009703/downloads/20080009703.pdf), pages 5, 15, 17) |
| The test's F/A-18 | 16 t, span efficiency 0.7, `C_lp` -0.25 | ours, the usual size for that wing |
| Lead schedule | 290 kt, M 0.78 | as `plane/PHYSICS.md` |
| Lead's gains | `LEAD_AUTOPILOT` | the 3D heading expert's on 2026-11-06, stale until PR 9 |

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

### Task: `wake_surf`

Surf a lead airliner's wake: hold an A320 at a station 3 km behind another, where
the lead's upwash lifts it, and pay for fuel and time as an operator does. The
station sits 0.87 of a span outboard and below by the vortex's sink; inside a
30 m box about it only the economy counts, so the cost falls where the follower
finds the upwash, and the vortex core a span inboard can roll it past 60 deg, an
upset (plane3d/PHYSICS.md, Amendment 4, wake surfing).

| obs | meaning |
|---|---|
| 0-9 | the follower's own state, as in `plane3d_heading` |
| 10-12 | offset from the station in the lead's frame: back, right, up (m) |
| 13-15 | velocity relative to the lead in the lead's frame: forward, right, up (m/s) |
| 16-18 | the station in the lead's frame: back, right, up (m) |
| 19 | the lead's heading relative to the follower's (rad) |
| 20 | the lead's speed (m/s) |
| 21 | throttle, 0 to 1 |
| 22 | elevator deflection (rad) |
| 23 | aileron deflection (rad) |
| 24 | distance to the station (m) |
| 25 | target distance, always 0 |

### Task: `patrol_range_bearing`

The `patrol` task flown by sight and a TCAS-style sensor: the wingman sees range,
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
the **measured range** for the range-and-bearing one, each against its commanded
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
speed, a bank demand limited to 45 deg and stall prevention, as ArduPilot flies
them, over the plant's fixed inner loop (`plane/PHYSICS.md`). A policy, the
scripted lead and the twin law all send it demands, so nothing here touches the
surfaces. The height and speed demands go to TECS as
they are, with only their rates filtered. Unfiltered, every step-to-step wobble
of the lead's track reaches the pitch loop.

On `patrol_range_bearing` an estimator feeds the same hold. Range, azimuth and
elevation fix the relative position. The lead's velocity is the filtered rate of
that position, with the follower's own motion added back, and its heading is
that track less the crab the follower flies in the same air.

The gains were tuned on training seeds 100-109 and checked once on held-out
seeds 200-209 (`scripts/tune_formation_hold.py`,
`tests/patrol/test_formation_hold.py`). The limit is NASA's F/A-18 figure
scaled by span: 10 ft one-sigma on a 12.3 m span is 8.5 m on the A320's
34.1 m. Over the scored part of the episode, on exact readings and with a raw
difference in the estimator, the full-observation expert held an RMS slot error
of 3.4 to 5.2 m (median 4.0) and the range-and-bearing one 3.8 to 5.7 m (median 4.4),
with no trip. The pursuit-guidance PID it replaced
(2026-10) held 44 to 68 m and 40 to 57 m.

**Those figures are the old plant's**: a 4 deg route, one common gust and no
envelope. On this plant, before PR 9 retunes it, the full-observation expert
holds 14.5 to 25.6 m RMS on seeds 200-209 (median 19.2) with one trip, a 2.3 g
pull on seed 206's first step from a spawn 27 m below the slot. The
range-and-bearing expert loses the lead, tripping 12 to 33 times an episode: its
20 s filter lags a 0.03 rad/s turn by far more than the slot. The scripted lead
does not trip, in 256 episodes of 200 steps and 64 of 1500 on these seeds' route
draws, with a peak load of 1.65 g, 36 deg of bank and 17 m of height.

**The 8.5 m limit, re-read for the 40 m separation.** The number stays, and the
trip does not touch it: the nearest slot is 150 m from the lead, 110 m from the
trip, which is 13 limits. The owner kept it on the whole route on 2026-11-06, turns
included, which makes it stricter than NASA's requirement, because this route is
not the flight it came from:

- **It is a requirement for steady flight.** The 10 ft is NASA's design requirement
  for steady-state tracking, 1 standard deviation, and the gain sets beat it,
  typically under +-5 ft. The autopilot "was designed strictly for nonmaneuvering
  flight", and in the lead's heading sweeps all four gain sets fell short of the
  goal ([TM-2002-210729](https://ntrs.nasa.gov/api/citations/20030005820/downloads/20030005820.pdf)).
  The route turns on most legs at up to 30 deg of bank, so this limit is asked of
  flight its source says it did not cover.
- **The span is not settled.** The plant's `wingspan` is 35.8 m, on which the same
  ratio gives 8.9 m. NASA gives the F/A-18's as 37.5 ft, 11.4 m, not 12.3 m, which
  gives 9.1 m on 34.1 m ([TM-2003-210734](https://ntrs.nasa.gov/api/citations/20030075684/downloads/20030075684.pdf)).
- **Its geometry fits.** NASA's trail positions run 2 to 6.6 spans, and these
  slots 4.4 to 8.8.

**The oracle** in the MPC slot is `PatrolTwinOracle` (`oracle.py`), a
feedback law with a short planner on top. **It predates this plant, and PR 9
rebuilds it.** Its law still flies the old lead's heading PID
(`TWIN_AUTOPILOT`), the gust is no longer common-mode, and the figures below
are the old plant's. The law flies the lead's own heading
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

On `patrol_range_bearing` the same oracle reads the true state. It is a
full-state bound, not a range-and-bearing controller: on a task defined by what the
observation withholds, its score measures information and control together.

## 6. Known deviations

**D1: the expert is research practice, not a product.** No deployed
autopilot holds a wingman's slot; the hold reconstructs the laws flight
research published (AFIT, NASA's AFF, West Virginia University's
three-aircraft trials) on this plant, tuned here. Until 2026-10 the expert
was a pursuit-guidance PID that departed on some geometries; the hold
replaced it.

**D2 — the lead is scripted, not reactive.** It flies its pattern regardless of
the follower, including through a collision. Formation flight against a lead
that reacts (a break turn, a station change) is a different and harder task,
and is not modelled.

**D3: the expert cannot fly a slot in the wake yet.** The slot draws are not shaped
around the wake, so about 20% of them sit within 60 m of the lead's path and 30 m of
its height, and 14% within a span of the pair. There the expert holds 54 to 119 m RMS
against 17 to 25 m with the wake off, and banks to the 60 deg trip. Held-out seed
208, a slot 16 m off the path, goes from no trips to 9. At 100 m off the path nothing
changes. The aileron has the authority, so this is the hold's, and PR 9's. The 8.5 m
limit comes from station-keeping held outside the vortex, so it does not speak for
those slots: it is stricter than NASA's requirement there too.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `patrol` | 200 | 1 | 3 min | 3 in [-1, 1] | 26 | 219 |
| `patrol_range_bearing` | 200 | 1 | 3 min | 3 in [-1, 1] | 21 | 219 |
| `wake_surf` | 1800 | 1 | 30 min | 3 in [-1, 1] | 26 | 219 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`: slot position plus heading alignment with the lead, see docs/reward-shaping.md. `PatrolParams` inherits the 3D aircraft's reward parameters (`e_floor_altitude`, `e_tol_altitude`, `e_floor_heading`, `e_tol_heading`, `e_floor_path`, `e_tol_path`, `tracking_exponent`, `failure_cost`, `rho_floor_tracking`, `rho_floor`, `floor_is_documented_minimum`); the heading term uses `e_floor_heading`, the others are unused here.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_slot` | 3 m; 30 m on `wake_surf`, its station box (ours, a span) | relative position from differenced GPS fixes (`slot_precision_floor`), an instrument resolution: the oracle holds 0.10 m (lowest per-seed hold in the test turbulence, `scripts/measure_hold.py`; the expert 3.4 to 5.2 m RMS on held-out seeds), so the instrument sets the scale. Until version 3 it was 18.6 m, an earlier planner's hold; the oracle audit (2026-10) found the current oracle holding 0.41 m under it |
| `e_tol_slot` | 0; 30 m on `wake_surf` | **provisional** on the patrol: the formation's station-keeping radius is a procedural number to be supplied. On `wake_surf` the station box, inside which only the economy counts (plane3d/PHYSICS.md, wake surfing) |
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

## Readings

The observation reports both aircraft through their instruments, with the slot
offsets and the commanded slot range exact
(`docs/proposals/measurement-noise.md`). `sensor_noise_scale = 0` gives the
noise-free task, rounding included. The follower's readings, on both tasks:

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `z` | 0.2 m | 1 ft | Air data computer baro altitude | Provisional: a ground test set's figure, no airborne one found | [DPST-8000](https://www.avionteq.com/Document/DPST-8000-specification-sheet.pdf) |
| `x_dot` | 0.03 m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling: receiver accuracy, not scatter | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `y_dot` | 0.03 m/s | 0.125 kt | IRS/GNSS ground velocity | Ceiling: as `x_dot` | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `z_dot` | 0.03 m/s | 1 ft/min | IRS inertial vertical speed | No source: a placeholder, not a bound | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `theta` | none | 0.010986 deg | IRS pitch attitude | Corrected: rounding only | [HG9900](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `theta_dot` | 4.3e-6 rad/s | 0.015625 deg/s | IRS body pitch rate | Confirmed | [Honeywell Mk VI guide](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `phi` | none | 0.010986 deg | IRS roll attitude | Corrected: rounding only | [HG9900](https://www.honeywellaerospace.com/us/en/products-and-services/products/navigation-and-sensors/inertial-measurement-units/hg9900-inertial-measurement-unit) |
| `phi_dot` | 4.3e-6 rad/s | 0.015625 deg/s | IRS body roll rate | Confirmed | [Honeywell Mk VI guide](https://fitness.manualsonline.com/manuals/mfg/honeywell/mk_vi.html?p=254) |
| `gamma` | 1.3e-4 rad at 230 m/s, as 0.03 / V | from the velocities | IRS flight-path angle | Confirmed | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `psi` | 1.3e-4 rad at 230 m/s, as 0.03 / V | from the velocities | IRS true track angle | Confirmed | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `power` | none | 0.003 | Engine thrust indication via FADEC | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `stick` | none | 1.6e-3 rad | Elevator surface position sensor | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `aileron` | none | 2.6e-3 rad | Aileron surface position sensor | No source: rounding only | [14 CFR 121 App. M](https://www.law.cornell.edu/cfr/text/14/appendix-M_to_part_121) |
| `x`, `y` | 0.27 m | none | GNSS position | Corrected | [NASA AFF](https://ntrs.nasa.gov/api/citations/20030005804/downloads/20030005804.pdf) |

`gamma` and `psi`, the lead's included, are worked out from the velocities read,
as an IRS does. The slot errors, relative velocities, relative heading and lead
speed are computed from these readings, so they inherit the noise. That includes
the slot frame, which turns with the lead's measured track.

The lead's readings on `patrol`:

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| `lead.x`, `lead.y` | 0.27 m | none | Lead GNSS fix over the air-to-air datalink | Corrected | [NASA AFF](https://ntrs.nasa.gov/api/citations/20030005820/downloads/20030005820.pdf) |
| `lead.z` | 0.2 m | 1 ft | Lead baro altitude over the datalink | Provisional: as `z` | [DPST-8000](https://www.avionteq.com/Document/DPST-8000-specification-sheet.pdf) |
| `lead.x_dot`, `lead.y_dot` | 0.03 m/s | 0.125 kt | Lead IRS/GNSS velocity over the datalink | Ceiling: as `x_dot` | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `lead.z_dot` | 0.03 m/s | 1 ft/min | Lead IRS vertical speed over the datalink | No source: a placeholder | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |
| `lead.psi` | 1.3e-4 rad at 230 m/s, as 0.03 / V | from the lead's velocities | Lead true track over the datalink | Confirmed | [NovAtel OEM7](https://docs.novatel.com/OEM7/Content/Technical_Specs_Receiver/SMART7_Performance_Specs.htm) |

On `patrol_range_bearing` the follower has no datalink: it sees the lead, and a
TCAS-style sensor reports its bearing and altitude. **These figures are
provisional, marked pending a real sensor.** None has been sourced for a wingman
40 to 80 m out:

| reading | sigma | step | instrument | status | source |
| --- | --- | --- | --- | --- | --- |
| range | none | none | Seen: the wingman judges the lead's distance by eye (the owner, 2026-11-10; it was TCAS II's, 15 m, set for slots 150 to 300 m out) | Assumed | none |
| azimuth | 0.026 rad | none | TCAS II four-monopole direction-finding antenna | Corrected: the random part, provisional | [Harman 1984](https://archive.ll.mit.edu/mission/aviation/publications/publication-files/ms-papers/Harman_1984_CACA_MS-6526_WW-18698A.pdf) |
| elevation | 0.0088 rad at 250 m, as 1 / range | through `lead.z` | TCAS relative altitude from Mode S reports | No source: provisional | none opened |

Range and azimuth are channel readings, drawn on the true geometry, because
computed from the fixes they would inherit GNSS-grade noise. The tracked value
is that same range reading, not a second instrument.

Elevation has no draw of its own. TCAS measures no angle: it takes the lead's
altitude from a Mode S report coded in 25 ft steps. So on this task `lead.z` is
that report, 0.2 m of baro scatter rounded to 25 ft, and elevation is computed
from it. This is one altitude path: the rounding's own scatter, 25 ft / sqrt(12)
or 2.2 m, over a 250 m range gives the 0.0088 rad, and it falls as 1 / range. It
replaces the datalink table's 1 ft step for `lead.z`, which on this task would
contradict elevation by a factor of ten. The 25 ft coding comes from no opened
source. The rounding also leaves a steady error of up to 3.8 m while the lead
holds its altitude.

`lead.x`, `lead.y` and the lead's velocities are declared on this task too, with
the figures above, but the observation does not show them. They reach only the
measured reward, through the slot error and `lead.psi`.

The figure the tracked altitude rests on is weak. No air data computer publishes
its scatter, so 0.2 m comes from a ground test set, and the benchmark runs at
`sensor_noise_scale` 0.5 and 2 to show how far a result depends on it. The
ground velocity figure is a receiver's accuracy, so true scatter is smaller, and
the vertical speed figure is a placeholder.

The expert reads these values, and so does every tuner. The range-and-bearing
estimator takes the lead's velocity as the filtered rate of the lead's position
(`t_rate`): 20 s while the range carried TCAS's 15 m, whose raw difference was
21 m/s of speed noise; 0, the raw difference, since the range comes from vision.
The difference's step of lag still leaves this expert short of the 10 ft limit: the
expert protocol on noisy plants has to replace this estimator. The oracle plans and acts on the true state, so it stays
a full-state bound. `reward`, trips and the score read the true state.
`info["measured_reward"]` is the same cost on the field readings, with any trip
taken from the true state. A channel reading stays out of it, so on
`patrol_range_bearing` it does not see the 15 m of range noise.

## The stream (docs/proposals/long-strata.md, S1b)

As a continuing plant (`target_gym.stream`) `patrol`, `patrol_range_bearing` and
`wake_surf` fly a sequence of sorties. Sources and searches: plane/REALITY.md,
"The stream".

- **Sorties.** A Hornet sortie lasts 7.8 h on average: the RAAF's more than 2 700
  sorties and 21 000 flying hours on Operation Okra, with tanker support [OKRA]; no
  training sortie length was found. `wake_surf`'s A320s fly the A320 fleet's
  1.90 h [AIRBUS-A320]. A sortie starts the pair in cruise on a fresh route and
  station, each aircraft at a mass drawn over `sortie_mass_range` and held: for the
  Hornet its zero-fuel mass with a quarter to all of the internal 4 926 kg aboard
  (ours), for the A320 60 to 78 t. The air (`gust_history`, `gust_own`) and every
  aircraft's clock carry on.
- **The route** cycles: the lead's leg reads its clock wrapped on whole routes, so
  the legs repeat exactly however long the plant runs.
- **No other event.** The patrol has no turbulence zones; its gust is continuous.
- **Restart.** `restart_plant` re-forms the follower on the lead (unchanged); in a
  stream it keeps the sortie's mass.

In a window nothing changes (PR description: bit-identical, traced and constant).
