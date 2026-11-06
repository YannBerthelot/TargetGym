# Formation: physics model, provenance and validation

A lead and two wingmen in a V, every aircraft flown by the learner. The lead
flies a patrol route, an altitude and a heading that turns onto a new rate
every leg; each wingman holds a slot fixed in the lead's body frame. The team
shares one cost.

This is the patrol (`../patrol/PHYSICS.md`) with the scripted lead replaced by
a learner, so it is a **multi-agent** target MDP: each wingman's reference is
a co-player, not an exogenous signal (see `docs/target-mdp.md`, *Multi-agent
target MDPs*).

## 1. Model and provenance

**The flight physics is not this module's.** Every aircraft is advanced by
`compute_next_state_3d`, reused verbatim from
[the 3D aircraft](../plane3d/PHYSICS.md), and each flies in the part of the air
it is in, as in the patrol (`../patrol/PHYSICS.md`, *The air*). The lead draws the
gust from the step clock. Each wingman meets it a slot depth later, shares what
its offset from the lead's path leaves shared, and draws the rest, with its own
rolling gust. Each still gets its own altitude-dependent shear. The lead's wake
reaches the wingmen as in the patrol (`wake_air`). The V's lateral spacing of at
least 60 m keeps every wingman outboard of the vortex pair, in its upwash. A
wingman's own wake is left out: only the unscored ranks behind it would meet it.

**The slot geometry and each wingman's observation are the patrol's**
(`slot_error_vector`, `get_obs_full`), applied to each wingman against the
lead, so a wingman faces exactly the patrol's follower problem whenever the
lead flies as the patrol's scripted lead does
(`test_with_the_autopilot_in_the_lead_seat_each_wingman_flies_a_patrol`).

What this module adds:

**The lead is a learner.** Its reference is the patrol's route: the target
heading advances by the current leg's turn rate every step, whatever the lead
does, and the turn-rate schedule is in the state. The lead is scored on its
altitude and heading against that reference, as the 3D heading task scores
its aircraft.

**The V.** Wingman `i` flies on the right when `i` is even and on the left
when it is odd, at rank `i // 2 + 1`. The V's depth, lateral spacing and
height are drawn once per episode (`slot_back_range`, `slot_lateral_range`,
`slot_up_range`), the same for every rank, so the formation is symmetric.
With at least 60 m of lateral spacing per rank, the two wingmen of a rank
start at least 120 m apart.

**Trips.** The formation trips when any aircraft leaves the 3D tasks' envelope
(altitude, 60 deg of bank, the load limits, VMO and MMO), any two aircraft come
within `min_separation` (40 m, over the 34.1 to 35.8 m span), or any wingman is
`max_slot_error` (1500 m) from its slot. The whole formation restarts
(`base.failure_kernel`).

**Two views.** `Formation` is the registered, joint form: one action of 3 per
aircraft and the observations concatenated, lead first. `FormationMARL`
(`marl.py`) is the same step with one observation and one action per
aircraft, in the JaxMARL style; every agent receives the team reward and
`info["costs"]` splits it by aircraft.

## 2. Validation targets

| Property | Target | Asserted by |
|---|---|---|
| With the patrol's lead autopilot in the lead seat, each wingman moves as the patrol's follower and observes the patrol's 26 values | equal to float32 rounding | `test_with_the_autopilot_in_the_lead_seat_each_wingman_flies_a_patrol` |
| The lead's reference follows the route whatever the lead does | the commanded heading, the leg's rate | `test_the_reference_follows_the_route_whatever_the_lead_does` |
| The team cost is the lead's cost plus each wingman's patrol cost | exact | `test_the_team_cost_is_the_lead_cost_plus_each_wingman_cost` |
| Wingmen colliding with each other, or with the lead, trip | at `min_separation` | `test_wingmen_colliding_with_each_other_trip`, `test_a_wingman_colliding_with_the_lead_trips` |
| A trip is charged once and the whole formation restarts | `restart_steps x failure_cost`, a fresh formation on the same clock | `test_a_trip_is_charged_once_and_the_whole_formation_restarts` |
| The V is symmetric, drawn per episode, and starts clear of a collision | 64 draws | `test_the_v_is_drawn_per_episode_within_the_ranges`, `test_the_formation_starts_clear_of_a_collision` |
| The gust comes from the step clock and reaches any size of formation | distinct draws under a constant key | `test_the_gust_comes_from_the_step_clock_and_any_size_of_formation_flies_in_it` |
| Each wingman flies in its own air | gusts differ | `test_formation_wingmen_are_in_their_own_air` |
| Each wingman meets the lead's wake on its own side | one upwash in still air | `test_each_wingman_meets_the_wake_on_its_own_side` |
| The expert's lead seat flies the schedule | 2 m/s of it at 3000 m, no trip | `test_the_formation_lead_seat_flies_the_schedule` |
| The multi-agent view is the joint task split by seat | same observations, same reward | `tests/formation/test_formation_marl.py` |
| The expert's lead holds the route on held-out seeds | mean altitude error at most 2 m, heading at most 0.5 deg, seeds 200-209 (all strict expected failures until PR 9, D3) | `test_the_formation_lead_holds_altitude_and_heading` |
| The expert's wingmen hold their slots on held-out seeds | RMS slot error after the burn-in at most 8.5 m, no trip, seeds 200-209 (all strict expected failures until PR 9, D3) | `test_every_formation_wingman_holds_its_slot` |

The environment also passes every shared contract in
`tests/test_env_conformance.py`, including that its PID beats the best
constant action.

## 3. Parameter table

Flight parameters are inherited from `PlaneParams3D`
([3D aircraft](../plane3d/PHYSICS.md)) and the route, slot depth and height,
spawn noise and trip thresholds from `PatrolParams`
([patrol](../patrol/PHYSICS.md)). The formation adds:

| Parameter | Default | Meaning |
|---|---|---|
| `slot_lateral_range` | (60, 150) m | Lateral offset of each rank from the lead, drawn per episode; `slot_right_range` is not read |
| `w_track` | 0.6 | Version 1 only: weight of the wingmen's mean slot-keeping |
| `w_lead` | 0.4 | Version 1 only: weight of the lead's tracking |

`Formation(num_wingmen=k)` takes 1 to 4 wingmen; only two are scored.

## 4. Task design

### Task: `formation`

Fly a lead and two wingmen in a V, every aircraft controlled by the policy.
The lead holds an altitude and a heading that follows a patrol route, and
each wingman holds a slot fixed in the lead's frame. Because the lead is
learned too, each wingman's reference is a co-player rather than a script;
the team shares one cost, and a collision between any two aircraft trips all
three. The action is the three aircraft's controls, the lead's first.

| obs | meaning |
|---|---|
| 0-14 | the lead's observation as in `plane3d_heading`, its target heading being the one the route commands this step |
| 15 | the route's current turn rate (rad per step) |
| 16-18 | wingman 0's offset from its slot in the lead's frame: back, right, up (m) |
| 19-21 | wingman 1's offset from its slot, likewise (m) |
| 22-47 | wingman 0's observation, as in `patrol` |
| 48-73 | wingman 1's observation, as in `patrol` |

The tracked values are the lead's altitude, against its target, and each
wingman's slot error, against zero. The lead observes 22 values: the 3D
heading task's 15, whose target is the heading the route commands this step;
the route's current turn rate; and each wingman's slot error vector in its
frame. Each wingman observes the patrol's 26. The wingmen do not observe each
other.

## 5. Baselines

**The PID** flies the lead seat as a formation lead flies, and each wingman
seat with the patrol's formation hold (`experts.py`, patrol's PHYSICS.md
section 5). As in the patrol, this is the practice of formation-flight
research (AFIT, NASA's F/A-18 Autonomous Formation Flight trials, West
Virginia University's three-aircraft trials), not of a deployed product.

The lead holds the route with the 3D heading task's autopilot: a course hold
on the heading the route commands, with the leg's turn rate fed forward, and
TECS on height and the 3D tasks' speed schedule. With the engine's 1.7 s lag
TECS's throttle flies as smooth and predictable as a formation lead should, so
the set power the old 20 s lag called for is gone.

The lead and the hold were tuned on training seeds 100-109 and checked once
on held-out seeds 200-209 (`scripts/tune_formation_hold.py`,
`tests/patrol/test_formation_hold.py`), against the 3D heading task's limits
for the lead (2 m, 0.5 deg) and the patrol's for the wingmen (8.5 m RMS).
Over the scored part of the episode the lead holds its altitude 0.94 to
1.56 m off on average and its heading 0.05 to 0.14 deg off. Nineteen of the
twenty wingmen held 2.7 to 7.7 m RMS (median 4.2) and no seed tripped, on the old
plant: a 4 deg route, one common gust and no envelope (D3 has this plant's). The
team it replaced (2026-10), the patrol's scripted lead
autopilot and pursuit PID, held the lead's altitude 141 to 155 m off and the
slots 35 to 56 m off.

**The oracle** in the MPC slot is `FormationOracle` (`oracle.py`), the
patrol oracle's design applied to the whole formation. Each seat flies a law:
the patrol's lead autopilot in the lead seat and the patrol's twin-autopilot
law in each wingman seat, which flies the lead's own autopilot on the
wingman's state, aimed at its slot. One receding-horizon planner then adds a
residual to every seat inside its rollout of `step_env` (turbulence zeroed)
and descends the team cost itself. A wingman's residual is added on top of the
lead's, so with its own at zero a wingman in its slot copies the lead's
correction and the correction, like the shared gust, drops out of the
relative position.

The planner is what makes the lead's term an oracle's. The lead autopilot
alone, the patrol's scripted lead, holds its altitude about 140 m off target
after the climb out of the untrimmed start, which never mattered in the
patrol, where only relative position is scored. Here it would be about 2e4 per
step at the 1 m floor. With the planner the oracle holds the lead's altitude
to 0.80 to 0.91 m, its heading to 1.0e-3 to 1.1e-3 rad, each slot to 0.23 to
0.27 m and each wingman's alignment to 1.1e-3 to 1.4e-3 rad, with no trip
(lowest to highest per-seed hold, `scripts/measure_hold.py`, seeds 0-1). Its
residual bound is the full action range, not the patrol's +-0.3: at +-0.3 the
lead is still about 50 m off at the end of the episode (training seeds
100-103).

The oracle can coordinate, since it descends the team cost over every seat at
once, but it does not reason past its 30-step horizon, and its laws do not
coordinate on their own.

## 6. Known deviations

**D1: the deviations of the 3D aircraft and the patrol apply.** The airframe
and the slot geometry are theirs, and so are their known deviations.

**D2: the wingmen do not observe each other.** A wingman far out of its slot
can fly into the other wingman's path without seeing it. Its own slot error,
which it does observe, is what keeps it out of that region, and the trip on
a collision is what charges a policy that lets it happen.

**D3: the expert does not fly this plant yet.** PR 9 retunes it. On held-out
seeds 200-209, 7 of 10 trip, 12 to 80 times an episode, and the wingmen hold 25 to
153 m RMS. A wingman spawned up to about 40 m off its slot asks TECS for a height
capture whose 2 s time constant pulls past 2.5 g, as on the 3D tasks, and after a
trip the expert's memory stays stale across the restart, which turns one trip into
a cascade. The lead's seat flies 9 to 12 m off its altitude on the seeds that do
not trip. All of it is kept as strict expected failures. The join from a poor spawn
was already the expert's weakness on the old plant.

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `formation` | 200 | 1 | 3 min | 9 in [-1, 1] | 74 | 81 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are the scored configuration's (`env.default_params`),
which is what the recorded baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`: the lead's altitude and heading
against its route, plus each wingman's slot position and heading alignment
with the lead, see docs/reward-shaping.md. `FormationParams` inherits the 3D
aircraft's and the patrol's reward parameters; `e_floor_path` is unused here.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_altitude` | 1 m | the barometric altimeter resolution, as in the 3D tasks: the oracle holds the lead's altitude below it (0.80 m, lowest per-seed hold in the test turbulence, `scripts/measure_hold.py`; the expert 0.94 to 1.56 m on held-out seeds), so the instrument sets the scale |
| `e_tol_altitude` | 0 | inherited; no altitude band |
| `e_floor_heading` | 0.0087 rad | the 0.5 deg AHRS resolution, for the lead's heading against its route and each wingman's alignment with the lead; the oracle holds both below it |
| `e_floor_slot` | 3 m | the relative-GPS resolution, the patrol's: the oracle holds each slot below it |
| `e_tol_slot` | 0 | **provisional**, the patrol's: the station-keeping radius is a procedural number to be supplied |
| `e_floor_path` | 3 m | inherited, unused |
| `tracking_exponent` | 2 | quadratic, as every aircraft task |
| `failure_cost` | 2 x the largest team tracking cost = 2.99e8 per step | twice the lead at the altitude envelope and half a turn off heading, plus each wingman at the slot-loss bound and half a turn off: `2 ((12192 / 1)^2 + (pi / 0.0087)^2 + 2 ((1500 / 3)^2 + (pi / 0.0087)^2))`; it still covers four wingmen |
| `restart_steps` | 3600 (1 h) | the patrol's: a crash, collision or lost wingman loses the sortie, an hour of flight (provisional) |
| `rho_floor_tracking` | 0.704 | the oracle's lowest per-seed holds in floor units (`scripts/measure_hold.py`): `(0.8036 / 1)^2 + (1.006e-3 / 0.0087)^2 + (0.2305 / 3)^2 + (1.113e-3 / 0.0087)^2 + (0.2347 / 3)^2 + (1.133e-3 / 0.0087)^2`, of which the lead's altitude is 0.646 |
| `rho_floor` | 0.704 | the same; there is no running cost |

`reward_version = 1` reconstructs the team reward the unregistered formation
module shipped (`compute_reward_v1`): `w_lead` times the lead's log-scaled
altitude and heading tracking plus `w_track` times the wingmen's mean
log-scaled slot-keeping, each times its heading alignment. It reads
`precision_floor`, `heading_precision_floor`, `slot_precision_floor` and
`heading_tolerance`. That module ended the episode on a trip; under the
restart a tripped step scores `-restart_steps` instead, as on `unstable_cstr`
and `compressor_surge`, since a 0 would make a trip that restarts the
formation in its slots nearly free.

## Readings

The observation reports every aircraft through its instruments, with the slot
offsets, the commanded altitude and heading and the route's turn rate exact
(`docs/proposals/measurement-noise.md`). `sensor_noise_scale = 0` gives the
noise-free task, rounding included. The same instruments sit on the lead and on
each wingman:

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

The lead's seat reads these directly, and each wingman gets them over the
air-to-air datalink, so one draw of the lead's readings reaches every seat, and
the slot errors in the lead's seat match the wingmen's. The datalink carries the
figures above: `lead.x` and `lead.y` rest on [NASA's Autonomous Formation Flight
relative GPS](https://ntrs.nasa.gov/api/citations/20030005820/downloads/20030005820.pdf)
(Corrected), and the rest are as the table's.

`gamma` and `psi` have no noise of their own: on every aircraft they are worked
out from the velocities read, as an IRS does. The slot errors, relative
velocities, relative heading and lead speed are computed from these readings, so
they inherit the noise. That includes the slot frame, which turns with the lead's
measured track.

The figure the tracked altitude rests on is weak. No air data computer publishes
its scatter, so 0.2 m comes from a ground test set, and the benchmark runs at
`sensor_noise_scale` 0.5 and 2 to show how far a result depends on it. The
ground velocity figure is a receiver's accuracy, so true scatter is smaller, and
the vertical speed figure is a placeholder.

The expert reads these values, and so does every tuner. The oracle plans and acts
on the true state, so it stays a full-state bound. `reward`, trips and the score
read the true state. `info["measured_reward"]` is the same cost on the readings,
with any trip taken from the true state.
