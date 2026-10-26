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
[the 3D aircraft](../plane3d/PHYSICS.md), and all of them fly in one air mass:
one turbulence gust, drawn from the step clock as in the patrol, is folded
into the wind every aircraft sees, and each still gets its own
altitude-dependent shear.

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

**Trips.** The formation trips when any aircraft leaves the altitude
envelope, any two aircraft come within `min_separation` (25 m), or any
wingman is `max_slot_error` (1500 m) from its slot. The whole formation
restarts (`base.failure_kernel`).

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
| One gust for the whole formation, from the step clock | distinct draws under a constant key | `test_one_gust_for_the_whole_formation_and_any_size` |
| The multi-agent view is the joint task split by seat | same observations, same reward | `tests/formation/test_formation_marl.py` |

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

The tracked values are the lead's altitude, against its target, and each
wingman's slot error, against zero. The lead observes 22 values: the 3D
heading task's 15, whose target is the heading the route commands this step;
the route's current turn rate; and each wingman's slot error vector in its
frame. Each wingman observes the patrol's 26. The wingmen do not observe each
other.

## 5. Baselines

**The PID** is the patrol's, seat for seat: the patrol's lead autopilot in the
lead seat, aimed at the heading the route commands, and the patrol pursuit PID
in each wingman seat. Over the scored part of the test episode it holds the
lead's altitude 141 to 155 m off target and the slots 35 to 56 m off, with no
trip (lowest to highest per-seed hold, `scripts/measure_hold.py`, seeds 0-1).

The lead seat keeps the patrol's autopilot although the 3D heading task's own
PID holds the lead's altitude better (about 38 m late in the test episode,
against 141 m): behind that lead the patrol PID's wingmen collided in 5 of the
10 baseline seeds, at about 5e11 per episode in trip costs, where behind the
autopilot they never trip (about 9e6 per episode, measured on seeds 0-9). The
patrol PID was tuned behind the autopilot.

**The oracle** in the MPC slot is `FormationOracle` (`experts.py`), the
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
to 0.82 to 0.86 m, its heading to 1.1e-3 to 1.2e-3 rad, each slot to 0.24 to
0.32 m and each wingman's alignment to 1.1e-3 to 1.3e-3 rad, with no trip
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

---

<!-- BEGIN GENERATED FACTS -->

<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:
     `make ci-docs` fails if this block does not match the code. Prose
     about *why* these numbers are what they are belongs outside it. -->

### Facts, generated from the code

| environment | steps | step (s) | episode | action | obs | float state |
| --- | --- | --- | --- | --- | --- | --- |
| `formation` | 200 | 1 | 3 min | 9 in [-1, 1] | 74 | 18 |

`float state` counts the scalar and array float fields the state carries,
`time` excluded; the gap between it and `obs` is what the controller cannot
see. Episode lengths are `EnvSpec.test_params`, which is what the recorded
baselines use.

<!-- END GENERATED FACTS -->

## Reward (version 2)

`compute_reward = -(tracking + failure)`: the lead's altitude and heading
against its route, plus each wingman's slot position and heading alignment
with the lead, see docs/reward-shaping.md. `FormationParams` inherits the 3D
aircraft's and the patrol's reward parameters; `e_floor_path` is unused here.

| parameter | value | source |
| --- | --- | --- |
| `e_floor_altitude` | 1 m | the barometric altimeter resolution, as in the 3D tasks: the oracle holds the lead's altitude below it (0.82 m, lowest per-seed hold in the test turbulence, `scripts/measure_hold.py`; PID 141 m), so the instrument sets the scale |
| `e_tol_altitude` | 0 | inherited; no altitude band |
| `e_floor_heading` | 0.0087 rad | the 0.5 deg AHRS resolution, for the lead's heading against its route and each wingman's alignment with the lead; the oracle holds both below it |
| `e_floor_slot` | 3 m | the relative-GPS resolution, the patrol's: the oracle holds each slot below it |
| `e_tol_slot` | 0 | **provisional**, the patrol's: the station-keeping radius is a procedural number to be supplied |
| `e_floor_path` | 3 m | inherited, unused |
| `tracking_exponent` | 2 | quadratic, as every aircraft task |
| `failure_cost` | 2 x the largest team tracking cost = 2.99e8 per step | twice the lead at the altitude envelope and half a turn off heading, plus each wingman at the slot-loss bound and half a turn off: `2 ((12192 / 1)^2 + (pi / 0.0087)^2 + 2 ((1500 / 3)^2 + (pi / 0.0087)^2))`; it still covers four wingmen |
| `restart_steps` | 3600 (1 h) | the patrol's: a crash, collision or lost wingman loses the sortie, an hour of flight (provisional) |
| `rho_floor_tracking` | 0.735 | the oracle's lowest per-seed holds in floor units (`scripts/measure_hold.py`): `(0.8187 / 1)^2 + (1.125e-3 / 0.0087)^2 + (0.2388 / 3)^2 + (1.096e-3 / 0.0087)^2 + (0.2743 / 3)^2 + (1.165e-3 / 0.0087)^2`, of which the lead's altitude is 0.670 |
| `rho_floor` | 0.735 | the same; there is no running cost |

`reward_version = 1` reconstructs the team reward the unregistered formation
module shipped (`compute_reward_v1`): `w_lead` times the lead's log-scaled
altitude and heading tracking plus `w_track` times the wingmen's mean
log-scaled slot-keeping, each times its heading alignment. It reads
`precision_floor`, `heading_precision_floor`, `slot_precision_floor` and
`heading_tolerance`. That module ended the episode on a trip; under the
restart a tripped step scores `-restart_steps` instead, as on `unstable_cstr`
and `compressor_surge`, since a 0 would make a trip that restarts the
formation in its slots nearly free.
