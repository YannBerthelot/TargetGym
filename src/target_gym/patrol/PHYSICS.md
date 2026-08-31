# Close patrol — physics model, provenance and validation

Two aircraft in formation. The **lead** flies a scripted patrol pattern; the
**follower** is the agent and must reach and hold a *slot* — a position fixed
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
the right coordinates. The reward multiplies the slot-position term by a
Gaussian on the heading difference, so occupying the slot while pointing the
wrong way does not score.

**Collision as an irrecoverable state.** Separation below `min_separation`
terminates. This is the qualitative addition over the single-aircraft tasks:
the follower is rewarded for approaching something it must not hit, so the
optimal policy runs toward a terminal state and stops short.

**The lead's motion.** The lead is driven by the tuned 3D heading autopilot at
a turn rate sampled per episode from `lead_turn_rate_range`. It is scripted,
not adversarial — the difficulty is tracking a manoeuvring reference, not
fighting an opponent.

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
| Reward approaches 1 in the slot and aligned | ≈ 1 | `test_reward_near_one_in_slot` |
| Collision terminates and is penalised | `-max_steps_in_episode` | `test_collision_terminates_with_penalty` |
| Losing formation terminates | at `max_slot_error` | `test_lost_formation_terminates` |
| The shipped expert holds formation through a turn | across sampled turn rates | `test_expert_holds_formation` |
| The expert never collides | separation > `min_separation` | `test_expert_never_collides` |
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
| `slot_tolerance` | 60 m | σ of the Gaussian slot reward |
| `heading_tolerance` | 0.5236 rad (30°) | σ of the heading-alignment factor |
| `min_separation` | 25 m | Collision distance → terminal |
| `max_slot_error` | 1500 m | Formation lost → terminal |
| `lead_turn_rate_range` | (−0.003, 0.003) rad/step | Lead's turn rate; 0 is straight and level |
| `follower_spawn_noise` | 40 m | Isotropic spawn offset, so the episode starts solvable but untrimmed |
| `delta_t` | 1.0 s | Step size |

The defaults place the follower in a trailing echelon roughly 200 m back and
120 m to one side. At `delta_t = 1 s`, the turn-rate bound of 0.003 rad/step is
about 0.17 °/s — a gentle orbit, not an aerobatic one.

## 4. Task design

The tracked scalar is the **slot error** for the full-observation variant and
the **measured range** for the bearing-only one, each against its commanded
value. Reward is a Gaussian in the tracking error multiplied by the heading
alignment, with the suite's usual `-max_steps_in_episode` on an irrecoverable
state.

## 5. Baselines

A **PID** ships for both variants: a stateful wrapper around the functional
pursuit expert for the full-observation task, and for the bearing-only task the
same pursuit law fed by a lead-state estimator. Measured performance of the two
is close — about 229 m settled slot error for the bearing-only expert against
about 260 m with full observation — so the partial observation costs
essentially nothing once the heading is reconstructed.

**No MPC.** The follower's plant is the full 3D aircraft and its reference is a
manoeuvring lead, so an MPC needs the lead's future trajectory as a
time-varying parameter. That is not yet wired, and `EnvSpec.baselines_note`
records it so the gap is documented rather than silent.

## 6. Known deviations

**D1 — the expert completes 4 of 8 seeds.** The shipped pursuit expert holds
formation on the seeds it completes but departs on the others. The departures
are a lateral bank pilot-induced oscillation: the roll command and the
cross-track error feed each other until the bank angle diverges. Two candidate
fixes were tried during development and neither held, so both were removed
rather than left in place as dead configuration. The remaining diagnosis is
that the expert needs energy management — the follower trades speed for
position without regard to the thrust required to recover it — which is a
change to the controller, not to the physics.

This is a limitation of the *baseline*, not of the environment: the plant is
the validated 3D aircraft, and the failing seeds are flyable. It is recorded
here because the baseline is what a learned policy is measured against, and a
baseline that completes half its seeds is a weaker bar than it appears.

**D2 — the lead is scripted, not reactive.** It flies its pattern regardless of
the follower, including through a collision. Formation flight against a lead
that reacts (a break turn, a station change) is a different and harder task,
and is not modelled.
