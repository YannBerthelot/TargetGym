# Model review checklist

A day spent correcting the 2D aircraft turned up seven defects in one
environment, none of them found by inspection and every one of them outside the
regime the model was designed for. This page turns those into checks, and
records what each one finds when run against the other seventeen environments.

The checks are ordered by what they cost to run, not by how clever they are.

## 1. Is the reward normalised by a tolerance, or by the state space?

**What went wrong.** The aircraft's tracking reward divided the altitude error
by the whole 12 km envelope. Over any realistic error that is effectively
linear, worth 8e-4 per metre whether the aircraft is 10 m or 400 m out, so
closing the last 9 m gained 0.007 where halving a 1600 m error gained 0.26. Fine
tracking was invisible next to coarse approach.

**How to check.** Look for the tracked error divided by a difference of two
*limits* rather than by a tolerance. `grep` for `_max - ..._min` inside a
`compute_reward`.

**What it finds here.** It found three more instances after the 2D aircraft: the
three 3D tasks in `plane3d/env.py` — whose docstring claimed it "mirrors Plane
(2D)" while doing the opposite — and the lead term of the patrol formation. All
are now on the shared `log_scaled_reward`. The four-tank had the same defect and
records it as its own D1, so this shape has now appeared in four separate
places, which is the argument for the check rather than for another one-off fix.

## 2. Does the reward still pay for precision once you are close?

**What went wrong.** Two candidate replacements looked fine and were not. A
band-scaled kernel concentrates its resolution *at* the band and collapses
inside it, so a policy holding 1 m scores almost the same as one holding 10 m —
which makes the comparison the benchmark exists for invisible.

**How to check.** Tabulate the reward gained per *halving* of the error across
several decades. Constant means "closer is always better"; a collapse means
"close enough".

**What it finds here.** See `docs/reward-shaping.md`. Only the log-scaled form
is flat across decades. Every environment with a `band` or `tolerance`
parameter is worth putting through this table.

## 3. Is any state field written but never read?

**What went wrong.** `state.m` was set to `initial_mass + fuel` = 92 588 kg,
above the aircraft's maximum takeoff weight, while the dynamics integrated
`initial_mass` directly. Nothing read the field, so a 20-tonne disagreement sat
there until fuel burn made mass load-bearing.

**How to check.** For each `*State` dataclass, count attribute reads of each
field across the package. Automatable in about fifteen lines.

**What it finds here.** Two fields are written and never read — `hvac.T_surface`
and `glass_furnace.T_stack`. Both are documented as algebraic or diagnostic, so
neither is a defect. The check is still worth keeping: it is cheap, and the one
time it mattered it would have caught a 20-tonne error.

## 4. Is any state discarded where it is unpacked?

**What went wrong.** `compute_acceleration` began `x_dot, z_dot, _ = velocities`.
The pitch rate was passed in and thrown away on the first line, which is why the
aircraft had no pitch damping and a departed airframe tumbled indefinitely.

**How to check.** `grep` for `_` in tuple unpacking of state vectors, then ask
whether the physics genuinely does not depend on it.

**What it finds here.** Only position components (`x`, `y`), which the dynamics
correctly do not depend on. No remaining rate is discarded.

## 5. Do the regimes join?

**What went wrong.** Past the stall the model collapsed lift, and because drag
was defined as `cd0 + k·CL²`, it collapsed drag with it. A separated wing had
*less* drag than in cruise, so a departed aircraft fell at 300 m/s against an
implied terminal velocity of 767 m/s and tumbled without damping.

**How to check.** Sweep the regime boundary finely and measure the largest step
in dissipated power against the typical step. A rapid transition is fine; a jump
is not. The aircraft's blend gives 4.7x, a naive piecewise switch 32x, and the
defect infinity.

**What it applies to.** Any plant assembled from more than one description:
laminar and turbulent, charging and discharging, calcining and inert, boiling
and single-phase. Most of these environments have such a seam.

## 6. Is the model still physical where an optimiser can drive it?

**What went wrong.** Every drag test probed attached flow, where `CD` is
*defined* from `CL` and is self-consistent whatever the values. The two tests
that did reach past the stall were satisfied *by* the defect: a lift collapse to
zero is a maximal collapse, and a sweep asserting `isfinite` and `cd > 0` is
content with a wing producing less drag than in cruise.

**How to check.** State contracts over the *reachable* state space, not the
design point: drive the plant to its action limits and require the state stay
physical or the episode end. `test_attitude_rates_stay_bounded_under_extreme_actions`
in the conformance suite is the general form.

## 7. Can the energy budget be bounded from outside?

**What went wrong.** Nothing — but only because it was never checked.

**How to check.** Two bounds, neither referencing the model's own forces.
Energy may only enter through the actuator, so it cannot rise faster than the
actuator can supply it. And it cannot fall faster than the largest dissipation
the geometry admits. The weak form, `dE/dt = T·V − D·V`, closes by construction
and tests the integrator rather than the physics.

**What it applies to.** Every environment has an energy or an equivalent
conserved quantity — charge for the battery, enthalpy for the thermal plants,
neutrons for the reactor.

## 8. Is actuator authority validated, or merely plausible?

**What went wrong.** The aileron moment applied the *wing's* lift-curve slope to
the control deflection, implying a section lift change of 2.20 at full throw —
larger than the entire wing's CL_max of 1.5. The aircraft rolled at 84 deg/s
against a transport's 25-30.

**How to check.** Compute what full actuator travel commands and compare it with
what the plant can physically produce. Then validate the resulting rate against
a published figure.

**What it applies to.** Valve authority, heater duty, pump head, rod worth —
any actuator whose gain was written down rather than derived.

## 9. Does a control loop's gain depend on an operating variable?

**What went wrong.** The patrol follower's heading loop commanded *bank*, and a
banked aircraft turns at `g·tan φ / V`, so the loop gain went as `1/V` — a 45 %
swing across the speeds it flies. That is why the controller looked like a coin
flip on the seed. Commanding a turn rate and inverting the relation removed it.

**How to check.** Ask what the inner loop actually delivers per unit of the
commanded quantity, and whether that ratio moves with speed, level, temperature
or load.

**What it applies to.** Any cascade. Gain-scheduled controllers are already
acknowledging this; the ones that are not scheduled are the ones to look at.

## 10. What does the tuning objective's behaviour tell you?

**What went wrong.** A gain search on the patrol follower was chaotic — a 0.1 %
change in one gain moved the objective by a factor of two — and a "best" point
found in one run scored 2.6x worse when re-evaluated. That was not noise to be
averaged away: it was the search finding which side of a *saturation* boundary
each seed fell on.

**The lesson.** A well-conditioned search means a genuine gain problem. A
chaotic one means a structural fault, and tuning will not fix it. The
figure-eight's search converged smoothly (185 → 153 → 110 → 98 → 90 m) and its
gains really were the problem; patrol's did not, and its guidance law was.
Structure first, then gains.

## 11. Can the error metric resolve what the reward is asking for?

**What went wrong.** Fixing check 1 on the figure-8 exposed a second defect
*underneath* it. Cross-track error there is an `argmin` over 400 samples of a
44 km curve with no sub-sample refinement, so the reported distance is quantised
by the sample spacing: an aircraft flying the commanded curve **exactly** was
told it was up to 66 m off it. That is larger than the expert's own settled
error, so the reward had been scoring its own discretisation rather than the
controller, and no reward shape could have fixed it. Projecting onto the two
adjacent chords brought the floor under a millimetre.

**How to check.** Feed the metric a state you know is perfect and check it
returns zero, then feed it known offsets and check it returns them. Any metric
built on `argmin`, a lookup table, a fixed grid or a finite-difference step has a
resolution, and it has to be finer than the precision the reward is trying to
buy.

**What it applies to.** Every reward whose error is *searched for* rather than
computed in closed form. The circle task is safe because its distance is
analytic; the figure-8 was not.

---

## Open items this produced

- The patrol slot reward is still a Gaussian on `slot_tolerance`. It is anchored
  to a real tolerance rather than to the state space, so it is not a check-1
  defect, but it is precision-blind in the sense of check 2. `max_slot_error`
  (the error at which the formation is declared lost) is the natural envelope if
  it is converted.
- **The circle and figure-8 guidance laws do not hold their path.** Found by
  measuring the tracking error the reward had never been able to see: over three
  laps the circle expert wanders 640-1670 m from an 8.4 km circle and the
  figure-8 expert sits 6-12 km from a curve 8.4 km across. Both altitude loops
  are fine (0.2-1.5 m), which is what the tuning runs measured -- the
  cross-track error was never measured. Recorded as a strict xfail in
  `tests/plane3d/test_plane3d_env.py`. By check 10 this is a structural fault,
  not a gains fault.
- **A test episode shorter than the task's own period proves nothing.** These
  tasks are exercised over `max_steps_in_episode=200`, which at `dt = 1 s` is
  200 s against a 264 s lap, and the aircraft is initialised exactly on the
  path -- so a controller that simply flies straight ahead looks correct for the
  whole episode. Every periodic or path-following task needs an episode of
  several periods before any expert-quality claim about it means anything.
- The reward-shaping phase in the roadmap should apply checks 1 and 2 to every
  environment with a band or tolerance parameter.
- Checks 3, 4 and 6 are automatable and could join the conformance suite.
