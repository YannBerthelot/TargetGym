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

**What it finds here.** Four fields, once the scan was actually run rather than
eyeballed: `hvac.T_surface`, `glass_furnace.T_stack`, `hvac.Q_command` and
`wind_turbine.torque_cmd`. All four are benign, and the last two in an
instructive way — the *local* variable of that name carries the value into the
dynamics, and the state field is only a record of what was commanded. So the
scan's hit rate is poor and its cost is a few seconds; keep it, but read every
hit before believing it. The one time it mattered it would have caught a
20-tonne error.

## 4. Is any state discarded where it is unpacked?

**What went wrong.** `compute_acceleration` began `x_dot, z_dot, _ = velocities`.
The pitch rate was passed in and thrown away on the first line, which is why the
aircraft had no pitch damping and a departed airframe tumbled indefinitely.

**How to check.** `grep` for `_` in tuple unpacking of state vectors, then ask
whether the physics genuinely does not depend on it.

**What it finds here.** Three sites across the package, and all three are
correct: `x` in the 2D dynamics, `x` and `y` in the 3D dynamics — which the
equations of motion genuinely do not depend on — and a `lax.scan` carry in
`utils.py`. No rate is discarded anywhere. Now verified mechanically rather
than by reading.

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

**Do this with autodiff, not finite differences.** The prescription above --
"measure the largest step against the typical step" -- cannot tell a kink from a
steep curve, and run across all eighteen it ranked the cstr first at a ratio of
8.5e18. That is not a discontinuity, it is Arrhenius: the metric flags any
function whose derivative *grows*, and an exponential does that perfectly
smoothly.

Every plant here is differentiable, so ask the dynamics instead of sampling
them. At a point `x`, take the exact Jacobian of one step just below and just
above it:

    jump(x) = |J(x+eps) - J(x-eps)| / (|J(x+eps)| + |J(x-eps)|)

For *any* smooth function, however steep, the two one-sided derivatives converge
to the same value and `jump` falls to zero with `eps`. Across a genuine kink
they converge to different values and `jump` holds. It is scale-free, so a plant
whose derivative spans twenty orders of magnitude is not penalised for having
one; and `jnp.where` switches are caught precisely *because* autodiff returns
the selected branch's derivative. The thing that makes autodiff "wrong" at a
seam is what makes it a detector for one. Shrinking `eps` by 4x settles each
case: a smooth point decays with it, a seam does not.

Two filters make the output readable. Score only where the increment is real --
where a field is unaffected by the swept variable, both derivatives are float
noise and their ratio is meaningless. And require *both* one-sided derivatives
to be active: where one is exactly zero the ratio is 1 by construction, and that
is a **saturation**, a clip on an action or a rate, not two physical
descriptions failing to join. Every environment clips something, so scoring
those buries what the check is for.

**What it finds here.** Nine candidates across eighteen environments, and most
are benign:

| environment | seam | `jump` | held under refinement |
| --- | --- | --- | --- |
| aircraft (all four) | `x_dot` near 308 m/s | 1.00 | yes |
| `battery` | `soc` near 0.115 | 1.00 | yes |
| `reactor` | `T_fuel` near 718 K | 0.89 | yes |
| `glass_furnace` | `T_crown` near 1527 K | 0.17 | yes |
| `boiler_drum` | `pressure` | 0.20 | **no** (0.45) -- smooth |
| `wind_turbine` | `v_wind` | 0.12 | **no** (0.54) -- smooth |

The furnace's is `m_batch = jnp.maximum(position[3], 0.0)`, the batch blanket
running out -- a mass that cannot go negative, which is a legitimate kink. The
battery's is its state-of-charge-dependent power limit binding. The boiler drum
and wind turbine decay under refinement and are simply steep.

The aircraft's is the one that needed checking, and it is **outside the
reachable state space**: it sits at 308 m/s, and holding every actuator hard
over from cruise reaches 251.5 m/s on the 2D aircraft and 285.8 m/s on the 3D
one before the episode ends. A sustained dive is the case this does not cover.

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

**How to run it without naming each plant's energy.** Hold the actuator at zero
and let the plant run. With nothing being supplied, nothing may grow without
bound; anything that does is producing energy from its own equations, which is
the class of defect that let a departed aircraft fall at 300 m/s against an
implied terminal velocity of 767.

The trap is that growth alone proves nothing. The first run of this flagged all
four aircraft, because position grows -- the aircraft flies forward -- and
flagged the battery, because cumulative capacity fade counts upward. Both are
integrators, not instabilities. What separates them is whether the growth is
linear or accelerating: compare the mean increment over the last fifth of the
run against the first. An integrator holds near 1; an unstable mode runs away.

**What it finds here.** Nothing, which is the answer worth having. Over 3000
unforced steps no environment accelerates: the largest is the glass furnace's
`T_work` at 3.25x and the circle task's `psi` at 2.40x, both far below anything
suggesting an unbounded mode, and the rest sit near 1. Five plants are driven by
an exogenous input -- wind, dispatch, weather, a manoeuvring lead -- and are not
unforced even at zero action, so they are marked rather than scored.

**Plants unstable by design.** `KNOWN_OPEN_LOOP_UNSTABLE` in
`tests/test_env_conformance.py` lists the plants whose unforced mode grows on
purpose, and the check skips them. It admits no other plant, since a plant
that runs away without being designed to is the defect this check exists to
find. The check could not score such a plant anyway, because a trip restarts
it (`base.failure_kernel`) and the unforced run becomes a runaway-and-restart
sawtooth whose ratio depends on where the restarts fall. So each entry records
the measured growth rate and names the test in the plant's own suite that
asserts it. The one entry today is `unstable_cstr`, held on a saddle at every
target with lambda+ 1.39 to 3.16 /min (derived by
`scripts/unstable_cstr_numbers.py --section targets`). The rate is asserted
from the Jacobian by `test_every_target_is_a_saddle` and by stepping the env by
`test_unforced_error_grows_at_the_saddle_rate`.

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

**The plant-agnostic form** is a time constant. Hold the actuator hard over from
reset and measure how far it drives the tracked variable and whether it destroys
the plant. Both tails are suspicious: an actuator that can trip the plant in a
few steps leaves the episode decided before the controller has acted, and one
that cannot move it is decorative.

**What it finds here.** No environment is inert, since every actuator moves its
tracked variable, but the margin varies by three orders of magnitude, and three
plants trip within twenty steps:

| environment | steps to trip at full travel | in seconds |
| --- | --- | --- |
| `boiler_drum` | **3** | 6 s |
| `unstable_cstr` | 5 to 20 (median 9) | 15 to 60 s |
| `wind_turbine` | 11 | 2.75 s |
| aircraft (3D) | 27 | 27 s |
| `hvac` | 56 | 14 h |
| `distillation` | 231 | 231 s |
| `glass_furnace` | 3162 | 53 min |
| `cstr`, `first_order`, `ph_neutralization` | never | -- |

The boiler drum trips on `|level| >= 0.25 m` at `dt = 2 s`, so a saturated
feedwater valve destroys it in six seconds out of a 400-step episode. That is
not obviously wrong -- drum shrink-and-swell really is that fast, and the drum
is meant to be one of the hardest environments here -- but it does mean a
controller that saturates has already lost, which is worth knowing before
reading a poor score as a tuning problem. The three plants that cannot be
tripped at all are the three simplest, as expected.

The unstable CSTR's row is full heating from each of its 135 reset corners
(five targets, nine corners, three feed drifts; derived by
`scripts/unstable_cstr_numbers.py --section reach`). A fast trip under full
heating is intended, since the plant is held a few kelvin below a point of no
return. Full cooling from the same corners never trips it.

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

**What it finds here.** The other three aircraft lateral loops still command
bank, so the defect is present in all of them. Measuring the speeds each task
actually flies puts a number on it: heading and circle range 201-244 m/s for a
1.22x gain swing, and the figure-8 ranges 96-228 m/s for **2.38x** — larger than
the 45% swing that was judged worth fixing on patrol.

**And converting them changed nothing.** Applying patrol's exact fix to all
three — command a turn rate, invert `g*tan(phi)/V`, clip the rate rather than
the bank — moved the three-lap path error by less than 1% and the ten-seed
returns by less than the seed noise (heading -1.7, circle +0.1, figure-8 +0.9,
patrol -1.6, against standard deviations of 14-76). It was reverted, on the same
grounds the aircraft's stall-margin barrier was: a change that measures as
nothing does not earn its complexity, and here it would have been *five* copies
of it (see check 12).

That is worth recording as a result rather than quietly dropping, because the
check is still correct. A 2.38x gain swing is real and would matter to anyone
re-tuning these loops or training against them. It simply was not what was
holding the path — which is the lesson of check 10 arriving a second time, and
this time it was *me* reaching for the nearest plausible structural candidate.

**What was actually wrong with the circle is one line of geometry.** A level
coordinated turn needs `tan(phi) = V^2/(g R)`. The task samples its radius from
`target_radius_range = (8000, 12000)` m while the autopilot holds a 230 m/s
cruise and the bank limit is 30 deg — and the smallest radius that combination
can fly is **9.34 km**, rising to 10.5 km at the top of the speed band. So a
third to a half of the environment's own parameter range is unflyable at cruise:

| seed | radius | speed | bank needed | pinned at limit | settled error |
| --- | --- | --- | --- | --- | --- |
| 0 | 8.42 km | 235 m/s | **33.8 deg** | **100%** of the episode | 1860 m |
| 1 | 11.31 km | 230 m/s | 25.5 deg | 0% | 31 m |
| 2 | 11.46 km | 230 m/s | 25.1 deg | 0% | 52 m |

No guidance law can fix that, and the coin-flip-on-seed signature was check 10
pointing at a saturation boundary exactly as it says it does. The aircraft can
fly an 8 km circle — it just has to slow to 213 m/s first, and nothing told it
to. The circle law now publishes the speed its radius admits and the shared
airspeed channel holds it, which takes the failing seed from **1860 m to 73 m**
and lets the strict xfail be narrowed to the figure-8 alone.

**The figure-8's target is reachable; its expert simply cannot fly there.** The
MPC holds the same curve to **0.5 m** over the same 800-step episode, using up
to 76.7 deg of bank at 160-204 m/s. The geometry is not asking for a path the
aircraft cannot fly, so changing the task would be fixing the wrong thing --
worth stating plainly, because "make the target reachable" is the tempting
response to a controller that misses by kilometres, and here it would have made
the benchmark easier while leaving the actual defect in place.

**The same treatment made the figure-8 worse, which is the more interesting
half.** Its lemniscate has a minimum radius of curvature of `a/3`, so at the
shipped 8.4 km lobe radius the lobes admit only 114 m/s — and on a fixed
throttle the aircraft arrives at them having accelerated to 228 m/s. The defect
is real and quantified. But holding one speed for the whole curve took the
three-lap error from 1779 m to 2824 m, and scheduling it against local curvature
(differentiating the observed tangent heading with respect to distance
travelled) still gave 2098 m. Both were reverted. Speed is *not* the binding
constraint on that curve. Nor is its bank limit: raising the expert's 25 deg cap
through 75 deg bottoms out at 1547 m and then gets worse. Nor is a missing
coordinated-turn feedforward -- the circle law has had one all along and this one
never did, and adding it, with curvature estimated from the rate of change of
the observed tangent heading, reaches 1532 m, which is real but does not change
the outcome, so it went the way of the stall barrier.

Five candidates are now eliminated by measurement rather than argument: the
integrator, the loop-gain scheduling, the speed schedule, the bank limit and the
feedforward. What remains untried is the blend at the centre of the law: beyond
5% of the lobe radius it chases the *bearing to the nearest curve point* and
ignores the tangent entirely. That is pure pursuit, and pure pursuit lags a
curved path by construction -- which is the symptom. That is where the next
attempt should go.

## 10. What does the tuning objective's behaviour tell you?

**What went wrong.** A gain search on the patrol follower was chaotic — a 0.1 %
change in one gain moved the objective by a factor of two — and a "best" point
found in one run scored 2.6x worse when re-evaluated. That was not noise to be
averaged away: it was the search finding which side of a *saturation* boundary
each seed fell on.

**What it finds here.** Its own advice, applied to the path-following xfail,
cleared the integrator immediately: the circle's three-lap error is identical to
four significant figures at `rk4_2`, `rk4_4` and `rk4_8`, so that trajectory is
converged and the fault was genuinely downstream. Cheap, and it stopped a second
integration hunt before it started.

**The lesson.** A well-conditioned search means a genuine gain problem. A
chaotic one means a structural fault, and tuning will not fix it. The
figure-eight's search converged smoothly (185 → 153 → 110 → 98 → 90 m) and its
gains really were the problem; patrol's did not. Structure first, then gains.

**But the search does not tell you *which* structure.** That last sentence used
to end "and its guidance law was", and that was wrong. Patrol's expert sat 139 m
from a 60 m slot, a grid search over every gain scaling got no closer, and the
conclusion drawn — recorded in a strict xfail and left standing for a long time —
was that the guidance law needed rework. It did not. The plant was integrated
with one RK4 substep where two are needed, and correcting that took the settled
slot error to 41 m with no change to the controller at all.

An unimprovable search is evidence that something upstream of the gains is
wrong. The guidance law is the nearest candidate and therefore the tempting one,
but the plant, the observation and the integrator are all upstream too. Before
concluding it is the controller, check that the *simulation* is converged: the
cheapest version is to halve the integration step and see whether the controller
you were about to rewrite suddenly works.

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

## 12. Is this logic implemented more than once?

**What went wrong.** Check 9's fix was applied to `plane3d_heading_pid_step`,
`plane3d_circle_pid_step` and `plane3d_figure8_pid_step`, measured, and found to
change *nothing at all* — the before and after numbers were identical to the
decimal. Not "within noise": identical. That is not a result, it is a symptom,
and the cause was that `EnvSpec.make_pid` returns
`StatefulCascadedPlane3DPID`, a **separate** host-side implementation of the
same three control laws. The edited functions are real and are used — by
`env_jax.py`'s `FunctionalExpertPolicy` and by the patrol lead — just not by
anything the benchmark runs.

The heading law exists in three places, the circle law in three, the figure-8 in
two. They must agree, and nothing enforces it.

**How to check.** Before editing a control law or a piece of physics, `grep` for
a distinctive line of it and count the hits. Afterwards, confirm the measurement
moved: a change that alters *nothing* has usually missed its target, and an
identical number is much stronger evidence of that than a plausible one.

**What it applies to.** Every environment here has a `env.py` / `env_jax.py`
pair, and the experts have a traced form and a host-side form. The split is
deliberate — one is differentiable and jit-friendly, the other is cheap to step
from Python — so the duplication is not itself the defect. Silently diverging
is.

## 13. Does a reduced prediction model still answer the actuator like the plant?

**What went wrong.** Each CasADi MPC in `target_gym.experts.mpc` optimises
against a hand-reduced model of its environment, and the glass furnace's
reduction changed the model's response to fuel. That does not make the
controller noisy, which is what one looks for; it makes it settle in the
*wrong place*, because an MPC drives its
own **predicted** error to zero and predicted is not actual. It re-measures the
true state every step, so nothing accumulates and nothing looks wrong -- it
simply picks the same slightly-wrong fuel flow again, forever, with no
integrator to notice. A PID needs no model and its integrator does not care
about gain error at all, so the two diverge only at steady state.

Measured, the furnace MPC finished 16% behind its own PID and lost 10 seeds out
of 10.

**How to check.** No optimiser required, which is what makes it cheap. Take
states from a real rollout, hand the same state and the same action to the MPC's
own prediction model, advance both by one control interval, and compare the
tracked output. A model that tracks the plant gives a mean signed error near
zero. A reduction that changed the response gives a persistent *one-signed*
error, and that is what integrates into a standing offset.

    do_mpc.simulator.Simulator(mpc._mpc.model)   # integration_tool="idas" if
                                                 # the model has algebraic vars

**What it finds here.** On the glass furnace the plant runs **+0.0275 K per
control interval hotter than the model**, one-signed on 71% of settled steps.
Per step that is nothing. Multiplied by the plant's own 132-step time constant
it is **3.6 K** -- which is precisely the 2-6 K plateau the closed-loop deciles
showed, so the mechanism is quantified rather than argued.

**Why the reduction looked safe when it was written.** The reasoning is in the
class docstring and is worth reading as a specimen. One bullet keeps the
regenerator *because* "a controller blind to it mis-predicts the steady-state
gain badly"; the next averages away the reversal cycle because "predicting its
phase buys nothing". The second is true about *phase* and slides silently from
"I do not need the phase" to "I may replace the oscillation with its mean",
which holds only if everything downstream is linear in those temperatures.
Radiative coupling goes as T^4. Measured, the two chambers sit 86 K apart on
average and 242 K at worst, and their average has a third of the real swing.

**And nothing could have caught it.** The MPC's contract at the time ran a
240-step episode, which ends while *both* controllers are still on their way to
the setpoint -- the deciles show them identical over exactly that span. A
steady-state offset cannot be found by a measurement that never reaches steady
state. It sat at "1.3% behind, passing" until the episode-length audit
lengthened that episode to 1600 steps.

**What it applies to.** Every environment whose MPC plans against a model of
its own, which here is all eight CasADi ones. The seven in
`target_gym.experts.mpc` plan against reduced models, and none of them had this
comparison before. The unstable CSTR's CasADi model restates the env's
equations without reducing them, because CasADi cannot trace the env's `jnp`
code. Two tests in `tests/pc_gym/unstable_cstr/test_unstable_cstr_experts.py`
hold that copy to the plant. `test_mpc_model_is_the_env_velocity` compares its
right-hand side with the env's `compute_velocity` at 1000 random points, and
`test_mpc_predicts_one_step_like_the_plant` runs this check on 120 states of a
PID episode with the feed drift on. One control interval of the model lands
within `e_floor` of `step_env` in C_a, and the mean signed error stays under a
tenth of `e_floor`.

---

## Open items this produced

- The patrol slot reward is still a Gaussian on `slot_tolerance`. It is anchored
  to a real tolerance rather than to the state space, so it is not a check-1
  defect, but it is precision-blind in the sense of check 2. `max_slot_error`
  (the error at which the formation is declared lost) is the natural envelope if
  it is converted.
- ~~**The circle and figure-8 guidance laws do not hold their path.**~~ Half
  resolved. The circle was never a guidance fault at all: a third to a half of
  its own radius range is unflyable at the cruise speed the autopilot holds, and
  trading speed for radius takes the failing seed from 1860 m to 73 m. Its half
  of the strict xfail is now a passing test. **The figure-8 remains open** and is
  still a strict xfail at 1.3-2.7 km, with the integrator, the loop-gain
  scheduling and the speed schedule all now ruled out by measurement.
- ~~**A test episode shorter than the task's own period proves nothing.**~~
  Fixed, and it needed a second criterion to find. These tasks were exercised
  over `max_steps_in_episode=200`, which at `dt = 1 s` is less than one lap of a
  221-264 s circuit, with the aircraft initialised exactly on the path -- so a
  controller that simply flew straight ahead looked correct for the whole
  episode. The circle is the instructive case: at 14.3 open-loop time constants
  it passed every settling-based check comfortably while being scored over
  **0.76 of a single lap**. Benchmark episodes now satisfy
  `N >= max(10 * tau_actuator, 1 * T_period)`, where a settling time exists;
  six were below it and were
  lengthened. See the episode-length section of docs/rl-protocol.md.
- ~~The reward-shaping phase should apply checks 1 and 2 to every environment
  with a band or tolerance parameter.~~ Done: all eighteen now share one
  log-scaled, bounded reward contract. See `docs/reward-shaping.md`.
- ~~Checks 3, 4 and 6 are automatable and could join the conformance suite.~~
  Done, and more than those three. `tests/test_env_conformance.py` now runs
  checks 3, 4, 5, 7 and 8 against every registered environment, so a new
  environment inherits them by adding one line to the registry. Checks 3 and 4
  are repository-wide scans and cost seconds, so they run in the fast job;
  checks 5, 7 and 8 step or differentiate every plant and are marked `slow`,
  adding about 100 s to that job.

  Each carries an allowlist (`KNOWN_WRITE_ONLY_STATE_FIELDS`,
  `KNOWN_DISCARDED_UNPACKS`, `KNOWN_SEAMS`, `KNOWN_OPEN_LOOP_UNSTABLE`), so
  the tests report *new* defects instead of restating the known-benign ones
  every run, and so that admitting a finding is a deliberate act with a reason
  recorded next to it.
- ~~Checks 5, 7 and 8 have still not been run against the other seventeen
  environments.~~ Run. All twelve checks have now been applied to all eighteen.
  Each turned out to have a plant-agnostic form needing no per-plant seam,
  conserved quantity or actuator figure identified by hand -- autodiff for 5, an
  unforced run for 7, a hard-over actuator for 8 -- which is why they were
  cheaper than they looked, and why they should be re-run whenever the dynamics
  change rather than treated as a one-off audit.
- **Each of checks 5, 7 and 8 needed its first metric thrown away.** Largest
  step against typical step ranked Arrhenius above every real seam; growth under
  zero input flagged four aircraft for flying forwards; and the actuator scan
  needed the tracked index the runners already use rather than a guess at which
  state field is the output. In all three the fix was to ask what distinguishes
  the defect from the benign thing that resembles it -- a kink from a steep
  curve, an instability from an integrator -- rather than to raise a threshold.
  A check whose output is a long list is usually measuring the wrong quantity.
- The seams checks 5 and 8 flagged are recorded but **not resolved**: the
  aircraft's is outside the reachable envelope only for level flight, and the
  boiler drum's six-second trip is plausible but unverified against a real drum.
  Both want a source rather than an argument.
