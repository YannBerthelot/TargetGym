# Changelog

Notable changes to TargetGym. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/).

Tags before this file were named `vX.Y.Z`, except `0.5.0`; future tags use the
`vX.Y.Z` form.

## [Unreleased]

The work between `0.5.0` and 1.0. Grouped by what it changes for a user rather
than by commit.

### Added

- **Eleven environments**: building HVAC, pH neutralisation, Skogestad's
  distillation Column A, the NREL 5 MW reference wind turbine, a grid battery,
  a boiler drum, and a cement kiln; then four aircraft tasks that move the
  setpoint instead of holding it, since a tuned PID finishes an altitude hold
  with 0.0 m of settled error and can no longer tell two controllers apart:
  `plane_steps` (a staircase), `plane_sine` (a sinusoid, whose amplitude ratio
  and phase lag are the closed loop's frequency response), `plane_energy`
  (altitude and airspeed together, which removes the spare actuator), and
  `plane3d_racetrack` (a holding pattern, which is heading hold and a sustained
  coordinated turn in one task). Twenty-one in total.
- **Environment versioning.** `EnvSpec.version` and `spec.versioned_name` give
  every environment a public identity such as `plane-v1`, stamped in
  `data/env_versions.json` by `scripts/stamp_env_versions.py`.
  `tests/test_env_versions.py` fails when an environment's fingerprint moves
  without its version being bumped, so a published number keeps meaning what it
  meant. Everything ships as `v1`.
- **A harness for learned-policy results**: `data/rl_results.json`, written
  through `target_gym.rl_results.record_result` and guarded by an environment
  fingerprint deliberately narrower than the one the shipped baselines use, so
  re-tuning a controller cannot throw away a training run. The experimental
  design is fixed in advance in `docs/rl-protocol.md`. No results are published
  yet, and nothing in the suite presumes RL beats a PID.
- **`docs/target-mdp.md`**, stating the definition the whole suite is built on:
  the target set, the tracking shape, the admissibility criterion on the target
  and the feasibility condition on the plant.
- **A Colab quickstart** (`notebooks/quickstart.ipynb`), linked from the README
  and the documentation index, and executed by the test suite so it cannot rot.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- **A central registry** (`target_gym.registry`) describing every environment,
  its parameters and its baselines, and a **shared conformance suite** that runs
  the same contracts against all of them -- PRNG hygiene, determinism, the
  gymnax six-value step API, and a PID that beats the best constant action.
- **Physics contracts.** Every environment carries a `PHYSICS.md` with a sourced
  parameter table, published validation targets asserted by tests, and numbered
  known deviations. The method is written up in `docs/PHYSICS_METHODOLOGY.md`.
- **Documentation** under `docs/`: getting started, a public API contract, a
  baselines guide, and an environment reference generated from the registry.
  Every runnable example is executed by the test suite.
- `CONTRIBUTING.md`, `LICENSE`, `CITATION.cff`, and a pre-commit configuration.
- `tracked_names`, `obs_value_index` and `obs_target_index` on every
  environment, so generic tooling can find the tracked variable and its
  setpoint without naming the environment.
- Actuator lag parameters for the aircraft (`power_response_rate`,
  `stick_response_rate`, `aileron_response_rate`), previously literals.
- `CHANGELOG.md`, issue templates and a pull-request template.

### Changed

- **Every reward is now a floor-normalised cost, and every environment is a
  new version (`-v2`; the reactor `-v3`).** `reward = -(tracking + running +
  failure)`: tracking as `(max(|e| - e_tol, 0) / e_floor) ** p` with
  `e_floor` the achievable hold floor measured or certified under the plant's
  own reference and disturbance (`scripts/measure_hold.py`,
  `scripts/floor_reactor_hold.py`; `data/hold_measurements.json`), running
  cost charged only above the hold-phase consumption `c_hold` -- at weight 1
  on the dimensionless plants, at the owner's prices on the reactor (imbalance
  at 3x spot), the building (gas at EUR 0.10/kWh), the battery (imbalance
  $100/MWh, fade $300/kWh) and the wind turbine -- and a trip that never ends
  the window: the plant is frozen at `failure_cost` per step for a documented
  `restart_steps` and restarts, or stays down (`base.failure_kernel`), so no
  plant raises `terminated` any more. Convex, additive, in
  defensible units; the per-step scale is no longer capped at 1 and differs
  by orders of magnitude between plants, so cross-plant comparison is the
  protocol's job (`target_gym.eval`), not the return's. The positioning and
  the per-plant table are in docs/reward-shaping.md; each PHYSICS.md carries
  its floor, tolerance, hold-phase consumption, exponent and prices with the
  script or source behind them, and marks the ones that are provisional
  stand-ins for numbers a plant owner would supply (tolerances, wear and
  comfort prices, the reactor's rod-wear weight). The version-1 reward stays
  constructible with `reward_version=1` on any params, its baselines are kept
  in `data/baseline_returns_v1.json`, and a contract test
  (`tests/test_reward_contract.py`) holds every plant to additivity,
  convexity, documentation and -- on the slow job -- to the shipped MPC never
  beating a measured floor. Per plant, old and new PID / MPC costs:

  | plant | v1 PID / MPC cost (1 − share) | v2 PID / MPC cost per step | units | MPC ranking |
  | --- | --- | --- | --- | --- |
  | `plane` | 0.110 / 0.075 (9/10) | 6333 / 6068 (5/10) | floor-widths | unchanged |
  | `plane_energy` | 0.382 / 0.244 (10/10) | 1.575e+04 / 2677 (10/10) | floor-widths | unchanged |
  | `plane_sine` | 0.386 / 0.046 (10/10) | 4881 / 3147 (7/10) | floor-widths | unchanged |
  | `plane3d_heading` | 0.840 / 0.149 (10/10) | 4.028e+04 / 8644 (10/10) | floor-widths | unchanged |
  | `plane3d_circle` | 0.565 / 0.080 (10/10) | 1.538e+04 / 2947 (10/10) | floor-widths | unchanged |
  | `plane3d_racetrack` | 0.533 / 0.055 (10/10) | 9.094e+05 / 1265 (10/10) | floor-widths | unchanged |
  | `plane3d_figure8` | 0.795 / 0.040 (10/10) | 1.153e+06 / 82.36 (10/10) | floor-widths | unchanged |
  | `patrol` | 0.444 / 0.159 (10/10) | 293.8 / 51.11 (10/10) | floor-widths | unchanged |
  | `cstr` | 0.106 / 0.054 (10/10) | 6319 / 5803 (10/10) | floor-widths | unchanged |
  | `first_order` | 0.070 / 0.046 (10/10) | 1018 / 1002 (10/10) | floor-widths | unchanged |
  | `four_tank` | 0.223 / 0.106 (10/10) | 1167 / 344.5 (10/10) | floor-widths | unchanged |
  | `ph_neutralization` | 0.246 / 0.131 (9/10) | 157.8 / 47.15 (10/10) | floor-widths | unchanged |
  | `distillation` | 0.389 / 0.227 (10/10) | 1019 / 127 (10/10) | floor-widths | unchanged |
  | `glass_furnace` | 0.098 / 0.054 (10/10) | 32.35 / 11.46 (10/10) | floor-widths | unchanged |
  | `reactor` | 0.667 / 0.365 (10/10) | 58.48 / 3.294 (10/10) | $/step | unchanged |
  | `hvac` | 0.475 / 0.443 (10/10) | 0.06478 / 0.04603 (10/10) | EUR/step | unchanged |
  | `cement_kiln` | 0.112 / 0.075 (10/10) | 8.202 / 2.213 (10/10) | floor-widths | unchanged |
  | `boiler_drum` | 0.381 / 0.242 (10/10) | 308.6 / 54.78 (10/10) | floor-widths | unchanged |
  | `wind_turbine` | 0.171 / 0.129 (8/10) | 2.622e-05 / 9.44e-05 (4/10) | $/step | **flipped** |
  | `battery` | 0.272 / 0.262 (8/10) | 0.003457 / 0.001569 (10/10) | $/step | unchanged |

- **The MPCs are ceilings under the new reward.** Four things in the
  planners had been written against the version-1 reward and stopped being
  upper bounds under version 2 (the wind turbine's MPC lost to its PID on 6 of
  10 seeds; the 2D aircraft's parked at the edge of the altitude tolerance
  55 m/s below cruise). The gradient and sampling planners now descend the
  plant's own version-2 cost in floor units, with their barriers kept and the
  linear (p = 1) tracking terms squared for a gradient that vanishes at the
  optimum; the HVAC CasADi planner minimises the priced dead-zone comfort and
  the gas; the 2D aircraft plans without its 60-step open-loop tail, which
  under an unbounded cost dominated the objective; the wind and 2D aircraft
  planners start from, and at every step are compared against, the shipped
  PID's rollout plan, so their plan is never worse than the PID's under the
  planner's model; and the turbine planner carries move suppression on the
  pitch command priced like the reward's fatigue term, plus a mild soft box
  on rotor speed (what its supervisory logic does), because re-planning
  creates activity no open-loop plan can see. Every MPC now beats its PID on
  episode return and on hold cost (docs/reward-shaping.md,
  docs/baselines.md). `runners.baseline_policy` also built the MPC on the raw
  params rather than `plan_params`, so hand-run MPCs on the wind turbine and
  the battery planned against one fixed noise realisation; fixed.
- **`target_gym.eval`, the reach-and-hold protocol.** Gain after a per-plant
  burn-in over every step (the long-run cost), and over the settled steps only
  (the hold), each split into tracking and running cost, with each controller's
  transient measured on its own cycles; reach cost per target change; reach
  fraction; trip rate; the normalised expert advantage
  `(PID - x) / (PID - rho_floor)`; time-in-band as a KPI only.
  `scripts/evaluate_baselines.py` scores the shipped controllers with it and
  writes `data/protocol_results.json`; `docs/baselines.md` carries the table.

- **Migrated to the gymnax 1.0 six-value step API.** `step_env` now reports
  natural termination alone; the time limit is gymnax's, via `step`.
- **Every renderer rebuilt** on a shared control-room toolkit
  (`target_gym.render_kit`), and the README gallery regenerated and extended to
  every environment.
- **Seven per-environment figure/video runners consolidated** into one
  registry-driven module, which covers all twenty-one environments rather than
  eight.
- The PyPI development status classifier moves from `3 - Alpha` to
  `4 - Beta`. Not a promise of an API freeze, which stays deferred; a statement
  that the surface is settled enough to build against.
- Python support is 3.11 through 3.14, tested on all four in CI.
- The test suite runs in parallel; full-suite wall time went from about
  fifteen minutes to under two.

### Fixed

- **The reactor's recorded baselines were understated about fivefold, and it
  is now `reactor-v2`.** The plant integrates ten 1 s physics sub-steps per
  environment step, and both `state.time` and `max_steps_in_episode` counted
  the sub-steps while every rollout in the suite counted environment steps.
  `runners.rollout` therefore ran 8640 env steps against a limit that fired at
  864, and `step_env` spent the remaining 7776 returning a frozen plant scored
  at a tenth of the reward: the published means (PID 0.081, MPC 0.125 per
  step) were exactly (864 × 0.33 + 7776 × 0.033)/8640. Both counters now count
  environment steps, as on every other plant and as gymnax's `is_truncated`
  assumes; the physics clock lives in `state.physics_time`. Within an episode
  nothing changed -- the trajectories are bit-identical -- but a number
  published against `reactor-v1` was taken over a different task, so the
  version is bumped and the baseline re-recorded on the same 2.4 h episode:
  PID 0.333, MPC 0.635 per step. `runners.rollout` also loops on the
  environment's clock rather than a step count, so this class of bug cannot
  score a plant past its own time limit again.
- **The generated facts tables labelled minutes as seconds for the PC-gym
  plants, and would have labelled the reactor's control step as 1 s.** The CSTR
  and distillation column integrate in their models' native minutes; their
  params now declare `time_unit_seconds = 60`, and `registry.control_step_seconds`
  folds that and the reactor's `control_period` into the seconds one env step
  advances. The distillation episode reads 200 min rather than "3 min", the
  CSTR's 25 min rather than "25 s". The two fingerprints moved with the new
  field; the baselines were re-recorded and reproduced to the last digit.
- **An installed package could not find its own tuned gains.** Both
  `experts/pid.py` and `provenance.py` resolved the data directory relative to
  the repository root, which is correct from a source checkout and nonsense from
  site-packages: an installed target-gym looked in
  `/usr/local/lib/python3.13/data/`, found nothing, and silently re-ran
  gradient tuning for minutes on every user's first PID, producing controllers
  that need not match the published baselines. The wheel did not ship the files
  either. `data/` now lives inside the package, is included in the wheel, and a
  clean install builds a PID in 1.8 s with no tuning. `scripts/tune_pid.py` was
  also writing to a path the loader no longer read, so tuning would have
  appeared to succeed and changed nothing.
- **Installing on a Colab TPU runtime breaks it, and the notebook now refuses.**
  A TPU VM ships jax and jaxlib 0.7.2 matched to its `libtpu`; gymnax 1.0.0 caps
  `jax<0.7`, so installing downgrades both to 0.6.2 and every later JAX call
  aborts the kernel with `Unexpected PJRT_Plugin_Attributes_Args size: expected
  32, got 24`. pip writes to the VM disk, so a plain `import jax` keeps crashing
  until the runtime is recreated. This clears when gymnax releases with its jax
  cap lifted.
- **Environment pages showed broken images on the published site.** The
  generator emitted `../videos/...`, and mkdocs serves these pages at
  `/environments/<name>/`, so the browser asked for
  `/environments/videos/...`. The homepage worked because it sits at the site
  root, which is why this went unnoticed.
- **The docs deploy died rendering the clips.** All twenty-one ran in one
  process, and since each environment compiles its own `step_env` and XLA keeps
  the executables alive, the cost climbed from 9 s to 92 s per environment until
  the runner killed the job. One process per environment keeps it flat.


- **Patrol had no MPC, for a reason that was wrong.** The obstacle on record was
  that its reference is a manoeuvring lead, so a planner would need the lead's
  future trajectory as a time-varying parameter. That is true of a CasADi model
  and irrelevant to a gradient planner: the lead is scripted and deterministic,
  so differentiating `step_env` propagates it for free. `patrol` now ships a
  `GradientMPC` leading its PID by 51% at 0.84 of ceiling. `patrol_bearing_only`
  still has none, and now says why: it withholds the slot error a planner reads,
  which is the point of the variant.
- **The patrol PID was missing a term, not mistuned.** Its settled error was
  exactly linear in the lead's turn rate and exactly symmetric in its sign, 25.9 m
  per 0.001 rad/step, which is proportional control against a rotating reference.
  A grid search over the gains had never closed it because no gain could.
  Feeding the lead's turn rate forward takes the hardest case from 77.8 m to
  2.4 m against a 60 m tolerance and a 3 m reward precision floor.
- **The patrol lead never manoeuvred.** It drew one turn rate at reset and held
  it for the whole episode, which a single feedforward term cancels outright, so
  the task rewarded no anticipation and its MPC had nothing to plan against. The
  lead now flies a routed circuit of eight legs, and settled error goes from
  2.4 m to 8-13 m with turn onsets measurably worse than mid-leg.
- **Default parameters now match the configuration the baselines are recorded
  at**, for the eleven environments that do not share a params class with a
  sibling, and their redundant `test_params` entries are gone. The episode
  lengths that became defaults are the reasoned ones; the old defaults were not,
  all nine aircraft having said exactly 10 000.
- **`requires-python` allowed 3.14, which jaxlib has no wheel for.** A clean
  `uv sync` resolved CPython 3.14.6 and failed on jaxlib; the test matrix listed
  3.14 as well, so that job could never have passed. Both stop at 3.13.
- **The episode-length rule stated a criterion nobody could check.** It asked for
  `N >= max(10 * tau_actuator, 3 * T_period)` while a later section of the same
  document recorded relaxing the period clause to one lap, and `tau_actuator`
  does not exist for two thirds of the suite: nine environments integrate, the
  aircraft oscillate under a held elevator, and the glass furnace does not settle
  inside eight thousand steps. The rule now says where it applies and what sets
  the episode elsewhere.


- **The MPC could see the future, once every ten seeds.** `GradientMPC` and
  `SamplingMPC` roll the real environment forward to score a plan, under a
  hardcoded `jax.random.PRNGKey(0)`, while `rollout` drives the plant with
  `PRNGKey(seed)`. On seed 0 those coincide, so the planner's simulated
  disturbance *was* the plant's actual disturbance and the MPC had perfect
  foresight. On the battery, whose tracked target is the noise, that was worth
  350.4 against an honest 151.8: a median tracking error of 22 W where the
  truth is 60 630 W. It inflated seed 0 of every environment with one of those
  planners, and with it every published mean. Planners now take a copy of the
  parameters with the fields named in `EnvSpec.noise_fields` zeroed, so they
  predict the mean disturbance. That is certainty equivalence, and
  `docs/baselines.md` had already flagged its absence as a caveat.
- **`plan_params` and `noise_fields` are inside the fingerprint.**
  `runners/runners.py` and `registry.py` are in neither `_SHARED_SOURCES` nor
  an environment's own sources, so a change to either altered every MPC number
  and invalidated no record. `plan_params` now lives in `experts/mpc.py`, which
  is fingerprinted, and `baseline_fingerprint` hashes `spec.noise_fields`.
- **The battery task was measuring the dice, not the controller.** Its dispatch
  target was an Ornstein-Uhlenbeck process whose one-step innovation had a
  standard deviation of 63.6 kW against a 150 kW tracking band, which puts the
  best attainable tracking reward at 0.429. The shipped PID scored 0.447 and
  the MPC 0.430: both were pinned on an irreducible noise floor and the
  environment could not tell a good controller from a mediocre one. The signal
  is now a schedule of twelve 300 s dispatch blocks drawn in ±0.8 MW with 2 kW
  of regulation jitter, which is what a grid battery is actually handed. The
  PID goes to 262.0 and the MPC to 265.8, from 160.8 and 174.2.


- **The MPC solver could hang, and nothing noticed.** IPOPT was left at its
  default of 3000 iterations and no time limit, so a single badly conditioned
  step could run for half an hour while its neighbours took a tenth of a
  second: nine glass-furnace seeds finished in about three and a half minutes
  each and the tenth was still going after seventy. `CasadiMPC` now caps
  iterations at 150 and CPU time as a backstop, the way a controller with a
  sample period has to. The iteration cap is deterministic, so a baseline
  recorded on one machine still reproduces on another.
- **A failed solve was indistinguishable from a converged one.** do-mpc neither
  raises nor warns when IPOPT gives up: it stores the failed iterate, returns it
  as the action, and warm-starts the next step from it. An MPC baseline could
  therefore quietly stop being the upper bound it is presented as. Every record
  now carries `solver_calls`, `solver_failures`, `solver_capped` and
  `solver_mean_iters`, and a solve that failed for any reason other than the cap
  makes the controller hold its previous action and restore its previous warm
  start instead of planning from the wreckage.
- **No CasADi MPC declared variable scaling.** IPOPT auto-scales the objective
  and constraints but not the decision variables. The reactor was handing it a
  vector spanning `rho_ext` around 0.0016 up to a precursor concentration around
  377, a factor of 605 000, with hard bounds on the smallest entry of it. Each
  subclass now declares a `SCALING` table of typical magnitudes.
- **The four-tank MPC was blind to half a termination condition.** The plant
  ends the episode when any level reaches `h_min` *or* `h_max`; the controller
  bounded only `h_min`, and bounded it hard. Both bounds are now present and
  soft, so the controller can see an overflow coming and a level that touches a
  bound cannot make the NLP infeasible at `x0`.
- **Recording lost everything when interrupted.** `scripts/record_baselines.py`
  wrote its results once, after the loop, so a run that was killed threw away
  every finished environment. It now checkpoints after each one, and runs
  environments concurrently rather than one after another, so a slow seed holds
  a single worker instead of stopping every other environment from starting.

- **Four-tank**: the target range sat entirely above what the plant can reach,
  so every episode was unwinnable. The range, the loop pairing (the RGA puts
  λ11 at −0.067, so the loops must be crossed) and the tuner objective were all
  corrected.
- **Aircraft lift curve**: `cl_alpha` was 54 % below what the wing's own aspect
  ratio implies, and the stall clamp was applied before the Prandtl--Glauert
  factor, so peak lift *rose* with Mach instead of falling past `M_crit`.
- **Reactor renderer** produced no frames for short episodes and never reset its
  history between episodes: it advances `state.time` by a control period, so the
  `time == 1` episode-start signal never fired.
- **Bearing-only patrol** gained a baseline, via a lead-state estimator feeding
  the same pursuit law the full-observation variant uses.
- The glass furnace setpoint band was narrowed and given a working tuner.
- **Four-tank gradients.** Outflow goes as the square root of the level and a
  tank can sit empty; written as `sqrt(max(h, 0))` the forward value is right
  but the reverse-mode derivative is NaN at zero, which made gradient-based PID
  tuning return NaN gains from a loss that evaluated perfectly well. Forward
  results are unchanged.
- **Gradient MPC could park an actuator at a limit and never move it again.**
  These plants saturate, an engine cannot make less than zero thrust, and
  saturation is written with `clip` or `maximum`, whose derivative at the kink
  is exactly zero. `GradientMPC._optimize` projected its iterates onto the
  closed action interval, so any overshooting step put an action exactly on a
  bound, where its derivative was then zero and gradient descent could never
  move it again. Measured on `plane_steps`, at the plan where the aircraft gave
  up: the true one-sided slope in thrust is +3.0 and autodiff returns 0.0, with
  thrust pinned at -1.000 for 800 steps while the elevator went on being
  optimised normally. Nothing looks wrong from outside, because a planner that
  has stopped searching still emits finite, in-bounds actions. Iterates are now
  held 1e-3 inside the bounds, which is what interior-point solvers do and for
  this reason. Twelve of the twenty environments with an MPC share it. The
  four plants among them were re-recorded and moved by under half a point of
  return, so the defect was latent there and real only on the aircraft.
- **The aircraft MPC flew the plan into the ground.** With altitude scored and
  two actuators available, the best plan over a 90 s window is a zoom climb
  that trades airspeed for altitude faster than the engines can replace it: on
  `plane_steps` it reached the commanded altitude at t=90 with 30 m/s of
  airspeed left, departed at 91 degrees angle of attack and hit the ground at
  t=372. The planner's objective now carries a barrier on airspeed against the
  stall speed at the current mass and altitude, the pattern
  `make_wind_turbine_mpc` already used. Fencing angle of attack instead does
  not work, since it sits at 4-8 degrees throughout the manoeuvre and only
  crosses 15 degrees one step before the departure. Together with the bound fix
  above, `plane_steps` goes from terminating early on all ten seeds and scoring
  656 against the PID's 1593, to flying full episodes and leading the PID on
  every seed tried.
- **`plane3d_racetrack` could not be rolled out at all.** The class declared
  `obs_value_index` and no `obs_target_index`, which `runners.rollout` reads to
  find the setpoint. The gain search scores candidates inside a `try`, so it
  reported every one as `-inf` and finished successfully having changed
  nothing, and no baseline could have been recorded for the environment. The
  conformance suite now checks both indices across the registry; the tests that
  covered this named three classes by hand, which is how a fourth got past
  them.
- **`plane3d_racetrack` guidance gains**, never previously searched. Settled
  cross-track error goes from 3.02 km to 0.31 km against an 8.4 km turn radius,
  and the return from 269.2 to 319.0, so it holds the pattern rather than
  flying its shape. The roll loop is deliberately held out of the search: left
  free it stiffens the loop and strips its damping, buying return while taking
  the achieved bank to 48 degrees against a 30 degree command limit and making
  the tracking worse. The environment's `expert_degraded` note is gone with it.
- **`render_mode="human"` raised on every environment whose renderer was
  rebuilt on the shared toolkit.** Those draw to an image and return no pygame
  screen, while the Gymnasium wrapper pumped the video system unconditionally,
  so `render()` failed with "video system not initialized". It now pumps only
  when a renderer actually opened a window.

- **The glass furnace was missing the two things that make furnace control
  hard.** It had thermal inertia but no dead time, and a first-order plant with
  no transport delay has no bandwidth limit, so a PID could be tuned arbitrarily
  tight against it: one held the crown to 0.078 K mean against a 10 K open-loop
  drift, roughly thirty times better than a real furnace is held, leaving the
  task no headroom. Its only load variation was an AR(1) drift on pull at a
  50 min correlation time, slower than the plant, and a slow smooth load is
  precisely what integral action cancels perfectly. Now: a 120 s crown
  thermocouple lag on the observation (the reward still scores the true crown
  temperature), 60 s of fuel transport delay, discrete batch charging on a 300 s
  charger cycle with dose-to-dose mass jitter, and the 40 s firing interruption
  at each reversal that deviation D2 had recorded as missing. Open-loop crown
  swing went from 10 K to 31 K, and the PID from 99% of the reward ceiling to
  90%.

- **Setpoint schedules re-derived against what the plants can actually be
  asked.** The 2D aircraft's staircase was a square wave between two altitudes
  2.4 km apart; it is now a ladder of eight levels with adjacent changes of 0.2
  to 0.8 of the amplitude. `plane_sine` commanded a peak climb rate of 20.9 m/s
  against an aircraft that can sustain 14.6, so the target was unreachable by
  construction; its amplitude is now 300 m, peaking at 7.9 m/s. The glass
  furnace's five setpoints were independent draws from a 45 K band, spanning
  30 K on average against the 10-20 K trim its own comment described; the
  schedule is now a bounded walk of at most 6 K a step. Every aircraft task and
  the furnace now start near the level they are commanded, rather than drawing
  the start independently of the target: the 3D tasks began a median 1.6 km
  from their assigned altitude, so the episode opened with minutes of open-loop
  climb.

- **Episode lengths cut where the protocol's own criterion says they are too
  long.** `plane_energy` ran 104 time constants against a suite that clusters at
  10 to 15; the path-following tasks ran up to 3.6 laps where one shows whether
  the path can be flown. Five aircraft episodes shortened, taking the aircraft
  recording from 12.9 h to about 4.5 h. See `docs/rl-protocol.md`.

- **One renderer for every aircraft panel.** `render_aircraft.py` now holds the
  palette, the console chrome, the A320 mesh and its projections; the 2D tasks,
  the 3D tasks and both patrol variants draw with it. `patrol` had been reaching
  into `plane3d.rendering` for eleven private names, so a palette change in one
  aircraft environment silently restyled another and nothing said so. Gallery
  clips are regenerated for both the PID and the MPC, all at 10 fps and about
  ten seconds.

### Removed

- **`plane_steps`**, absorbed into `plane_energy`. They were the same
  environment: same plant, same setpoint schedule, same disturbances, same
  episode length, differing only in `speed_weight` being 0.0 rather than 0.5.
  Two registered environments for one reward coefficient is not two tasks, and
  the pair cost 9 h of the 12.9 h it took to record the aircraft. The pure
  altitude staircase is still reachable as `PlaneParams(speed_weight=0.0)`.

- **Running-cost terms from seven rewards**, for this release line: fuel on the
  glass furnace, the boiler drum and the cement kiln, energy on the building,
  reboiler duty on the column, and reagent on the pH loop. All six weights are
  zero and those tasks score setpoint tracking alone.

  The battery was briefly in this list by mistake. Its `cost_weight` is not a
  consumption cost: it gates the degradation and state-of-charge terms, which
  are what keep that control problem well posed, and zeroing it made the
  optimal policy follow dispatch until the pack hit a limit. Restored, and the
  field now says what it is. Cost is real, but its weight against tracking accuracy silently picks a
  point on a Pareto front, and none of the seven had an argument behind its
  number. The glass furnace showed what that costs: a 0.1 fuel weight made a
  3.3 K standing error the optimum of what its MPC was asked to minimise, so the
  controller sat 6 K cold with fuel at minimum 80% of the time and lost to its
  own PID. The fields and the terms are still wired, so a weight can be restored
  once there is a defensible way to set one.

- Dead modules carrying no importers: `experts/degradation.py`,
  `experts/cpg.py` and `experts/pd.py` (the latter two were Brax/MuJoCo
  locomotion experts for environments this project does not have), and
  `scripts/benchmark_integration.py`, which imported an undeclared dependency
  and could not run.

### Known gaps

- ~~Both patrol variants hold formation only loosely, roughly 139 m of settled
  slot error against a 60 m tolerance.~~ **Resolved.** The error was exactly
  linear in the lead's turn rate and exactly symmetric in its sign, 25.9 m per
  0.001 rad/step, which is proportional control against a rotating reference
  rather than a mistuning; no gain could close it because the term was missing.
  Feeding the lead's turn rate forward takes the hardest case from 77.8 m to
  2.4 m, against a 3 m reward precision floor, and both variants now settle
  around 40 m at the benchmark settings, inside the 60 m tolerance. The strict
  xfail is removed.
- ~~Two of the seven gradient PID tuners return NaN gains.~~ **Resolved**, and
  the cause turned out to be one line. `plane.dynamics.aero_coefficients` wrote
  its stall blend as `CL_linear / (1 + exp(u))`; past about 77 degrees of
  incidence the exponent overflows float32, `exp` returns `inf`, and the
  reverse-mode derivative of `x / (1 + inf)` is NaN even though the forward value
  is a perfectly good 0. Any rollout long enough for the aircraft to depart
  reached that incidence, so every gradient through it came back NaN. Written as
  `jax.nn.sigmoid`, the same function evaluated stably, all seven tuners pass.
- No published RL baseline results yet.

[Unreleased]: https://github.com/YannBerthelot/TargetGym/compare/0.5.0...HEAD
