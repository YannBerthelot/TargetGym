# Oracle audit: is every oracle a problem definition?

Audit of 2026-11-03, read-only, on main at `8b6fb53` (the CasADi port, #124,
merged). It checks the 24 oracles (`REGISTRY[name].make_mpc`,
`target_gym.make_oracle(name)`) against the rule the owner adopted before any
results:

> An oracle is a problem definition, not a tuned controller.
>
> 1. Objective: the task's own reward.
> 2. Horizon: from the plant by a stated rule, e.g. covering the measured
>    settle time used by the episode-length rule.
> 3. Limits: the task's failure conditions become hard state constraints,
>    with margins taken from the task definition.
> 4. Model: the plant with noise zeroed (`plan_params`).
> 5. Solver: one constrained solver for every task, run to convergence,
>    certified by its convergence measure (KKT residual), constraint
>    violation, and the fact that it finds nothing better, never by the score.
> 6. Scores are reported only; nothing is changed after seeing them, and
>    held-out seeds only report.

This reverses the stance of yesterday's expert audit, which left the oracles
out of scope because "tuning them for performance is their purpose". Most of
what follows is the trace of that earlier stance.

**Owner decisions, 2026-11-03** (section 7):

- **Order (11):** the oracle rule comes first, then the expert protocols,
  and all of it ships in 0.8.0.
- **Margins and failed solves (3, 5):** a hard constraint at each trip
  limit; elastic slack priced at the trip cost when no plan is feasible; on
  a failed solve, the previous plan's next action, never the expert.
  Amended 2026-11-04: a warm solve that fails or hits the cap is first solved
  again cold (G11 in [oracle-solver.md](oracle-solver.md)); the constraint sits inside the limit by the planner's
  worst one-step error on it and the solver's tolerance (K11 in
  [oracle-smoothing-protocol.md](oracle-smoothing-protocol.md), Amendment
  1), and an oracle under the rule beats its expert on every recorded seed
  (a records check).
- **Horizon (2):** none of the three options. The owner asked for a clear
  protocol from practice: [oracle-horizon-protocol.md](oracle-horizon-protocol.md),
  Cooper's DMC recipe at Cooper's sample time, accepted the same day.
- **Kinks and the formation:** [oracle-smoothing-protocol.md](oracle-smoothing-protocol.md)
  (exact rewrites first, then softplus at a resolution-based width) and
  [oracle-formation-protocol.md](oracle-formation-protocol.md) (cooperative
  per-seat sub-problems), both accepted the same day.
- **Solver:** [oracle-solver.md](oracle-solver.md), the IPOPT port with
  exact Hessians computed forward-over-forward, accepted the same day after
  a probe of compile cost, memory and convergence.
- **Everything else is accepted as recommended:**
  - (1) the objective is the episode reward;
  - (4) plant kinks are smoothed under one width rule fixed in advance;
  - (5, 9) one shared solver, once oracle rows have their own fingerprint;
  - (6) per-solve certificates stored with each row, and records compared
    within a stated tolerance;
  - (7) e_floor, c_hold and rho_floor frozen as task constants;
  - (8) the wind turbine, battery, patrol and formation oracles replaced by
    the shared solver;
  - (10) the kiln's derivatives fixed.

The section on compute calls for a timing and memory probe before
committing to the solver; that comes first in the implementation.

## How to read the verdicts

Each task gets one verdict per rule element, six in all: objective, horizon,
limits, model, solver, and score independence (element 6).

- **PASS**: the element meets the rule as it stands.
- **FAIL**: it contradicts the rule. Either it was chosen or changed after
  seeing scores (on protocol seeds 0 to 2, recorded seeds 0 to 9, or training
  seeds), or it is a different thing from what the rule asks (a surrogate
  objective, a zero-gradient trip charge, a fixed iteration count).
- **UNCLEAR**: set by hand with no stated source, or from a source that is not
  a plant rule (an inherited default). It needs a decision, not a fix.

Assumptions:

- "The task's own reward" means the per-step reward summed over the episode,
  not the protocol's scored window. Two oracles weight the window (cstr,
  four_tank); I mark them UNCLEAR and list the choice as a decision.
- A failure condition the task documents as unreachable (cstr, first_order,
  ph_neutralization) passes the limits element: there is nothing to
  constrain.
- A trip charged inside the rollout passes nothing on limits. The trip cost is
  a constant on a boolean, so its gradient is zero, and after a trip the
  state is a fresh reset draw that does not depend on the plan
  (`base.failure_kernel`). A planner sees it only by comparing whole plans.
- Provenance comes from docstrings, PHYSICS.md, CHANGELOG, commit messages
  and `git log -S`. Where a setting has no stated source I say so; I did not
  infer intent.

## Summary

No task passes all six elements. Over the 144 verdicts:

| element | PASS | FAIL | UNCLEAR |
| --- | --- | --- | --- |
| 1. objective | 14 | 8 | 2 |
| 2. horizon | 5 | 7 | 12 |
| 3. limits | 3 | 20 | 1 |
| 4. model | 22 | 1 | 1 |
| 5. solver | 1 | 23 | 0 |
| 6. score independence | 0 | 18 | 6 |
| **total** | **45** | **77** | **22** |

The model element is the one the suite already meets: every planner plans on
the plant's own step with its declared noise zeroed. Solver and limits fail
almost everywhere, and 18 of 24 oracles had settings chosen by score.

Closest to the rule: ph_neutralization (four PASS, solver FAIL, score
UNCLEAR), first_order and cstr (solver FAIL, the rest PASS or UNCLEAR), and
compressor_surge, the only constrained solver run to a tolerance.

## 1. Today's setup

Eight designs serve 24 tasks.

| family | tasks |
| --- | --- |
| `NewtonMPC`: box-constrained iLQR on `step_env`, fixed iterations, action bounds only | cstr, first_order, four_tank, ph_neutralization, glass_furnace, reactor, hvac, unstable_cstr |
| `GradientMPC`: normalised projected gradient, fixed iterations | distillation, boiler_drum, plane_energy, plane3d_heading, plane3d_circle, plane3d_racetrack, plane3d_figure8 |
| Two `GradientMPC`s with a handover | plane, plane_sine |
| `SamplingMPC`: cross-entropy method | cement_kiln |
| Feedback law plus an Adam residual planner | patrol, patrol_bearing_only, formation |
| Feedback law, no planner | wind_turbine |
| Feedforward, no planner | battery |
| Interior-point NLP (JAX port of IPOPT), multiple shooting | compressor_surge |

Per task (H is the horizon in steps; "reward/floor" means the step's reward
divided by `rho_floor_tracking`):

| task | solver | H | iterations | stage cost | barrier or penalty | guide or warm start | other tricks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plane | 2 x GradientMPC | 20 | capture 50 at lr 0.05; hold 100, lr 0.05 to 0.002 | reward / 0.84^2 (fixed scale) | stall barrier, weight 10, from 1.3 Vs | PID rollout as initial plan and guide | hold plans the elevator only, throttle on the PID's speed loop; handover after 5 steps within 3 m, back past 20 m; bumpless integrator |
| plane_energy | GradientMPC | 20 | 50, fixed lr 0.05 | reward / 1.0 | stall barrier | PID rollout | no tail |
| plane_sine | as plane | 20 | as plane | reward / 1.0 | stall barrier | PID rollout | as plane |
| plane3d_heading, _circle, _racetrack, _figure8 | GradientMPC | 30 | 200, lr 0.05 to 0.002 | reward from `step_env` | none | shifted plan, zero start | NaN gradients zeroed, best iterate kept |
| patrol, patrol_bearing_only | law + Adam residual | 30 | 40, lr 0.02 | -reward from `step_env` | none | twin-autopilot law (9 searched gains), residual within +-0.3 | reads the true state on the bearing-only task |
| formation | law + Adam residual | 30 | 40, lr 0.02 | team -reward | none | each seat's law, residual within +-1.0 | wingman residual added on top of the lead's |
| cstr | NewtonMPC | rest of episode (150) | 20 | reward/floor, burn-in steps weighted 1e-3 | none | shifted plan with feedback | re-solve every step |
| first_order | NewtonMPC | 5 | 10 | reward/floor | none | shifted plan | |
| four_tank | NewtonMPC | rest of episode (1200) | 20, cold 60 | reward/floor, burn-in weighted 1e-3 | none (the CasADi build's 5 mm level margins were dropped in #124) | shifted plan | re-solve every 25 steps |
| ph_neutralization | NewtonMPC | 20 | 10 | reward's tracking and running terms, pH with its implicit-function slope | none | shifted plan | `planner_pH` custom JVP |
| distillation | GradientMPC | 15 | 80, lr 0.08 to 0.004 | reward | none | shifted plan | |
| glass_furnace | NewtonMPC | 120 | 10 | reward/floor | none | shifted plan | |
| reactor | NewtonMPC | 20 | 10 | tracking and rod wear, each abs value rounded by `PLANNER_ROUNDING` = 1.0 (times e_floor) | trip charged as a constant, state kept | shifted plan | sub-step scan of the physics |
| hvac | NewtonMPC | 24 | 10 | reward/floor | none | shifted plan | |
| cement_kiln | SamplingMPC | 40 | 32 x 96 samples, 12 elite, std 0.1 | reward/floor on the returned state | none | shifted mean | returns the best sample |
| boiler_drum | GradientMPC subclass | 30 | 1280, lr 0.15 to 0.005 | reward/floor | none | zero start | |
| wind_turbine | feedback law | 1 step | 4 Newton steps on torque | next-step power equal to target x low-speed protection | slew cap 0.13 deg a step inside 0.97 to 1.13 x rated, ramp 0.04 | pitch PI Kp 12, Ki 0.6, setpoint 1.05 x rated | v1 params get a GradientMPC |
| battery | feedforward | none | none | none: commands the next block's level | none | none | state of charge not guarded |
| unstable_cstr | NewtonMPC | 32 | 10 | reward/floor + barrier | past the point-of-no-return line: quadratic over 1 K, then 9.891e6 per K | cascade PID rollout as guide | |
| compressor_surge | IPM, float64 | 60 + 1 | to tol 1e-8, cap 150 | own: (dp error / 1 kPa)^2, energy above 49.6 kW, move weight 1e-3, terminal weight 20 | surge as a soft constraint, 5 % margin, slack priced at the trip cost | cold guess from the PID; warm shifted plan and multipliers | restated model with clips smoothed (0.1); PID fallback on a failed solve |

## 2. Failure conditions and how the oracle treats them

No task defines a planner margin. Every margin in use is the oracle's own.

| task | trips (`check_is_terminal`) | oracle's treatment |
| --- | --- | --- |
| plane, plane_energy, plane_sine | altitude outside [min_alt, max_alt] | invisible: the objective is scored on the returned state, which after a trip is the restart draw. A stall barrier (not a task limit) is the only guard |
| plane3d (4 tasks) | altitude band | charged in the rollout, no gradient |
| patrol, patrol_bearing_only | either aircraft's altitude band; separation at most 25 m; slot error at least 1500 m | charged, no gradient |
| formation | every aircraft's altitude band; any pair within 25 m; any slot error at least 1500 m | charged, no gradient |
| cstr | none (documented unreachable) | nothing to do |
| first_order | abs(x) at least 3 (documented unreachable) | charged; never binds |
| four_tank | any level at h_min or h_max | charged, no gradient. The CasADi build held planned levels 5 mm inside as hard constraints; the port removed that |
| ph_neutralization | none (documented unreachable) | nothing to do |
| distillation | yD at most 0.90, xB at least 0.10 | charged, no gradient |
| glass_furnace | crown, melt and working-end temperature bands | charged, no gradient |
| reactor | flux, fuel and coolant temperature bands | charged as a constant in `reactor_stage`, no gradient |
| hvac | air temperature band (restart in place, every step outside charged) | charged, no gradient |
| cement_kiln | burning zone above T_bz_max or below T_bz_min; discharge free lime at least lime_max | invisible, as on the 2D aircraft |
| boiler_drum | abs(level) at least 0.25 m; pressure band | charged, no gradient |
| wind_turbine | rotor speed outside 0.40 to 1.25 x rated | not modelled; the slew band is the only guard |
| battery | state of charge, cell voltage, cell temperature | ignored ("never trips over 2000 seeds") |
| unstable_cstr | T at least 365 K, plus validity guards | charged, plus a soft barrier at a point-of-no-return line defined in `experts.py`, not the task |
| compressor_surge | phi_min below 2W | soft constraint with an exact penalty at 2W x 1.05; margin derived by a gate on training seeds |

## 3. Provenance of the settings

| task | setting | source | verdict for the setting |
| --- | --- | --- | --- |
| plane | horizon 30 to 20 | "measured on seed 0 over the hold" (`plane/experts.py`) | FAIL |
| plane | iterations, decay, speed loop, handover, thresholds | oracle audit 2026-10, protocol seeds 0 to 2 and the ten recorded seeds (`_TASK_SETTINGS` comment) | FAIL |
| plane | stall barrier, tail 60 then 0 | measured on `plane_steps` seed 0 and on `plane` | FAIL |
| plane_energy | kept the shipped planner | compared on protocol seeds | FAIL |
| plane_sine | as plane | protocol and recorded seeds | FAIL |
| plane3d (4) | horizon 30 | first commit (`3528d53`, 2026-04-17), no reason given | UNCLEAR |
| plane3d (4) | 200 iterations, decay to 0.002 | protocol seeds 0 to 2 (`faf89f8`) | FAIL |
| patrol (2) | law gains | cross-entropy search on episode cost, training seeds 100 to 115 | FAIL (searched, though not on scored seeds) |
| patrol (2) | H 30, 40 iterations, lr 0.02, residual 0.3 | `378e051`, no source; accepted on protocol seeds 0 to 2 | FAIL |
| formation | residual bound 1.0 | episode cost on training seeds 100 to 103 (`0468283`) | FAIL |
| cstr | whole episode, window weighting | a stated design, "optimise the window the protocol scores" (`80ad7c6`) | UNCLEAR |
| cstr | 20 iterations, pre-weight 1e-3 | no source; the port was accepted by matching the records on seeds 0 to 9 | UNCLEAR |
| first_order | horizon 5 | PC-gym's default, below the plant's 10-step time constant | UNCLEAR |
| four_tank | re-solve every 25, 60 cold iterations | no source | UNCLEAR |
| ph_neutralization | horizon 20 | "slightly more than one residence time", at creation | UNCLEAR (iterations unsourced) |
| distillation | horizon 15 | at creation (`2557b5c`), no reason | UNCLEAR |
| distillation | 80 iterations, decay | "measured on the protocol seeds" (`23cd878`) | FAIL |
| glass_furnace | horizon 60 to 120 | protocol seeds 0 to 2 and recorded seeds (`d8da2d8`) | FAIL |
| reactor | `PLANNER_ROUNDING` 1.0 | episode cost on seeds 2, 3 and 7, all recorded seeds (`88ccf37`) | FAIL |
| hvac | horizon 24 | "enough to see the morning setback and the solar peak" (`9951617`), a schedule reason, not a plant rule | UNCLEAR |
| cement_kiln | horizon 40 | "most of the ~25 minute transport delay": a plant rule, but 40 steps are 20 minutes and do not meet it | UNCLEAR |
| cement_kiln | 96 x 32, std 0.1, best sample | "measured on the protocol seeds" (`23cd878`) | FAIL |
| boiler_drum | 1280 iterations, decay; horizon 45 rejected | protocol seeds, oracle audit 2026-10-01 | FAIL |
| wind_turbine | the law itself; slew band and ramp | replaced the planner because it "beats it on every protocol seed" (`2bb4dfb`); ramp from 197 screened seeds and the protocol seeds; PI gains grid-searched | FAIL |
| battery | the feedforward | replaced a planner on protocol seeds 0 to 2 | FAIL |
| unstable_cstr | horizon 32 | stated rule: the smallest whole number at least 1.5 x the 21-step recovery | PASS for the horizon |
| unstable_cstr | point-of-no-return barrier | added after one trip on the ten recorded seeds; penalty "ours" | FAIL |
| compressor_surge | horizon 60 | stated rule: the slowest actuator move a block asks for | PASS for the horizon |
| compressor_surge | 5 % margin | sized by a stated noise rule, candidates gated on training seeds 1000 to 1127 at equal cost | not score-chosen |
| compressor_surge | error scale, move and terminal weights, power reference 49.6 kW | "ours"; the power reference is a provisional c_hold, now 62.27 kW in the params | UNCLEAR |

The port itself (#124) was accepted on score: "over the ten seeds each matched
or beat its 0.7.0 record" (CHANGELOG). Under the rule that is a report, not a
certificate.

## 4. Verdicts per task

O objective, H horizon, L limits, M model, S solver, 6 score independence.

| task | O | H | L | M | S | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| plane | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| plane_energy | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| plane_sine | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| plane3d_heading | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| plane3d_circle | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| plane3d_racetrack | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| plane3d_figure8 | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| patrol | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| patrol_bearing_only | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| formation | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| cstr | UNCLEAR | PASS | PASS | PASS | FAIL | UNCLEAR |
| first_order | PASS | UNCLEAR | PASS | PASS | FAIL | UNCLEAR |
| four_tank | UNCLEAR | PASS | FAIL | PASS | FAIL | UNCLEAR |
| ph_neutralization | PASS | PASS | PASS | PASS | FAIL | UNCLEAR |
| distillation | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| glass_furnace | PASS | FAIL | FAIL | PASS | FAIL | FAIL |
| reactor | FAIL | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| hvac | PASS | UNCLEAR | FAIL | PASS | FAIL | UNCLEAR |
| cement_kiln | PASS | UNCLEAR | FAIL | PASS | FAIL | FAIL |
| boiler_drum | PASS | FAIL | FAIL | PASS | FAIL | FAIL |
| wind_turbine | FAIL | FAIL | FAIL | PASS | FAIL | FAIL |
| battery | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| unstable_cstr | FAIL | PASS | FAIL | PASS | FAIL | FAIL |
| compressor_surge | FAIL | PASS | UNCLEAR | UNCLEAR | PASS | UNCLEAR |

Notes on the less obvious cells:

- **Objective PASS** means the planner's per-step cost has the reward's value
  on steps without a trip. Where trips are invisible (2D aircraft, kiln) the
  limits cell carries it. The 2D aircraft fail objective on the stall barrier
  and the fixed objective scale; the reactor on its rounding; unstable_cstr on
  its barrier; the compressor on its own quadratic, its move and terminal
  weights, and a power reference 12.7 kW below the params' `c_hold`.
- **Model**: the battery's feedforward uses no plant model. The compressor
  plans on a restatement of the step with smoothed rate limits, held to the
  env by tests (at most 0.555 Pa on the next pressure); whether a smoothed
  restatement counts as "the plant" is a decision.
- **Solver PASS** for the compressor means it runs to IPOPT's tolerance and
  counts failures and capped solves. It still hands a failed solve to the PID
  (one in 12,000 on CI) and does not record the KKT residual per solve.
- **Horizon PASS** for cstr and four_tank is the whole remaining episode, a
  rule with no free parameter. It would not survive a settle-based rule
  unchanged (section 5).

## 5. What the rule implies

### Horizon

The example rule, "cover the measured settle time", has two problems in this
repository. `eval.fixed_settle` is the slower of the PID's and the oracle's
measured settles, so a horizon set from it depends on the oracle it sizes.
And measured settles read 0 to 4 steps on fast plants (reactor, hvac,
wind_turbine, battery), where the scored behaviour is disturbance rejection.

Three candidate rules, all stated before any result:

- **A**: H = `fixed_settle` (the episode-length rule's S). Circular, and
  nearly zero on four tasks.
- **B**: H = max(expert's measured settle, 5 x tau_63), capped at the episode.
  tau_63 is the open-loop actuator-to-output step response
  (`scripts/measure_time_constants.py`, the table in docs/rl-protocol.md
  section 6). Not circular, never zero. It moves when the expert is retuned.
- **C**: H = the rest of the episode, as the discount rule does
  (gamma = 1 - 1/N). No free parameter, highest cost.

| task | today | A: fixed_settle | expert settle | 5 x tau_63 | B | episode N |
| --- | --- | --- | --- | --- | --- | --- |
| plane | 20 | 47 | 47 | 115 | 115 | 280 |
| plane_energy | 20 | 85 | 85 | 115 | 115 | 1600 |
| plane_sine | 20 | 81 | 81 | 115 | 115 | 480 |
| plane3d_heading | 30 | 136 | 136 | 70 | 136 | 300 |
| plane3d_circle | 30 | 66 | 66 | 70 | 70 | 300 |
| plane3d_racetrack | 30 | 42 | 42 | 70 | 70 | 650 |
| plane3d_figure8 | 30 | 17 | 0 | 190 | 190 | 400 |
| patrol | 30 | 84 | 84 | 65 | 84 | 200 |
| patrol_bearing_only | 30 | 84 | 84 | 60 | 84 | 200 |
| formation | 30 | 45 | 45 | 180 | 180 | 200 |
| cstr | 150 (episode) | 67 | 67 | 25 | 67 | 150 |
| first_order | 5 | 63 | 63 | 50 | 63 | 150 |
| four_tank | 1200 (episode) | 581 | 581 | 190 | 581 | 1200 |
| ph_neutralization | 20 | 12 | 12 | 70 | 70 | 300 |
| distillation | 15 | 46 | 46 | 40 | 46 | 200 |
| glass_furnace | 120 | 18 | 7 | 660 | 660 | 1600 |
| reactor | 20 | 1 | 0 | 5 | 5 | 864 |
| hvac | 24 | 4 | 0 | 310 | 310 | 720 |
| cement_kiln | 40 | 8 | 8 | 290 | 290 | 700 |
| boiler_drum | 30 | 8 | 5 | 15 | 15 | 400 |
| wind_turbine | 1 (law) | 0 | 0 | 5 | 5 | 400 |
| battery | none | 6 | 6 | 5 | 6 | 360 |
| unstable_cstr | 32 | 99 | 99 | 32 to 72 | 99 | 1200 |
| compressor_surge | 60 | 100 | 100 | 247 | 247 | 1200 |

Caveats. The settles come from `hold_measurements/`, whose MPC rows on eight
tasks still describe the CasADi oracles (dated before the port). The expert
settles will move with the expert-protocol work planned for 0.8.0. Rule B
roughly triples or quadruples today's horizons on the aircraft, glass,
hvac, kiln and compressor, and shortens reactor and wind.

### Limits, and the effort to state each task for the IPM

The IPM (`compressor_surge/experts_ipm.py`) solves min f(z) subject to
c(z) = 0 and G z + g0 >= 0: its inequalities are linear. A bound on a state
fits directly. A nonlinear limit needs an auxiliary variable defined by an
equality (h(x_k) - y_k = 0) and a bound on y_k, as the compressor carries
phi_min as a sixth state. A nonconvex limit (a minimum separation, a maximum
over a profile) is solvable but only to a local optimum.

Kinks in the reward can be written exactly with epigraph slacks rather than
smoothed: the tolerance band max(abs(e) - e_tol, 0) and the running cost
max(q - c_hold, 0) become a slack with two linear bounds. That keeps the
objective exactly the reward. Kinks inside the plant (clips, rate limits,
saturations) cannot be removed that way; the compressor smooths them.

A trip cannot happen in a feasible plan, so the trip charge drops out of the
objective. When no plan is feasible, an elastic slack priced at the trip cost
(the compressor's design) keeps the reward's price on a violation.

Effort, as a rough size: S is a few states with bounds and a smooth step;
M has a large state, a long horizon, or an algebraic limit; L has nonconvex
limits or a model whose derivatives fail today. "nx" counts every floating
leaf of the state, schedules and controller memory included; the dynamic
state is smaller.

| task | limits as constraints | form | nx | effort |
| --- | --- | --- | --- | --- |
| plane, plane_energy, plane_sine | min_alt <= z <= max_alt | state bound | 16 | M: 29 non-smooth operations in the dynamics and reward, stall regime |
| plane3d (4) | min_alt <= z <= max_alt | state bound | 26 | M: as above, 3D |
| patrol (2) | both altitude bands; separation > 25 m; slot error < 1500 m | bounds; separation nonconvex; slot error nonlinear | 74 | L |
| formation | all altitude bands; three pairwise separations; two slot errors | as patrol, more seats | 96 | L |
| cstr | none | none | 4 | S |
| first_order | none in practice | state bound | 3 | S |
| four_tank | h_min < h_i < h_max, four tanks | state bounds | 8 | S, but H 581 to 1200 |
| ph_neutralization | none | none | 6 | S: pH needs the implicit-function derivative, which exists |
| distillation | yD > 0.90, xB < 0.10 | state bounds | 46 | M |
| glass_furnace | crown, melt, working-end bands | state bounds | 26 | M: 22 non-smooth operations, H up to 660 |
| reactor | flux, fuel and coolant bands | state bounds | 14 | S to M: ten sub-steps a stage, abs values in the reward |
| hvac | air temperature band | state bound | 9 | S, H up to 310 |
| cement_kiln | max T_solid < T_bz_max; max T_solid > T_bz_min; lime < lime_max | one bound per cell; the "cold" limit is a max, nonconvex | 85 | L: reverse-mode gradients overflow to NaN through the Arrhenius term and the transport delay (`SamplingMPC` docstring), so a Newton solver needs a model fix first |
| boiler_drum | abs(level) < 0.25 m; pressure band | state bounds | 10 | S to M: 18 non-smooth operations, ill-conditioned |
| wind_turbine | 0.40 < omega / omega_rated < 1.25 | state bound | 8 | S |
| battery | SoC band; cell voltage band; T_cell < T_max | voltage is algebraic: auxiliary variable | 19 | M |
| unstable_cstr | T < 365 K | state bound | 10 | S; margin decision matters (section 6) |
| compressor_surge | phi_min >= 2W | already in the NLP | 17 (6 planned) | S: point the objective at the reward and drop the hand weights |

The IPM as written is dense. It forms the full Hessian of the objective
(`jax.hessian(pr.objective)(z)`), the dense inequality Jacobian G, and
K = H + G' diag(w/s) G as an n x n matrix; only the Newton step uses a
stage-wise Riccati solve. The compressor's n is 9 x 61 = 549, so this is
2.4 MB. Under rule B, n reaches about 17,800 on glass_furnace (660 stages of
27) and 25,000 on the kiln, which is 2.5 to 5 GB per dense matrix in
float64. A shared IPM needs block-sparse, stage-wise assembly before it can
serve the long-horizon tasks.

## 6. Cross-cutting findings

### Fixed iterations do not converge on a cold start

Every planner except the compressor's runs a fixed number of iterations and
reports nothing about convergence. I measured the iLQR's planned cost on a
cold solve from one reset (training seed 100, the planner's own model, the
shipped horizon), after 10, 40 and 160 iterations:

| task | shipped iterations | 10 | 40 | 160 |
| --- | --- | --- | --- | --- |
| first_order | 10 | 98,025 | 98,025 | 98,025 |
| ph_neutralization | 10 | 132,355 | 132,355 | 132,355 |
| hvac | 10 | 0 | 0 | 0 |
| glass_furnace | 10 | 62,860 | 62,814 | 62,811 |
| reactor | 10 | 9.618 | 9.539 | 9.539 |
| cstr | 20 | 701.8 | 372.5 | 372.5 |
| four_tank | 60 cold | 5,416 | 137.7 | 137.7 |
| unstable_cstr | 10 | 9,938,117 | 2,171,915 | 2,170,990 |

On unstable_cstr the shipped 10 iterations leave the first plan at 4.6 times
its converged cost. Warm starts recover over later steps, but the closed loop
the record measures starts from an unconverged plan. One seed, one state:
this shows the gap exists, not how large it is over an episode.

### Reproducibility across machines

A convergence stop makes the iteration count depend on rounding, so
machines can branch. CI saw one compressor solve in 12,000 handed to the PID
that a laptop did not. Fixed iterations do not buy bit-identity either: the
recorded rows were produced on at least four CPU models (AMD EPYC 7763,
9V74 and 9V45, and Intel Xeon Platinum), and XLA's CPU code differs between them. The
rule's certificate (KKT residual, violation, iterations per solve) is what
makes a record checkable; the check then compares returns within a stated
tolerance. On a failed solve, applying the previous plan's next action keeps
the oracle independent of the expert; handing the step to the PID does not.

### Oracles that depend on the expert

Five designs read expert gains: the plane's speed loop, initial plan and
guide; the patrol and formation laws (the lead autopilot's
`plane3d_heading` gains, declared in `gains_keys`); the wind turbine's
pitch PI; unstable_cstr's guide; the compressor's cold guess and fallback.
Their oracle rows move whenever the expert is retuned, which is why
yesterday's plan sequences the expert work after the port. Under the rule
none of them is needed.

### Fingerprints

`experts/mpc.py` and `experts/pid.py` are in every task's baseline
fingerprint (`provenance._SHARED_SOURCES`), and one fingerprint covers both
the PID and the oracle rows of a record. So moving the IPM into `experts/`
re-records all 24 tasks, PID rows included, and so does any later edit to
it. Today's records cost about 26,700 CPU-seconds on CI for ten seeds of all
24 tasks (sum of `seconds` in `baseline_returns/`). The environment version
stamps do not hash expert code, so oracle changes bump no version unless a
task constant that derives from the oracle moves (next finding).

### Oracle results feed the metrics and the reward

| quantity | where | how the oracle enters |
| --- | --- | --- |
| `best_known` | `scoring.best_known` | the oracle's move and settling ratios against the expert, reported beside a policy's score |
| Reach points | `suite.MIN_REACH_ROOM` = 1.25 | a task gets a Reach point only if the oracle's move ratio is at least 1.25; first_order, cstr, ph_neutralization and four_tank are out today |
| fixed settle S | `eval.fixed_settle` | max of the PID's and the oracle's settle: sets the hold window of every score and the episode-length rule's cycle check (cycle at least 2S) |
| `e_floor` | params, per PHYSICS.md | max(instrument resolution, oracle hold): an earlier oracle's hold on wind_turbine (1680 W), boiler_drum level (2.67 mm) and cement_kiln; the ceiling is (resolution / e_floor)^p, so the ceiling and headroom move with it |
| `c_hold` | params; `reward.running_cost` | "a measured MPC consumption" by the reward's own docstring; the compressor's is the NMPC's mean recycle power. It sets the running cost, so it is part of the reward |
| `rho_floor_tracking` | params | the oracle's best hold on every task. The score does not use it, but the planners divide by it, so the oracle's own hold scales its objective. The plane oracle documents that this loop did not settle and froze the scale |

So a rule-compliant oracle that holds differently would, if the floors keep
following it, move e_floor and c_hold (rewards, so version bumps), the
ceiling, S, the hold windows and the Reach radar.

### Compute

The compressor's IPM costs about 103 s per episode on CI (1029 s for ten
seeds) and about 67 s on a laptop, at H 61 with six planned states and 5.8
iterations a solve. The aircraft gradient planners are the most expensive
today (racetrack 3818 s, plane_energy 3068 s, formation 2941 s for ten
seeds). Rule B lengthens most horizons, and per-iteration cost grows with
the horizon and steeply with the state size. A timing probe on one
mid-size task (glass_furnace or plane3d_circle) on training seeds should
come before any commitment.

## 7. Decisions for the owner

1. **Objective scope.** The episode reward (recommended, as the rule reads),
   or the protocol's scored window, as cstr and four_tank weight it today.
2. **Horizon rule.** B, max(expert settle, 5 x tau_63) (recommended: not
   circular, never zero); A, `fixed_settle` (circular, near zero on four
   tasks); or C, the rest of the episode (no parameter, costliest).
3. **Margins.** No task defines one. Zero (hard at the trip limit, recommended
   as the literal rule, with trips then reported); a noise-derived margin by a
   rule fixed in advance (the compressor's gate is one); or a margin written
   into each task's params. Plus: elastic slack priced at the trip cost when
   no plan is feasible (recommended).
4. **Plant kinks.** Allow smoothing of clips and rate limits in the planner's
   model under one stated width rule (as the compressor does), or require the
   exact step. Reward kinks go through epigraph slacks either way.
5. **Solver home and failure handling.** Lift the IPM into a shared module
   (re-records all 24, PID rows too) or keep it per package; tolerance and
   iteration cap; on a failed solve, the shifted plan (recommended) or the
   expert.
6. **What a record certifies.** Per-solve KKT residual, violation and
   iterations stored with each row, and records compared within a stated
   tolerance instead of bit-identity.
7. **Decouple task constants from the oracle.** Freeze e_floor, c_hold and
   rho_floor as task constants (recommended: no version bumps, and the
   reward stops depending on the oracle), or keep re-deriving them from each
   new oracle's hold (every task bumps).
8. **Oracles that are not planners.** Replace the wind turbine's law, the
   battery's feedforward and the patrol and formation residual planners with
   the shared solver (the rule's reading), or keep any of them as a stated
   exception.
9. **Fingerprint split.** Give the oracle rows their own fingerprint, so an
   oracle change stops staling the PID rows, before the re-record.
10. **The cement kiln.** Fix the model's derivatives (needed for any Newton
    solver) or keep it outside the rule until then.
11. **Sequencing and budget.** The expert-protocol work also lands in 0.8.0.
    Doing the oracle rule first, with the oracles decoupled from the expert,
    lets the expert work re-record PID rows only.
