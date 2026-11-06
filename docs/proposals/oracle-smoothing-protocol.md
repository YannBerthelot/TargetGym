# Oracle smoothing protocol

Specification of 2026-11-03, for owner review. It implements the accepted
audit decision (4), "plant kinks are smoothed under one width rule fixed in
advance" (`oracle-audit.md`, owner decisions). Nothing here changes the
repository or records anything. Two read-only probes were run in the
project's `.venv` (JAX 0.6.2): one step of every task under x64, to read
dtypes, and `jax.grad` of `clip`, `maximum` and `abs` at their kinks.

The owner's rule applies throughout. A setting comes from published or
real-life practice, run as a fixed recipe, and is never chosen by score. A
value no source gives is a rule fixed here, before any run, and labelled
ours. It builds on the accepted decisions: the objective is the episode
reward; failure limits are hard constraints with slack priced at the trip
cost; one shared constrained solver runs to convergence; Cooper's sample time
T_c, with the move held between solves.

**Owner decisions, 2026-11-03.** Every recommendation in section 9 is
accepted (S1 to S10):

- S1: the resolution width rule K5;
- S2: softplus everywhere;
- S3: guards dropped and checked, promoted on a flag;
- S4: the min over turn directions smoothed;
- S5: the kiln's cold limit as a conservative smoothed max;
- S6: the reactor made float64-clean before the re-record;
- S7: the exact rewrites verified first on first_order, battery and glass_furnace;
- S8: the distillate clip smoothed;
- S9: the racetrack joins left C1;
- S10: the compressor brought under the rule.

**S7, first result, the same day.** With the exact rewrites T1 and T2 only,
in the scratchpad probe, first_order converges in 11 iterations cold and 1
on every warm solve (KKT error 1e-9 to 1e-11). Without them, its first three
solves hit the cap or fail. The JAX derivative of a clip at its bound is 0.5
times the unclipped slope, and 0 just past it, as section 3 states. battery
and glass_furnace need more than one slack per stage for their epigraph
rewrites, which the stage-wise solve does not yet support; they are checked
when the shared solver is built.

Labels, as in the horizon protocol:

- **read**: a source states it.
- **derived**: computed from a read value, or a direct consequence of an
  accepted decision or of the code.
- **ours**: no source gives it, so it is fixed here, before any run.

A **kink** is any point where the planner's model (`step_env` with
`plan_params`, and the reward) is not twice continuously differentiable:
clip, max, min, abs, where and cond branches, floors and integer counters,
sqrt near zero, angle wraps, trip and reset kernels, implicit solves with no
derivative. A **site** is one source line on the planning path, counted once
however many tasks or evaluations use it.

## 1. What practice prescribes

| id | source | what it says | where |
| --- | --- | --- | --- |
| IPOPT | Ipopt documentation, [coin-or.github.io/Ipopt](https://coin-or.github.io/Ipopt/) | the objective and constraint functions may be nonconvex but "should be twice continuously differentiable" | Overview |
| CASADI | J. Andersson (CasADi lead developer), casadi-users, [18 Nov 2013](https://groups.google.com/g/casadi-users/c/CKA5Zzum9Ew) | the derivative rule of `fabs` is equal to sign(x) only "almost everywhere"; AD works with nonsmooth functions if that is kept in mind | thread "if_else usage" |
| JAXFAQ | JAX FAQ, [docs.jax.dev/en/latest/faq.html](https://docs.jax.dev/en/latest/faq.html) | in `jnp.where`, the unselected branch's NaN adjoint is accumulated into the gradient; guard the argument with an inner `where` | "Gradients contain NaN where using where" |
| QB | S. J. Qin and T. A. Badgwell, *Control Eng. Practice* 11 (2003) 733-764 ([PDF](https://skoge.folk.ntnu.no/prost/publications_various/mpc_survey_qin.pdf)) | every industrial MPC product allows hard MV maximum, minimum and rate-of-change constraints; heuristics that set an input to its limit or clip it prevent violations but lose performance and the solutions do not satisfy the KKT conditions; least squares followed by clipping is "clearly sub-optimal" | p. 755; p. 753; p. 757 |
| MW | MathWorks MPC Toolbox, ["Specify Constraints"](https://www.mathworks.com/help/mpc/ug/specifying-constraints.html) | put known physical MV limits in as hard MV bounds; add MV increment bounds for a known physical rate limit; hard MV bounds and hard increment bounds on one MV can conflict, so soften one; increment bounds are soft by default | "Input and Output Constraints: Tips"; "Constraint Softening" |
| CM | C. Chen and O. L. Mangasarian, "A class of smoothing functions for nonlinear and mixed complementarity problems", UW-Madison Math. Prog. TR 94-11 (1994) ([PDF](https://minds.wisc.edu/items/bfdba5d0-3603-4491-acb0-89e7aa84bbf3)); *Comput. Optim. Appl.* 5 (1996) 97-138 | smooth plus functions from a density; the gap to x_+ lies in [-D2 beta, D1 beta]; Ex. 2.1, the softplus x + beta log(1 + e^(-x/beta)), has D1 = log 2, D2 = 0 and is infinitely differentiable; Ex. 2.2, Chen-Harker-Kanzow-Smale (x + sqrt(x^2 + 4 beta^2)) / 2, has D1 = 1; Ex. 2.3 and 2.4 are only once differentiable; Alg. 4.1 sets the smoothing parameter "in reverse proportion" to the residual, capped at the value the target accuracy needs | Prop. 2.2, p. 4; Ex. 2.1 to 2.3, p. 8; Ex. 2.4, p. 10; Alg. 4.1, p. 20 |
| GB | V. Gopal and L. T. Biegler, "Smoothing methods for complementarity problems in process engineering", *AIChE J.* 45(7) (1999) 1535-1547 (abstract via [OpenAlex](https://api.openalex.org/works/doi:10.1002/aic.690450715)) | smoothing solved by a continuation algorithm; nested derivative discontinuities approximated with simple functions of a single smoothing parameter | abstract |
| BIEGLER | L. T. Biegler, *Nonlinear Programming*, SIAM 2010, ch. 11, as relayed by A. Nurkanović, lecture 5 of the Winter School on Nonsmooth Optimal Control, Paris, Feb. 2025 ([PDF](https://www.syscop.de/files/users/armin.nurkanovic/05_mpec_formulations_theory.pdf)) | guidelines for MPEC modelling of piecewise smooth functions are in ch. 11; abs and max written with complementarity pairs | slides 8/41 to 12/41 (secondary) |
| SCH | S. Scholtes, *SIAM J. Optim.* 11 (2001) 918-936; A. U. Raghunathan and L. T. Biegler, *SIAM J. Optim.* 15 (2005) 720-750 (abstracts via OpenAlex) | a sequence of regularised NLPs converges to stationary points of the complementarity problem; an interior-point method is adapted to complementarity constraints by relaxing them | abstracts |
| NOSNOC | nosnoc_py, `nosnoc/nosnoc_opts.py` and `nosnoc/solver.py` at commit `c8a2294` (2026-08-11), [github.com/FreyJo/nosnoc_py](https://github.com/FreyJo/nosnoc_py) | homotopy from sigma_0 = 1.0, sigma times 0.1 per stage, to sigma_N = 1e-8 or a complementarity residual below comp_tol = 1e-8; IPOPT's tol is comp_tol x 1e-2 | opts lines 51-54, 137; solver lines 185-192, 425-432 |
| SA | D. E. Stewart and M. Anitescu, "Optimal control of systems with discontinuous differential equations", *Numer. Math.* 114 (2010) 653-695 (abstract at [OSTI 972578](https://www.osti.gov/biblio/972578)) | differentiating fixed-step simulations of a discontinuous right-hand side gives gradient errors that do not shrink with the step; after smoothing, gradients converge if the step is "sufficiently small" compared with the smoothing parameter | abstract |
| NK4 | A. Nurkanović, lecture 4, same school ([PDF](https://syscop.de/files/users/armin.nurkanovic/04_direct_methods_nonsmooth.pdf)) | smoothing an example OCP at a fixed step h for decreasing sigma: when h is much larger than sigma the smoothed problem "behaves the same as the nonsmooth" one | slide 6/41 |
| NK6 | A. Nurkanović, lecture 6, same school ([PDF](https://www.syscop.de/files/users/armin.nurkanovic/06_mpec_active_set.pdf)) | regularisation, smoothing and penalty sequences: easy to implement, but ill-conditioned with weaker theory; smoothing (w1 w2 = sigma) has drawbacks against relaxation (w1 w2 <= sigma) | slides 3/38, 7/38 |
| GUM | JCGM 100:2008 (GUM), [BIPM PDF](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf) | an indication of resolution delta x stands for any value in X - delta x/2 to X + delta x/2 | F.2.2.1, p. 64 |

Measured here (derived, JAX 0.6.2): `grad(clip(x, -1, 1))` is 0.5 at x = 1
and 0 at x = 1 + 1e-8; `grad(maximum(x, 0))` is 0.5 at 0; the second
derivative of `maximum(abs(e), 0)**2` at e = 0 is 0.5, not 2.

What the sources share: an interior-point solver needs C2 functions (IPOPT).
Industrial MPC states input limits as constraints, not clips (QB, MW).
Where a kink stays, practice either smooths it with a C-infinity plus function
(CM, GB) or models it with complementarity and a regularisation driven to
zero (BIEGLER, SCH, NOSNOC). No source gives a smoothing width as a number for
a plant model.

## 2. The protocol

Four steps, in this order, for every task.

1. **Rewrite exactly** every kink that has an exact form on the feasible set
   (section 4: T1, T1s, T2, T2p, T3).
2. **Drop guards** that a hard limit or the physics keeps away from their
   edge (T4). The planner's model evaluates each dropped guard's argument
   along every plan, and flags a crossing.
3. **Smooth what remains** (T5) with one form and the width rule (section 3).
4. **Certify** every plan by replaying it on the exact `step_env` and
   recording the gaps (section 3, K8). Certificates are reported only.

Exactness here means: the planner's model and the env agree on every plan
that meets the constraints, so the optimum of the rewritten problem is an
optimum of the env's problem.

## 3. Shared choices, fixed before any run

| # | choice | value | label and ground |
| --- | --- | --- | --- |
| K1 | Input limits | remove each range clip from the planner's model and bound the planned move instead. Keeping an inactive clip is not enough: at an active bound its JAX derivative is 0.5, and 0 once IPOPT's bound relaxation (`MPC_BOUND_RELAX` = 1e-8) steps past it | read: QB p. 755, MW. Derived: the JAX probe; on the feasible set clip(u) = u |
| K2 | Smoothing form | the softplus plus function p_w(x) = w log(1 + exp(x / w)), CM Ex. 2.1, for every smoothed site. Compositions: max(a, b) = b + p_w(a - b); min(a, b) = a - p_w(a - b); clip(z, lo, hi) = z - p_w(z - hi) + p_w(lo - z); abs(x) = p_w(x) + p_w(-x); a switch between two joined pieces blends with the logistic s(x / w), which is p_w's derivative; a max over a set is w log sum exp(x_i / w) | read: CM (C-infinity, gap at most w log 2). Derived: the compositions. Ours: one family for all sites (CM also offers Ex. 2.2). The compressor's `_smooth_clip` is already this form |
| K3 | Conservative side | a smoothed limit must not admit a plan the exact limit forbids: a smoothed max used in a "greater than" limit is shifted down by w ln n, its largest overstatement | derived: w log sum exp(x / w) - w ln n <= max(x). The compressor's own trip variable is the mirror case (PHYSICS.md: conservative by T ln 10) |
| K4 | Scale per site | s_k, in the argument's own units, from the task definition: the clipped interval's span, else the span of the argument's limit band or actuator range | ours |
| K5 | Width | w_k is the largest value in {s_k / 10, s_k / 100, ..., s_k / 1e8} such that, at every probe point, w_k log 2 times the sum over the site's evaluations within one step of abs(d y / d delta_k) is at most q_y / 2 for every tracked output y. delta_k is an additive perturbation of the site's output, differentiated by autodiff; q_y is y's precision floor (the resolution field in `scoring.TRACKED_TERMS`). Probe points: the states the expert visits on training seeds 100 to 109, every 10th step, with its command | read: CM Prop. 2.2 (gap bound), GUM F.2.2.1 (half a resolution step is unresolvable). Derived: the bound. Ours: the probe set, the ladder, per step rather than per horizon |
| K6 | Discretisation check | for a site whose argument is integrated inside the step, record the argument's change over one integrator substep at the probe points. If w_k is below it, flag the site. The width is not changed | read: SA, NK4 (the step must be small against the width). Ours: accuracy wins, the flag is reported |
| K7 | Continuation | cold solves only: solve at s_k / 10, then at each tenth down to w_k, warm-starting each from the last. Warm (shifted) solves run at w_k | read: CM Alg. 4.1 and GB (continuation on the smoothing parameter); NOSNOC (factor 0.1). Ours: cold solves only |
| K8 | Certificate | per solve: the planned and the replayed horizon reward and their relative gap; the largest abs(y_plan - y_replay) / q_y over the horizon per tracked output; any trip in the replay; every dropped guard whose argument crossed its edge along the plan; plus the solver certificate of audit decision 6. Replay: the planned moves, held per T_c, on `step_env` with `plan_params` from the same start | ours; derived from rule 5 (certified by convergence measures, never by score) |
| K9 | Guard promotion | a T4 site flagged on any training-seed run moves to T5 for that task before any scored run. This is the only change a run may cause, and it is mechanical | ours |
| K10 | Fixed per task | one width per site per task, written to the task's oracle fingerprint before the re-record | ours |

Limitation of K5: it bounds one step, not the accumulation over a horizon of
up to several hundred steps. K8 reports the accumulated gap; nothing is tuned
on it.

## 4. Classes and treatments

| class | what | treatment |
| --- | --- | --- |
| A, actuator limit | a clip of a command to its range, a state-dependent command limit, an actuator rate limit | **T1**: bound the move (exact). **T1s**: a state-dependent limit as a nonlinear constraint on the move (exact when every clipped outcome is reachable by an unclipped command). Redundant clips: **T4**. Rate limits that act inside the step: **T5** |
| R, reward kink | abs and max in tracking and running costs, hinge bands, a min over paths or turn directions | **T2**: an epigraph slack (t >= x, t >= -x, cost t; or s >= q - c, s >= 0), or the identity e^2 where e_tol = 0 and p = 2 (exact). **T2p**: a distance to a path, which is a min, becomes a path parameter chosen by the planner (exact: a joint minimum). A min over turn directions has neither form: **T5** |
| P, physics kink | saturations of states, piecewise models, validity guards | guard off its edge: **T4**. Reachable: **T5** |
| D, discrete event | trip tests, the trip and reset kernel, in-step freezes, implicit solves with no derivative, angle wraps | **T3**: a hard limit, with elastic slack at the trip cost (accepted decision); an "all of" limit (min separation, max error, max over cells from above) becomes one constraint per element; an implicit solve becomes an algebraic variable with its residual as an equality. Unreachable wraps: **T4** |
| X, exogenous | schedules, clocks, clipped disturbances | not a kink for the planner: no derivative with respect to the plan. Precomputed along the horizon; noise zeroed by `plan_params` |

Two notes that apply to every task:

- The trip branch drops out of a feasible plan (T3), and with it the reset
  draw that `base.failure_kernel` evaluates on every step (base.py:99). On
  the kiln that reset marches the axial profile 4000 times
  (cement_kiln/env.py:653, called from env_jax.py:106).
- `reward.select` (reward.py:76) traces the version-1 reward beside the
  current one inside `jnp.where`. The planner writes the current reward only,
  which removes the where-NaN hazard of JAXFAQ (derived; not counted as a
  site).

### Correction to the draft: no rate limit acts between steps

> Amended 2026-11-04 (Amendment 1, at the end): the bound that makes these
> rate limits exact is the travel over the hold, not over one substep (T1r).

The draft turns rate limits into constraints on planned moves. In the 24
tasks no rate limit acts between steps. All four act inside the step, so the
planner's move does not reach them:

- compressor_surge drive and valve, per substep (env.py:511, 514);
- reactor rod, per physics sub-step, ten per step (env.py:586);
- wind_turbine pitch rate, inside the ODE right-hand side (env.py:309).

Bounding the command to one substep's travel would make them exact, but
would forbid moves the env allows, which is not exact. They are T5. The
compressor's oracle already smooths its pair (experts.py:711-724).

## 5. Inventory

Counting rule: one site per source line on the planning path (`step_env`,
`compute_reward`, `failure_kernel`), shared lines counted once. Reset code,
observations and the version-1 rewards are left out. All paths are under
`src/target_gym/`.

### 5.1 Shared sites

| site | kink | class | treatment | tasks |
| --- | --- | --- | --- | --- |
| utils.py:115 | clip of the raw action to [-1, 1] | A | T1 | 14: every task outside the aircraft |
| reward.py:31 | abs and max in `tracking_cost` | R | T2: e^2 on the 20 tasks with e_tol = 0 and p = 2; epigraph on reactor, wind_turbine, battery (p = 1) | 23 (not hvac) |
| reward.py:42 | max(q - c_hold, 0) in `running_cost` | R | T2 epigraph | 10: plane x3, ph, distillation, glass, kiln, boiler, wind, compressor |
| reward.py:60-61 | `with_trip`: costs zeroed, trip charged | D | T3 | 24 |
| base.py:93-101 | `failure_kernel`: trip test, reset draw, select | D | T3 | 24 |

### 5.2 Aircraft (10 tasks)

Shared by all ten (plane/dynamics.py, imported by plane3d, patrol, formation):

| site | kink | class | treatment |
| --- | --- | --- | --- |
| :154 | throttle clip [1e-8, 1] | A | T1 (raw throttle bound at -1 + 2e-8) |
| :188 | max(M - M_crit, 0)^2 in thrust | P | T4, past M_crit (checklist: 308 m/s seam unreachable; 251.5 and 285.8 m/s reached) |
| :211 | altitude clip [-50 km, 40 km] | P | T4, by the altitude band [0, 12191 m] |
| :235 | incidence wrap, mod 360 | D | T4, wraps at +-180 deg |
| :266 | CL clip [CL_min, CL_max], the stall | P | T5 |
| :272 | sqrt(max(1e-6, 1 - M^2)) | P | T4, M < 1 |
| :287, :292, :297, :302 | shock-stall clip and max, CL cap and floor switches at M_crit, Mach-limited CL clip | P | T4, past M_crit (4 sites) |
| :304 | drag-rise switch at M_crit (C1) | P | T4, past M_crit |
| :330 | abs(incidence) in the separation blend | P | T5; its slope jump at 0 deg is 3.3e-12 per deg (derived from aoa_stall 15, width 3), so any width passes K5 |
| :344, :347 | CL clip +-2, CD clip [0, 2.1] | P | T4, post-stall guards (2 sites) |
| :372 | sound speed switch at 11 km | P | T5: the tropopause lies inside the altitude band |

2D only (plane, plane_energy, plane_sine): :354 flight-path arctan2 and :362
incidence wrap (D, T4, unreachable cuts); :462 max(V, 1e-3) (P, T4); :522
acceleration clip (P, T4, promoted to T5 if K9 flags it); plane/env.py:634
fuel clip (P, T4, tanks never run dry in an episode, checked by K8);
plane/env.py:375 altitude trip (D, T3); plane/env.py:406 running cost on
abs(speed - target) (R, T2, with speed as an algebraic variable).

3D (plane3d x4, patrol x2, formation): plane3d/dynamics.py:66 incidence wrap
(D, T4); :209 and :256 speed guards (P, T4); :299 acceleration clip (P, T4,
K9); plane3d/env.py:939 fuel clip (P, T4); plane3d/env.py:932 heading by
arctan2, with its cut at +-180 deg (D, T3: the planner forms the heading
error from the velocity components and the target direction).

Per task:

| task | sites | class | treatment |
| --- | --- | --- | --- |
| plane3d_heading | plane3d/env.py:992 wrapped heading error (a min over turn directions, kink at 180 deg); :345 altitude trip | R; D | T5; T3 |
| plane3d_circle | plane3d/env.py:428 sqrt distance, kink only at the centre; :345 | R; D | T4; T3 |
| plane3d_racetrack | plane3d/env.py:457-459 abs (kink on the centre line), :460 leg or arc switch (C1); :345 | R; D | T2p: path parameter on the stadium curve. Its leg-arc joins stay C1 (curvature steps from 0 to 1/r), left as is (ours); T3 |
| plane3d_figure8 | plane3d/env.py:625 argmin over 400 samples, :640 chord clip, :650 argmin of two; :345 | R; D | T2p on the analytic lemniscate the samples come from; the replay gap includes the polyline's sagitta, sub-metre by the env's own comment; T3 |
| patrol, patrol_bearing_only | patrol/env.py:337 wrapped heading error; :300-315 trips (both altitude bands, separation <= 25 m, slot error >= 1500 m). Exogenous: the lead's legs (:73-74, :550) | R; D | T5; T3 (separation is nonconvex, so a local optimum) |
| formation | formation/env.py:243, :252 wrapped heading errors; :204-206 min pairwise separation; :222 max slot error; :212 trips. The lead is planned here, not exogenous | R; D | T5; T3 with one constraint per pair and per wingman |

Exogenous on the aircraft: plane/dynamics.py:39 (gust model switch) and the
altitude schedule plane/env.py:518-540 (float32 clock, floor, mod, clip).

### 5.3 Process tasks

| task | site | kink | class | treatment |
| --- | --- | --- | --- | --- |
| cstr | pc_gym/cstr/env.py:168 | trip test (documented unreachable) | D | T3, nothing binds |
| first_order | pc_gym/first_order/env.py:117 | abs(x) >= 3 | D | T3, never binds |
| four_tank | pc_gym/four_tank/env.py:204 (used at :162-165) | sqrt(max(h, 0)) with a where guard | P | T4: the trip band keeps h >= 0.05 |
| | :243 | level band | D | T3 |
| ph_neutralization | pc_gym/ph_neutralization/env.py:245 | pH by 20 bisection steps: derivative zero almost everywhere | D | T3: pH as an algebraic variable with the charge balance as an equality. The env's midpoint differs by at most 14 / 2^21 = 6.7e-6 pH (derived). `experts.planner_pH` already gives this derivative |
| | :332 | trip test (documented unreachable) | D | T3 |
| distillation | pc_gym/distillation/env.py:228 | D = clip(V - L, 0.1, 0.9); V - L spans -0.3 to 1.3, so both edges are reachable | P | T5 (decision S8) |
| | :311 | compositions clipped to [0, 1] | P | T4: the env notes a converged solution never leaves [0, 1] |
| | :355 | yD and xB limits | D | T3 |
| glass_furnace | glass_furnace/env.py:522 | gas temperature clip | P | T4 |
| | :543, :806 | batch mass max(., 0): the blanket runs out (checklist: a real kink) | P | T5 (2 sites) |
| | :562, :758 | batch coverage clip [0, 1] | P | T5 (2 sites) |
| | :593 | heat to batch max(., 0) | P | T4, K9 |
| | :855 | crown, melt and working-end bands | D | T3 |
| unstable_cstr | pc_gym/unstable_cstr/env.py:326 | T < T_trip, plus validity guards | D | T3 |

Exogenous: ph :290 (buffer flow clip); distillation :294 (feed clip) and
:244 (a static stage switch, not a kink); glass :418, :425, :469, :654-655,
:668-715, :734, :747 (reversal and charge schedules, regenerator side, firing
interruption, pull disturbance); unstable_cstr :220 (setpoint blocks).

### 5.4 Industrial and energy tasks

| task | site | kink | class | treatment |
| --- | --- | --- | --- | --- |
| reactor | reactor/env.py:586 | rod rate limit per sub-step, ten per step | A | T5 |
| | :587 | rod range clip, redundant (the demand is already in range) | A | T4 |
| | :704 | rod wear abs(command - position) | R | T2 |
| | :676 | flux, fuel and coolant temperature bands | D | T3 |
| | reactor/env_jax.py:113-119 | state frozen at the first sub-step that trips | D | T3 at every sub-step state |
| hvac | hvac/env.py:537, :538 | comfort hinge max(abs(e) - tol, 0) and setback shortfall max(-e, 0), both squared (C1) | R | T2: slack t >= e - tol, t >= -e - tol, t >= 0, cost t^2 |
| | :523 | air temperature band, restart in place | D | T3 |
| cement_kiln | cement_kiln/env.py:311, :361, :364, :394, :395, :407, :409, :410, :475, :476, :477, :478 | rpm, gas-march, fraction and temperature guards | P | T4 (12 sites). By limit: rpm >= 2 > 0.1; every cell below T_bz_max = 1900 K < 2400 K. By reach: the gas-march clips and the lower edges |
| | :548 (T_bz = max over cells, :556) | overheat: max <= T_bz_max; cold: max >= T_bz_min or lime >= lime_max | D | T3 per cell for the upper limit. The cold limit is a disjunction (some cell hot enough): conservative smoothed max, K2 and K3 |
| boiler_drum | boiler_drum/env.py:289, :301-302, :329, :334, :360, :369, :378, :429, :430, :431, :432 | void fraction, circulation, pressure, enthalpy, determinant, volume and mass guards | P | T4 (11 sites). By limit: the pressure clips [20, 200] bar against the band [65, 105]; the drum volume by the level band |
| | :379 | condensation flow max(Q - q_dc subcool, 0) | P | T4, K9 (reach not known) |
| | :485-487 | abs(level) >= 0.25 m; pressure band | D | T3: two bounds on the level |
| wind_turbine | energy/wind_turbine/env.py:309 | pitch rate clip inside the right-hand side | A | T5 |
| | :356, :357 | pitch and torque clips after the step | A | T4: a lag toward an in-range command stays in range |
| | :271 | cp clip [0, 0.593]; the floor is reachable at high pitch | P | T5 |
| | :262, :264, :276, :277, :355 | tip-speed and speed guards | P | T4 (5 sites) |
| | :421 | activity abs(pitch command - pitch) | R | T2 |
| | :399 | rotor speed band | D | T3 |
| battery | energy/battery/env.py:313 | command clipped to 0.999 of the deliverable power, which falls with state of charge (checklist seam at soc 0.115) | A | T1s: every clipped outcome is reachable by commanding it, so the constraint loses nothing (derived). Imposed at the step's nodes; interior RK stages checked by K8 |
| | :283, :314, :378 | state-of-charge and discriminant guards | P | T4: the band [0.05, 0.95] and T1s keep them off their edges |
| | :336 | abs(current) in the fade rate | R | T2: fade enters only the reward, through a non-decreasing max, so the slack is exact in value (derived; `q_loss` feeds nothing) |
| | :465 | max(fade - c_hold, 0) | R | T2 |
| | :436 | state of charge, cell voltage, cell temperature | D | T3; the voltage needs an algebraic variable |
| compressor_surge | compressor_surge/env.py:511, :514 | drive and valve rate limits per substep | A | T5. Today's `MPC_CLIP_SMOOTHING` = 0.1 is replaced by K5 |
| | :314 | characteristic switches to its tangent at phi_join 0.78 (C1); the plant reaches 0.7775 | P | T4, K9 |
| | :610 | surge, the soft-min of Phi below 2W | D | T3 (already a constraint) |

Exogenous: reactor :630-634 (demand); hvac :289, :311-312, :319, :327, :437,
:539-540 (clock, sun, occupancy, setpoint, weather); kiln :462 (feed);
boiler :418 (steam demand); wind :338 (wind); battery :276-277, :390
(dispatch blocks); compressor :363-390 (demand schedule).

Not kinks: the compressor's check valve (env.py:317-332) and its trip
variable (:547) are smooth by the task's own definition (`check_width`,
`soft_min_temperature`). Their widths are task constants, outside this
protocol. The compressor's `MPC_POWER_SMOOTHING` = 0.05 smooths a reward
kink, so T2 replaces it with an exact slack.

### 5.5 Counts

Per task, sites by class (shared sites included):

| task | A | R | P | D | total | T5 sites |
| --- | --- | --- | --- | --- | --- | --- |
| plane, plane_energy, plane_sine | 1 | 3 | 16 | 6 | 26 | 3 (+1 by K9) |
| plane3d_heading | 1 | 2 | 17 | 6 | 26 | 4 (+1) |
| plane3d_circle | 1 | 2 | 17 | 6 | 26 | 3 (+1) |
| plane3d_racetrack | 1 | 3 | 17 | 6 | 27 | 3 (+1) |
| plane3d_figure8 | 1 | 4 | 17 | 6 | 28 | 3 (+1) |
| patrol, patrol_bearing_only | 1 | 2 | 17 | 6 | 26 | 4 (+1) |
| formation | 1 | 3 | 17 | 8 | 29 | 5 (+1) |
| cstr | 1 | 1 | 0 | 3 | 5 | 0 |
| first_order | 1 | 1 | 0 | 3 | 5 | 0 |
| four_tank | 1 | 1 | 1 | 3 | 6 | 0 |
| ph_neutralization | 1 | 2 | 0 | 4 | 7 | 0 |
| distillation | 1 | 2 | 2 | 3 | 8 | 1 |
| glass_furnace | 1 | 2 | 6 | 3 | 12 | 4 (+1) |
| reactor | 3 | 2 | 0 | 4 | 9 | 1 |
| hvac | 1 | 2 | 0 | 3 | 6 | 0 |
| cement_kiln | 1 | 2 | 12 | 3 | 18 | 0, plus the cold limit |
| boiler_drum | 1 | 2 | 12 | 3 | 18 | 0 (+1) |
| wind_turbine | 4 | 3 | 6 | 3 | 16 | 2 |
| battery | 2 | 3 | 3 | 3 | 11 | 0 |
| unstable_cstr | 1 | 1 | 0 | 3 | 5 | 0 |
| compressor_surge | 3 | 2 | 1 | 3 | 9 | 2 |

Unique sites over the 24 tasks: **121**.

| class | sites | exact rewrite (T1, T1s, T2, T2p, T3) | guard dropped (T4) | T4, promoted if flagged (K9) | smoothed (T5) |
| --- | --- | --- | --- | --- | --- |
| A, actuator limit | 10 | 3 | 3 | 0 | 4 (all in-step rate limits) |
| R, reward kink | 19 | 14 | 1 | 0 | 4 (heading wraps) |
| P, physics kink | 63 | 0 | 50 | 4 | 9 |
| D, discrete event | 29 | 25 | 4 | 0 | 0 (plus the kiln's cold limit) |
| **total** | **121** | **42** | **58** | **4** | **17** |

About 30 further sites are exogenous (X) and need no treatment. The audit's
counts (29 on the 2D aircraft, 22 on the furnace, 18 on the boiler) used a
different rule and are not reconciled here.

## 6. Float64

The probe stepped all 24 tasks once under x64 from a reset. 23 return
float64 states and rewards. The reactor fails: `jax.lax.scan` raises a
carry dtype mismatch, because its reset builds float32 state fields
(reactor/env_jax.py:234-241) and its sub-step loop seeds float32 sums
(:125, :129), while the step returns float64. Without x64 it runs, in
float32. The IPM's tolerance of 1e-8 on its scaled error is below float32's
1.2e-7, so the reactor cannot join the shared solver until those seeds take
the state's dtype (decision S6).

Explicit float32 elsewhere, on exogenous values only: plane/env.py:518 and
glass_furnace/env.py:670, :704 (schedule clocks), plane/env.py:473
(observation), formation/env.py:166 (a constant of small integers),
reactor/env.py:630 (noise, zeroed when planning). The planner computes these
exactly as the env does, so they need no change (derived).

## 7. What the first probe suggests

The IPOPT port, on the raw step and reward, converged on hvac and capped or
failed on battery, first_order and glass_furnace. The inventory gives each a
candidate cause. None is verified.

- **battery**: tracking with p = 1 puts the optimum (zero error) on the
  kink of abs(e). The fade carries abs(current) at every charge-discharge
  switch, inside a max at c_hold, and the deliverable-power clip binds at
  low charge. All four sites have exact rewrites (T2, T1s).
- **glass_furnace**: the running cost max(fuel - c_hold, 0) has its kink at
  c_hold, the shipped oracle's own mean consumption, which is where an
  optimum should sit. The batch blanket adds two reachable clips (T5).
- **first_order**: a linear plant whose only kinks are the input clip and
  the reward's e^2 written as max(abs(e), 0)^2. With no input cost the
  optimum starts on the input bound (u_max = 2, target near 1, tau = 0.5 s),
  where JAX gives the clip a derivative of 0.5, and 0 just past it. T1 and
  T2 remove both.
- **hvac**: its reward kinks are C1 squared hinges, its energy is linear,
  and it has no p = 1 term and no c_hold kink. It also clips its input, so
  the clip alone does not explain first_order.

Prior evidence in the repository: with the CasADi build, the compressor's
exact substep clips hit IPOPT's 150-iteration cap on 4 of 48 solves; rounded
at 0.1 or 0.01, none did (experts.py:713-724).

The cheapest test of the hypothesis is the same probe on first_order,
battery and glass_furnace with T1, T1s and T2 only, no smoothing (decision
S7).

## 8. Where the sources disagree

1. **Smooth, or model exactly with complementarity.** CM and GB smooth with
   one parameter and continuation. BIEGLER and SCH model piecewise functions
   with complementarity pairs and drive a regularisation to zero. NK6 finds
   smoothing has drawbacks against relaxation. The protocol avoids the
   question where it can: 100 of 121 sites have an exact rewrite or a guard
   that a limit or the plant keeps off its edge.
2. **How small the width.** Accuracy wants it small (CM: the gap is at most
   w log 2; NOSNOC drives sigma to 1e-8). Conditioning and discretisation
   want it large (NK6: ill-conditioning; SA and NK4: below the integrator's
   step the smoothed model behaves like the nonsmooth one). K5 takes the
   largest width an instrument could not tell from the exact model, one
   step at a time.
3. **The continuation endpoint.** CM caps the parameter at the target
   accuracy; NOSNOC continues to the complementarity tolerance. The draft's
   first width option (continuation to the solver tolerance) is NOSNOC's
   practice for complementarity models, not for smoothed ones.
4. **Hard rate limits.** QB: every product allows hard rate limits. MW:
   increment bounds are soft by default because hard ones can conflict with
   hard bounds. Here the conflict cannot arise, since the actuator's position
   is the env's state and stays in range (derived), and no rate limit acts
   between steps anyway (section 4).
5. **No width as a number.** No source gives one for a plant model. The
   compressor's 0.1 and 0.05 are labelled ours in its own code. K5 is ours,
   built from read pieces (CM's bound, GUM's half resolution).

## 9. Decisions for the owner

Recommended option first.

| # | decision | options |
| --- | --- | --- |
| S1 | Width rule | K5, the resolution rule: largest width whose worst one-step effect is under half the output's resolution (ours, from CM and GUM). Or the draft's continuation to the solver tolerance via complementarity (BIEGLER, SCH, NOSNOC: exact in the limit, but needs an MPCC-aware IPM step as in Raghunathan and Biegler, about nine solves per cold solve, ill-conditioned, NK6). Or fixed relative widths as the compressor's (no source) |
| S2 | Form | softplus, CM Ex. 2.1, everywhere (matches the compressor's clip). Or Chen-Harker-Kanzow-Smale, CM Ex. 2.2 (matches its power smoothing) |
| S3 | Guards kept off their edge only by measured reach | drop them (T4), check each plan's arguments, promote on a flag (K9). Or smooth every guard (17 smoothed sites become 79; more smoothing error, no flags) |
| S4 | Heading wraps (4 sites) | smooth the min over turn directions (T5). Or keep each solve's starting branch (exact unless a plan crosses 180 deg error, which a reversal does) |
| S5 | Kiln cold limit | conservative smoothed max over cells (K3). Or a constraint on the hottest cell at the solve's start (not exact). Or keep the kiln outside the rule until its derivatives are fixed (audit decision 10) |
| S6 | Reactor dtype | make reset and loop seeds take the state's dtype before the re-record (changes reactor/env_jax.py, so its stamps and baselines go stale). Or plan the reactor in float32 at a looser tolerance (breaks the one shared tolerance) |
| S7 | Verify first | rerun the probe on first_order, battery and glass_furnace with the exact rewrites only, before building T5 and K5. Or build the whole protocol now |
| S8 | Distillation distillate clip | smooth it as the model is written (T5). Or bound V - L to [0.1, 0.9] as a derived input limit (keeps plans physical, but forbids moves the env allows) |
| S9 | Racetrack leg-arc joins | leave them C1. Or round the path's curvature step (changes the scored path) |
| S10 | Compressor | bring it under the rule: T2 for its power excess, K5 for its substep clips, which re-records its oracle row. Or keep its widths as a stated exception |

## 10. Source check

Each key source was fetched twice on 2026-11-03, the second time by a
separate download.

- **PDFs** (CM, QB, GUM, NK4, NK5, NK6): the cited phrases were found in
  pdftotext output of both copies. CM Prop. 2.2 and Ex. 2.1 to 2.3 were also
  read as page images, which confirmed the bound -D2 beta <= p - x_+ <= D1
  beta and both forms.
- **Web pages** (IPOPT, MW, JAXFAQ, SA at OSTI, CASADI): fetched twice, the
  phrase found both times.
- **Code** (NOSNOC): read through the GitHub API and again as raw files at
  the pinned commit.
- **Abstracts** (GB, SCH, Raghunathan and Biegler): OpenAlex, twice.

Not read, and not relied on beyond what is stated:

- **Biegler 2010.** SIAM's pages refused the fetch (403). Only the secondary
  account in NK5 and the abstracts of GB and Raghunathan and Biegler 2005.
  The chapter's section titles and any width it may give are unknown.
- **Betts 2010.** No online text. Google Books answered with a bot check,
  which was not attempted. Betts is not cited for anything, so how he treats
  discontinuities is an open check.
- **Nocedal and Wright.** No legitimate online copy found. The epigraph
  rewrite is labelled derived; it does not rest on them.
- **Maciejowski 2002.** Not online. QB and MW carry the input-constraint
  practice.
- **Boyd and Vandenberghe.** The author's PDF failed on a TLS certificate
  error.
- **CasADi documentation.** No FAQ entry on nonsmooth functions was found;
  only the maintainer's post above.
- **Stewart and Anitescu**, full text: paywalled; the abstract only. Its
  setting is a discontinuous right-hand side and Euler steps; applying it to
  the continuous kinks here is a reading, not a statement of the paper.

## Amendment 1, 2026-11-04: rate limits, and a back-off on every limit

C1 rebuilt the compressor's oracle under this protocol. It lost to its
expert by 20 to 60 times on training seeds 100 and 101, where main's oracle
beat it on all ten test seeds. The owner accepted both changes below on
2026-11-04, and with them a records gate: an oracle under the rule beats its
expert on every recorded seed.

**T1r, a rate limit's reach.** Section 4 bounded a command to one
substep's travel, found that it forbids moves the env allows, and smoothed
the four in-step rate limits instead (T5). The exact bound is the travel
over the hold, the steps a move is held. A command beyond it moves the
actuator at full rate in every substep, exactly as a command at it does, so
bounding each command to that reach from the actuator's position forbids no
trajectory. Without it, a command past the reach changes nothing in the
model, and IPOPT stopped at plans that did nothing: at step 50 of seed 100,
a converged plan (KKT error 6e-11) earned -3.07e6 over its horizon, worse
than holding still (-3.01e6), while the expert's moves earned -1.46e6.
Main's NLP had these rows ("the rate reach", one step's travel), as QB's
hard rate-of-change limits and MW's increment bounds put them. T5 still
smooths the corner left inside the reach, where the actuator arrives
mid-step. Sites: the compressor's drive and valve (C1); the reactor's rod
and the wind turbine's pitch take the same treatment when C2 rebuilds them.
For the pitch, a rate limit inside the right-hand side, the reach is where
the rate stays saturated for the whole hold, derived with that rebuild.

**K11, a limit's back-off.** With T1r the compressor tracked exactly but
held its trip variable on the trip line. Two things then tripped it. On the
noise-free plant the plan passed the line by the solver's tolerance, since
the trip test is a strict comparison. On the noisy plant one innovation of
the demand deviation moved the next step's trip variable, which the
noise-free plan cannot foresee: 7.9e-4 at one sigma, against a smoothing gap
of 6.7e-5. Each hard limit is therefore planned inside its bound by the
planner's worst one-step error on the limited quantity q, measured at K5's
probe points, as the sum of:

- the smoothing's gap: sum over sites of w log 2 times the sum of
  |dq / d delta| over the site's evaluations (K5's bound, on q);
- the noise: z s max |dq / d d|, s the standard deviation of one innovation
  of the plant's noise-driven deviation d, and z the normal quantile of a
  1 % risk of any such trip over the ten recorded seeds, split evenly over
  their steps (Boole's inequality). The 1 % is ours; a tenfold change moves
  the compressor's back-off by about 10 %;
- the solver's tolerance, (bound_relax_factor + tol) max(1, |bound|).

This is constraint tightening by a bound on the model's error, as robust
MPC does (Mayne, Seron and Rakovic, Automatica 41 (2005); risk split by
Boole's inequality as in Blackmore and Ono, AIAA GNC 2009; neither re-read
here). A task with no smoothing and no noise keeps the solver's term alone,
and a limit with no inside (lo = hi, as a line-out) is not backed off. The
values are in `limit_backoffs` in each task's widths file: 0.0040 on the
compressor's trip line, 0.055 K on the unstable CSTR's temperature band.

**Scaling.** The builder scales variables by typical magnitudes. Under T1r a
move's typical size is its reach, not its range, so a task with move rows
gives it (`move_scale`). On the compressor this took 174 capped and 15
failed solves down to 52 and 0, seed 100, with median iterations 28 to 7.

**Capped solves stay applied.** A solve the cap stops applies its last
iterate, as before. Keeping the shifted plan instead was tried and rejected
on evidence: it froze the valve through a run of capped solves as demand
fell, and the plant tripped.

## Amendment 2, 2026-11-05: what C2 found

C2 rebuilt hvac, the battery, the boiler, the wind turbine, the glass furnace
and the kiln under this protocol. Each change below was forced by something
measured, and each is exact or derived; none was chosen by score.

**T1r on the wind pitch.** A rate limit inside the right-hand side acts on
(cmd - pitch) / tau, so the actuator closes the gap at the rate limit until the
gap is rate x tau. A command rate x (hold + tau) from the pitch keeps it at
full rate over the whole hold, as any command past it does. At the 0.25 s step
that reach is 3.6 deg.

**K5 on a site inside the right-hand side.** The wind's two sites and the
glass blanket are evaluated at every RK stage. Their delta enters all of a
step's evaluations at once, with evaluations = 1. It bounds the sum by the
step's total sensitivity, which equals K5's sum where the sensitivities share
their sign over one step, as they do here (derived).

**The glass blanket is one site.** Sections 5.3 and 5.5 count four. The
right-hand side and the flame read the blanket through its floor and its
coverage clip (env.py:543, :562, :758), so one softplus on the mass they read
smooths all three: a mass inside (0, m_batch_full) passes the env's guards
exactly. The fourth, the stored floor at env.py:806, stays a dropped guard
(T4), checked on every plan: a softplus on the state would add its gap every
step, where in the right-hand side it costs it once. K5 gives width 3900 kg,
the ladder's first rung. The expert's blanket stays at 0.22 to 0.34 of full.

**The kiln's fractions keep their clip.** Section 5.4 dropped the clip of alpha
and lime after the step (env.py:477, :478) as T4. It is active in every hot
cell: there the reaction's rate times the step is far past RK4's stability, one
step multiplies a fraction's error by about 1.3e6, and only the clip bounds it.
K9 would move it to T5, but no softplus can stand in for a clip that holds the
state on its bound against that gain: K5's perturbation measures a first-order
effect the clip removes. So the model keeps the env's clip, exact, kink and
all. Without it the replay gap was 7e-5, cold solves failed at their first
iteration, and the first 11 steps of a seed applied held moves. With it the
replay matches to rounding and solves converge in 10 to 30 iterations.

**The battery's fade.** Section 5.4 puts a slack on abs(current). One hinge per
sign of the current is simpler and exact while the calendar rate stays under
c_hold, which makes both hinges' sum max(calendar + cycling |I| - c_hold, 0).
That is a dropped guard (T4), and it is checked on every plan.

**The kiln's soft max.** The cold limit's site moves no tracked output, so K5
takes the ladder's first rung, 28 K. K3's shift already keeps it conservative,
by at most 28 ln 16 = 78 K against a burning zone 134 K above the limit, so it
takes no K11 gap.
