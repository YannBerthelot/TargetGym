# Oracle solver: the Hessian of the shared interior-point method

Specification of 2026-11-03, for owner review. It covers the Hessian that the
shared oracle solver uses: the compressor's IPOPT port,
`src/target_gym/compressor_surge/experts_ipm.py`, lifted to serve all 24
oracles (audit decision 5). Nothing here changes the repository or records
anything.

**Status.** The owner chose a Gauss-Newton Hessian earlier today, from the
first probe. Later probes the same day (section 9) found that Gauss-Newton
does not converge on plane3d_circle, while the exact Hessian does, and that
an exact Hessian taken forward-over-forward costs a fraction of what
`jax.hessian` cost. So this spec sets out both forms side by side and
recommends one. The choice is the owner's (decision G1).

**Owner decisions, 2026-11-03.**

- **The Hessian:** G1, form A, the exact Lagrangian Hessian, for all 24
  tasks; G2, computed forward-over-forward. This replaces the Gauss-Newton
  choice taken earlier the same day, which the probe showed does not
  converge on the aircraft. G6 falls away with it.
- **G3 to G5 and G7, as recommended:**
  - G3: indefinite curvature handled by the port's IPOPT inertia correction;
  - G4: the 150-iteration cap kept, and capped solves recorded in the
    certificate;
  - G5: one algebraic variable per kinked term per visited state, with
    `stage_kkt_solve` extended to carry them;
  - G7: the build starts by measuring glass_furnace with its running-cost
    kink rewritten, and reports each task's per-episode cost before the
    re-record.

The owner's rule applies throughout: one constrained solver for every task,
run to convergence, certified by its KKT residual, never by the score. It
builds on the accepted decisions: the episode reward as objective, hard trip
limits with elastic slack priced at the trip cost, Cooper's sample time T_c
with the move held over each T_c, and the smoothing protocol's exact
rewrites (T1 to T3) and widths (K5).

Labels, as in the horizon and smoothing protocols:

- **read**: a source states it.
- **derived**: computed from a read value, or a direct consequence of an
  accepted decision, of the code, or of a measurement in section 9.
- **ours**: no source gives it, so it is fixed here, before any run.

Notation. Stage k of the multiple-shooting NLP has the variables
v_k = (x_k, u_k), n_v of them (29 on plane3d_circle, 27 on glass_furnace).
One stage runs b task steps with u_k held (b = T_c in steps) and visits
states s_1 .. s_b. F_k(v_k) = s_b is the shooting map. The stage cost is
l_k(v_k) = sum over j of phi(s_j), with phi = -compute_reward. lambda are
the multipliers of the shooting equalities.

## 1. What the sources say

| id | source | what it says | where |
| --- | --- | --- | --- |
| BP84 | H. G. Bock and K. J. Plitt, "A multiple shooting algorithm for direct solution of optimal control problems", 9th IFAC World Congress, 1984, *IFAC Proc. Vol.* 17(2) 1603-1608 | not read (see section 12). As relayed by DIEHL02 and RMD: it introduced the partitioned (block-wise) high-rank Hessian updates, and the exact Hessian of a multiple-shooting Lagrangian is block diagonal | DIEHL02 p. 580; RMD p. 556 |
| BOCK83 | H. G. Bock, "Recent advances in parameter identification techniques for ODE", Birkhauser 1983, pp. 95-121 | not read. RMD and MBD21 credit it with the constrained Gauss-Newton method | RMD p. 549; MBD21 p. 72 |
| DIEHL02 | M. Diehl, H. G. Bock, J. P. Schloeder, R. Findeisen, Z. Nagy, F. Allgoewer, *J. Process Control* 12 (2002) 577-585 ([PDF](https://lirias.kuleuven.be/retrieve/80043)) | three Hessian versions for multiple shooting: (a) the exact Hessian, Newton's method, with "excellent local convergence", but it "may become indefinite" far from the solution, so globalisation is needed; (b) Bock and Plitt's partitioned high-rank updates; (c) constrained Gauss-Newton, an excellent approximation if the residual is sufficiently small (error O(norm of the residual)), "especially recommended for tracking problems"; least-squares integrals over an interval get their Gauss-Newton matrix from adapted integrators | section 3.3, p. 580 |
| DIEHL05 | M. Diehl, H. G. Bock, J. P. Schloeder, *SIAM J. Control Optim.* 43(5) (2005) 1714-1736 ([PDF](https://lirias.kuleuven.be/retrieve/80044)) | the Lagrangian Hessian is block diagonal per stage and the KKT system is solved by a Riccati recursion; the constrained Gauss-Newton blocks are the residual Jacobian's Gram matrix and "do not depend on" the multipliers; a Newton-type iteration contracts if the Hessian error, measured through the inverse KKT matrix, stays below kappa < 1 | section 2.1 p. 1718; 2.2 pp. 1718-1719; Theorem 4.1 p. 1720 |
| MBD21 | F. Messerer, K. Baumgaertner, M. Diehl, "Survey of sequential convex programming and generalized Gauss-Newton methods", *ESAIM: Proc. Surveys* 71 (2021) 64-88 ([PDF](https://optimization-online.org/wp-content/uploads/2020/04/7733.pdf)) | GGN splits the objective into a convex outer function of a nonlinear inner function; B = J' Hess(phi) J is PSD "due to convexity" of phi; the neglected term is small if the inner curvature or the outer slope at the residual is small. Constrained GGN (CGGN): GGN Hessian on the objective, constraints linearized, constraint curvature thrown away. The local rate is linear, the smallest alpha with -alpha B <= E <= alpha B on the reduced space; B >= half the true reduced Hessian is necessary for convergence; quadratic when the solution is fully determined by the active constraints; the methods can diverge from some local minima ("desirable divergence"). In an epigraph reformulation, keeping the slack's own lower bound significantly improves convergence | p. 66; eq. (8) p. 69; Thm 2.2 p. 70; section 3.2 p. 72; Ex. 3.2 p. 74; Thm 4.5 p. 78; Thm 4.9 p. 82; section 5 |
| RMD | Rawlings, Mayne and Diehl, *Model Predictive Control*, 2nd ed., 6th printing, ch. 8 ([PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-6th-printing.pdf)) | Newton-type contraction: rate kappa set by the Hessian error, quadratic for exact Newton; the Gauss-Newton Hessian of a sum of squares drops the residual curvature and the constraint curvature lambda' Hess G, is PSD, multiplier-free, good "for problems with small residuals", and converges only linearly; Hessian approximations still use the exact constraint Jacobian; Gauss-Newton from integrator sensitivities on a fine grid inside each interval; exact Hessian block diagonal (Bock and Plitt, 1984); real-time iterations with a non-exact Hessian converge linearly; "in practice" Gauss-Newton is often used | Thm 8.7 p. 512, Cor. 8.8 p. 513; section 8.5.1 p. 533; 8.6.3 pp. 547-550; 8.7.1 p. 551; 8.8.1 p. 556; 8.9.2 p. 573 |
| NW | J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed., Springer 2006, ch. 10 | not read. A course's slides that follow the chapter ([PDF](https://www.cs.nthu.edu.tw/~cherung/teaching/2009cs5321/link/opt10.ppt.pdf)): J'J "is a good approximation if" the residual is small; global convergence to a stationary point under a uniformly full-rank Jacobian and a Wolfe line search; large-residual problems call for other methods (quasi-Newton on the second term) | slides 7, 8, 11 (secondary) |
| ACADOS | acados, `interfaces/acados_template` at commit `51db6db` (2026-10-29), the source of [docs.acados.org](https://docs.acados.org/python_interface/index.html) | `hessian_approx` is `GAUSS_NEWTON` (default) or `EXACT`; under Gauss-Newton with a general cost, acados uses the cost's exact Hessian and "no Hessian contribution from constraints and dynamics"; the `CONVEX_OVER_NONLINEAR` cost psi(y(x, u)) takes Gauss-Newton or an exact Hessian without the cost's own term; `regularize_method` defaults to none, `PROJECT` raises each eigenvalue to at least `reg_epsilon` (1e-4), `MIRROR` takes absolute values; `exact_hess_dyn`, `exact_hess_cost`, `exact_hess_constr` switch off parts of the exact Hessian | `acados_ocp_options.py` (hessian_approx, regularize_method, reg_epsilon, exact_hess_*); `acados_ocp.py` (GN checks); `acados_ocp_cost.py` |
| IPOPT | Ipopt options, [coin-or.github.io/Ipopt/OPTIONS.html](https://coin-or.github.io/Ipopt/OPTIONS.html) | `hessian_approximation`: `exact` (default, second derivatives from the NLP) or `limited-memory` (quasi-Newton, history 6); no Gauss-Newton option. Inertia correction: if the inertia is wrong, "a multiple of the identity" is added to the Lagrangian Hessian, which steers away from maximizers. `tol` default 1e-8; `max_iter` default 3000 | sections Hessian Approximation, Hessian Perturbation, Termination |
| JAXAD | JAX, "The Autodiff Cookbook", [docs.jax.dev](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html) | `jacfwd` suits tall Jacobians, `jacrev` wide ones, and for near-square ones `jacfwd` "probably has an edge"; forward-over-reverse is "typically the most efficient" Hessian; forward mode's memory "is independent of the depth" of the computation, while reverse mode's memory "scales with the depth" | sections jacfwd and jacrev; hessian; JVPs; VJPs |
| GW | A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM 2008 | not read (section 12) | |

What the sources share. An exact Hessian gives Newton's local quadratic rate
but can be indefinite away from a solution, which IPOPT handles by its
inertia correction (DIEHL02, RMD, IPOPT). Gauss-Newton needs only first
derivatives of the nonlinear part, gives a PSD and multiplier-free matrix,
and converges linearly, fast when residuals and multipliers are small and
possibly not at all when they are large (DIEHL02, DIEHL05, MBD21, RMD, NW).
Practice uses Gauss-Newton for tracking NMPC (DIEHL02, RMD p. 573) and keeps
the exact Hessian as the other standard option (ACADOS, IPOPT).

## 2. The two forms

Both keep everything else in `experts_ipm.py` unchanged: the exact gradient,
the exact constraint Jacobian, the filter line search, the inertia
correction, the stage-wise Riccati solve, and IPOPT's error measure and
termination. Only `derivs` returns a different H.

### Form A: exact Lagrangian Hessian, forward-over-forward

    H_k = Hess over v_k of [ l_k(v_k) + lambda_{k+1}' F_k(v_k) ]
        = jax.jacfwd(jax.jacfwd(stage Lagrangian))(v_k)

- What it is: Newton's Hessian, IPOPT's default (read: IPOPT; DIEHL02 (a);
  RMD Cor. 8.8). Block diagonal per stage (read: RMD p. 556, BP84 as
  relayed), so it fits `stage_kkt_solve` as the compressor's does today.
- AD mode: forward-over-forward, not `jax.hessian` (forward-over-reverse).
  JAXAD prefers forward-over-reverse for flops but notes reverse mode's
  memory grows with the depth of the computation, and that `jacfwd` has the
  edge on near-square Jacobians (read). The stage is a `lax.scan` over b
  steps of a scanned integrator, so it is deep, and n_v is about 30
  (derived). The probe measured 14 s and 2.4 GB to compile one copy, against
  105 s and 10 GB for `jax.hessian` (section 9). The flop cost grows with
  n_v squared rather than n_v (derived), which the probe's 0.07 s against
  0.02 s per evaluation reflects (assumed attribution, section 9).
- Indefiniteness: handled by the port's inertia correction (section 5).

### Form B: constrained generalized Gauss-Newton (CGGN)

    H_k = sum over j of J_j' Hess(psi)(r_j) J_j,   J_j = d r_j / d v_k,
    r_j = r(s_j(v_k))

with r the reward's inner quantities (the tracking errors, the consumptions)
and psi the convex outer function the reward applies to them.

- What it is: MBD21's CGGN (section 3.2), which for a sum of squares is
  Bock's constrained Gauss-Newton (read: MBD21 p. 72; RMD p. 549; DIEHL05
  section 2.2). acados implements it as `CONVEX_OVER_NONLINEAR` with
  `GAUSS_NEWTON` (read).
- The gradient is exact (read: RMD p. 548). The dynamics' curvature
  lambda' Hess F is dropped, as are the curvatures of any other equality
  (read: RMD p. 548, ACADOS).
- Derivatives: first derivatives of the stage only, plus the second
  derivative of psi, which is small and cheap (derived).
- Sub-step sum: the cost over the b visited states enters as a sum of
  per-state terms, each with its own Jacobian. That is RMD's and DIEHL02's
  Gauss-Newton for a least-squares integral on a finer grid than the
  shooting grid (read: RMD p. 533; DIEHL02 p. 580).

## 3. The planned form, checked against the sources

| element of the plan | sources | verdict |
| --- | --- | --- |
| "Bock and Plitt 1984 (multiple shooting, Gauss-Newton)" | DIEHL02, RMD | partly off. BP84 gives multiple shooting, the block structure and partitioned high-rank updates. Gauss-Newton in this setting is BOCK83's constrained Gauss-Newton |
| inner map v_k to the b visited states, first derivatives only | MBD21, RMD p. 533, DIEHL02 | matches |
| outer function -compute_reward(state) on the whole state | MBD21 eq. (1), p. 69 | not a GGN in general: B is PSD only if the outer function is convex, and -reward is not convex in the state (a squared path distance has negative curvature inside the circle). The split must be at the reward's convex terms (Form B above). On plane3d_circle at reset the state-level outer Hessian was PSD anyway (eigenvalues 0 to 4, measured), so this is not what failed there |
| H_k = sum J' Hess phi J, exact gradient | MBD21 eq. (8)-(9), RMD p. 548 | matches |
| drop lambda' Hess F | RMD p. 548, DIEHL05 p. 1719, ACADOS | matches CGN and CGGN |
| the probe's version | section 9 | the plan as written, split at the state |

## 4. When Form B is valid

Read:

- the outer function is convex, so B is PSD (MBD21 p. 69);
- the neglected curvature is small: small inner curvature, or residuals near
  the outer function's minimum (MBD21 p. 69), small residuals and small
  multipliers (RMD p. 549), residual "sufficiently small" (DIEHL02);
- necessary for convergence: on the reduced space, B at least half the true
  Hessian; the linear rate is the smallest alpha with
  -alpha B <= E <= alpha B (MBD21 Thm 4.5);
- quadratic convergence when the solution is fully determined by the active
  constraints (MBD21 Thm 4.9).

Derived for these tasks. The process plants start near their setpoint, with
small tracking errors and a well-damped response, which is the small-residual
case. The aircraft start far from the path or slot, so the cold solve's
residuals and its multipliers are large, and the dynamics' curvature is
weighted by those multipliers. That is the regime in which the sources say
Gauss-Newton converges slowly or not at all. The probe agrees (section 9).
This explanation is not measured: nobody computed alpha.

## 5. Indefinite curvature

| case | Form A | Form B |
| --- | --- | --- |
| where it comes from | any nonconvex term of the Lagrangian: the dynamics, a path distance, a smoothed max in a "greater than" limit, a smoothed heading wrap | only from a non-convex outer function |
| what the sources do | IPOPT adds delta I until the inertia is right (read: IPOPT). acados offers per-block projection for its SQP, off by default (read: ACADOS) | MBD21: choose the split so the outer function is convex; then B is PSD by construction. acados `PROJECT` as a fallback |
| this spec | rely on the port's inertia correction, already IPOPT's (`experts_ipm.py`, first shift 1e-4, growth 100 then 8). No projection (derived: IPOPT's design for exact Hessians; projection would change the Newton step) | split at the reward's convex terms. A term with no convex split (the smoothed min in a heading wrap) gets its outer Hessian projected, eigenvalues floored at 0 (ours; ACADOS `PROJECT`). Projection on plane3d_circle changed nothing (measured) |

A smoothed max in a "greater than" limit (the kiln's cold limit, smoothing
K3) is a constraint, not a cost. Form A carries its curvature with its
multiplier, exactly. Form B drops it with the other constraint curvature
(read: MBD21 p. 72, CGGN throws it away).

## 6. Slacks and epigraph terms

The IPM accepts only linear inequalities, G z + g0 >= 0 (read:
`experts_ipm.py`). So a nonlinear row needs an algebraic variable defined by
an equality, as the compressor carries phi_min (derived; audit section 5).

| term | how it enters | Form A | Form B |
| --- | --- | --- | --- |
| elastic trip slack e_k, priced at the trip cost (accepted) | linear cost, bound e_k >= 0 | zero Hessian; the barrier's G' diag(w/s) G gives its curvature | same |
| epigraph slack t for p = 1 tracking or the running-cost hinge (smoothing T2) | linear cost; y_j = e(s_j(v_k)) as an equality in c(z); bounds t >= y_j, t >= -y_j (or t >= q - c_hold), and t >= 0 | the curvature of y_j's equality enters with its multiplier | dropped, as all constraint curvature; the outer function there is linear, so B gets nothing (read: MBD21, phi = v gives B = 0) |
| t >= 0 | linear bound | keep it explicit | keep it explicit (read: MBD21 Ex. 3.2, it "significantly improves" convergence of the quadratic variants) |
| squared tracking (p = 2, e_tol = 0, smoothing T2) | (e / e_floor)^2 in the stage cost | exact | 2 J_e' J_e / e_floor^2, the classic Gauss-Newton term |

Layout consequence (derived). The epigraph rows sit on every visited state,
so a stage carries b algebraic variables per kinked term (13 per stage on
glass_furnace for its running cost). `stage_kkt_solve` takes one scalar per
stage today; it needs a block of per-stage algebraic variables before either
form can carry the epigraph rows. Their equalities couple only v_k, so the
Riccati structure is kept.

## 7. Convergence certificate

- **Stopping test:** IPOPT's scaled error, dual infeasibility, primal
  infeasibility and complementarity, at most `MPC_TOL` = 1e-8 (read: IPOPT
  `tol` default). It is the same for both forms: the Hessian enters the step,
  not the test. With the exact gradient and Jacobian, a fixed point of either
  iteration is a KKT point of the true problem (read: MBD21 Lemma 4.1; RMD
  p. 548).
- **Rate:** Form A quadratic near a solution (read: RMD Cor. 8.8); Form B
  linear at rate alpha (read: MBD21 Thm 4.5).
- **Cap:** `MPC_MAX_ITER` = 150, the CasADi build's cap (ours, inherited).
  IPOPT's own default is 3000 (read). Derived: from an error of order 1, a
  linear rate alpha needs ln(1e-8) / ln(alpha) iterations: 27 at 0.5, 83 at
  0.8, 175 at 0.9. So under Form B the cap of 150 fails any solve with alpha
  above about 0.88. Under Form A the probe's converged solves took 26 to 88.
- **Recorded per solve:** iterations, final error, the largest constraint
  violation, whether the cap stopped it, and under Form A the largest inertia
  shift delta (ours; audit decision 6 stores the certificate with each row).
- **Float64:** a 1e-8 tolerance needs x64, which 23 of 24 tasks run; the
  reactor needs its float32 seeds fixed first (smoothing spec S6).

## 8. Exact Hessians for small tasks only?

No (derived). The rule asks for one solver for every task, and a per-task
choice of Hessian is a per-task setting of the kind the rule removes. If Form
A is chosen, every task gets it and the question does not arise. If Form B
is chosen, the aircraft need something else, which breaks the rule.

## 9. Solver probe, 2026-11-03

Read-only scratchpad probes (`ipm_probe.py`, `deriv_alone.py`): the port as
it stands, the planner's noise-free params, float64, training seed 100, one
cold solve and three or four warm (shifted) solves, each warm solve one T_c
after the last, on one laptop (JAX 0.6.2). Inputs bounded by the action box,
one elastic slack per stage, no limit rows yet. Times include no
compilation unless stated.

**Assumed, not measured:** the horizons and T_c. Nothing has run the horizon
protocol yet. plane3d_circle: 60 stages at T_c = 1 step (assumed).
glass_furnace: 62 stages at T_c = 13 steps; hvac: 63 at T_c = 6 (the
protocol's provisional 0.1 tau; assumed). Also assumed: which of the two
per-evaluation times (0.07 s, 0.02 s) belongs to which AD mode; this spec
reads 0.07 s as forward-over-forward, which theory predicts.

### Exact Hessians by `jax.hessian` (first probe)

| measurement | plane3d_circle (26 states + 3 inputs, 60 stages, 1,800 variables) |
| --- | --- |
| full solver | compile 224 s; peak 12.2 GB, all reached during compile; 31 to 88 iterations to error 1e-11 to 9.6e-9; warm solve 1.0 to 1.7 s |
| one copy of the per-stage exact derivatives alone | 105 s compile, 10 GB |
| second derivatives of the dynamics part only, or of the cost part only | about 45 s and 6 GB each |
| first derivatives (Jacobians) only | 4 s, 0.82 GB |
| stage-by-stage `lax.map` with `jax.checkpoint` | memory unchanged (12.35 GB); solves 2 to 3 times slower |
| dropping the trip-and-reset branch | no change |

- glass_furnace (1,736 variables): compile 58 s, 4.3 GB, about 90 ms per
  iteration; capped at 150 on every solve (error 6.7e-4 to 1.8e-3), with its
  running-cost kink not yet rewritten.
- hvac (693 variables), raw step and reward: 15 iterations cold, then one
  failed warm solve and 23 to 55 iterations; compile 10 s, 1.56 GB.
- formation (about 6,400 variables): did not finish compile plus one solve
  in 15 minutes. The owner chose cooperative per-seat sub-problems for it.
- first_order: with the smoothing spec's T1 and T2, 11 iterations cold and 1
  warm; without them it fails or hits the cap.
- Environments: 23 of 24 run under x64; the reactor's step seeds float32.

### Gauss-Newton against exact forward-over-forward

| | A: exact, forward-over-forward | B: Gauss-Newton, as planned (split at the state) | exact, `jax.hessian` (reference) |
| --- | --- | --- | --- |
| plane3d_circle, convergence | converged: 88 iterations cold, 26 to 47 warm, error 8.4e-11 to 2.5e-9 | never: cap 150 (error 0.016 to 1.2) and cap 500 (0.069 to 0.28); over all variants, with or without T1 to T3, error 0.01 to 1.8; PSD projection no help | converged: 31 to 88 |
| plane3d_circle, full-solver compile | 44 s | 22 s | 224 s |
| plane3d_circle, peak memory | 6.1 GB (4.0 after compile) | 3.2 GB | 12.2 GB |
| plane3d_circle, warm solve | 1.5 to 2.5 s | 2.4 to 2.7 s, all capped | 1.0 to 1.7 s |
| one copy of the stage derivatives | 14 s, 2.4 GB | 4 s, 0.82 GB (Jacobians; psi's Hessian is small) | 105 s, 10 GB |
| one evaluation of the stage derivatives | 0.07 s (assumed attribution) | not measured | 0.02 s |
| glass_furnace, convergence | capped at 150 (error 6.7e-4 to 1.8e-3) | capped at 150 (error 6.7e-4 to 1.8e-3) | capped at 150 (same range) |
| glass_furnace, compile and peak | 20 s, 3.1 GB | 14 s, 2.4 GB | 58 s, 4.3 GB |
| glass_furnace, per capped solve | 27 to 28 s, about 180 ms an iteration | 5 s, about 34 ms an iteration | 13 to 14 s, about 90 ms an iteration |
| hvac | not run | with T1 and T2: 15 iterations cold, 1 warm, error 1e-10 to 1e-11; compile 5 s, 1.35 GB | 15 cold, 23 to 55 warm, one failed (raw step, not comparable) |

Form A's rows are read from `probe_fwd_plane3d_circle.log` and
`probe_fwd_glass_furnace.log`, both written while this spec was drafted; the
coordinator may replace them with the final run's numbers. glass_furnace caps
under every form with its running-cost kink unrewritten, so it does not
separate them on convergence. It does show Form A's cost per iteration where
a stage holds many steps (b = 13): twice `jax.hessian`'s and five times Form
B's (derived from the per-solve times), since forward-over-forward pays for
n_v squared directions through all b steps.

## 10. Recommendation

**Form A, the exact Lagrangian Hessian computed forward-over-forward, for all
24 tasks** (derived).

- It converges where Form B does not: one aircraft task, one seed, which is
  the evidence available. The ten aircraft tasks share the dynamics, so the
  same failure is expected on them (assumed).
- The sources explain the split: Gauss-Newton is for small residuals
  (DIEHL02, RMD, MBD21), and an aircraft's cold solve is not one.
- Forward-over-forward removes most of the cost that motivated Gauss-Newton:
  44 s and 6.1 GB against 224 s and 12.2 GB, within a factor of 2 of Form
  B's compile and memory.
- It keeps one solver with no per-task setting, and IPOPT's default Hessian,
  with the inertia correction the port already has.

The cost it carries: on glass_furnace an iteration takes about 180 ms, against
34 ms under Form B (measured), so a slow process task pays several times
more per iteration for a Hessian it may not need. Whether that fits depends
on iteration counts once glass_furnace's kink is rewritten, which no run has
measured. What would change the recommendation: Form A failing a compute
budget on the long-horizon process tasks once the horizon protocol runs, or
a damped Form B converging on the aircraft, which nobody has tried.

## 11. Where the sources disagree

1. **Which Hessian for NMPC.** DIEHL02 recommends Gauss-Newton for tracking
   problems and RMD calls it common in practice; IPOPT and DIEHL02 (a) give
   the exact Hessian the best local convergence. The probe sides with the
   exact Hessian on the aircraft and finds both fine on hvac.
2. **AD mode.** JAXAD calls forward-over-reverse typically the most
   efficient. That is a flop statement; its own memory remark and the probe
   favour forward-over-forward for deep stages with n_v near 30.
3. **Handling indefiniteness.** IPOPT shifts the whole Hessian by delta I;
   acados projects each block for its SQP. With an interior-point method and
   an exact Hessian, IPOPT's is the native choice.
4. **Iteration cap.** The compressor's 150 against IPOPT's 3000.
5. **Attribution.** The plan credits Gauss-Newton to BP84; DIEHL02 and RMD
   credit BP84 with the block structure and partitioned updates, and BOCK83
   with constrained Gauss-Newton.

## 12. Source check

Each key source was fetched twice on 2026-11-03.

- **PDFs:** MBD21 (Syscop preprint twice, byte-identical, plus the published
  ESAIM text from Optimization Online; every page number above is the
  journal's); DIEHL02 and DIEHL05 (two downloads each); RMD (a fresh download,
  byte-identical to the one fetched earlier today). Phrases were checked in
  pdftotext output of both copies.
- **Code:** ACADOS, at commit `51db6db`, once as raw files and once through
  the GitHub API; byte-identical.
- **Web pages:** IPOPT options and JAXAD, two downloads each, phrases found
  both times.
- **Secondary, twice:** the NW course slides.

Not read, and not relied on beyond what is stated:

- **BP84.** ScienceDirect refused the fetch (403 and a Cloudflare block);
  OpenAlex, Crossref and Semantic Scholar carry no abstract. What it contains
  comes from DIEHL02 and RMD, both co-written by Bock's group or by Diehl.
- **BOCK83.** Not online; cited through RMD and MBD21.
- **Nocedal and Wright.** No legitimate copy found; Springer's page needs
  JavaScript. Only the course slides; the local-rate statements rest on RMD
  and MBD21 instead.
- **Griewank and Walther, 2nd ed.** SIAM's pages refused the fetch, and the
  related Math. Comp. paper (Griewank, Utke and Walther 2000) was not
  retrievable. The AD-mode choice rests on JAXAD and the probe.
- **Waechter and Biegler 2006.** The inertia correction is cited through
  IPOPT's documentation and the port's docstring.
- **ESAIM's own site** served a bot check, which was not attempted.

## 13. Decisions for the owner

Recommended option first.

| # | decision | options |
| --- | --- | --- |
| G1 | Hessian form | A: exact, forward-over-forward, for all 24 (converges on the aircraft; IPOPT's default). Or B: Gauss-Newton for all (cheaper, PSD, but fails on plane3d_circle). Or A on the aircraft and B elsewhere (two settings; against the one-solver rule) |
| G2 | AD mode for second derivatives | forward-over-forward (14 s, 2.4 GB per copy). Or `jax.hessian` (105 s, 10 GB; faster per evaluation) |
| G3 | Indefinite Hessian under A | the port's IPOPT inertia correction only. Or per-block projection as acados offers (changes the Newton step) |
| G4 | Iteration cap | keep 150 and record capped solves in the certificate. Or IPOPT's default 3000 (needed under B if alpha exceeds about 0.88) |
| G5 | Epigraph rows on visited states | one algebraic variable per kinked term per visited state, with `stage_kkt_solve` extended to carry them. Or epigraph at stage boundaries only (fewer variables, not exact for b > 1) |
| G6 | If B is chosen: where to split | at the reward's convex terms (MBD21). Or at the whole state, as the probe did (not a GGN where -reward is nonconvex) |
| G7 | Budget check | set a per-episode compute budget, then rerun glass_furnace under Form A with its kink rewritten (smoothing T2) before G1 is final. Or decide G1 now on plane3d_circle alone |

## 14. Amendment 1, 2026-11-04: what C1's validation found

C1 ran the five process oracles on the solver above for training seeds 100
and 101 and found three faults, none in the rules. The owner accepted the
three fixes on 2026-11-04. Each was checked first on a copy of the code,
then rebuilt test first.

**The port left out three parts of IPOPT's line search** (read: IPOPT 3.14
source, IpBacktrackingLineSearch.cpp, IpFilterLSAcceptor.cpp, IpUtils.cpp):

- when the line search fails at a point that meets the acceptable tolerance,
  IPOPT stops there ("Solved To Acceptable Level") instead of restoring;
- its accept tests allow 10 eps of the reference value (Compare_le);
- a step below 10 eps of the variables, with the violation at most 1e-4, is
  taken whole (DetectTinyStep).

Its backtracking also runs only while the step exceeds alpha_min. The port
compared with >=, and since alpha_min is 0 when the violation is, four_tank's
solve looped forever at a step of 0. Without the first part, cstr's warm
solves at a held target ended failed after four restorations: 18 and 49 of
150. With it, none failed. The other two parts changed nothing measurable
there and are kept because they are IPOPT's. IPOPT's soft restoration and
its stop after repeated tiny steps are still not ported.

**G8, cold start.** A cold solve started from the held moves simulated
forward: four_tank's reset has both pumps off, the simulation drained the
tanks, and the solve never recovered (seed 101: no solve converged, 47
trips). Multiple shooting keeps every state a variable so that no such
simulation is needed. It can start from a dynamically infeasible guess,
which single shooting cannot (Vanroye et al., FATROP, arXiv 2303.16746).
Every state now starts at the measured one, with the moves still held.
four_tank then had no failure and no trip.

**G9, the stages past the episode.** As the horizon shrinks at the episode's
end, the stages past it still ran the plant. On the open-loop unstable CSTR
their states reached 1e41, and the last 8 solves of seed 100 failed. Those
stages now hold their state, since no plant runs past the episode. The
shrinking horizon (H_k) is unchanged.

The compressor's solves still hit the cap at about the same rate, and none
of the three changes it. Its rate-limit widths are K6-flagged (the smoothed
corner is crossed within one integrator substep), and it is diagnosed next.

## 15. Amendment 2, 2026-11-04: G10, a tie-break on moves

With T1r and K11 in place, 2 to 4 % of the compressor's solves still hit the
cap. The cause is not the surge line: the capped solves sat 0.07 to 0.13 above
it. Two moves track one output, and while recycle power is under `c_hold`
nothing prices how the work is split between them, so the optimum is not
unique: 26 of the 52 eigenvalues of the move Hessian are zero at a plan. With
the barrier at its floor, Newton wandered along that valley, its error
bouncing between 2e-7 and 5e-5 until the cap. Corners 10 times wider also
cleared 8 of 8 captured solves, by adding curvature, at the cost of K5's
accuracy and a tenfold K11 back-off. They were not taken.

**G10.** The objective adds tie x sum over planned stages of |du|^2, du a
move's change from the one before, in units of its scale (its reach under T1r,
else its range), u_hold before the first. This is input move suppression, one
of the two strategies industrial MPC uses against ill-conditioning. It raises
the matrix's diagonal, lowers the condition number and "tends to minimize the
norm of the MV moves" (QB, section 3.4.3, p. 752; QB's "fat plant", more MVs
than CVs, is this case). Here it is sized to pick one plan among equal ones
and nothing else: tie = 0.25 / (moves x solves in the episode), so that every
move of every solve at full size costs 0.25 reward units over the episode.
That is what an error of half the resolution costs in one step, (q / 2)^2 /
e_floor^2, with e_floor clamped at the resolution q, as on all five tasks so
far. The term enters the solve only: the plan's cost and the record's reward
do not contain it.

It applies to every task on the shared builder, with no per-task condition.
A task whose reward already fixes its moves has no flat direction to pick
along, and the term cannot cost it more than the budget above. The aircraft,
kiln, boiler, wind turbine and formation have more moves than tracked
outputs, so C2 and D may need it.

**Provenance.** The value was derived after 1e-4 had been seen to clear 8 of 8
captured solves (1e-6 cleared 7, 1e-8 cleared 5). It uses task constants only,
and no return was used in choosing it, but it is not strictly pre-hoc.
Compressor, training seeds 100 to 102: capped 49, 26, 51 down to 0, 0, 0.

## 16. Amendment 3, 2026-11-04: G11, a cold retry

The records gate failed on the first re-record: unstable_cstr tripped once on
the ten test seeds and lost that seed to its expert, with 10 failed and 7
capped solves. It is an exothermic reactor, open-loop unstable, and the
failures sat at one event: the step of the target at step 600, where the plan
heats the jacket and then brakes at full cooling. The plan's active set has
to change there, and warm starts begin with their slacks 1e-9 off their
bounds, so the steps collapsed to 1e-3 and below and the iterates jammed. At
six captured states the warm solve ended at a cost of about 3e13 and the cold
solve, from the same state, converged in about 30 iterations to about 3e6.

**G11.** A warm solve that fails or hits the cap is solved again cold (G8's
start, with K7's rungs). If that solve fails too, the shifted plan applies, as
the owner decided (oracle-audit.md, decision 5). A superseded solve is counted
in `solver_retries`, not in the failed or capped counts. Main's compressor
oracle retried in the same way before it handed a step to its expert.

Measured, ten test seeds and ten training seeds: unstable_cstr 20 of 20 wins,
no trip, no failed or capped solve, 7 retries in 24,000 solves. The plan's
heat-then-brake around the target's step stays aggressive by design: it is the
noise-free optimum. T reached 357 K against a 365 K trip with the jacket at
its minimum on seed 8, and this CPU did not trip where CI's did. The gate on
the recorded seeds remains the check on that margin.

## 17. Amendment 4, 2026-11-05: a resting target, and C2's other finds

C2 put six more tasks on the builder. Three of them price their reward in
money, about 1e-5 per step, and one has a spare move. That exposed what the
process tasks could not.

**G12. A spare move gets a resting target.** Count a plant's moves and the
outputs its reward prices at a steady state, running cost included. With more
moves than priced outputs the plant is fat, in Qin and Badgwell's word (2003,
Fig. 3): its steady states form a family the reward cannot tell apart. Their
survey lists driving the inputs to steady-state targets with the remaining
degrees of freedom as an objective above move suppression (section 3.4), and
says all the surveyed algorithms do it (section 3.4.7). So the output the
reward leaves free gets a squared target at the plant's design point, scaled
to 1 at the edge of its limit, weighted so that it costs G10's budget over an
episode spent at that edge. It lives in the plan only: K8's replay and the
reward never see it. The survey's targets sit on the inputs. On a wind turbine
the wind moves the inputs a steady state needs, so the target sits on the speed
they hold.

Wind is the one task this reaches: two moves, and only power priced, since its
fatigue term prices the pitch command's gap, which a steady state closes. Its
target is rated speed, where the reference turbine's region 3 holds the rotor
(Jonkman et al. 2009, ch. 7). `tests/oracles/test_process_oracles.py` keeps the
count for every task on the builder, so a new task has to state its own.

The need showed on seed 1. With the speed free, the plan knows only the wind's
mean, rides the speed down to about 0.82 of rated in a lull, and pays there
with one-step torque dips of 625 and 825 kW, the same plan warm or cold: a
converged plan, not a stalled one. With the target the speed stays between 0.96
and 1.22 of rated on seeds 0 and 1, the worst step falls from -0.006 to -1e-4
dollars, and the return on seed 1 goes from -0.016 to -0.0053. The weight is
the first and only value tried, and the score chose nothing.

It does not cure every seed. The term settles a tie, so it cannot hold the rotor
against a real cost. Seed 9 asks for 4.96 MW, close to the torque limit, and a
long lull takes the speed to 0.91 of rated, where the torque reaches its ceiling
and the plan pays with dips again: -0.064 against -0.0071 for the previous
oracle and -0.070 for the expert. The ten seeds' mean is -0.011 against -0.0059.
Nine seeds beat the previous oracle. The gate passes, and the weight stayed as
written.

The count has one soft spot. A running cost above `c_hold` prices a level only
above it, and the compressor's was active on 36 % of the steps of seed 0. Below
it the spare move is free. Nothing has gone wrong there, and the count stands
as written.

**The solve stays in the reward's units.** The tolerance is absolute and the
scaling only scales down (MAX_GRADIENT), so on a reward in dollars 1e-8 leaves
about three digits of the gradient. Solving in floors, as IPOPT's scaling
advice suggests, was tried and dropped: wind then hit the cap on about 70 % of
its solves at about 140 iterations, two hours a seed against 100 s, for a
return 0.8 % better on seed 0 and equal on seed 1, and hvac and battery came out
identical. G10's budget still follows the reward: `tie_budget` is what half a
resolution costs in one step, in the reward's units.

**A hinge's redundant row.** On an absolute hinge with no tolerance, t >= r and
t >= -r already give t >= 0, and at t = r = 0, where a p = 1 tracking term sits
when it tracks exactly, the third row is active too. Three active rows on two
variables break LICQ, the condition an interior-point step relies on. The row
is left out there.

**P stops at the episode.** H12 shrinks the plan to N_ep - k, so a P past the
episode only ever adds dead stages. hvac's P is 932 against 720 samples. The
builder is sized to min(P, N_ep), which changes no plan.

**What stays open: the kiln.** About 3 % of its solves fail (234 of 7000 on the
ten test seeds). The env's explicit 30 s step amplifies the hot cells'
calcination about 3000 times a step, and only the clip at 1 holds it. A plan's
free states can sit just off that clip,
where the exact Hessian reaches 1e18. No Hessian shift then gives a usable step
at step 0 of seed 0, nor at steps 587 to 602 once the start's multipliers are
guarded (there the solve dies at iteration 0 on non-finite multipliers). Keeping
only shifts that worked, retrying in the warm mode and a clip with no slope at
its bound did not cure it. Failed solves apply the shifted plan (G11), and the
oracle was recorded as it stands, on the owner's decision of 2026-11-05: it
averages -406 against -110 for the previous oracle, and beats the expert by at
least 11 to 1 on every seed. The cure belongs in the env.
