# Why the IPOPT solves are slow, and how to speed them up

Status: investigation, 2026-11-12. Prototype on `investigate/ipopt-speed`
(draft PR into `int/amendment3`). Nothing re-recorded.

## The question

The reactor's ceiling solves (`scripts/reactor_ipopt.py`, the whole day, 2 880
stages) ran 4.5 to 6 hours per 300-iteration rung on bob, at about 10 cores and
16 GB each. Every rung hit the cap (KKT error 0.03 to 52 000). Two seeds' final
1 500-iteration rungs took 17 to 19 hours each and stayed capped.

## What "IPOPT" is here

No IPOPT library is involved. `oracles/ipm.py` is a JAX port of IPOPT's
algorithm, and `oracles/shooting.py` builds the multiple-shooting NLP that every
task oracle uses (14 tasks, the compressor's included). So:

- The linear solver is our own Riccati recursion (`stage_kkt_solve`), not MUMPS.
  HSL (MA27/57/86/97) and Pardiso don't apply, and no licence question arises.
  They would only matter if we went back to real IPOPT through CasADi.
- The Hessian is exact: per-stage blocks from JAX autodiff (`stage_derivatives`).

The reactor's NLP has 2 881 stages. Each has 92 augmented states (47 plant and
queue states, plus 45 outputs: 6 sub-steps of hinge arguments and limited
quantities), 2 moves and 45 slacks: 400 459 variables and 139-wide stage blocks.

## Profile (bob, 8 cores per job, JAX 0.6.2, float64)

Times are medians after compiling (`scripts/profile_ipm.py`).

| piece, per call | 97 stages | 385 stages | 2 881 stages |
| --- | --- | --- | --- |
| stage simulation (c, f; each line-search trial) | 0.3 ms | 1.2 ms | 13 ms |
| stage Jacobians | 26 ms | 98 ms | 0.64 s |
| stage Hessians, forward over forward (current) | 1.98 s | 5.74 s | not run (about 50 s, linear) |
| stage Hessians, reverse over forward (prototype) | 0.28 s | 1.12 s | 6.0 s |
| one Riccati KKT solve | 0.17 s | 0.52 s | 2.5 s |
| one iteration, current code | | 4.72 s | 31.4 s |
| one iteration, prototype | | 1.94 s | 10.3 s |
| peak RSS of the profiling process, current / prototype | | 3.8 / 3.5 GB | 17.3 / 11.7 GB |

Per iteration at 2 881 stages, before the fix:

- **The Hessian takes most of the time.** `stage_derivatives` took the Hessian
  forward over forward. That pushes 49 x 49 = 2 401 tangents through each stage's
  6 sub-steps of TR-BDF2: 15 to 20 ms per stage, about 50 s per iteration for the
  whole day. It also explains the 16 GB: 2 881 stages x 2 401 lanes of
  intermediates.
- **The Riccati solves come next.** Each is 2.5 s, and an iteration takes 2 to 8
  of them: the inertia-correction trials, then the step, plus the affine and
  centring directions in adaptive mode and up to 4 second-order corrections.
  Each refactorises the 94 x 94 recursion.
- **The rest is small.** Function evaluations, Jacobians and line-search trials
  are under 1 s.

There is no recompilation inside a solve: the whole solve is one `while_loop`.
A first call compiles for 1 to 2 minutes.

## Why it does not converge

At 96 steps the same problem converges in 40, 30 and 13 iterations per rung
(KKT about 1e-9). At 960 steps it already caps at 300. A per-iteration trace
(seed 102, 960 steps) shows why:

1. **The cold start sets the error.** The start puts every state at x0 under
   the held moves. Against a day's load profile, that start breaks the Tavg DNB
   limit by tens of kelvin at some stage. The elastic slack on that limit starts
   at a scaled value of 3.5e7: the slack is in dollars at the trip price. Its
   s·w term (3e7) is then the whole KKT error, while dual and primal
   infeasibility are about 100 and 0.1. In monotone mode mu only falls once the
   barrier problem is solved to 10 mu. With that slack still large, that never
   happens, so mu sits at 0.1 for the whole rung and the slack creeps down by
   about 1% per iteration.
2. **The inertia correction is large.** From iteration 9 on, the Hessian needs
   a shift of 1e2 to 9e3 to be positive definite on the reduced space. The shift
   is uniform over all 2 881 stages, so a few non-convex stages turn every
   stage's step into a short gradient step (alpha about 1e-3), and each
   iteration spends 2 to 5 extra factorisations finding the shift.
3. **One rung was repeated.** `SCALE * 0.01` equals `WIDTH`, so the script's
   rungs 2 and 3 were the same problem. On the whole day the repeat ran 300
   more iterations (3.6 to 5.8 hours per seed) and ended at a higher KKT error
   than it started.

Measured at 960 steps, seed 102, first rung (boron width 0.1), 300-iteration cap:

| cold start | iterations | KKT error | planned cost |
| --- | --- | --- | --- |
| every state at x0 (current) | 300, capped | 0.18 | 1 647.9 |
| held moves, states simulated | 271 | 2.7e-9 | 1 647.6 |
| reference plan, states simulated | 241 | 2.5e-9 | 1 647.6 |

Scaling (`nlp_scaling_method`): the builder already applies IPOPT's
gradient-based scaling. The trouble is the price on the elastic slacks, which
puts trip-sized numbers into the complementarity, not a missing scaling.

## A bug found on the way: forward over reverse is wrong here

`jax.hessian` and `jacfwd(grad)` give the reactor stage a Hessian that is 2%
off central differences, and not symmetric. They double exactly one diagonal
entry: the boron-queue slot `queue[delay - 1]` that arrives during the step.
That holds on JAX 0.6.2 and 0.11.2. Forward over forward, `jacrev(jacfwd)` and
the per-direction `grad` of a JVP all agree with central differences to 2.5e-8.
The prototype uses the last of these. I could not reproduce the doubling in a
small example (a gather inside a scan, softplus at 0, the boron step alone all
agree), so the cause is still open. `scripts/check_stage_hessian.py` reproduces
it. Forward over reverse must not be used on these stages until it is
understood.

## Options, ranked

| # | change | measured or estimated effect | cost | touches |
| --- | --- | --- | --- | --- |
| 1 | Stage Hessian by reverse over forward, one `grad` of a JVP per input (prototyped) | measured: 3.0x per iteration on the whole day (31.4 to 10.3 s), 2.4x at 385 stages; peak RSS 17.3 to 11.7 GB; identical iterates (KKT error equal to 12 digits) | done: 6 lines in `shooting.py`, test oracle changed to central differences | every oracle and ceiling built on `Shooting` (14 tasks). Solver fingerprints move. Plans should be the same to rounding, but a borderline solve could flip. |
| 2 | Cold start from simulated states (prototyped as `Shooting.solve(..., start=moves)` and `--start` in the script) | measured at 960 steps: the first rung converges in 241 to 271 iterations instead of capping at 300 | done for the ceiling script. For task oracles G8 says never start from an expert, so only the "held, simulated" start would be eligible there, and only for plants stable under held moves. | the reactor ceiling script. Oracles only if the owner amends G8. |
| 3 | Drop the repeated rung (done in the script) | saves one full rung per seed: 3.6 to 5.8 h at the old speed | none | the reactor ceiling script only |
| 4 | Factorise the Riccati recursion once per shift and reuse it for every right-hand side (step, affine, centring, second-order corrections) (built, owner's decision) | measured, 10 iterations, same 8 cores side by side: warm (adaptive) solves 1.27x at 385 stages (2.28 to 1.79 s) and 1.52x on the whole day (20.8 to 13.7 s); cold solves no gain (12.8 against 13.0 s), since they take only two directions per iteration and each stage scan costs about the same with or without the matrix work; identical iterates (KKT error equal to every digit after 2, 5 and 12 iterations) | about 100 lines in `ipm.py`: `stage_kkt_factor` and `stage_kkt_solve_factored`, `NLP.factor` and `NLP.solve` | every oracle (fingerprints), landed with option 1 so they move once |
| 5 | Skip structurally zero Hessian directions (the reactor's 30 queue states are a shift register: zero curvature except the arriving slot) | estimated 2x on the Hessian (49 to about 20 directions) | moderate: detect zero rows once per build, or a task hook | `shooting.py` (fingerprints) |
| 6 | Per-stage inertia correction (shift only the stages whose reduced Hessian is indefinite) | could cut iterations a lot where a few stages are non-convex (trace item 2), not measured | a departure from IPOPT's algorithm. The solver docs would need an amendment, and the owner should decide. | `ipm.py`, every oracle |
| 7 | Condense the 45 output states out of the Riccati state (94 to 49 wide) | estimated 4 to 7x on each KKT solve | large: the rows on outputs would become nonlinear in (x_k, u_k), which the port does not support | `shooting.py`, `ipm.py` |

Not worth doing: switching linear solvers (there is none to switch), HSL
(irrelevant without real IPOPT), more threads (the Riccati scan is sequential
over stages, and the Hessians already vectorise over stages).

## Full day with options 1 to 3

Seed 102, the whole day, reference-plan start, 600-iteration cap, 10 cores on bob
at load 150 to 380 (tmux `ipm-ref2880` in `~/ipopt-new`; that tree's `ipm.py`
prints a trace, so never copy it back). Option 4 was not in this run.

| rung (rod width, boron width) | old code (#174 runs) | options 1 to 3 |
| --- | --- | --- |
| 1 (0.0105, 0.1) | 300 iterations, capped, KKT 1.4, 4.25 h, planned cost 1 903.4 | **372 iterations, converged, KKT 9.9e-9**, 1.37 h, planned cost 1 896.2 |
| 2 (0.00105, 0.01) | 300, capped, KKT 0.71, 4.6 h | 600, capped, KKT 2.2, 2.61 h, planned cost 1 796.5 (violation 1.1e-5) |

Rung 3 never finished: the run was stopped at about 18:05 on 2026-11-12, when the
owner cleared every job on bob for a clean start. So the full day still has no
converged plan at the narrow rungs. Rung 1 converges now; rung 2 runs twice as
many iterations in about half the old time, but stays capped with KKT 2.2. The
narrow widths are the open problem, and a rerun of rungs 2 and 3 (with option 4)
belongs on smithers or flanders, not on a shared bob.

## pH oracle (second job, from the owner)

The pH oracle (branch `wip/ph-oracle`) failed on the shared solver because
`planner_pH` returned the env's bisection value: 20 halvings of [0, 14] give a
staircase of 1.3e-5 pH, while its custom derivative is the exact slope. Near the
optimum the IPM's error then stalls at about 3e-5 against a barrier test of
1.8e-5, the line search fails, and the restoration fails the solve. Polishing the
root with 3 Newton steps on the same charge balance makes the value exact to
rounding. Seed 0's first solve then converges in 59 iterations (KKT 2.5e-9,
replay gap 3e-10). The env is unchanged. Fix and test: branch
`fix/ph-oracle-solver`.

Over seeds 0 to 4 the solver then had no failed or capped solve. The oracle
still lost to its expert, and the cause was the buffer noise between re-plans,
not the upsets: without upsets or noise it beat the expert (-82.5 against -112).
The owner chose to size the permit back-off for the noise over a T_c hold, at
2 sigma. With that, it beats the expert on seed 0 only. At K11's quantile (4.8
sigma) it wins seeds 0 to 3 and loses seed 4, even with upsets off. Numbers and
next steps are on draft PR #199.
