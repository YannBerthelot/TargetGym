# Formation oracle: cooperative distributed MPC

Specification of 2026-11-03, for owner review. Nothing here has been run or
implemented. It covers the `formation` oracle only, plus a check of whether
`patrol` and `patrol_bearing_only` need the same treatment (section 7).

**Why.** A joint NLP over the three seats (about 6,400 variables at 60
stages) did not compile and solve once in 15 minutes with the shared
IPOPT-port solver. The owner chose on 2026-11-03 to split the formation by
aircraft, each seat planning with its own full model, in the cooperative
variant: every seat minimises the team cost over its own inputs, and the
seats re-solve until their plans stop changing.

**Owner decisions, 2026-11-03.** Every recommendation in section 9 is
accepted (F1 to F7):

- F1: Jacobi with w = 1/3 and Rawlings, Mayne and Diehl's test;
- F2: stop at a plan change of 1e-6, cap 20;
- F3: the warm-start tail repeats the last move;
- F4: the cold start holds each seat's reset actuators;
- F5: per-seat elastic copies of the coupled constraints;
- F6: the patrol lead's gains frozen as patrol task constants;
- F7: a timing probe before the full run.

The shared solver uses a Gauss-Newton Hessian (owner decision the same day;
see `oracle-solver.md` when it lands).

**What still applies** (owner decisions, `oracle-audit.md` and
`oracle-horizon-protocol.md`): the objective is the episode reward; trip
limits are hard constraints, elastic, with slack priced at the trip cost; one
shared constrained solver run to convergence; Cooper's T_c, one solve per
T_c with the move held; a failed solve applies the previous plan's next
action.

**One correction up front.** The sources guarantee that cooperative
iteration reaches the centralized optimum only for linear models, convex
costs and uncoupled input constraints. The formation has a nonlinear model
and coupled state constraints. What the sources still give, and what this
protocol keeps, is a team cost that never rises within a solve and a usable
plan at any iterate. At a fixed point with no coupled constraint active, the
plan is a stationary point of the team problem, the same kind of point the
centralized IPM returns, though not necessarily the same one (section 4).

Labels, as in the horizon protocol:

- **read**: a source states it.
- **derived**: computed from a read value, from the code, or a direct
  consequence of an accepted decision.
- **ours**: no source gives it, so it is fixed here, before any run.

## 1. Sources

| id | source | what it says | where |
| --- | --- | --- | --- |
| SVRWP | B. T. Stewart, A. N. Venkat, J. B. Rawlings, S. J. Wright and G. Pannocchia, "Cooperative distributed model predictive control", *Systems and Control Letters* 59 (2010) 460-469 ([in-press PDF](https://pages.cs.wisc.edu/~swright/papers/SteVRWP10.pdf), pages 1 to 10 = 460 to 469) | Each subsystem minimises the plantwide objective over its own inputs, the others held at the last iterate (3.6); next iterate is the convex combination (3.7) with w_1 + w_2 = 1, w_i > 0, "of the Gauss-Jacobi type"; given a feasible initial guess every iterate is feasible (Lemma 5), the cost is nonincreasing (Lemma 6), and it converges to the centralized optimum (Lemma 7). Warm start: shift the last plan one sample and append 0, from local information only. Stopping early retains stability. Uncoupled input constraints assumed (Remark 2). Coupled constraints give stable "yet suboptimal" feedback (Remark 5); the fix augments each subsystem's decision with the coupled inputs. Table 3: 1 iterate costs 76.8 % more than centralized, 10 iterates 9.88 % | abstract p. 1; sec. 2 p. 2; sec. 3.1.2-3.1.3 p. 3; sec. 3.1.7 p. 4; sec. 5 and 6 pp. 6-7; Table 3 p. 9 |
| VHRW | A. N. Venkat, I. A. Hiskens, J. B. Rawlings and S. J. Wright, "Distributed MPC strategies with application to power system automatic generation control", *IEEE TCST* 16(6) (2008) 1192-1206 ([PDF](https://web.eecs.umich.edu/~hiskens/publications/cst_nov08.pdf)) | Algorithm 1: given p_max and epsilon > 0, iterate while rho_i > epsilon for some i and p <= p_max; u_i^p = w_i u_i* + (1 - w_i) u_i^(p-1); rho_i is the norm of the change of subsystem i's input trajectory. p_max is "a design limit"; the user may stop earlier. Cost nonincreasing (Lemma 1). With a coupled input constraint, iterates stay feasible and the cost still falls, but "limit points ... need not be optimal" | p. 1194; Algorithm 1 and Lemmas 1-2 p. 1196 (read as a page image); sec. VII p. 1203 |
| RMD | J. B. Rawlings, D. Q. Mayne and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., 6th printing, chapter 6 ([PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-6th-printing.pdf)) | Output constraints are "softened with exact soft constraints" and moved into the objective. Uncoupled input constraints: iterates feasible, cost decreases, converged = centralized. Coupled: the first two hold, the third is lost (Figure 6.6, "stuck on the boundary"); one remedy adds a player that parcels out the shared resource. Conclusions hold "regardless of how many iterations". Nonlinear: nonconvexity can make the convex combination raise the cost (Figure 6.7). Distributed gradient algorithm: each player takes a projected gradient step with Armijo backtracking; the combined step must pass test (6.33), else the player with the least improvement gets w = 0 and the rest are renormalised; Lemma 6.18: feasible, cost decreasing, every accumulation point stationary; different starts reach different local minima. A terminal state constraint is a coupled input constraint, so it is moved into the penalty. Nonlinear warm start: shift and append a terminal controller's move. The problem is not expected to be solved to optimality each sample (suboptimal MPC). Example 6.22: p = 3 stabilises, p = 10 is close to centralized | sec. 6.3 p. 400; 6.3.1 pp. 402-404; 6.3.2 pp. 405-406; 6.4 pp. 412-414; 6.5.1 pp. 415-417; 6.5.2 pp. 417-419; 6.5.3 pp. 419-422; 6.5.4 pp. 422-425 |
| CSML | P. D. Christofides, R. Scattolini, D. Munoz de la Pena and J. Liu, "Distributed model predictive control: A tutorial review and future research directions", *Comput. Chem. Eng.* 51 (2013) 21-41; read in its CPC VIII (2012) copy ([PDF](https://skoge.folk.ntnu.no/prost/proceedings/cpc8-focapo-2012/data/papers/002.pdf)), cited by PDF page | Cooperative DMPC: "the same global cost function is optimized" by every controller, each needing the full dynamics and objective; the first iterate starts from the shifted last plans; the convex combination is "of the Gauss-Jacobi type"; the problem statement carries state constraints x in X. Properties are "strongly based on convexity"; for nonlinear constrained systems there is no guaranteed convergence to the centralized optimum. Termination: an iteration cap, and/or a threshold on the change of cost or solution between iterates, and/or a time limit; one scheme applies the plan with the smallest cost found. Formation practice (non-cooperative): vehicles with decoupled dynamics and coupled constraints, controllers evaluated in sequence (Richards and How 2007), decentralized RHC for vehicle teams (Keviczky et al. 2006, 2008), compatibility constraints (Dunbar and Murray 2006); these reach Nash equilibria when the costs are selfish | sec. V-A p. 7; V-B pp. 8-9; V-B.2 pp. 10-11 |

What the sources share: every seat minimises the one team objective over its
own inputs, with the others' latest plans fixed; the seats' answers are
combined by a convex combination (Jacobi); the cost never rises; any iterate
can be applied. What they do not share: optimality at convergence, which
needs convexity and uncoupled constraints.

## 2. Decomposition

### 2.1 The formation's real coupling

Read from `src/target_gym/formation/env.py` and `env_jax.py`.

| element | depends on | coupling |
| --- | --- | --- |
| dynamics | each aircraft on its own three controls (`compute_next_state_formation`, a vmap of `compute_next_state_3d` over the fleet) | none |
| gust | the step clock only; in the plan model (noise zeroed) it decays as (1 - theta dt) g, the same for every seat | exogenous |
| route: target heading, turn schedule, time; slots (back, right, up) | drawn at reset or advanced by the clock | exogenous |
| lead altitude and heading cost | lead | lead only |
| wingman i slot cost: offset from the lead, in the lead's heading frame (`slot_error_vector`) | wingman i, and the lead's position and heading | lead and wingman i |
| wingman i alignment cost: psi_i - psi_lead | wingman i, lead | lead and wingman i |
| altitude band, each aircraft | that aircraft | none |
| slot loss: wingman i at 1500 m from its slot | wingman i, lead | lead and wingman i |
| separation, any pair within 25 m (`min_separation`) | the pair | lead-W0, lead-W1, W0-W1 |

So the formation is dynamically decoupled and coupled only through the cost
and the constraints (derived). The wingmen share no cost term; they couple
only through their separation. This is the class the review calls decoupled
systems with coupled constraints, with vehicle formations as its example
(read: CSML p. 7).

### 2.2 What each seat owns and holds fixed

| | lead sub-problem | wingman i sub-problem |
| --- | --- | --- |
| decision variables | lead's 3 inputs over the horizon, its state at each shooting node, its own slacks | wingman i's 3 inputs, its states, its own slacks |
| objective | every team-cost term that depends on the lead: its altitude and heading, and both wingmen's slot and alignment terms | wingman i's slot and alignment terms |
| own constraints | lead's dynamics, input box, altitude band | wingman i's dynamics, input box, altitude band |
| coupled constraints, against fixed trajectories | slot loss of W0 and W1; separations lead-W0, lead-W1 | its slot loss; separations with the lead and with the other wingman |
| held fixed | both wingmen's latest plans and the state trajectories they imply | the lead's and the other wingman's latest plans and trajectories |

- **Each sub-problem minimises the team cost** (read: SVRWP (3.6a), RMD p.
  413, CSML p. 9). Terms that do not depend on the seat's variables are
  constants and are dropped from its sub-problem; they do not change its
  minimiser (derived). The full team cost is still evaluated in the
  combination test (section 3).
- **The other seats' trajectories are parameters, not variables.** SVRWP
  (3.6b) puts the whole plant model in each sub-problem. Here the dynamics are
  decoupled, so another seat's states depend only on its own inputs. They are
  simulated once per iterate with the plan model and passed in (derived).
- **Horizon and grid.** One horizon H_k = min(P, N_ep - k) and one T_c for all
  seats, from the horizon protocol (the lead's rows as plane3d_heading, each
  wingman's as patrol, the largest P over rows, H5). Moves sit on the T_c grid
  and are held within a stage (derived).
- **Objective weights.** The team cost is the plain sum the reward defines, so
  rho_i = 1 for every seat (derived: the objective is the episode reward).
- **Size.** By the count that gives the joint 6,400 variables at 60 stages
  (26 state leaves and 3 inputs per aircraft), a seat has about 1,800, fewer
  once constant leaves are dropped (derived). The two wingman sub-problems
  have one structure, so one compiled program serves both, vmapped; the lead
  needs a second (derived). Whether this compiles and solves in budget is for
  the timing probe (decision F7).

## 3. The iteration

At each solve (once per T_c), starting from the warm start u^0:

1. **Sub-solves, in parallel.** For each seat i, solve its sub-problem with the
   shared IPM, warm-started at its current plan and last multipliers, with
   the others at u^p. Call the result u_i*. If the solve fails, or if
   V(u_i*, u_-i^p) > V(u^p), set u_i* = u_i^p (ours: keeps each direction a
   descent direction, which RMD's test assumes).
2. **Combine.** u^(p+1) = sum_i w_i (u_1^p, ..., u_i*, ..., u_M^p) (read:
   SVRWP (3.7) and p. 7; RMD p. 413). The convex combination is taken on the
   input sequences; each seat's states are then re-simulated from the
   combined inputs (derived: the sources combine inputs and states follow
   from the model).
3. **Test.** Accept u^(p+1) only if V(u^(p+1)) <= sum_i w_i V(u_i*, u_-i^p),
   with V the full team cost including every elastic slack at its least value
   for that plan. If it fails, set w = 0 for the seat whose own candidate
   improved least, renormalise the others, and recombine; repeat (read: RMD
   (6.33) and the drop rule, p. 418). With one seat left the test holds
   trivially (derived).
4. **Stop or repeat** (section 4).

| # | choice | value | label and ground |
| --- | --- | --- | --- |
| I1 | Type | Jacobi: all seats solve against the same iterate, then combine | read: SVRWP p. 4, "Gauss-Jacobi type"; RMD pp. 413, 418; CSML p. 9. Gauss-Seidel is decision F1 |
| I2 | Weights | w_i = 1/3 for each seat; after a drop, equal shares of the rest (1/2, then 1) | read: w_i > 0 and sum 1 (SVRWP (3.7), VHRW Algorithm 1, RMD p. 418). Ours: the values; equal weights favour no seat |
| I3 | Sub-solve | the shared IPM to its own tolerance and cap, as the audit's decision 5 fixes them | accepted decision. RMD's own algorithm takes one projected gradient step per player instead (p. 417); see section 7 |
| I4 | Nonconvex safeguard | test (6.33) with the drop rule | read: RMD pp. 415-418. Needed because the convex combination can raise a nonconvex cost (RMD Figure 6.7) |
| I5 | Warm start | each seat's last applied plan shifted one T_c sample, the last move repeated; once H_k reaches the episode end, shifted with nothing appended | read: shift (SVRWP pp. 2, 4; RMD pp. 404, 422; CSML p. 9). Ours: the appended move. SVRWP appends 0 in deviation variables and RMD a terminal controller's move; neither exists here, and action 0 decodes to half power, not trim (`patrol.env.decode_action`) |
| I6 | Cold start (k = 0) | every seat holds the action that commands its reset actuator positions over the whole horizon | read: a feasible initial sequence is required (SVRWP p. 2; RMD p. 421). Ours: the sequence. It needs no expert, and the elastic form makes any sequence feasible |
| I7 | Synchrony | all seats solve on the same T_c grid from the same measured state | read: CSML p. 9, step 1 (one full state to all controllers) |

## 4. Stopping and what holds at any iterate

**What the sources guarantee, here.**

- **Cost decrease at every iterate:** V(u^(p+1)) <= V(u^p). Read: SVRWP Lemma
  6 and RMD p. 403 for the convex case; RMD Lemma 6.18(b) for the nonconvex
  case with the test. Derived here: step 1's guard plus step 3's test give the
  same inequality with IPM sub-solves.
- **Feasibility at every iterate.** Read: SVRWP Lemma 5, RMD Lemma 6.18(a),
  for convex input constraints. Derived here: with elastic slacks every plan
  is feasible in the elastic sense, and the test rejects any combination whose
  trip-priced violation outweighs its gain (section 5).
- **Any iterate may be applied.** Read: SVRWP abstract and sec. 2; VHRW p.
  1194; RMD pp. 414, 422 (suboptimal MPC).
- **What the iterate converges to.** Read: the centralized optimum only for
  convex costs and uncoupled constraints (SVRWP Lemma 7; RMD p. 414); a
  stationary point for the gradient algorithm (RMD Lemma 6.18(c)); no
  guarantee for nonlinear constrained problems (CSML p. 11). Derived: when no
  coupled constraint is active, the constraints left are each seat's own, so a
  plan where no seat can improve satisfies every seat's KKT conditions, which
  together are the team problem's.

**The stopping rule, fixed in advance.**

| # | rule | value | label and ground |
| --- | --- | --- | --- |
| S1 | Plans agree | stop when max over seats and moves of abs(u_i^p - u_i^(p-1)) <= eps_u, in action units ([-1, 1]) | read: the form, VHRW Algorithm 1 (rho_i, the change of each subsystem's input trajectory); CSML p. 10 (change of solution below a threshold). Ours: the infinity norm |
| S2 | Tolerance | eps_u = 1e-6 | ours: IPOPT's default `acceptable_tol` reused, so the outer loop stops no looser than the solver's acceptable level. About 8 float32 steps at 1.0, so the applied float32 action barely changes |
| S3 | Cap | p_max = 20 iterates per solve | read: a cap is a design limit (VHRW; CSML p. 10). Ours: twice the largest count the sources report (SVRWP Table 3 and RMD Example 6.22: 10, still 9.88 % off centralized in SVRWP; VHRW: 5) |
| S4 | No seat moves | stop when step 1 leaves every seat at its plan | derived: the next iterate would repeat this one |
| S5 | At the cap | apply the current plan and record that the cap ended the solve | read: any iterate may be applied (as above) |
| S6 | Failed solve | if every seat fails on the first iterate, the plan is the warm start, so the move applied is the previous plan's next action | derived: identical to the accepted failed-solve rule |

Only one stopping test is used. The team cost is recorded but does not stop
the loop, so a plateau where plans still move is not mistaken for agreement
(ours).

## 5. Coupled constraints

What the sources allow:

- **Convex coupled input constraints** keep feasibility and cost decrease but
  lose centralized optimality (read: RMD pp. 405-406, Figure 6.6; VHRW p.
  1203; SVRWP Remark 5).
- **State and output constraints** are softened into the objective (read: RMD
  p. 400), and a terminal state constraint counts as a coupled input
  constraint, so RMD moves it into the penalty (read: RMD p. 421). CSML writes
  x in X into each controller's problem but gives no property for it (read:
  p. 9).
- **Remedies for optimality:** augment each seat with the coupled inputs
  (read: SVRWP sec. 5), or add a coordinating player (read: RMD pp. 406, 417).
  Here a separation involves every input of both aircraft, so augmentation
  merges the coupled seats, and the lead's slot-loss and separation terms
  reach both wingmen: the result is the joint problem that failed (derived).
  RMD notes a coordinator may be no simpler than the centralized controller
  (p. 417). Neither is used.

What this protocol does (ours, built on the accepted elastic decision):

| # | constraint | treatment |
| --- | --- | --- |
| C1 | own altitude band, input box | hard in the seat's own sub-problem, elastic at the trip cost (accepted) |
| C2 | slot loss, separations | in every sub-problem of a seat the constraint involves, hard against the other seat's fixed trajectory, elastic at the trip cost. Each seat has its own slack copy |
| C3 | slack in the combined plan | not combined: recomputed at its least value for the combined trajectories, max(0, limit violation), and priced in V. Derived: for fixed trajectories that is the elastic form's optimal slack |
| C4 | effect on the test | a seat's own candidate meets every coupled constraint against the others' current plans whenever its sub-problem is feasible. A combination that breaks one pays the trip price and fails (6.33), and drops end at a single seat's candidate. So an accepted iterate creates no coupled violation the sub-solves avoided (derived) |

**Where optimality can stop short.** With a coupled constraint active, two
seats can each be optimal against the other and still miss a joint
improvement (RMD Figure 6.6). Each seat's multiplier on that constraint then
differs, which the record shows (section 6). In normal flight the V keeps
these constraints far from active: lateral spacing is at least 60 m per rank,
so wingmen in their slots are at least 120 m apart and each at least 160 m
from the lead (slot back 150 to 300 m), against a 25 m separation limit and a
1500 m slot-loss limit (derived from `FormationParams` and `PatrolParams`).
Spawn noise of 40 m (one sigma) can bring a pair closer early in an episode.

## 6. Certification: what a record stores per solve

| field | meaning | label and ground |
| --- | --- | --- |
| k, H_k, T_c | sample, horizon in stages, sample time | horizon protocol |
| iterates | p used | read: VHRW, p and p_max |
| stop reason | S1 tolerance, S3 cap, S4 no seat moved, S6 all failed | ours |
| V_warm, V_applied | team cost of the warm start and of the applied plan; V_applied <= V_warm must hold | read: cost decrease (section 4); a check, not a tuning knob |
| plan gap | the S1 quantity at the last iterate | read form: VHRW rho_i. This is the agreement gap |
| seat gain | at the last iterate, the largest V(u^p) - V(u_i*, u_-i^p) over seats: what any one seat could still gain | derived: the rule's "finds nothing better", per seat |
| multiplier gap | for each active coupled constraint, the difference between the two seats' multipliers | derived: zero at a joint KKT point; nonzero flags the RMD Figure 6.6 case |
| drops | per iterate, how many seats the test dropped | read: RMD drop rule |
| per seat, per iterate | IPM status, iterations, KKT error, constraint violation, largest slack | the audit's rule 5 certificate (decision 6) |
| fallback | whether the applied move is the previous plan's next action | accepted decision |

Per episode, the record summarises: the share of solves stopped by S1, S3,
S4 and S6; the largest plan gap, seat gain and multiplier gap; and the
number of fallbacks. Records are compared within a stated tolerance, as the
audit's decision 6 says.

## 7. Patrol and patrol_bearing_only

**Single-seat already.** `PatrolState` carries the follower, the lead and
the lead autopilot's memory (`lead_pid`). The action is the follower's three
inputs only. `step_env` advances the lead with the 3D heading autopilot inside
the transition (`patrol/env.py`, `step_lead` and
`compute_next_state_patrol`), and nothing in `step_lead` reads the follower.
So the oracle controls one seat, and the lead's future is a function of the
state alone in the plan model (derived).

- The shared solver applies directly, with n_u = 3 and no cooperative loop
  (derived).
- Separation and slot loss become the follower's own state constraints
  against a lead trajectory that does not depend on the decision. It can be
  simulated once per solve (derived).
- The lead autopilot's clips (bank limit 25 deg) sit in the plan model, so
  the accepted smoothing rule for plant kinks applies to them (derived).
- `patrol_bearing_only` has the same transition (`PlanePatrolBearingOnly`
  differs only in `get_obs`, `patrol/env_jax.py`), and its oracle reads the
  true state (`patrol/PHYSICS.md`, section 5). Same problem (derived).

**One finding outside this spec.** The scripted lead's gains come from the
`plane3d_heading` entry of `src/target_gym/data/pid_gains.json`
(`make_plane3d_heading_pid`, called in `patrol/env_jax.py`). So a write to
that entry changes the patrol's plant. `gains_keys` feeds the baseline
fingerprint only, not the version stamp (`registry.py` docstring). Decision
F6.

## 8. Where the sources stop

1. **Converged sub-solves on a nonconvex problem.** RMD's nonlinear algorithm
   takes one projected gradient step per player, and Lemma 6.18(c)
   (stationary limits) is proven for that. With IPM sub-solves the cost
   decrease still holds (section 4); stationarity of limits is not covered.
2. **Hard coupled state constraints.** The sources soften them (RMD p. 400)
   or treat them as coupled input constraints (RMD p. 421). Per-seat elastic
   copies with a recomputed slack (C2, C3) are ours.
3. **Numbers.** No source gives weight values, eps or p_max (I2, S2, S3).
4. **Warm-start tail and cold start for aircraft.** The sources append 0 or a
   terminal controller's move and require some feasible initial sequence (I5,
   I6 are ours).
5. **Gauss-Seidel.** The review describes sequential schemes (Liu et al.,
   Lyapunov-based; Richards and How, non-cooperative), but no source read
   here states the properties of a cooperative Gauss-Seidel sweep.
6. **Formation practice is non-cooperative.** Per CSML p. 7, vehicle-formation
   DMPC uses selfish costs with sequential updates or compatibility
   constraints and reaches Nash equilibria. The cooperative scheme comes from
   process control (SVRWP, VHRW, RMD), not from formation flight.

## 9. Decisions for the owner

Recommended option first.

| # | decision | options |
| --- | --- | --- |
| F1 | Iteration type | Jacobi, w = 1/3, with RMD's test (the cooperative MPC of every source; parallel; one program for both wingmen). Or Gauss-Seidel, lead then W0 then W1, each against the latest plans (no weights or test needed for cost decrease; serial; the order is ours; no source here states its properties) |
| F2 | Stopping | eps_u = 1e-6 on the plan change, cap 20. Or cap 10, the sources' largest reported count (cheaper; SVRWP still 9.88 % off centralized at 10). Or no cap (agreement always reached, unbounded time) |
| F3 | Warm-start tail | Repeat the last move. Or append the action that commands the reset trim. Or append action 0 (SVRWP literally; half power, so it biases the tail) |
| F4 | Cold start | Hold each seat's reset actuator positions (no expert). Or one cold joint solve at k = 0 (the problem that failed to compile) |
| F5 | Coupled constraints | Per-seat elastic copies with the test (C2 to C4; optimality can stop short only while one is active, which the multiplier gap shows). Or merge coupled seats (SVRWP sec. 5; becomes the joint problem). Or a coordinator (RMD p. 406; RMD warns it may be no simpler than centralized) |
| F6 | Patrol lead gains | Freeze the scripted lead's gains as patrol task constants, so an expert retune cannot move the plant. Or keep reading `pid_gains.json`, and bump the patrol tasks whenever that entry changes |
| F7 | Sequencing | A timing probe first: compile both sub-problem programs and run one cooperative solve from a training-seed reset, against the 15 minutes the joint NLP missed. Then the full formation run. Or implement directly |

## 10. Source check

Each source was downloaded twice on 2026-11-03 (identical checksums), and
every quoted phrase was found in pdftotext output of the second copy.

- **Confirmed:** SVRWP, VHRW, RMD (chapter 6), CSML. VHRW's Algorithm 1 was
  read as a page image, since its symbols do not survive text extraction.
- **SVRWP** was read in its in-press layout (pages 1 to 10). The published
  pages 460 to 469 are from CSML's reference list.
- **CSML** was read in its CPC VIII conference copy, whose text matches the
  journal abstract. The journal's own page numbers were not checked.

Not read, and not relied on beyond what is stated:

- Stewart, Wright and Rawlings, "Cooperative distributed model predictive
  control for nonlinear systems", *J. Process Control* 21 (2011) 698-704: the
  primary for RMD section 6.5. Only RMD's summary was used.
- Richards and How 2007 (*Int. J. Control* 80, 1517-1531), Keviczky,
  Borrelli and Balas 2006 (*Automatica* 42), Keviczky, Borrelli, Fregene,
  Godbole and Balas 2008 (*IEEE TCST* 16(1)), Dunbar and Murray 2006
  (*Automatica* 42): known only through CSML's summary.
- Bertsekas and Tsitsiklis, *Parallel and Distributed Computation*, which
  SVRWP cites for the Gauss-Jacobi iteration: not read.
