# Oracle horizon protocol

Specification of 2026-11-03, for owner review. Nothing here has been run or
implemented. It replaces the audit's rule B, max(expert settle, 5 x tau_63),
which had no source (`oracle-audit.md`, section 5, Horizon).

> Amendment 3, 2026-11-04 (section 12): a rate limit's ramp joins P (H21).
>
> Amendment 1, 2026-11-03: after the first run (8 of 14 identifications
> failed), the owner accepted H14 to H20 and every recommendation D8 to D14.
> Bump tests start from steady operation, after one episode under the
> expert with the moves held at its last 20 commands, which amends H8.

> Amendment 2, 2026-11-03: D9's sigma counted plant disturbances that the
> subtracted test (C1) already cancels, so good fits failed V1 (hvac 91 %,
> unstable_cstr 96 %). The owner replaced it, added the pure-gain case, and
> changed D13. All three rules are ours: no source covers a subtracted step
> test. They are judged on measuring noise, not on the horizons they give.

The owner's rule applies throughout. Every value comes from one fixed recipe,
run once on bump tests from training seeds 100 to 109. No score, no held-out
seed and no menu picks a value. Held-out seeds 200 to 209 are not used at all.

**Owner decisions, 2026-11-03.**

- **Sample time:** the controller's sample time is COOPER's
  T_c = max(0.1 tau, 0.5 theta), not the task's step. This replaces H2 as
  first drafted (T = the task's step, labelled ours), which put horizons in the
  hundreds of steps against the 30 to 120 samples that COOPER, SEB and DMCP
  give.
- **Execution:** the oracle solves once every T_c and holds its move in
  between, as a DMC controller executes. The task still steps every T.
- **Everything else as recommended:**
  - D1: free moves at every T_c sample;
  - D2: P = N + M;
  - D3: trip-limited CVs count in the maximum;
  - D4: an elastic terminal line-out on unstable_cstr and the boiler level;
  - D6: the aircraft tests through ArduPlane 4.0.9 default loops;
  - D7: this protocol runs before the oracle re-record, independent of the
    expert retune.
- **D5 (blocking) falls away:** moves sit on the T_c grid by construction.

Where the text below still says T, read T_c. Section 6's provisional numbers
are in task steps at T = the task step and are superseded; the protocol
recomputes every horizon in T_c samples.

Labels, as in the expert protocols:

- **read**: a source states it.
- **derived**: computed from a read value, or a direct consequence of an
  accepted decision.
- **ours**: no source gives it, so it is fixed here, before any run.

Notation. T is the controller's sample time T_c (H2), and dt the task's step
(`delta_t`), fixed by the task. tau and theta
are a fit's time constant and dead time. Int(x) is the integer part (floor),
as in Cooper. N_ep is the episode length in T_c samples, and k the current
sample.

## 1. What practice prescribes

| id | source | what it says | where |
| --- | --- | --- | --- |
| COOPER | D. J. Cooper, *Practical Process Control*, Control Station, 2005 ([PDF](https://apmonitor.com/che436/uploads/Main/PPC_Textbook.pdf)). Cooper is the author of Shridhar and Cooper 1997 and 1998 and of Dougherty and Cooper 2003; this book states their DMC recipes as practitioner guides | Self-regulating: fit FOPDT; P = N = Int(5 tau / T) + k with k = Int(theta / T) + 1, "the process settling time in samples"; M = Int(tau / T) + k, the time to 60 % of the open-loop response; T near the larger of 0.1 tau and 0.5 theta. Integrating: stabilise with a simple P-only loop if needed, step the setpoint, fit a FOPDT-integrating model; tau_CL = theta sqrt(10); P = N = Int(5 tau_CL / T) + k; M = Int(tau_CL / T) + k; T <= 0.5 theta. N must equal the open-loop settling time to avoid truncation | Appendix C.1 p. 283, C.2 p. 284; section 23.4 pp. 252-253 |
| SEB | Seborg, Edgar, Mellichamp and Doyle, *Process Dynamics and Control*, 3rd ed., Wiley 2011, ch. 20 ([PDF](https://skoge.folk.ntnu.no/vgprosessregulering/papers-pensum/seborg-c20ModelPredictiveControl.pdf)) | N dt = t_s, the open-loop settling time, typically 30 to 120; P = N + M "so that the full effect of the last MV move is taken into account"; a different P per output if settling times differ; infinite P has theoretical advantages; M: 5 to 20 and N/3 to N/2; input blocking cuts computation. Pretest: each MV bumped by a small step, loops on the outputs in manual, gains and settling times estimated; plant-test duration set by the longest settling time seen. Exercises use the 99 % settling time t99 | section 20.6 p. 429; section 20.7 pp. 434-435; exercises 20.1 and 20.2 |
| QB | S. J. Qin and T. A. Badgwell, "A survey of industrial model predictive control technology", *Control Eng. Practice* 11 (2003) 733-764 ([PDF](https://skoge.folk.ntnu.no/prost/publications_various/mpc_survey_qin.pdf)) | P is set long enough to capture the steady-state effects of all computed future moves; performance improves as M grows, at the expense of computation; move blocking options exist; the pretest obtains the time to steady state of each CV; DMC-plus forces ramp (integrating) CVs to line out at steady state; step and impulse models are limited to strictly stable processes | p. 747; p. 753; p. 757; p. 759 |
| DMCP | AspenTech DMCplus Build documentation, entries reproduced in V. J. Cuadrado Alvarado, Univ. Sevilla thesis, Anexo IV ([PDF](https://biblus.us.es/bibing/proyectos/abreproy/3886/fichero/Cap%C3%ADtulos%252FAnexo+IV.+Entradas+de+Build.pdf)) | SSMINS is the model "time to steady state"; the control interval is SSMINS / IPXNCI, so the model spans exactly the time to steady state in 30 or more coefficients; the move calculation may run at a coarser interval than the prediction (IPSKIP, above 5 not recommended); without extended moves 30 coefficients, 15 predictions, 8 moves; with them up to 120 coefficients and 14 moves; with the composite optimiser, the move calculation drives each variable to its steady-state target at the end of the horizon | annex pp. 173, 182, 220, 269 |
| MW | MathWorks, MPC Toolbox, ["Choose Sample Time and Horizons"](https://www.mathworks.com/help/mpc/ug/choosing-sample-time-and-horizons.html) | try p with p Ts near the desired closed-loop response time; increase p until further increases change little; p above 50 rarely needed unless Ts is too small; for an open-loop unstable plant, p is bounded above by the steps before the open-loop step response becomes infinite; default m = 2; blocking is a more flexible alternative to m; p must let the controller anticipate constraint violations | web page, fetched twice 2026-11-03 |
| RMD | Rawlings, Mayne and Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Nob Hill, 6th printing ([PDF](https://sites.engineering.ucsb.edu/~jbraw/mpc/MPC-book-2nd-edition-6th-printing.pdf)) | a batch process that stops at N uses the finite-horizon controller, and stability is then not relevant; adding a terminal cost is generally advantageous; the infinite-horizon optimum is reached if the horizon is long enough; for unstable systems, constraints zero the unstable modes at the horizon's end, N at least the number of unstable modes; batch processes are run on shrinking horizons | p. 22; p. 166; p. 168 (notes); section 6.3 p. 401; p. 574 |
| AS | M. Alhajeri and M. Soroush, "Tuning guidelines for model-predictive control", *Ind. Eng. Chem. Res.* 59 (2020) 4177-4191, the authors' update of Garriga and Soroush 2010 ([accepted manuscript](https://par.nsf.gov/servlets/purl/10299968)) | upper prediction horizons "should be large enough"; Table 2 lists Shridhar and Cooper's MIMO rule as the maximum over inputs j of (theta_ij + 5 tau_ij) / ts + 1, Yamashita et al. 0.8 times the largest open-loop settling time, and others; Table 3 lists control horizons from 1 to P; longer control horizons raise aggressiveness and lower robustness (citing Seborg) | section 3.1.1 Tables 1-2; 3.1.2 Table 3; 3.1.3 |
| SS | T. Stastny and R. Siegwart, "Nonlinear model predictive guidance for fixed-wing UAVs using identified control augmented dynamics", ICUAS 2018 ([arXiv 1802.02624](https://arxiv.org/abs/1802.02624)) | guidance NMPC predicts on the autopilot-augmented (stabilised) dynamics, identified from flight tests; the 7 s horizon was set by computation, not by a rule | abstract; section V |

What the sources share: the horizon must see the full effect of a move on
every controlled variable. For a stable plant that means the open-loop time to
steady state (COOPER, SEB, QB, DMCP). For an integrating or unstable plant,
which has no open-loop steady state, practice stabilises it first and sizes the
horizon on the stabilised response (COOPER C.2, QB, SS). The episode end cuts
the horizon (RMD).

## 2. The rule

For each pair of a manipulated variable (MV) and a controlled variable (CV),
on each training seed, a bump test gives a fit, and the fit gives

    N = Int(5 tau_x / T) + k        settling time in samples
    M = Int(tau_x / T) + k          control horizon
    P = N + M                       prediction horizon

with tau_x = tau for a self-regulating CV and tau_x = tau_CL = sqrt(10)
theta_e for an integrating one (section 4). The task's horizon is the largest P
over all pairs and all ten seeds. At step k of an episode the planner uses

    H_k = min(P, N_ep - k).

So a task whose P reaches the episode end plans to the end, and every task
shrinks over its last P steps.

## 3. Shared choices, fixed before any run

| # | choice | value | label and ground |
| --- | --- | --- | --- |
| H1 | Settling | 99 % of the final change, computed as 5 tau + theta on the fit | read: COOPER step 3 ("5 tau", 99.3 % for a FOPDT); SEB exercises use t99 |
| H2 | Sample time | T_c = max(0.1 tau, 0.5 theta) from the fit, rounded down to a whole number of task steps, at least one. With several pairs, the smallest T_c over pairs and seeds, so no pair is undersampled, unless Dougherty and Cooper 2003 gives its own MIMO rule, which then wins. The oracle solves every T_c and holds the move between solves | read: COOPER step 2 (T_c); DMC executes once per sample (COOPER, SEB, DMCP). Ours: the rounding, and the pair rule unless the MIMO source gives one. Owner decision, 2026-11-03 |
| H3 | Dead time in k | k = Int(theta_fit / T) + 1, theta_fit without the expert protocols' half sample | derived: COOPER's "+1" already counts the sample |
| H4 | Dead time in tau_CL | theta_e = theta_fit + T/2 + tau2/2 | read: SIMC eq. 5.12 half sample (expert protocols C4) and SIMC's half rule for a lag, as the aircraft protocol 3.5 applies it. Derived: theta_e never vanishes, so tau_CL > 0 |
| H5 | Which fit | the largest P over the ten training seeds and over every MV-CV pair | read: SEB, the longest settling time sets the test; AS Table 2, Shridhar and Cooper's maximum over pairs; DMCP, one time to steady state per controller |
| H6 | CVs | every tracked output, plus every variable with a trip limit in `check_is_terminal` | read: QB, CVs carry limits; MW, anticipate constraint violations. Decision D3 |
| H7 | Step | expert protocols C2: existing steps stay; a new test steps 5 % of the MV's range, away from the nearer limit | ours (C2) |
| H8 | Response | stepped run minus unstepped run on the same noise, from each seed's reset state | read: bump test (expert protocols C1) |
| H9 | Window | the whole episode after the step; truncated at the first trip of either run | ours: SEB sizes tests by the longest settle, and no horizon can exceed the episode |
| H10 | Class | integrating if, over the window, the mean slope of the last quarter is at least half that of the first quarter; else self-regulating | ours: the criterion `scripts/measure_time_constants.py` already uses |
| H11 | Fits | FOPDT by the expert protocols' `fit_fopdt`; integrating by `fit_integrating_lag` (k', theta, tau2). No fit is dropped, whatever its residual | ours, as the expert protocols |
| H12 | Episode end | H_k = min(P, N_ep - k) | read: RMD p. 22 and p. 574. Derived: the objective is the episode reward, so nothing after N_ep counts |
| H13 | Record | per task, two files. A compact summary in the package, `data/oracle_horizons/<task>.json`: T_c, N, M, P, the pair and seed that set each, compliance, the class per CV and the label counts. The full evidence, every test, retest and fit, in `evidence/oracle_horizons/<task>.json` at the repository root: kept in git, not shipped in the wheel (review of #133: the evidence was nine times the package data) | ours |

## 4. Per class

### 4.1 Self-regulating SISO

COOPER C.1 as written, with T = T_c (H2). One MV, one tracked CV, plus the
limit CVs of H6. The open-loop bump of H8 is SEB's pretest. Tasks:
first_order, cstr, ph_neutralization, glass_furnace, reactor, hvac, battery.

For a FOPDT with no dead time, P is about 6 tau / T + 2. That is close to the
audit's 5 tau_63, but the new value comes from the ten-seed maximum and the
protocol's own test, and it adds the control horizon (D2).

### 4.2 MIMO

Each MV is stepped on its own, the others held (read: SEB p. 435, the
traditional practice; QB p. 747). Each MV-CV pair gets its fit and its P.
The task takes the largest (H5). SEB allows a different P per output, but one
plan has one horizon, so the longest output sets it (derived). Tasks:
four_tank, distillation, boiler_drum, cement_kiln, compressor_surge,
wind_turbine.

Distillation is ill-conditioned (LV). Single-MV steps excite the slow
direction, which is the one that sets the horizon. That is what H5 asks.

### 4.3 Integrating

COOPER C.2: the CV has no open-loop steady state, so the horizon covers the
closed-loop settle the dead time allows, tau_CL = sqrt(10) theta_e. In
simulation the open-loop bump of H8 can be read before the CV drifts out of
range, so no stabilising loop is needed for the process tasks (as the expert
protocols do). COOPER's P-only stabiliser is needed only where the plant
trips first. Integrating CVs (H10 decides): boiler_drum level, battery state
of charge, and the aircraft CVs of 4.5. Others may appear when H10 runs.

DMC-plus forces ramp CVs to line out at steady state at the horizon's end
(QB p. 753). Whether the oracle does the same is decision D4.

### 4.4 Open-loop unstable

Practice does not plan an unstable plant from an open-loop step. Step and
impulse models cannot represent it (QB p. 759), and COOPER C.2 stabilises
first. So the test runs on the stabilised plant: the base loop closed, the
base loop's setpoint stepped, a FOPDT fitted to the CV, and C.1 applied
(read). For unstable_cstr the base loop is the inner temperature PD the
expert protocol fixes (SIMC, Kp_T 8.49, Kd_T 0.946 min), and the test is the
existing 0.5 K setpoint step with the inner loop closed. The oracle still plans
the raw coolant; only the horizon is measured this way.

Theory adds a terminal condition. RMD section 6.3 zeroes the unstable modes at
the horizon's end, with N at least the number of unstable modes (one here).
DMCP drives each variable to its steady-state target at the end of the
horizon. For the oracle this is "the last planned state is an equilibrium
under the last planned input", elastic, with its slack priced at the trip cost
as the accepted limit decision prices slack (derived). Decision D4.

MW bounds p from above for unstable plants. That bound comes from
single-shooting predictions that blow up. A multiple-shooting solver, as the
compressor's IPM is, does not have it (derived; RMD ch. 8).

### 4.5 Aircraft guidance

No source gives a horizon rule for guidance. SS plans on the autopilot's
closed-loop dynamics and sized its horizon by computation. So the protocol
uses the integrating rule of 4.3 on the stabilised airframe, measured through
the base loops ArduPilot flies (ours: the mapping; read: COOPER C.2 stabilise
then step; SEB, pretest moves go through the existing loops; SS).

The base loops run at ArduPlane 4.0.9 defaults, not the tuned expert gains
(read values from the aircraft protocol, section 3.2 step 1, 3.3 and 3.4: roll
P 1.0 [1.27], D 0.08 [0.102]; pitch P 1.0 [1.27], D 0.04 [0.051], I 0;
NAVL1_PERIOD 17 s, damping 0.75; TECS defaults). That keeps the horizon fixed
when the expert is retuned (ours; decision D6).

| CV | test MV (stabilised command) | loops closed during the test | step | fit |
| --- | --- | --- | --- | --- |
| altitude | pitch demand | pitch and roll | 5 % of the pitch-demand range (C2) | integrating with lag |
| altitude | throttle | pitch and roll | 5 % (C2) | integrating with lag |
| heading | bank demand | roll | 5 % of +-45 deg, 4.5 deg (C2) | integrating with lag |
| distance to path (circle, racetrack, figure-8) | course demand | roll and course hold | 2 deg (aircraft protocol 3.5) | integrating with lag |
| slot error, along track | speed demand | TECS, roll | +5 m/s (3.5) | integrating with lag |
| slot error, across track | course demand | roll and course hold | 2 deg (3.5) | integrating with lag |

Each test flies from each training seed's reset state, in its turbulence, with
the response read against an unstepped run (H8). Limit CVs: the altitude band
is covered by the altitude rows. Separation (patrol, formation) is covered by
the slot rows (ours). The formation runs the lead's rows as plane3d_heading
and each wingman's as patrol.

The path rows read the distance one integration below the heading. The course
hold closes the heading, so the distance is again a single integrator with
lag, which C.2 covers (derived).

### 4.6 Episode end and shrinking horizons

Every task is scored on a fixed episode, so H12 shrinks every horizon over the
last P steps. When P >= N_ep the planner plans to the end from the first step,
which is the shrinking-horizon practice for batch processes (RMD p. 574). The
oracle may then ease off near the end, because the reward stops there. That is
correct under the accepted objective, the episode reward (derived).

Today cstr and four_tank plan to the episode end. Under the provisional
numbers of section 6, cstr would plan 32 steps and four_tank 960. Distillation
would plan to the end; so would wind_turbine and glass_furnace if their slow
limit CVs enter (D3).

### 4.7 Control horizon and move blocking

The sources disagree (section 7). The formula keeps COOPER's M because
P = N + M needs it (SEB). What the planner does with its moves is decision D1:

- **Free moves at every step** over the whole horizon (recommended). QB:
  performance improves with M, at a computational cost. The oracle is a problem
  definition, not a tuned controller. With multiple shooting the states
  dominate the problem size: (n_x + n_u) x P variables, so freeing the inputs
  costs little (derived).
- **DMC hold**: M free moves, then the input held for N steps (read: COOPER,
  SEB). Cheaper. On unstable_cstr and the aircraft a held raw input diverges
  in the tail, which would bias the plan.
- **Blocking** (D5), only if the timing probe the audit asks for fails a
  budget: hold each move for b = min(5, Int(T_c / T)) steps, with
  T_c = max(0.1 tau, 0.5 theta) (read: COOPER step 2; DMCP, IPSKIP above 5
  not recommended).

## 5. Data collection

Every test reuses the expert protocols' machinery: seeds 100 to 109, C1
differences, C2 steps, the shared fitters. What is new: the oracle's own MVs
(raw actuators, except 4.4 and 4.5), the whole-episode window, every MV
stepped alone, and the limit CVs.

| task family | existing test reused | new |
| --- | --- | --- |
| first_order, ph_neutralization | the expert protocol's bump (0.1 raw; 0.1 mL/s) | window to N_ep; limit CVs |
| cstr | none on the raw coolant to C_a (the expert's tests are on its cascade) | coolant 5 % |
| four_tank, distillation | none on raw pairs (the expert's tests are on decoupled or cascade loops) | each pump, reflux, boilup 5 % |
| glass_furnace | +0.05 raw firing (expert protocol section 6) | window to N_ep, outside the 40-sample fit |
| boiler_drum | firing +0.02, feedwater trim +2.0 kg/s | window to N_ep |
| cement_kiln | fuel step | kiln speed 5 % |
| battery | +0.1 raw command | state of charge, voltage and temperature read as limit CVs |
| compressor_surge | speed +1.75 %, recycle +0.05 | window to N_ep |
| reactor, hvac, wind_turbine | none | each MV 5 % |
| unstable_cstr | 0.5 K setpoint step, inner PD closed | window to N_ep |
| aircraft (10) | aircraft protocol 3.5's speed and course steps | pitch, throttle and bank steps on default loops (4.5) |

Cost: (number of test MVs + 1) runs of N_ep steps per seed, ten seeds per
task. The largest is four_tank, 3 x 10 x 1200 steps. No score is computed.

## 6. The 24 tasks

"Provisional" numbers are read from existing measurements, which are not the
protocol's test. They are to be re-measured by the protocol and are shown only
to give the scale.

- `tau63` is from docs/rl-protocol.md section 6: one seed, all actuators
  stepped together by 0.4 raw, the first tracked output only, and the 63 %
  time taken as tau with theta = 0. Where theta is in fact large this
  overstates N by up to 4 theta / T.
- `fit` is a median SIMC fit from the task's PHYSICS.md, half sample removed
  (H3).

Columns: SR self-regulating, INT integrating, UNST unstable. B is the audit's
rule B, for comparison.

| task | class | MVs -> tracked CVs; limit CVs | test | inputs to measure | provisional P (steps) | today | B | N_ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plane | aircraft | throttle, stick -> altitude; altitude band | 4.5 altitude rows | k', theta, tau2 | none | 20 | 115 | 280 |
| plane_energy | aircraft | as plane | as plane | as plane | none | 20 | 115 | 1600 |
| plane_sine | aircraft | as plane | as plane | as plane | none | 20 | 115 | 480 |
| plane3d_heading | aircraft | throttle, stick, aileron -> altitude, heading; band | altitude, heading rows | per row | none | 30 | 136 | 300 |
| plane3d_circle | aircraft | -> altitude, path distance; band | altitude, path rows | per row | none | 30 | 70 | 300 |
| plane3d_racetrack | aircraft | as circle | as circle | per row | none | 30 | 70 | 650 |
| plane3d_figure8 | aircraft | -> path distance; band | altitude, path rows | per row | none | 30 | 190 | 400 |
| patrol | aircraft | -> slot error; band, separation, slot limit | slot, altitude rows | per row | none | 30 | 84 | 200 |
| patrol_bearing_only | aircraft | -> measured range; as patrol | as patrol | per row | none | 30 | 84 | 200 |
| formation | aircraft, 3 seats | 9 -> lead altitude, 2 slot errors; bands, 3 separations | lead as heading, wingmen as patrol | per row | none | 30 | 180 | 200 |
| cstr | SR SISO | coolant -> C_a | coolant 5 % | k, tau, theta | 32 (tau63 5) | 150 (episode) | 67 | 150 |
| first_order | SR SISO | u -> x; abs(x) < 3 | existing 0.1 | k, tau, theta | 62 (fit: tau 10 steps, theta 0) | 5 | 63 | 150 |
| four_tank | SR MIMO 2x2 | 2 pumps -> h1, h2; all 4 levels | each pump 5 % | 8 pairs | 960 (fit, decoupled loop h1: tau 146 s, theta 41.4 s) | 1200 (episode) | 581 | 1200 |
| ph_neutralization | SR SISO, varying gain | base flow -> pH | existing +0.1 mL/s | k, tau, theta | 142 (fit: tau 117 s, theta 0) | 20 | 70 | 300 |
| distillation | SR MIMO 2x2, ill-conditioned | L, V -> yD, xB; yD, xB limits | each MV 5 % | 4 pairs | 200, the episode (slow mode about 160 to 194 min: 5 tau > N_ep) | 15 | 46 | 200 |
| glass_furnace | SR SISO, slow | fuel -> T_crown; crown, melt, working-end bands | existing +0.05, window to N_ep | 4 pairs | 786 (tau63 132, theta 2 steps); 1600 if the melt band enters (D3) | 120 | 660 | 1600 |
| reactor | SR SISO, fast | rod -> n; flux, fuel, coolant temperatures | rod 5 % | 4 pairs | 8 (tau63 1), tracked CV only | 20 | 5 | 864 |
| hvac | SR SISO, slow | heating -> T_air (also the band) | heating 5 % | k, tau, theta | 374 (tau63 62) | 24 | 310 | 720 |
| cement_kiln | SR, 2 MVs, delay-dominant | fuel, speed -> free lime; T_bz band, lime limit | fuel existing, speed 5 % | 6 pairs | 150 to 350 (tau63 58 with theta 50 or 0 steps; delay about 25 min) | 40 | 290 | 700 |
| boiler_drum | MIMO 2x2: level INT, pressure | fuel, feedwater -> level, pressure; level, pressure band | existing firing and feedwater steps | 4 pairs | none (tau63 3 is on the level, which integrates) | 30 | 15 | 400 |
| wind_turbine | SR, 2 MVs, fast; slow limit CV | pitch, torque -> power; rotor speed band | each MV 5 % | 4 pairs | 8 (tau63 1), tracked only; 400, the episode, if rotor speed enters with tau near 100 steps (D3) | 1 (law) | 5 | 400 |
| battery | SR SISO, fast; INT limit CV | power -> delivered power; SoC, voltage, temperature | existing +0.1 | 4 pairs | 8 (power, tau63 1); 10 (SoC, theta_fit 0 so theta_e = T/2) | none | 6 | 360 |
| unstable_cstr | UNST, measured stabilised (4.4) | coolant -> C_a; T < 365 K | existing 0.5 K setpoint step, inner PD closed | k, tau, theta per level | 62 (fit: tau 9.42 steps, theta 2.28 steps) | 32 | 99 | 1200 |
| compressor_surge | SR, 2 MVs | speed, recycle -> header pressure; surge margin | existing speed and recycle steps | 4 pairs | 297 (dp tau63 49.3) | 61 | 247 | 1200 |

Two existing horizon rules the audit passed are replaced by this shared rule:
unstable_cstr's 1.5 times the 21-step recovery, and the compressor's slowest
actuator move.

**Consequence for the solver.** Several horizons grow by a factor of 4 to 16
(first_order, ph_neutralization, glass_furnace, hvac, cement_kiln,
compressor_surge). The
audit's dense IPM forms n x n matrices with n = (n_x + n_u) x P. At P 786 on
glass_furnace (27 per stage) that is about 21,000, so a block-sparse, stage-wise
assembly is needed before these tasks run. The timing probe the audit asks for
should come before D5.

## 7. Where the sources disagree

1. **Open-loop or closed-loop time.** COOPER C.1, SEB, DMCP and QB size the
   horizon on the open-loop time to steady state. MW sizes it on the desired
   closed-loop response time, and COOPER C.2 on a closed-loop time constant.
   The protocol uses each where its source applies: open loop for stable CVs,
   closed loop for integrating and unstable ones.
2. **P = N or P = N + M.** COOPER sets P = N. SEB sets P = N + M, and QB's
   wording (the effects of all computed moves) agrees with SEB. Decision D2.
3. **Control horizon.** COOPER: tau / T + k, the time to 60 % of the open-loop
   response, which is about N / 5 for a delay-free plant. SEB: 5 to 20, and
   N/3 to N/2. DMCP: 8 moves over 30 coefficients, up to 14 over 120. MW:
   default 2. AS Table 3: from 1 up to P, with state-space methods at P. QB:
   performance grows with M. Decision D1.
4. **Sample time.** COOPER picks T from the plant (near 0.1 tau), so N is near
   50. DMCP scales the interval so the model spans its time to steady state in
   30 to 120 coefficients. SEB gives 30 to 120. MW: p above 50 is rarely needed
   "unless Ts is too small". Here T is fixed by the task and often far below
   0.1 tau, so N in steps runs to the hundreds (glass_furnace's 0.1 tau is
   13 steps, hvac's 6, four_tank's 15). D5 is practice's answer if compute
   forces one.
5. **Unstable plants.** MW gives an upper bound on p. RMD gives a lower bound
   (the number of unstable modes) plus terminal conditions. Industry (QB,
   COOPER) stabilises with a base loop first. The protocol measures on the
   stabilised plant and leaves the terminal condition to D4.
6. **Other published rules.** AS Table 2 also lists Yamashita et al. (0.8
   times the largest open-loop settling time), Maurath et al. (between the 80 %
   and 90 % rise times) and pole-placement lower bounds (theta / ts + 1). These
   are shorter than COOPER's. The protocol follows COOPER because it is the
   industrial DMC recipe and the only one with a full recipe for both stable
   and integrating plants.
7. **Why DMC ties P to N.** In DMC, N is the length of the step-response model,
   and too short an N truncates the prediction (COOPER 23.4). The oracle
   predicts with the plant itself, so truncation does not arise. The reason
   that remains is seeing the full effect of a move (SEB, QB), and that is the
   one the rule keeps (derived).

## 8. Decisions for the owner

Recommended option first.

| # | decision | options |
| --- | --- | --- |
| D1 | Moves within the horizon | Free at every step (QB; little extra cost with multiple shooting). Or the DMC hold after M (COOPER, SEB; cheaper, but a held raw input diverges on unstable_cstr and the aircraft) |
| D2 | Horizon length | P = N + M (SEB, QB). Or P = N (COOPER), shorter by M, about tau |
| D3 | Limit CVs in the maximum | Yes (QB, MW): wind_turbine and probably glass_furnace then plan to the episode end. Or tracked CVs only (cheaper, but trips come into view later) |
| D4 | Terminal line-out | On unstable_cstr and on integrating CVs with a setpoint (boiler_drum level): the last planned state is an equilibrium under the last input, elastic, priced at the trip cost (RMD 6.3; DMCP; QB p. 753). Not on the aircraft, whose references are periodic or paths (ours). Or no terminal condition anywhere |
| D5 | Move blocking | Only if the timing probe fails a budget you set, with b = min(5, Int(max(0.1 tau, 0.5 theta) / T)) (COOPER, DMCP). Or never |
| D6 | Aircraft base loops for the tests | ArduPlane 4.0.9 defaults, so the horizon does not move when the expert is retuned. Or the tuned expert loops (closer to SEB's "existing loops"; re-measure after each retune) |
| D7 | Sequencing | Run this protocol before the oracle re-record and independently of the expert retune. Only unstable_cstr reads an expert value (its inner PD), which comes from a fixed recipe |

## 9. Source check

Each key source was fetched twice on 2026-11-03.

- **PDFs**: the cited phrases were found in pdftotext output of two separate
  downloads. COOPER (Appendix C) and SEB (p. 429) were also read as page
  images, which confirmed sqrt(10) and P = N + M.
- **Web page**: MW was fetched twice with different questions.
- **Confirmed**: COOPER, SEB, QB, DMCP, MW, RMD, AS, SS.

Not read, and not relied on beyond what is stated:

- **Paywalled primaries.** Shridhar and Cooper 1997 (doi 10.1021/ie9604280)
  and 1998 (doi 10.1021/ie980202s), and Dougherty and Cooper 2003 (doi
  10.1021/ie020546p): only their abstracts, via OpenAlex. The 1997 abstract
  limits the method to open-loop stable processes; the 2003 abstract covers
  integrating processes. The formulas used here come from COOPER (same author)
  and AS Table 2.
- **Garriga and Soroush 2010** (doi 10.1021/ie900323c): not obtained. Its
  tables are in the 2020 review's supplement, which I could not retrieve.
- **Camacho and Bordons, Maciejowski**: not found online; not cited. GPC's
  "N2 near the rise time" (Clarke et al. 1987) appears only in secondary
  sources; not used.
- **DMCP** is AspenTech text reproduced in a university thesis, not AspenTech's
  own site. Current DMC3 documentation is not public.
- **IJACSA restatement.** A 2019 paper (Vol. 10 No. 6, paper 46) restates the
  MIMO rule as P = N = max over pairs of (5 tau_ij / T + k_ij). It is consistent
  with AS but garbled in places, so it is not cited for any number.

## 10. Amendment 1, identification validity and retests

Specification of 2026-11-03, for owner review. Nothing here has been run or
implemented. It amends H7 to H11 and adds H14 to H20, so that a bump test is
judged before its fit can set a horizon, and repeated when it fails.

**Why.** The protocol ran on training seeds 100 to 109 for the 14
non-aircraft tasks (PR #133, `src/target_gym/data/oracle_horizons/`). Six
tasks gave plausible horizons. Eight failed through the protocol's own rules:

- H11 kept every fit. Integrating fits with their lag at the fitter's floor
  (1e-9 s) and zero delay forced T_c to one task step on glass_furnace,
  cement_kiln, boiler_drum, distillation, wind_turbine, battery and
  ph_neutralization. The kiln's lime FOPDT fit has tau 1.1e8 s on a response
  no bigger than the noise, which gave P = 22 million steps.
- Integrating outputs were bumped open loop. Every boiler_drum response
  tripped.
- Classes were mixed across seeds or pairs on pH, distillation, glass crown,
  kiln, boiler, wind and battery T_cell.
- Trips cut the windows on boiler, wind, distillation, kiln and hvac.

**Owner decisions, 2026-11-03.**

- **H8 amended.** Bump tests start from steady operation: one episode under
  the shipped expert, then the moves held at the mean of its last 20
  commands (the glass furnace: its last reversal cycle). This replaces "from
  each seed's reset state".
- **Amend, then rerun all 14 tasks.**

**The binding rule.** Every rule below is fixed before any rerun. It comes from
published or practitioner practice where a source exists, and is labelled ours
otherwise. No rule reads P, and none targets a horizon value. A fit is judged
on its own quality only.

### 10.1 Sources

The ids of section 1 still hold. New pages and three new sources:

| id | source | what it says | where |
| --- | --- | --- | --- |
| COOPER | as section 1 | Tests run at the design level of operation; the PV should move at least 10 times the noise band; if a disturbance affects the test, it is repeated. Five checks on test data: steady state before the test, first point at that state, the response "clearly dominates" the noise, quiet disturbances, the fit visually matches the data. The noise band is plus or minus 3 standard deviations of the PV with the controller output constant at steady state; the signal to noise ratio should exceed ten, "a minimum specification". Figure 6.4 draws the band as the full plus or minus 3 sigma width. Fits minimise the squared error of the model "forced by the manipulated variable data". Closed-loop tests step the setpoint, with the controller output moving the PV more than 10 noise bands. A non-self-regulating process does not reach a natural steady state. P-only gains: Kc = (0.2 / Kp)(tau / theta)^1.22 (ITAE, set point); for an integrating process Kc = 1 / (K* theta) (D.2) | p. 24 (3.1); p. 51 (6.1); p. 54 (6.4); p. 55 (Fig. 6.4); p. 56 (6.5); p. 59 (6.6); p. 17 (2.5); p. 43 eq. 5.4; C.1 p. 283; C.2 p. 284; D.2 p. 286 |
| SEB | as section 1 | Pretest bumps estimate gains and settling times. "Too small a move" is obscured by normal fluctuations and noise; too large a move violates output constraints or leaves the linear range. The largest allowed input change is (Delta u_j)max = (Delta y_i)max / K_ij (eq. 20-75). The longest step lasts the longest settling time seen in the pretest. Periods of upsets or abnormal situations (valve saturation) are removed. Model accuracy is characterised by confidence intervals on the parameters (Ljung 1999) | section 20.7, pp. 434-435 |
| QB | as section 1 | The pretest obtains the time to steady state of each CV. Each MV is stepped 8 to 15 times, with a CV signal to noise ratio "at least six". Some packages move several MVs at once. Operators may intervene to avoid critical situations; data from several test periods may be merged after cutting corrupted stretches. Parameters are fitted by least squares on the output error (eq. 24). Some products give model uncertainty bounds | p. 747; p. 748 |
| DMCP | as section 1 | ISRAMP, the ramp indicator, is set per dependent variable (CV): not a ramp, a true ramp, or a pseudo-ramp | annex p. 223 |
| ZP | Y. Zhu and R. Patwardhan, "Towards a low cost and high performance MPC: the role of system identification", CPC VIII / FOCAPO 2012 ([PDF](https://skoge.folk.ntnu.no/prost/proceedings/cpc8-focapo-2012/data/papers/059.pdf)) | Each identified transfer function gets a 3 sigma error bound and a grade, from the bound's size against the model at low and middle frequencies: A very good, B good, C marginal, D "poor, or, no model exists". C and D models are zeroed where no model is expected; where one is expected, the test is modified: larger amplitudes and a longer test reduce model errors. Industrial model matrices have many zeros. A model is used if graded A, B or C and its gain sign matches the expectation. Models that could not be identified leave empty columns, reported | section 3, PDF p. 3; Automatic Model Selection, PDF p. 6; PDF p. 4 |
| DHN | M. L. Darby, M. Harmse and M. Nikolaou, "MPC: current practice and challenges", ADCHEM 2009 preprint ([PDF](https://skoge.folk.ntnu.no/prost/proceedings/adchem09/cd/abstract/239.pdf)) | Manual tests move one MV at a time, moving other inputs as needed to keep the process in its region. Amplitudes keep inputs and outputs in range; larger moves are preferred while the response stays linear; aim for a signal to noise ratio of at least 6 to 1. Whether a CV is integrating is often set by process knowledge (level to flow); slow stable variables, slower than the prediction horizon, often control better as integrators | Plant testing, PDF p. 6; Identification methods, PDF p. 7 |
| MWC | MathWorks System Identification Toolbox, [`compare`](https://www.mathworks.com/help/ident/ref/compare.html) | The NRMSE fit, 100 (1 - norm(y - yhat) / norm(y - mean(y))) percent. No threshold is given | Output Arguments, `fit` |

What the sources share. A step test is accepted when the response clearly
dominates the noise, the test starts at steady state, it lasts the settling
time, and the model visibly fits (COOPER, SEB, QB, DHN). When a test fails,
practice repeats it with a larger move or a longer duration, inside the
limits (SEB, ZP, DHN). Integrating CVs are tested in closed loop with a
setpoint step (COOPER C.2, 6.6). Pairs with no meaningful effect get a zero
model (ZP). The ramp class belongs to the CV (DMCP).

### 10.2 Amended and new choices

Notation. W is the test window in task steps, dy the response (stepped run
minus unstepped run), sigma the CV's noise, B = 6 sigma the noise band
(COOPER's plus or minus 3 sigma), and SNR = max |dy| / B. A pair-seed is one
MV-CV pair on one training seed. A test is one MV stepped on one seed; it reads
every CV.

| # | choice | value | label and ground |
| --- | --- | --- | --- |
| H2, H5 | Which fits count | Only valid fits (H19) enter T_c's minimum, the dead time k and P's maximum | derived: a fit H14 rejects does not describe the plant |
| H7 | Step | The H7 step, then escalated by H15 | as before, plus H15 |
| H8 | Response | Stepped run minus unstepped run on the same noise, from steady operation (owner decision above) | read: COOPER p. 24, tests at the design level of operation; p. 51, steady state before the test |
| H9 | Window | N_ep task steps after the step, cut at the first trip of either run; the record names the run and the CV that tripped. Lengthened by H15 | ours (as before); read: SEB p. 435 and QB p. 748, abnormal stretches are cut |
| H10 | Class | Replaced by H17 | |
| H11 | Fits | Replaced by H14 and H19. Both structures (FOPDT; integrating with lag) are fitted to every response, by output-error least squares on the recorded input | read: QB eq. 24 p. 748; COOPER p. 56, the model is driven by the recorded MV |
| H13 | Record | Extended by H19 | ours |
| H14 | Fit validity | All four checks of 10.3 pass | read: COOPER 6.1, 6.4; SEB p. 435; MWC. Ours: the fit floor (D10) |
| H15 | Retests | At most two per test; remedies by failed check (10.4) | read: SEB p. 435; ZP; COOPER C.2, 6.6. Ours: the count and caps (D11) |
| H16 | Insignificant pair | At the largest step reached, SNR < 1 or the gain's 3 sigma interval holds zero | read: SEB p. 435; ZP. Ours: the threshold 1 |
| H17 | Class | Per CV, by majority of seed evidence from both structures (10.6) | read: COOPER p. 17; DMCP p. 223. Ours: the majority and the tie-break (D12) |
| H18 | Stabilised retest | P-only loop on a CV that tripped; setpoint step when the loop holds the pair's own CV (10.7) | read: COOPER C.2, D.2, eq. 5.4, 6.6; QB p. 748; DHN |
| H19 | Exclusion and report | A pair-seed is valid, insignificant or invalid. Only valid ones count; all are reported | read: ZP, empty model columns reported. Ours: the record |
| H20 | No valid pair | T_c one task step, plan to the episode end, labelled unidentified | ours (D13) |

### 10.3 H14, fit validity

A fit is valid when it passes all four checks. COOPER's five questions map onto
them as follows: questions 1 and 2 (steady start, first point) hold by
construction, since H8 starts from steady operation and dy starts at zero.
Question 4 (quiet disturbances) is what the common-noise difference gives: a
disturbance shared by both runs cancels to first order (derived). What it
leaves is judged by V4.

| check | passes when | label and ground |
| --- | --- | --- |
| V1 signal | SNR >= 10 | read: COOPER 6.1 q3, 6.4, C.1, C.2 (10 noise bands, a minimum); QB and DHN give 6 (D8) |
| V2 duration | The fit's own transient ends inside the window: 5 tau + theta <= W dt for a FOPDT; 5 tau2 + theta <= W dt for an integrating fit | read: SEB p. 435, the step lasts the settling time; QB p. 747, the pretest finds the time to steady state; H1 defines the settle. Derived: the same 99 % for the lag of an integrating fit |
| V3 bounds | No parameter at an upper bound of the fitter (theta or tau2 at the window's end) | derived: an estimate at its bound is the bound, and the data say nothing beyond it |
| V4 fit | NRMSE fit >= 80 % | read: COOPER 6.1 q5, the fit must visually match the data; MWC defines the measure. Ours: 80 % (D10) |

Noise. sigma is the standard deviation of the CV in the unstepped run over the
same window, about its least-squares straight line (read: COOPER 6.4, the
band is measured with the controller output constant, which is the unstepped
run; ours: removing the line, since a held MV leaves a drift on an
integrating CV that is not noise). D9 offers the other reading.

A delay or lag at its lower bound (zero, or the fitter's 1e-9 s floor) is not
a failure. It means "below one task step", since the data are sampled at dt
and cannot resolve less (derived; COOPER C.1 ties the sample rate to what a
test can see). The record writes it as "< dt", and H2 to H4 use it as zero, as
today.

### 10.4 H15, retests

A failed pair-seed is retested at most twice (ours, D11). A retest repeats
the whole test (one MV, one seed), so it re-reads every CV. Each pair keeps the
first test at which its fit passes H14 (ours). A retest applies the remedy of
every check failed by a pair of that test that is not yet insignificant
(H16). Remedies add up across retests.

| failed | remedy | label and ground |
| --- | --- | --- |
| V1 | Double the step, within the cap below | read: ZP, "increase the amplitudes"; SEB p. 435, too small a move is obscured; DHN, larger moves preferred while linear |
| V2 or V3, window not cut by a trip | Double the window, up to 4 N_ep | read: SEB p. 435, the step lasts the longest settling time; ZP, a longer test reduces model errors. Ours: 4 N_ep |
| Any check, window cut by a trip | Stabilised retest (H18) | read: COOPER C.2 and 6.6; QB p. 748, operators intervene in critical situations; DHN, other inputs move to keep the process in range |
| V4 only, V2 passed | No remedy. The fit is invalid | ours: more data does not cure a model that has the wrong shape |

Step cap, the smallest of:

1. the MV's range, stepping away from the nearer limit (C2);
2. SEB eq. 20-75 over every CV with a trip limit: |Delta u| <= (Delta y)max /
   |K|. (Delta y)max is the distance from the CV's value at the step to its
   nearer trip limit, less 3 sigma. K is the gain of the failed fit; for an
   integrating fit, K = |k'| W dt, its move over the window (read: SEB;
   derived: the 3 sigma margin and the integrating K);
3. four times the H7 step (ours; SEB and DHN warn that large moves leave the
   linear range).

If the task cannot run past its episode, the window stays at the longest the
task allows, and the record says so.

### 10.5 H16, insignificant pairs

A pair-seed is insignificant when, at the largest step it reached (the cap, or
two escalations), either:

- its SNR is below 1, so the response stays within one noise band
  (read: SEB p. 435, obscured by normal fluctuations; ZP's grade D, "no model
  exists"; ours: the threshold 1); or
- the 3 sigma interval of its gain (k, or k') contains zero, with the
  interval from the least-squares Jacobian (read: SEB p. 435, confidence
  intervals; ZP grades on the 3 sigma bound).

A response that is identically zero is insignificant. A pair whose ten seeds
are all insignificant is a zero model (ZP, "zero them"), and is reported as
one. Neither rule reads tau, theta or P.

### 10.6 H17, class decision

Replaces H10. The class belongs to the CV, as DMCplus sets the ramp flag per
dependent variable (DMCP p. 223).

1. **Evidence per pair-seed**, from both fits of its last test:
   - self-regulating: the FOPDT fit passes V2 to V4 (the response reaches a
     new steady state; COOPER p. 17);
   - integrating: the FOPDT fit fails V2 (no steady state in the window) and
     the integrating fit passes V2 to V4;
   - both pass: the lower squared error wins (derived: both have three
     parameters, so the least-squares criterion of QB eq. 24 compares them
     directly);
   - neither: no evidence. The window was already lengthened by H15.
2. **Class per CV:** the majority over its pair-seeds with evidence, across
   all MVs and seeds (ours, D12). On a tie, the structure with the smaller
   sum of (100 - fit) over those pair-seeds (derived: QB p. 748 merges test
   periods into one fit).
3. **Refit.** Every pair-seed of the CV is refitted with the CV's class and
   judged by H14 again, with no further retest. Those that fail are invalid.

A slow stable CV that does not settle within 4 N_ep, but whose response a
ramp with lag fits, is classed integrating. That is DMCplus's pseudo-ramp
case, reached from the data alone. DHN's criterion (slower than the
prediction horizon) is not used, because it reads the horizon.

### 10.7 H18, stabilised retest

Used when a test failed H14 and a trip cut its window. Practice stabilises
before it steps (COOPER C.2, 6.6).

1. **Loops.** A P-only loop holds each CV that tripped in either run. Its MV
   is the stepped MV when the tripped CV is the pair's own CV. Otherwise it is
   the MV the shipped expert uses for that CV; if no expert loop holds it, the
   MV whose earlier tests gave that CV the largest SNR (ours).
2. **Gain.** COOPER D.2 for an integrating fit, Kc = 1 / (k' theta_P).
   COOPER eq. 5.4 for a FOPDT fit, Kc = (0.2 / k)(tau / theta_P)^1.22.
   theta_P is theta_e of H4 at T = dt, so it is never zero (derived). The
   model is the failed test's better fit (higher V4) on that MV-CV pair. With
   fewer than four samples before the trip there is no model, and the
   pair-seed is invalid (ours).
3. **Start.** Steady operation as H8. The loop closes bumplessly (setpoint at
   the CV, bias at the held MV) and runs 5 sqrt(10) theta_P before the step,
   in both runs (read: COOPER C.2, steady state before the test, and its
   closed-loop settle; ours for a self-regulating CV).
4. **Step.** If the loop's MV is the stepped MV, the setpoint steps by
   Delta u / Kc, Delta u the step H15 sets, so the loop's first move equals
   it (derived: COOPER C.2, the output move from the setpoint step drives the
   test). Otherwise the stepped MV steps as in H8 while the loop holds.
5. **Fit.** Output-error least squares on the CV's response, each model
   driven by the recorded deviation of every MV that moved (COOPER p. 56;
   QB p. 747-748). With one moving MV the fit is single-input. With two, the
   loop MV's model is fitted alongside, six parameters in all.
6. **Judged** by H14. sigma comes from this test's unstepped run.

The loop only stabilises the test. The fit is the open-loop model of the
recorded MV to the CV, so its parameters do not depend on Kc beyond the
excitation it gives (derived).

### 10.8 H19 and H20, exclusion, report and fallback

**H19.** After H15 to H18, each pair-seed is one of:

- **valid:** passes H14 under its CV's class. Only these enter H2 to H5;
- **insignificant:** H16. A zero model when all ten seeds are;
- **invalid:** fails H14 after its retests. Excluded from H2 to H5.

The evidence file (H13) adds, per pair-seed: each test run (step, window, loops,
trip run and CV), sigma, SNR, both fits with their V1 to V4 values, the
evidence, the label and the failed check. Per CV: the class and its vote. Per
task: the counts of valid, insignificant and invalid pairs, and the CVs left
with no valid pair. A CV with no valid pair adds nothing to the horizon; the
oracle still carries its limit, as every limit CV.

**H20.** If a task has no valid pair at all, it is marked unidentified. The
recipe has no fit for T_c, so T_c is one task step, and the planner plans to
the episode end, H_k = N_ep - k (ours; RMD p. 574 for the shrinking
horizon). D13 offers holding the task out instead.

### 10.9 Order of the rerun

1. Base test per MV and seed: amended H8, H7 step, window N_ep, both runs,
   trips recorded.
2. Per pair-seed: sigma, SNR, both fits, V1 to V4 for each.
3. Retests (H15, H18), at most two per test.
4. Insignificance (H16) at the end of the retests.
5. Class per CV (H17); refit under the class; H14 again.
6. Valid fits go to H2 to H5 and H12; H20 if none.
7. Record (H19).

Cost. The base test is unchanged. A test that fails twice adds runs of 2 N_ep
and 4 N_ep at most, plus the closed-loop settle of H18, so at most about 7
times the base cost for that test. Only failing tests pay it.

### 10.10 What the first run's failures meet

| failure | tasks | rules |
| --- | --- | --- |
| Integrating fits at the lag floor with zero delay | glass_furnace, cement_kiln, boiler_drum, distillation, wind_turbine, battery, ph_neutralization | H17 tests both structures per CV; V2 and V4 judge the fit. A floor fit that passes every check stays (V3) |
| FOPDT tau 1.1e8 s on a response within the noise | cement_kiln (lime) | V1 and V2; H15 escalates; H16 if still within the band |
| Integrating CVs bumped open loop, every response tripped | boiler_drum | H18, COOPER C.2's P-only setpoint step |
| Mixed classes across seeds or pairs | ph_neutralization, distillation, glass crown, kiln, boiler, wind, battery T_cell | H17, one class per CV |
| Windows cut by trips | boiler, wind, distillation, kiln, hvac | V2 on the cut window; H18 if invalid |

### 10.11 Where the sources do not settle it

1. **Noise in a simulator.** COOPER's band is the PV's noise with the output
   constant. The common-noise difference is cleaner than any plant response,
   so the band from the unstepped run is strict here. D9.
2. **Signal to noise.** COOPER defines his band (plus or minus 3 sigma) and asks
   for 10. QB and DHN ask for 6 without defining the noise, and QB's 6 comes
   with 8 to 15 steps per MV. D8.
3. **Fit floor and retry count.** No source gives a fit percentage (MWC gives
   none) or a number of retries. ZP repeats until most expected models grade
   A or B, with no cap. D10, D11.
4. **Grade thresholds.** ZP's A to D cut-offs are in Zhu 1998 and Zhu 2001,
   which I did not read. H16 uses only grade D's meaning.
5. **Weak pairs.** COOPER rejects a test under 10 bands. ZP still uses a
   C-grade model when its gain sign matches an expectation matrix. D14.
6. **Slow stable CVs.** DHN models them as integrators when slower than the
   horizon, and DMCplus has a pseudo-ramp type, but I found no definition of
   it. H17 decides from the data, so the window cap (D11) decides when such a
   CV is integrating or invalid.
7. **A valid delay-free integrator** (battery state of charge: lag and delay
   below one step) still sets T_c to one task step, by C.2's T <= 0.5 theta.
   This amendment does not change H2 beyond keeping invalid fits out.
8. **Not read:** Seborg et al. ch. 7, Zhu's 2001 book, Ljung (cited through
   SEB and ZP), the 2012 journal version of DHN, and AspenTech step-test
   guidance beyond the DMCP annex.

### 10.12 Source check

Each PDF was downloaded twice on 2026-11-03; the two copies had the same
SHA-1, and every cited phrase was found in pdftotext output of the second
copy. COOPER's Figure 6.4 was also read as a page image, which confirmed that
the 0.25 degC band spans the full plus or minus 3 sigma. MWC was fetched
twice with different questions. Confirmed: COOPER, SEB, QB, DMCP, ZP, DHN,
MWC.

### 10.13 Decisions for the owner

Recommended option first.

| # | decision | options |
| --- | --- | --- |
| D8 | V1 threshold | 10 noise bands, 60 sigma (COOPER, the recipe's own source). Or 6 bands (QB, DHN; their noise is undefined, and QB's 6 assumes 8 to 15 steps) |
| D9 | Noise sigma | From the unstepped run, straight line removed (COOPER's band with the output constant). Or from the residual of the differenced response (the noise the fit sees; no source) |
| D10 | V4 fit floor | 80 % NRMSE fit (ours). Or 90 % (ours, stricter). Or none: the fit is reported, not judged (COOPER's check is visual) |
| D11 | Retests | Two per test, step up to 4 times, window up to 4 N_ep (ours). Or one retest (cheaper, more invalid pairs). Or three, up to 8 times and 8 N_ep (dearer) |
| D12 | Class | Per CV, majority of seed evidence, pooled fit on a tie (DMCP; QB). Or process knowledge first: a CV whose equation is a pure balance (drum level, state of charge) is integrating, the data decide the rest (DHN). Or integrating whenever any seed's evidence says so (no source) |
| D13 | Task with no valid pair | Plan to the episode end at T_c of one task step, labelled unidentified. Or hold the task out of the oracle re-record until a protocol change identifies it |
| D14 | Weak pairs, 1 <= SNR < 10 at the cap | Invalid and reported (COOPER's minimum). Or valid when the gain sign matches an expectation matrix written from PHYSICS.md before the rerun (ZP's grade C) |

## 11. Amendment 2, noise from the fit residual, pure gains, non-compliant tasks

**Why.** The rerun under Amendment 1 (PR #133) left 8 of 14 tasks
unidentified. D9 took sigma from the unstepped run, which carries the plant's
own disturbances: the weather on hvac, the demand blocks on compressor_surge,
the inlet drift on unstable_cstr, the turbulence on wind_turbine. The
subtracted test (C1, H8) already cancels them, so good fits failed V1: hvac
at a 91 % fit, unstable_cstr at 96 %. Battery's delivered power, a pure gain,
had no transient, so its NRMSE fit was 0 / 0.

**Owner decisions, 2026-11-03.** The three rules below replace D9, amend V4
and replace D13 (and H20). All three are ours: no source covers a subtracted
step test. They are judged on how they measure noise, not on the horizons
they give.

| # | rule | value | label |
| --- | --- | --- | --- |
| N1 | Noise (replaces D9) | sigma is the standard deviation of the residual between the subtracted response and the fitted model, over the fit window. Each structure's fit has its own sigma and SNR. V1's threshold is unchanged: 10 bands of plus or minus 3 sigma | ours |
| N2 | Pure gain (amends V4) | The fit measure is taken against the response's pre-step value instead of its mean: fit = 100 (1 - norm(dy - yhat) / norm(dy)), since dy is zero before the step. A response with no transient is then well-posed: a pure gain fitted exactly scores 100 % | ours |
| N3 | No valid pair (replaces D13 and H20) | The task keeps its current oracle and its horizon file labels it non-compliant. No protocol horizon (no T_c, N, M or P) is written for it, and C1 and C2 leave it as it is | ours |

What follows from the rules (derived):

- Wherever section 10 reads sigma, it is now the residual sigma of the fit in
  question: V1, H16's SNR below 1 (on the closer fit), the 3 sigma margin of
  the step cap (10.4), and H18's step 6.
- The unstepped run is still recorded, for the subtraction only.
- 10.11 item 1 and D9 are settled by N1; item 3's fit floor (D10, 80 %) is
  unchanged but now measured by N2.

## 12. Amendment 3, a rate limit's ramp

**Why.** With the rebuilt oracle converging, the compressor beat its expert on
every seed but sat 15 % below main's oracle on the mean, and up to 80 % below
it on a seed. Nearly all of that was one event per seed: the large upward step
of the setpoint, about 6 kPa, where main's oracle starts the slow drive
seconds ahead and P = 26 (2.6 s) sees the step too late. Run at P = 60, main's
horizon, the rebuilt oracle landed within 0.1 % of main's on seeds 0, 7 and 9,
so the horizon was the whole of the gap. That run was a diagnosis; 60 was not
adopted.

The protocol's gap: Cooper's N is the time for a move's effect to finish, and
it is fitted to bump tests, which are too small to reach a rate limit. A large
move on a rate-limited MV is not a step. It is a ramp of move / rate, and only
then the plant's settling. Here the pair that shows it, speed to header
pressure, failed V1 on all ten seeds (SNR about 7 of the 10 required, with a
fit of 97 %), so P came from a faster pair.

**H21 (owner decision, 2026-11-04).** For a task with a target range and
rate-limited MVs:

    P = N + M + ramp
    ramp = the largest, over the fits of the tracked CV and those MVs, of
           span / (|K| rate), rounded up to whole samples of T_c

span is the CV's target range and rate the MV's slower direction, in the
units of its fitted gain K. Taking the largest follows H5. T_c, N and M do not
change. A gain needs a settled fit, not a quiet one, so a fit that failed V1
alone counts here, as an exception to H16 for H21 only; it still does not set
T_c, N or M.

Compressor: 8000 Pa / (7206 Pa per unit x 0.171 units/s) = 6.5 s, or 65
samples, so P = 26 + 65 = 91 (9.1 s). The reactor's rod and the wind pitch take
the same rule when their rebuilds reach them.

**Provenance.** H21 was written after the failure, and after P = 60 had been
seen to close the gap. It derives from Cooper's definition of N, and it uses
only the task's range, its rate limit and a fitted gain. No return enters it,
and 91 is not the value that was tried. It is ours: no source read covers
horizons for rate-limited MVs.
