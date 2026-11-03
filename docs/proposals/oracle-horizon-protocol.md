# Oracle horizon protocol

Specification of 2026-11-03, for owner review. Nothing here has been run or
implemented. It replaces the audit's rule B, max(expert settle, 5 x tau_63),
which had no source (`oracle-audit.md`, section 5, Horizon).

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
| H13 | Record | per task: P, the pair and seed that set it, every fit, the class per CV | ours |

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
