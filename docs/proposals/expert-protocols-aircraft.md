# Tuning protocols for the aircraft experts

Proposal of 2026-11-02, for the owner's review; nothing in it has been run. It replaces every value
the audit (`expert-tuning-audit.md`, "Aircraft") marks FAIL or UNCLEAR on the ten aircraft tasks
with a recipe from practitioner procedures, run once on training data, whose result ships whatever
it is: the performance the best real procedure gives, not the best score.

**Owner decisions, 2026-11-02.** Every recommendation in section 6 is accepted (D1 to D10, the
first option each time). Where no source gives a number, the value comes from a rule fixed here
before any run and labelled ours, as section 2 does.

## 1. What the experts copy, and which ArduPilot

The experts fly ArduPilot's fixed-wing stack: a pitch loop and a bank loop, TECS over the pitch
loop, and L1 guidance or a course hold over the bank loop. Every default, range and procedure below
is pinned to **ArduPlane 4.0.9** (decision D1), for two reasons:

- Its roll and pitch controllers are our loops' law. `AP_RollController.cpp` outputs
  `desired_rate * kp_ff + rate_error * D`, `desired_rate = angle_err / tau`,
  `kp_ff = (P - I tau) tau - D`: with I = 0, exactly P on the angle error minus D on the body rate,
  as `Kp_roll (bank - phi) - Kd_roll phi_dot`; the pitch controller adds I, as `pitch_hold` does.
  Later firmware (the archived guide covers 4.1 and earlier) flies a rate loop we do not.
- Its TECS is the one `experts_tecs.py` ports, with the guide's defaults (PTCH_DAMP 0.0, INTEG_GAIN
  0.1); master changed both to 0.3.

ArduPilot's surface output is in degrees (45 at full throw) and its errors in degrees, so one
ArduPilot unit of P, I or D is (180 / pi) / 45 = **1.273** of ours (stick in [-1, 1] per rad, per
rad/s, per rad s). Values below give ArduPilot units, ours in brackets.

| id | source (fetched twice on 2026-11-02 unless marked) | used for |
| --- | --- | --- |
| G-TECS | ArduPilot, [TECS tuning guide](https://ardupilot.org/plane/docs/tecs-total-energy-control-system-for-speed-height-tuning-guide.html) | TECS numbers and checks |
| G-MAN | ArduPilot, [Roll, Pitch and Yaw Controller Tuning](https://ardupilot.org/plane/docs/roll-pitch-controller-tuning.html), archived for 4.1 and earlier | attitude loops |
| G-AT | ArduPilot, [AUTOTUNE](https://ardupilot.org/plane/docs/automatic-tuning-with-autotune.html) | why not AUTOTUNE |
| G-NAV | ArduPilot, [Navigation tuning](https://ardupilot.org/plane/docs/navigation-tuning.html) | L1 period |
| SRC | [ArduPlane-4.0.9 source](https://github.com/ArduPilot/ardupilot/tree/ArduPlane-4.0.9): `AP_RollController`, `AP_PitchController`, `AP_AutoTune`, `AP_TECS`, `AP_L1_Control`, `ArduPlane/Attitude.cpp`; master's `AP_AutoTune` and `AP_TECS` | laws, defaults, ranges |
| FAR | [14 CFR 25.207(c)](https://www.law.cornell.edu/cfr/text/14/25.207), stall warning margin | angle-of-attack limit |
| ICAO | ICAO Doc 8168 (PANS-OPS): holding turns at 25 deg of bank or 3 deg/s, whichever needs less bank; read through [Wikipedia](https://en.wikipedia.org/wiki/Standard_rate_turn), primary not fetched | nominal turn bank |
| SIMC | Skogestad, J. Process Control 13 (2003) 291-309; Skogestad and Grimholt 2012, as `tune_process_experts.py` cites; [NTNU lecture](https://skoge.folk.ntnu.no/prosessregulering/lectures/2024/SiS6SIMC_tuning.pdf), fetched once | follower slot loops |
| BM | Beard and McLain, *Small Unmanned Aircraft*, Princeton 2012, ch. 6: inner loop 5 to 10 times the outer's bandwidth (companion notes only, book not seen) | alternative to SIMC |
| OG | Ogata, *Modern Control Engineering*, 5th ed., sec. 5-3 (textbook, not fetched) | step-response terms |

## 2. Judgment calls, in numbers

Every reading the guides leave to the pilot's eye is fixed here, before any run. "Ours" marks a
number with no source.

| reading | number | basis |
| --- | --- | --- |
| overshoots | the peak passes the target by more than 5% of the step | ours, Ogata's 5% band; a second-order loop at damping 0.69 |
| oscillates (wags, porpoises, weaves) | after the first peak the error changes sign at least twice, each excursion leaving the +-5% band | ours; second order: overshoot above 22%, damping below 0.43 |
| settled, acquired | inside the +-5% band and stays there | OG |
| lag | time to 63.2% of a step | first-order time constant |
| height oscillates | height error beyond 10 m after the scored burn-in, or a capture overshoot beyond 10 m | G-TECS: "height changes greater than 10m" |
| pinned at a limit | at the limit for more than half of the manoeuvre | ours |
| speed drops | more than 5% below the demand | ours |
| "slightly less than" the top speed | 5% less | ours |
| turns too slowly | the new leg acquired more than 2 t90 after the corner; t90 = (pi/2) V / (g tan 45 deg), 37 s at 230 m/s | ours |
| weaves after a turn | the new leg's cross-track error oscillates, band 5% of that corner's peak | ours, as "oscillates" |
| gains or loses height in a turn | more than 10 m in the first 10 s of a 45 deg bank, in attitude mode | ours, the TECS 10 m band |

A check flown on several seeds fails if it fails on any of them (ours: a pilot acts on one bad
flight).

## 3. The recipe

In the guides' order: trim and flight test, attitude loops, TECS, navigation, then the followers.
One airframe gets one tune: all ten tasks share it, and the figure-8's overrides (`Kd_theta` 6,
`time_const` 1.5, `ptch_damp` 2.0, `turn_bank_margin` 0.75) go. Note that `ptch_damp` 2.0 and
`time_const` 1.5 and 2.0 lie outside 4.0.9's own ranges (PTCH_DAMP 0.1 to 1.0, TIME_CONST 3 to 10),
as does the patrol's `thr_damp` 6.0 (0.1 to 1.0).

### 3.1 Trim and flight test (keeps step 1 of `tune_aircraft_guidance.py`)

Still air, wings level, 230 m/s, the scripted pilot holding speed with pitch as today (its gains
only reach the steady states measured). G-TECS, airspeed sensor branch, steps 2 to 8, and FAR.

1. At 3000, 5500 and 8000 m, the task envelope's ends and middle (today 3500, 5500, 7500): level
   trim, full-throttle climb, idle sink. Sets `pitch_trim`, the starting `thr_trim` (5500 m),
   `sink_min` (method unchanged) and `climb_max` = the full-throttle climb at 8000 m, the envelope's
   worst case, as measured, not rounded (decision D9).
2. Level flight at full throttle until the speed settles; AIRSPEED_MAX is 5% below that top speed,
   per altitude.
3. Idle descent with pitch holding AIRSPEED_MAX, or at the lower pitch limit if that binds first.
   `sink_max` is the lowest of the three sink rates (G-TECS: "achieved without exceeding the lower
   pitch angle limit", nor AIRSPEED_MAX).
4. Deceleration at 5500 m: level, 1 g, throttle set so the speed falls at 1 kt/s (25.207(c)'s rate) until
   the lift stops rising. The samples give the stall speed Vs and the lift curve (CL from weight and
   dynamic pressure).
   - `alpha_max` = the angle of attack at 1.05 Vs, where 25.207(c) puts the onset of stall warning
     (5 kt or 5%, the greater; 5% at these speeds).
   - the nominal turn compensation, CL_trim / CL_alpha at cruise: the extra angle of attack per unit
     of extra load factor (used in 3.2 step 5).

### 3.2 Attitude loops: the manual procedure, not AUTOTUNE

**Why not AUTOTUNE.** G-MAN recommends AUTOTUNE over itself, so the choice needs a reason.

- 4.0.9's AUTOTUNE flies like FBWA under rapid stick inputs. While the demanded rate exceeds 80% of
  the level's `rmax` (75 deg/s at the default level 6), it raises P by 5% after 200 ms of undershoot
  with the surfaces unsaturated and cuts it by 8% after 100 ms of overshoot, P in [0.3, 5.0];
  D = Dratio * P (0.075 at level 6), I = 0.5 D / tau, tau from the level (0.45 s at level 6).
- Master's AUTOTUNE (G-AT) tunes the rate loop we do not fly. An event starts when the demanded
  rate passes 0.4 * min(attitude limit / tau, rmax). FF is actuator over achieved rate (at most 12%
  up, 15% down per event); D grows by 1.3 per event until a 150 deg/s slew limiter on the actuator
  detects oscillation, then drops to 0.3 of that; P is halved, then grows by 1.3 to its own
  oscillation; I = min(P, FF) on roll, max(0.75 P, FF) on pitch; tau 0.1 to 1 s, 1.5 times on pitch.

Their windows (100 and 200 ms), time constants and slew detection sit below the simulator's 1 s
step, and their rate targets (20 to 300 deg/s) are a model's. The manual readings, overshoot and
oscillation after a step, are visible at 1 s on an airliner, and its P, D, I map onto ours.

**The tests.** Still air, 5500 m, 230 m/s, throttle at trim. Roll: bank demand 0, +45, -45, 0 deg
(45 deg is full stick in FBWA), each held 30 s (ours; the guide's "couple of seconds" suits a
model). Pitch: pitch demand trim +5, -5, trim deg, each held 20 s, wings level (ours: the guide
gives no size; 5 deg keeps the speed within about 10 m/s). Roll is tuned first.

**The recipe** (G-MAN Method 2, then its turn check):

1. Start at 4.0.9's defaults: P 1.0 [1.27]; D 0.08 [0.102] on roll and 0.04 [0.051] on pitch; I 0.
   If the first test already oscillates, halve P until it does not (G-MAN, initial assessment).
2. P: +0.1 [0.127] per test until a test overshoots or oscillates, then back one step; never above
   5.0 [6.37], 4.0.9's `AUTOTUNE_MAX_P`. The guide's other stop, "happy with the response", has a
   numeric check (`roll_speed`) only under a rate limit 4.0.9 disables by default: none (ours).
3. D: +0.01 [0.0127] per test until a test oscillates, then halve. No cap: the guide's warning above
   0.1 guards servo heating, which the model lacks; a sampled loop always reaches oscillation.
4. I: +0.05 [0.0637] per test from 0 until a test overshoots or oscillates, then halve. Pitch
   always; roll only if the owner adds the roll integrator ArduPilot flies (D2).
5. Turn check (G-MAN pitch, Method 1 step 3): attitude mode, 45 deg of bank each way for 20 s. The
   compensation multiplier (PTCH2SRV_RLL, nominal 1.0) goes down 0.05 while the aircraft gains
   height and up 0.01 while it loses height at once, until neither or the guide's 0.7 and 1.4
   bounds. `k_bank_comp` = multiplier * the nominal from 3.1 step 4. ArduPilot adds a body pitch
   rate in turns; this model observes Euler rates, which are zero in a steady turn, so the angle
   offset stays (ours).
6. `curvature_lead` = the lag of the final roll test's 45 deg entry (the bank loop's lag, as the
   figure-8 comment already defines it).

Sets `Kp_roll`, `Kd_roll`, `Kp_theta`, `Ki_theta`, `Kd_theta`, `k_bank_comp`, `curvature_lead`.

**Amendment, 2026-11-06, the owner's decision.** A first run on the plant of
`industrial-fixes.md` PR 7, with the throttle fixed, showed two gaps. The speed mode carried pitch
past its target 15 to 20 s after a step, which the overshoot reading took for a loop that was too
strong. And D did not reach oscillation: past about 10 the roll loop only slows down. So, fixed
before the rerun and replacing what is above where they differ:

- **Speed is held through both tests,** as a real flight test does with the throttle managed, so
  the phugoid is not excited. The throttle is a PI on the error from 230 m/s, about the trim lever,
  read and updated at 1 Hz, the plant's action rate. The signal is airspeed, which in still air is
  the true velocity's magnitude. The integral runs only while the lever is off its stops. Ours: G-MAN
  sets a cruise throttle before the tests and says nothing of it during them, and G-TECS has TECS
  hold energy with the throttle but gives no hold for a test.
- **The hold's gains follow SIMC,** by the rule section 3.5 uses, from the plant's known numbers and
  no bump test. Lever to speed is k' / (s (1.7 s + 1)), k' the thrust slope over the mass at the test
  point. A 1 s sample adds 0.5 s of delay, and the half rule folds the 1.7 s engine lag in:
  theta_e = 1.35 s. The tight setting tau_c = theta_e gives a gain of 1 / (2.7 k') and an integral
  time of 10.8 s.
- **The overshoot reading stays as section 2 writes it,** on the whole hold.
- **D stops at the guide's limit,** 0.1 [0.127], and ships there if no test oscillates (section 4's
  rule for a cap). This replaces step 3's "no cap", whose reason did not hold.

### 3.3 TECS (G-TECS Parts C and D)

Start at 4.0.9's defaults (TIME_CONST 5, PTCH_DAMP 0.0, INTEG_GAIN 0.1, THR_DAMP 0.5, RLL2THR 10),
the numbers of 3.1, and `spd_weight` by D4. Two hand-set values become ArduPilot's own (SRC,
`AP_TECS` 4.0.9):

- `accel_max` becomes the speed-demand slew, half the energy rate: 0.5 g `climb_max` / V up and 0.5
  g `sink_min` / V down (about 0.17 and 0.49 m/s^2 at 230 m/s with today's numbers). One limit
  becomes two.
- `k_vdot` becomes 1. ArduPilot reads the speed rate from the accelerometer through a 5-point moving
  average at 50 Hz (0.1 s). The observation has no accelerometer; the one-step speed difference is
  the mean acceleration over the 1 s step, so no further smoothing is the match (ours).

1. Loiter check: `plane3d_circle`, a loiter, on training seeds 100-109 in their turbulence. Height
   oscillates: PTCH_DAMP +0.1 to at most 0.5, then TIME_CONST +1 to at most 10, re-flying after each
   change (the guide's oscillation step, which contains its loiter step; section 7).
2. Climb and descent checks: `plane_energy` on seeds 100-109, whose rungs climb and descend 360 to
   720 m (the guide asks at least 50 m). In a climb, throttle pinned at full with the speed
   dropping, or the pitch demand pinned at `pitch_max`, sets `climb_max` to the mean climb rate of
   the worst such rung (ours: the guide says "reduce" without an amount). In a descent, speed above
   AIRSPEED_MAX or pitch pinned low sets `sink_max` the same way. A capture overshoot beyond 10 m is
   a height oscillation: step 1's remedy, then steps 1 and 2 again.
3. Speed step (ours: the guide names THR_DAMP's symptom, not a test): level at 5500 m, still air,
   speed demand +10 m/s. Oscillates: THR_DAMP +0.1, its @Increment, to at most 1.0, its @Range top.
4. Trim: `thr_trim` = the mean throttle over step 1's loiters, all seeds (G-TECS Part C step 5). One
   pass.

INTEG_GAIN stays at 0.1: the guide describes it but gives no test.

### 3.4 Navigation (G-NAV)

The guide flies a looped rectangle; no task does. The script flies one on the 3D dynamics: four
waypoints around each training seed's start, legs of 4 and 3 minutes at cruise (55 by 41 km: each
lasts three of the longest period NAVL1_PERIOD's range allows, 60 s), at the seed's altitude and in
its turbulence, two laps. Guidance is `update_waypoint`'s L1 on the active leg, which switches when
the aircraft passes the perpendicular through its waypoint (ours).

1. Start at NAVL1_PERIOD 17, the 4.0.9 code default; the page says 20 in one place and 17 in
   another. Damping stays 0.75 (a pass).
2. A corner turns too slowly: period -5. A corner weaves: +1 (the guide says "1 or 2"; we take the
   finer). Stop when no corner of either lap on any seed does either, or when a change would return
   to a period already flown: then take the larger of the two (ours).
3. The guide's step 3 (+1 and damping in steps of 0.05 "for maximum performance") is not run: no
   test ends it.

The period then sets three more values.

- `tau_hdg` (heading task, formation lead, followers) goes. The course hold becomes
  `update_heading_hold`: lateral acceleration 2 V w sin(Nu), with w = sqrt(2) pi / NAVL1_PERIOD and
  Nu clipped to +-90 deg; for small errors a course time constant of period / 8.9.
- `turn_bank_margin`: a curved path is flown at the speed whose steady bank is 25 deg, the bank
  ICAO's holding procedures turn at, so 25/45 = 0.56 on the circle, the racetrack (a holding
  pattern) and the figure-8 (D10). Stall prevention still sets the slowest speed.
- `max_intercept` (followers) = 45 deg, L1's capture limit (`sine_Nu1` clipped to 0.7071 in SRC).

ArduPilot's cross-track integrator (NAVL1_XTRACK_I, 0.02) is missing from the experts; out of scope.

### 3.5 Followers and the formation lead

No published gains exist for a formation hold over an autopilot (D5). Closest practice: the inner
autopilot is the tune above, unchanged (the followers' `tau_hdg` 3, `thr_damp` 6.0, `spd_weight` 0.5
go), and each slot loop is a PI on position set by SIMC from a step test, as the process family is.
The test: `patrol` on training seeds 100-109, the lead held straight (`lead_turn_rate_range` (0, 0)),
the slot loops open, each response read against an unstepped run on the same gusts.

1. Along track: speed demand +5 m/s (ours, 2% of cruise). Fit the along-track position to k'
   e^(-theta s) / (s (tau2 s + 1)) with the process script's `fit_integrating_lag`, take the median
   over seeds, and add half a 1 s sample to theta (SIMC 5.12). The loop has no D, so the half rule
   folds tau2 into the delay: theta_e = theta + tau2 / 2. SIMC's tight setting tau_c = theta_e gives
   `k_x` = 1 / (2 k' theta_e) and an integral time of 8 theta_e, so `k_xi` = `k_x` / (8 theta_e).
2. Across track: a lateral speed demand step of V sin 2 deg (about 8 m/s); the same fit and rule
   give `k_y` and `k_yi`.
3. Vertically there is no gain: TECS takes the height demand, as today.

One set serves every follower: both patrol tasks and the formation's wingmen. The bearing-only
estimator has no gains.

| value | replacement | basis |
| --- | --- | --- |
| `a_x` | TECS's speed-rate limits (3.3) | SRC, `AP_TECS` |
| `dv_max` | the speed demand clamped to [stall-prevention minimum, AIRSPEED_MAX] | TECS clamps its own demand so |
| `max_intercept` | 45 deg | L1's capture limit (3.4) |
| `x_band`, `y_band`, `int_max` | integrate only while the demand the integral feeds is inside its limit; across track also only while the course correction is under 5 deg, output clamped to 0.1 rad | TECS's anti-windup; L1's XTRACK_I band and clamp (SRC) |
| `k_ref`, `k_omega` | first-order filter, 3 s: 1 - e^(-1/3) = 0.28 per step | TECS_HDEM_TCONST (master only, range 1 to 5 s); reused for `k_omega` (ours) |
| `k_eng` | 0: the lever goes to the engine as ArduPilot sends its throttle | D6 |

Alternative for steps 1 and 2 (D5): bandwidth separation, each slot loop at one fifth of its inner
loop's bandwidth (BM's 5 to 10, low end).

**The formation lead**, as the audit proposes: `power_set` = the trim throttle at 230 m/s from 3.1's
table, interpolated to the formation's altitude, as a power setting read from a cruise table;
`k_speed` = 0; `Kd_theta` and `time_const` are the shared tune (D7).

## 4. Stop rules, and shipping a poor result

Every step ends by its own rule above: a symptom gone, a cap reached, a halving, or a revisited
value. Whatever it ends on ships.

- A cap reached with the symptom still there ships at the cap, and the report says so. The guide's
  hint that TIME_CONST above 10 points to the pitch loop does not send the protocol back to 3.2.
- Nothing is rerun because a held-out or scored number disappoints.
- The held-out limit tests stay as written (`tests/plane3d/test_plane3d_guidance.py`: 2 m of height,
  3 m of path; `tests/patrol/test_formation_hold.py`: 8.5 m slot RMS). A seed that fails is a
  finding, marked with its measured value through `_finding` on the owner's approval (D8).

## 5. Code, seeds and records

- **Script.** `scripts/aircraft_protocol.py` replaces `tune_aircraft_guidance.py` (step 1 moves in;
  step 2's grid and step 3's figure-8 search go) and `tune_formation_hold.py` (its five grids go).
  It runs 3.1 to 3.5 in order, in still air or on training seeds 100-109 only, and writes each value
  with its measurements to one JSON the experts read, as `tune_process_experts.py --write` does; no
  value is typed by hand. `--check` only reports held-out seeds.
- **Structure changes** to confirm: two speed-rate limits for `accel_max`; `update_heading_hold` for
  `tau_hdg`; conditional integration for the follower bands; the roll integrator (D2); the airspeed
  scaler (D3).
- **Held-out seeds.** 200-209 for nine tasks. The figure-8's 200-209 were seen by its retune, so it
  reports on 300-309, used nowhere in the repo yet, and its guidance tests move there.
- **Records.** Every PID row of the ten tasks changes. The edited experts files are in all ten
  fingerprints, so all ten re-record in one run, oracle rows too (their values should not move: no
  oracle flies the expert). Per the audit's plan this lands after the process and industrial
  families, with the CasADi removal in 0.8.0.
- **Docs and tests.** "The expert" sections of the four PHYSICS.md files. Spec-derived unit tests
  for the readings of section 2, the SIMC rule and the unit conversion.

## 6. Open decisions (recommendation first)

- **D1** Pin ArduPlane 4.0.9, whose controllers we fly; or master.
- **D2** Add the roll integrator ArduPilot flies (default I 0.3) so 3.2 step 4 runs on roll; or keep
  the roll loop PD.
- **D3** Add ArduPilot's airspeed scaler, s = SCALING_SPEED / EAS with SCALING_SPEED the test's EAS,
  clamped to [min(0.5, 0.5 v_min / SS), max(2, 1.5 v_max / SS)], on the angle path and s squared on
  the rate path, as 4.0.9 does. It replaces the figure-8's slow-flight retune; without it, one tune
  at 5500 m flies every condition.
- **D4** `spd_weight` from the task's stated objective: 0 ("prioritises height") where only height
  is the objective (`plane`, `plane_sine`, the four 3D tasks, the formation lead); 1 (the default,
  "balances") where speed is too (`plane_energy`, the followers, whose slot is held through speed).
  Alternative: 1 everywhere.
- **D5** Follower slot loops by SIMC from step tests; or bandwidth separation at 5.
- **D6** Drop `k_eng`; or keep an N1 loop, which needs a source.
- **D7** Formation lead at fixed trim power; or on the shared TECS throttle.
- **D8** Failing held-out seeds become `_finding` marks with their values.
- **D9** `climb_max` as measured (the airspeed branch); or 80% of it (the no-airspeed branch, whose
  reason is battery sag).
- **D10** A 25 deg nominal bank on curved paths (ICAO holding); or keep speed planning at the full
  45 deg limit with no margin.

## 7. Inconsistencies found in the sources

- G-NAV gives NAVL1_PERIOD's default as 20 in its introduction and 17 in its steps; the 4.0.9 and
  master code both say 17.
- G-TECS gives PTCH_DAMP's default as 0.0, which matches 4.0.9's code but falls outside 4.0.9's own
  @Range (0.1 to 1.0); master's default is 0.3. INTEG_GAIN is 0.1 in 4.0.9 and 0.3 in master.
- G-TECS's 80% climb margin appears only in the no-airspeed section. Its loiter step and its
  oscillation step raise different parameters first.
- G-MAN gives RLL2SRV_RMAX's default as 60 deg/s; 4.0.9's code has 0 (off). It starts I "from its
  default value of zero"; the code's default is 0.3 on both axes. Its pitch_speed check names
  `RLL2SRV_P`, a typo.
- G-AT puts the learning threshold at 40% of the maximum rate in one place and 80% in another;
  master's code uses 0.4, 4.0.9's 0.8. The page names no firmware version.
