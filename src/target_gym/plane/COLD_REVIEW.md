# Aircraft family: cold review against REALITY.md

Date: 2026-11-08. Reviewer had not seen the design. Read, in this order:
`src/target_gym/plane/REALITY.md` (the reference), then `plane/env.py`, `plane/env_jax.py`,
`plane/dynamics.py`, `plane/PHYSICS.md`, `plane3d/env.py`, `plane3d/env_jax.py`,
`plane3d/dynamics.py`, `plane3d/PHYSICS.md`, the seven entries in `registry.py` (lines
507 to 626), `reward.py` and the parts of `base.py` the tasks call (`actuate`,
`failure_kernel`, `EnvParams`). Experts, oracles and rendering were not reviewed.

Four short probes of the plant were run (no controller, fixed actions, a few hundred
steps each); their results are quoted where used. Source tags in square brackets are
REALITY.md's. "Unsourced" means the record has nothing either way.

Paths below are relative to `src/target_gym/`.

---

## A. Items shared by all seven tasks

### Limits

**A1. Load factor is a trip; the record says the aircraft protects it. Major.**
Task: `outside_envelope` trips at n > 2.5 or n < -1.0 (`plane/dynamics.py:479-488`,
called from `plane/env.py:409-428` and `plane3d/env.py:390-407`), and the worst load
inside the step is read (`plane/env.py:715-717`). Nothing in the plant limits n: the
inner loop flies a pitch attitude demand with no load factor or g limiter
(`plane/env.py:662-678`). Record: the limit load is "a design load, not a trip", and
normal law limits load factor; "a protected limit, enforced by the aircraft, not a
trip" [CFR 25.337, NTSB 1.6.3, BEA447]. Probe: from the reset trim at 3 800 m in still
air, a pitch demand step of 9 deg (60 % of the action) pulled 2.57 g in the first
second and tripped; 15 deg pulled 2.9 g; 6 deg reached 2.1 g. Effect: a large part
of the action box is a crash for the agent. A real A320 at that stick input would
fly at the protected n and carry on. The trip, not the job, becomes the thing to
learn first. Missing fact: the load factor protection of normal law [NTSB 1.6.3,
BEA447], or the TECS vertical acceleration limit of 0.7 g [TECS].

**A2. Overspeed (VMO/MMO) is a trip; the record has a warning. Major.**
Task: trips above 350 kt EAS or M 0.82 (`plane/dynamics.py:479-488`,
`plane/env.py:177-178`). Record: VMO 350 kt and MMO 0.82 with a "VMO/MMO warning";
"an operating limit with a warning, not a trip" [TCDS item 10]; what the controls do
above VMO: not found. Probe: a 6 deg nose-down demand from reset tripped on VMO after
12 s, because the reset speed of 200 m/s TAS is already 172 m/s EAS at 3 800 m, 4 %
under VMO (see A9). Effect: a common descent manoeuvre ends in a trip within seconds.
Also, the trip reads EAS while VMO is a CAS limit (PHYSICS.md admits this). Minor on
its own.

**A3. Stall is not protected and not a trip. Major.**
Task: no AOA bound in the inner loop or in `outside_envelope`; the wing has a
post-stall branch (`plane/PHYSICS.md` section 5) and the aircraft can be flown into
it with a 15 deg pitch demand at idle. Record: "The airplane cannot be stalled in
normal law"; AOA "will not exceed the allowable maximum" [NTSB 1.6.3]. Effect: an agent
can stall, sink, and then trip on load factor, VMO or the ground. The real aircraft
removes this failure mode. Missing fact: the AOA protection [NTSB 1.6.3].

**A4. The trip cost dwarfs every other term. Major.**
Task: a trip costs `restart_steps * failure_cost` = 3600 x 2 x (12 192 / e_floor)^2,
about 1.07e12 on `plane` and the heading, circle and racetrack tasks, 3.9e12 on the
figure eight (`plane/env.py:213-216`, `plane3d/env.py:235-237`, `reward.py:60-63`).
The plant then restarts from a fresh reset, with a new target and offset
(`base.py:452-486`). Record: "No source describes a trip that ends a flight. Every
limit found is either protected by the flight controls, warned, or a performance
requirement." Effect: one trip costs as much as about a million episodes of altitude
capture (A10). Together with A1 to A3, the score mostly measures never touching the
unprotected limits, which a real A320 cannot exceed in normal law.

**A5. Altitude band. Minor.**
Task: trip below 0 m and above 40 000 ft (`plane/env.py:229-230`). Record: maximum
operating altitude 39 100 ft, 39 800 ft with modification 30748 [TCDS item 11]. The
targets stay at 3 000 to 8 000 m, so the ceiling is never near; hitting the sea-level
floor is a real loss and fair as a trip.

### Observations

**A6. Airspeed is not observed; ground velocity is. Minor (Major on `plane_energy`).**
Task: obs carries `x_dot` (and `y_dot` in 3D), the inertial ground velocity
(`plane/env.py:533-552`, `plane/dynamics.py:20-35`); the gusts are hidden
(`plane/PHYSICS.md`, observation section). Record: the autopilot reads air data; PFD
airspeed "generally derived from ADR1" [BEA447]; TECS fuses airspeed with longitudinal
acceleration and a synthetic airspeed is "generally not recommended" [TECS]; gusts are
"seen through airspeed, acceleration and the height filter". Effect: the real
aircraft sees a horizontal gust immediately as an airspeed change; the agent sees it
only once it has moved the trajectory. The task hides more than the aircraft does.

**A7. No acceleration reading. Minor.** Task: no normal or longitudinal
acceleration in obs. Record: TECS's height and speed filters use vertical and
longitudinal acceleration [TECS, P409]. Effect: rate information an autopilot has is
missing; the agent must difference velocities.

**A8. No sensor latency, instantaneous vertical speed. Unsourced.** Task: readings
carry noise and rounding but no lag (`plane/dynamics.py:20-35`). Record: an
instantaneous VSI has "little or no lag", a conventional one lags a few seconds;
A320 sensor and computer latencies: not found [IFH ch. 5-6, BEA447].

### Episode shape

**A9. The reset aircraft is not established. Minor.**
Task: every reset sets 200 m/s ground speed at any altitude, full power, pitch 0 deg
and zero vertical speed (`plane/env.py:284-289`, `plane3d/env.py:262-270`); only the
elevator is trimmed (`plane/env_jax.py:171-172`). Probe: with zero demand the first
steps fly at 0.84 g, since the trim AoA is about 2 deg and pitch 0 gives AoA 0.
Record: the cruise mission is flown established in ALT CRZ [BEA447]. Effect: every
episode, and every restart after a trip, opens with a thrust and lift transient no
real hold has. At 3 000 m the start is 172 m/s EAS, 29 m/s above the 290 kt CAS
schedule the task itself commands, and near VMO (A2).

**A10. The start is up to 600 m off the assigned altitude, and that capture is
nearly the whole score. Major.**
Task: the initial altitude is the target plus U(-600, 600) m (`plane/env.py:283`,
`plane/env_jax.py:141-150`, `plane3d/env.py:261`, `plane3d/env_jax.py:163-172`).
Tracking is quadratic in metres over a 1 m floor (`reward.py:30-33`). Estimate (my
arithmetic, not a run): with |e0| uniform on 0 to 600 m and a climb or descent at
rate v, the capture costs E[e0^3]/(3v) = 5.4e7/(3v), so 3.6e6 at 5 m/s (about
1 000 fpm) and 1.8e6 at 10 m/s. The oracle's whole-episode hold cost is about 60 on
`plane` (0.21 per step over 280 steps), 100 to 320 on the 3D tasks (`rho_floor`
times the steps). Capture is therefore more than 99.9 % of the tracking cost at any
realistic climb rate, and takes on average 300/v s: 30 to 60 s, so 11 to 21 % of a
280 or 300 s episode. Record: the altitude hold is flown established at the
assigned level; the level-off is a separate manoeuvre with a 10 % lead and a 6 s
flare at 1 000 fpm [IFH ch. 7, BEA447]. Effect: controllers are ranked by how hard
they climb and level off, which the unprotected limits (A1 to A3) make a trip risk,
not by how they hold in turbulence, which is what the record calls the duty.
Missing fact: the hold starts established, the capture is its own phase [IFH ch. 7].

### Disturbances

**A11. Turbulence is continuous for the whole episode, at the top of light. Major.**
Task: OU gust on every axis, sigma 1.2 m/s per step, theta 0.2 /s, so 1.9 m/s
stationary std, on in every scored episode (`plane/env.py:300-301`,
`plane/dynamics.py:57-84`, `plane3d/env.py:283-284`). Probe: holding a fixed pitch
demand and throttle for 120 s, the step-peak load factor varied with 0.11 g std and
0.46 to 0.57 g peak to peak. Record: light turbulence is under 0.5 g peak to peak
[BEA447]; recorded cruise turbulence came in zones of 3 to 30 min, separated by calm
of over an hour, mostly 0.2 g or below [BEA447 Figure 24]. Effect: each episode is
the worst recorded light-turbulence zone, without the calm that is most of real
cruise. The task never asks to hold in calm air or to ride through a zone onset.
Missing fact: the intermittency of [BEA447 Figure 24].

**A12. Gust spectrum and scale. Minor.**
Task: a first-order OU with a 5 s correlation time, which is a scale length of about
1 000 to 1 150 m at 200 to 230 m/s; one draw is held across the four inner sub-steps
(`plane/env.py:630-643`). Record: Dryden or von Karman, scale length 1 750 ft (533 m)
above 2 000 ft, about 2.3 s at 230 m/s [MADDEN]. Effect: gusts about twice as long as
the record's, which shifts energy toward the phugoid and away from the short period.
The vertical Dryden gust is second order; the OU is first order. The 3D rolling gust
does use the 533 m scale (`plane3d/env.py:285-288`, `plane3d/dynamics.py:30-47`).

**A13. No steady wind. Minor (Major on `plane3d_racetrack`).**
Task: `wind_x`, `wind_z` (and `wind_y`), shear all 0 in the scored configurations
(`plane/env.py:293-313`, `plane3d/env.py:276-291`). Record: holding legs are timed
and drift-corrected for wind [IFH ch. 10]; wind changes the pivotal altitude on
eights [ACS]. Effect: ground speed equals airspeed on average, so the observation gap
of A6 never shows, and no path task asks for a wind correction.

### Actuators

**A14. The agent commands pitch attitude, not load factor or a mode. Minor.**
Task: action = throttle in idle to full, pitch demand +-15 deg, held 1 s; a fixed
4 Hz inner loop flies the elevator (`plane/env_jax.py:101-111`,
`plane/dynamics.py:224-231`). Record: in normal law the stick commands load factor and
roll rate, and the autopilot works in modes (ALT, level change, HDG, NAV) [BEA447,
NTSB 1.6.3]; TECS outputs pitch and throttle demands [TECS]. Effect: the action is
TECS-like, which the record accepts as the reference controller, but without TECS's
vertical acceleration or climb and sink limits [TECS, P409] and without any rate limit
on the demand (`base.py:64-71`, all actuator fields 0), so a full reversal of pitch
demand in one step is allowed and feeds A1.

**A15. Throttle is a direct lever, not autothrust. Minor.** Task: lever to
power with a 1.7 s lag (`plane/env.py:162-165`). Record: autothrust sets thrust in
cruise [BEA447]. The lag figure itself is outside the record: Unsourced.

### Objective

**A16. "Smoothly" is not scored; precision is scored far below the requirement.
Major.**
Task: the costs are quadratic altitude (and heading or path) error over floors of 1 m,
0.5 deg and 3 m, plus on the 2D tasks an airspeed excess term; nothing charges load
factor, attitude rate, bank, or control activity (`plane/env.py:450-468`,
`plane3d/env.py:1109-1175`). Record: the job is to hold "inside the protected envelope
and smoothly"; the altitude requirement is +-65 ft (+-20 m) in still air, a
performance requirement [RVSM A.4.1.4]; a tolerance in turbulence: not found. Effect:
the score keeps paying for every metre below a 20 m requirement and never charges
the accelerations that buy it, so the best-scoring policy is the hardest-working one.
In the real job that would be rough flying. Missing fact: the comfort or smoothness
side of the job sentence, and the RVSM band as the scale [RVSM].

---

## B. `plane` (altitude hold)

Registry `registry.py:507-520`; scored params `plane/env_jax.py:76-78` (280 steps,
hold pattern 0, `e_floor` 1 m, `c_hold` 5.06 m/s).

- **B1. Mission framing. Minor.** The record calls this mission real: cruise in ALT
  CRZ through light turbulence, +-65 ft in still air [RVSM, BEA447]. The task's
  altitude band, 3 000 to 8 000 m (9 800 to 26 200 ft, `plane/env.py:239`), is not
  tied to cruise in the record: Unsourced.
- **B2. Episode shape. Major.** See A10. With the 600 m offset the scored episode is
  a level change followed by a hold, and the level change carries the score.
- **B3. Time scale. Minor.** 280 s with turbulence throughout. Real zones last 3 to
  30 min [BEA447 Figure 24], so 280 s is inside a zone; it cannot show the onset or
  the end of one.
- **B4. Speed term. Minor.** Charged on ground speed `sqrt(x_dot^2 + z_dot^2)`
  against a TAS schedule (`plane/env.py:457, 465`). With no steady wind and 1.9 m/s
  gust std inside the 5.06 m/s free band this is small, but it scores the wrong
  quantity: TECS holds airspeed [TECS].

## C. `plane_sine` (sinusoidal altitude)

Registry `registry.py:540-552`; scored params `plane/env_jax.py:290-317`: 480 steps,
300 m amplitude, 240 s period, `e_floor` 1.26 m.

- **C1. No real duty. Major.** The task asks the aircraft to follow
  z = base + 300 sin(2 pi t / 240) for two periods (`plane/env.py:593`): a continuous
  climb and descent peaking at 7.9 m/s (about 1 550 fpm), reversing every 120 s.
  Record: "No real duty found. Neither ATC, the RVSM rules nor the autopilot documents
  describe following a periodic altitude profile." Nothing in the record supports it.
  The task's own docstring calls it a bandwidth probe (`plane/env_jax.py:282-283`),
  which is a test of the controller, not a mission. Its score is mostly phase lag
  times a quadratic cost, so it rewards high gain (A16).
- **C2. Episode shape. Major.** The 600 m start offset (A10) applies as well, and the
  sinusoid starts moving at once, so the first period is capture plus tracking.
- **C3. Objective. Minor.** The speed term charges ground speed (B4) with a 8.15 m/s
  free band, which is wider than on `plane` because the sinusoid makes speed swing.
  The record has no speed figure for this profile (no such duty).

## D. `plane_energy` (altitude staircase with airspeed)

Registry `registry.py:524-538`; scored params `plane/env_jax.py:256-278`: 1 600 steps,
eight 200 s treads, amplitude 900 m, levels `_STEP_LEVELS` (`plane/env.py:563`), so
changes of 360 to 720 m, `e_floor` 1.20 m, `c_hold` 5.23 m/s.

- **D1. The score has almost no speed-against-height trade. Major.**
  Task: tracking is (e / 1.20)^2; speed costs max(|V - V_sched| - 5.23, 0) / 5.23,
  linear (`plane/env.py:460-467`, `reward.py:48-52`). `speed_weight = 0.5`
  (`plane/env_jax.py:272`) is read only by the version 1 reward
  (`plane/env.py:471-512`); the scored version 2 never reads it. One unit of cost is
  1.2 m of altitude or 10.5 m/s of speed; during a 450 m level change the altitude
  term is about 1.4e5 per step, so losing 50 m/s of speed (8.6 per step) costs
  nothing that matters. Record: TECS's job is to hold height and speed together,
  throttle on total energy, pitch on the split; real cases are altitude changes with
  speed held [TECS, P409]. Effect: the optimum trades speed for height freely during
  every change, which is the opposite of the TECS job; the only lower speed limit is
  the stall (A3), which the real aircraft protects and the task does not.
  Missing fact: the speed hold in an altitude change [TECS, TECS_SPDWEIGHT in P409].
- **D2. Speed is scored and observed as ground speed. Major for this task.** The
  airspeed term and the agent's speed reading are both inertial (A6, B4). On the one
  task whose stated job is speed, the agent cannot see airspeed, the quantity TECS
  holds [TECS].
- **D3. Level changes are instant steps of 360 to 720 m. Minor.** The commanded
  altitude jumps at each tread (`plane/env.py:582-591`). Real level changes are flown
  at a vertical speed with a level-off lead [IFH ch. 7]; a published A320 profile of
  such a mission: not found. Effect: the quadratic cost makes each step's first
  seconds dominate the tread (as A10).
- **D4. Speed schedule. Unsourced.** 290 kt CAS then M 0.78 (`plane/env.py:265-266`),
  cited to EUROCONTROL, which is not in the record. It stays under VMO and MMO [TCDS].
- **D5. Time scale. Minor.** 200 s treads; a real cruise altitude change is followed
  by a long hold. Not in the record either way.

## E. Items shared by the four 3D tasks

Plant: `plane3d/env.py:900-1107`; action [throttle, pitch demand +-15 deg, bank demand
+-45 deg] (`plane3d/env_jax.py:126-138`).

- **E1. Bank is a trip at 60 deg. Major.** `abs(phi) > bank_limit` trips
  (`plane3d/env.py:138, 403`), cited to 14 CFR 91.307(c). Record: normal law limits
  bank attitude; "a protected limit, not a trip" [BEA447]. The 91.307(c) figure is
  not in the record (it is a parachute rule, not an airframe limit). The bank demand
  stops at 45 deg (`plane/dynamics.py:228`), but the roll loop overshoots a 45 deg
  step by 12 % (`plane/PHYSICS.md`, inner loop) and the rolling gust adds about
  1 deg std (probe), so a full-bank demand flies within about 10 deg of a trip worth
  1e12.
- **E2. Bank demand of 45 deg is above every autopilot figure in the record. Minor.**
  Record: holding turns at 3 deg/s, 30 deg or the flight director's bank [IFH ch. 10];
  ArduPlane LIM_ROLL_CD [P409]; A320 HDG bank limit not found (25 deg only
  unconfirmed). With no cost on bank (A16), the score pushes every turn to the 45 deg
  demand limit.
- **E3. No speed command or speed cost. Minor (Major on the racetrack).** The 3D
  scores have no speed term (`plane3d/env.py:1127-1175`); the aircraft flies whatever
  speed results, about 200 m/s from reset. Record: the job holds "the assigned
  altitude and speed"; holding speeds are capped by altitude [IFH ch. 10].
- **E4. Heading is the ground track. Minor.** `psi` is computed from ground velocity
  (`plane3d/dynamics.py:66-68`). HDG select holds heading [IFH ch. 7]. With no steady
  wind and no sideslip state (`plane3d/PHYSICS.md` D1) the two nearly coincide; in
  wind they would not.
- **E5. Lateral disturbance. Minor.** No sideslip state (PHYSICS D1). Probe: 120 s
  wings level in default turbulence moved the track by 0.21 deg std and the aircraft
  21 m sideways, mostly through the rolling gust. A crosswind drift that has to be
  corrected [IFH ch. 10] does not exist in the scored tasks (A13).
- **E6. Capture share (open question). Major.** See A10 for altitude; the 3D tasks
  inherit the +-600 m offset. Per task below.

## F. `plane3d_heading`

Scored params `plane3d/env_jax.py:264-283`: 300 steps; target heading U(-pi, pi)
(`plane3d/env.py:204`), start heading 0 (`plane3d/env_jax.py:191`); costs
(e / 1 m)^2 + (dpsi / 0.0087 rad)^2 (`plane3d/env.py:1127-1139`).

- **F1. Mission. Real.** Heading select with a roll-out lead of half the bank [IFH
  ch. 7]. Capture is the job here, so a capture phase is legitimate.
- **F2. Heading capture plus an unrelated altitude capture. Major.** Each episode
  combines a turn of 0 to 180 deg (mean 90 deg) with a 0 to 600 m level change. My
  estimate: the turn costs E[dpsi^3]/(3 r floor^2), 7e5 at a 45 deg bank (about
  2.8 deg/s at 200 m/s) and 1.5e6 at 25 deg (1.3 deg/s); the altitude part 1.8e6 to
  3.6e6 (A10); the oracle's hold over the episode about 100. A 180 deg turn takes
  64 s at 45 deg and 137 s at 25 deg, up to 46 % of the 300 s episode. Record: HDG
  select is a lateral mode; the record does not pair it with a level change.
- **F3. The score rewards the steepest bank allowed. Major.** Quadratic heading cost
  with no bank, roll rate or load cost means the shortest turn wins, at the 45 deg
  demand limit and close to the 60 deg trip (E1). Real: 25 to 30 deg [IFH ch. 10;
  25 deg unconfirmed], and the roll-out lead of half the bank [IFH ch. 7].
- **F4. Heading floor 0.5 deg. Unsourced.** `e_floor_heading` 0.0087 rad
  (`plane3d/env.py:231`), cited as an AHRS resolution; no heading tolerance in the
  record.

## G. `plane3d_circle`

Scored params `plane3d/env_jax.py:459-473`: 300 steps; radius U(8, 12 km)
(`plane3d/env.py:205`); start on the circle, tangent, counter-clockwise, wings level
(`plane3d/env_jax.py:497-527`); costs altitude plus radial distance over 3 m.

- **G1. Mission. Unsourced for an airliner.** Record: ArduPlane LOITER circles at
  WP_LOITER_RAD [P409]; "for an airliner, a constant-radius orbit as a routine duty:
  not found". The 8 to 12 km radius and the left-hand direction have no source.
- **G2. Capture (open question). Major.** Laterally the start is on the circle and
  tangent, so the only lateral transient is the roll-in to about 19 to 27 deg at
  200 m/s, a few seconds and some metres. The altitude offset (A10) is the capture:
  more than 99.9 % of the tracking cost, 10 to 20 % of the 300 s. Unlike a real
  loiter entry, the task never asks for the lateral capture, but it does ask for an
  unrelated level change.
- **G3. Time scale. Minor.** One lap at 10 km and 200 m/s is 314 s, so the 300 s
  episode is about one orbit, a sixth of it spent on altitude capture.
- **G4. No wind. Minor.** In still air an orbit is a constant bank; ground-referenced
  loiters in wind need a varying bank. The record does not discuss wind on orbits,
  so this is Unsourced beyond A13.

## H. `plane3d_racetrack`

Scored params `plane3d/env_jax.py:360-378`: 650 steps; radius U(8, 12 km); straight
legs of 4 radii (`_RACETRACK_LEG` = 2 half-legs, `plane3d/env.py:647`,
`distance_to_racetrack` `plane3d/env.py:492-521`); orientation uniform; start on a leg
abeam, tracking along it; right turns.

- **H1. Mission. Real** (the holding pattern) [IFH ch. 10]. Right-hand turns match.
  The cost is the unsigned distance to the oval, so flying it in the wrong direction
  scores the same. Minor.
- **H2. Legs and turns are far longer than a hold. Major.** Legs are 4r = 32 to 48 km,
  160 to 240 s at 200 m/s, against the record's 1 min inbound leg at or below
  14 000 ft and 1.5 min above [IFH ch. 10]; the targets span 9 800 to 26 200 ft, so
  both rules apply. A half turn at r = 8 to 12 km takes 125 to 190 s at 200 m/s
  (1.0 to 1.4 deg/s), against 70 to 90 s at 25 to 30 deg and the 3 deg/s standard
  rate [IFH ch. 4, 10]. The bank needed, 19 to 27 deg, is realistic; the speed is not
  (H3), which is what makes the radius large.
- **H3. Holding speed limits absent. Major.** No speed is commanded or charged (E3).
  At 3 000 m, 200 m/s TAS is about 334 kt EAS; the record caps holding at 230 KIAS
  from 6 001 to 14 000 ft and 265 KIAS above [IFH ch. 10]. Missing fact: the holding
  speed limits [IFH ch. 10].
- **H4. No wind correction. Major.** Record: holding legs are timed to correct for
  wind and drift is corrected on both legs [IFH ch. 10]. The scored configuration has
  no steady wind (A13), so the defining skill of flying a hold is absent.
- **H5. Episode covers about one lap. Major.** A lap is 2 x 4r + 2 pi r = 14.3 r, 114
  to 171 km, 570 to 860 s at 200 m/s. The 650 s episode is 0.8 to 1.1 laps at that
  speed. The comment at `plane3d/env_jax.py:362-365` gives a lap of "8.3 r" and three
  laps; by the geometry in `distance_to_racetrack` that arithmetic is wrong (the legs
  are 4r each, not 2r). Each straight-to-turn transition, the stated point of the
  task, is seen about twice, and the first is during altitude capture.
- **H6. Capture (open question). Major.** Laterally the start is on a leg and
  tracking, so nearly zero lateral capture. The altitude offset (A10) still carries
  more than 99.9 % of the tracking cost (oracle hold about 320 over the episode
  against 1.8e6 to 3.6e6). A real hold is entered at its assigned altitude, and the
  entry is its own procedure [IFH ch. 10].

## I. `plane3d_figure8`

Scored params `plane3d/env_jax.py:564-581`: 1 000 steps; lemniscate half-width
34 to 36 km, rotated +-15 deg, altitude swung +-200 m along it
(`plane3d/env.py:250, 650-669`); start exactly on the curve and at its altitude, at
the tip, tangent, wings level (`plane3d/env_jax.py:618-646`); cost 3D distance to the
curve over 3 m, no altitude term (`plane3d/env.py:1170-1175`).

- **I1. Mission. Major.** Record: real "only as a training manoeuvre in light
  aircraft": eights on pylons, bank not over 40 deg, altitude moved on purpose to keep
  the pivotal altitude [ACS]; "for a transport aircraft: no duty found"; a UAV
  figure-eight loiter is unconfirmed. The task is not eights on pylons either: it is a
  ground-track lemniscate with a prescribed +-200 m altitude twist, not a pivotal
  altitude on two points. The twist amplitude is Unsourced.
- **I2. Capture (open question). Minor.** The start is on the curve, at its altitude
  and tangent, so there is no altitude or lateral capture. The only transient is
  starting wings level at the tip, where the tip radius a/3 (about 11.6 km) needs
  about 19 deg at 200 m/s. This task is the one 3D task whose score is not dominated
  by capture.
- **I3. Objective. Minor.** Height and lateral error are summed in one 3D distance
  over the 3 m GPS floor, although altitude is read to 1 ft (`plane/dynamics.py:22`).
  Not in the record either way.
- **I4. Time scale. Minor.** A lap is about 183 km; 1 000 s at 200 m/s is about 1.1
  laps, so each crossover and tip is flown once or twice.

---

## Open questions, answered

**Does capture dominate the 3D scores?** Yes on heading, circle and racetrack, no on
the figure eight. The +-600 m start offset with a quadratic 1 m floor makes the
altitude capture cost 1.8e6 to 3.6e6 per episode at 5 to 10 m/s climb, against an
oracle hold of 100 to 320, so more than 99.9 % of the tracking cost, in 10 to 20 % of
the episode time (5 to 9 % on the racetrack). The heading task adds a turn of up to
180 deg, another 7e5 to 1.5e6 and up to 46 % of the episode. This is unlike a real
mission: a hold, orbit or pattern is flown established at its assigned level, and the
level change or entry is a separate phase [IFH ch. 7, ch. 10]. The same applies to
`plane` and `plane_sine` (A10).

**`plane_sine`.** The task asks for a 300 m, 240 s altitude sinusoid peaking at about
1 550 fpm. Nothing in the record supports it: "No real duty found" (C1).

**`plane_energy`.** Structurally the plant has a speed-against-height trade (thrust is
finite), but the scored version 2 reward makes speed about five orders of magnitude
cheaper than height during a level change, a linear excess over a 5.23 m/s free
band against a quadratic over 1.2 m, and the version 1 `speed_weight` is not read.
That is not the TECS balance of height and speed [TECS, P409] (D1). Speed is also
measured on ground speed, not airspeed (D2).

**`plane3d_figure8`.** Agreed with the record: a light-aircraft training manoeuvre
with no transport duty [ACS]; and the task is not that manoeuvre either (I1).

**Limits as trips where the aircraft protects them.** Yes, all four: load factor
(A1, a 9 deg pitch demand step trips at 2.57 g), bank at 60 deg in 3D (E1), overspeed
VMO/MMO (A2, a 6 deg descent from reset trips in 12 s), and stall, which is neither
protected nor a trip and leads to the others (A3). The record finds every one of them
protected or warned, and no trip that ends a flight.

---

## Most important mismatches

1. Protected limits are trips: load factor, bank 60 deg, VMO/MMO, each costing about
   1e12; a 9 deg pitch demand step already trips on load factor (A1, A2, E1, A4)
   [NTSB 1.6.3, BEA447, TCDS].
2. Stall is possible and unprotected, where normal law makes it impossible (A3)
   [NTSB 1.6.3].
3. The +-600 m start offset makes altitude capture more than 99.9 % of the tracking
   cost on `plane`, `plane_sine` and three 3D tasks; the holds the record describes
   are flown established (A10, E6) [IFH ch. 7, BEA447].
4. The objective has no smoothness term and scores precision far below the +-20 m
   requirement, so aggressive control wins (A16, F3) [RVSM, job sentence].
5. `plane_energy` has no real speed-against-height trade in the scored reward and
   reads ground speed, not airspeed (D1, D2, A6) [TECS, P409].
6. Turbulence is on for the whole episode at the top of light (about 0.5 g peak to
   peak), with no calm, about twice the record's scale length (A11, A12)
   [BEA447 Figure 24, MADDEN].
7. `plane3d_racetrack` is not a hold in time scale or procedure: 2.7 to 4 min legs,
   no holding speed limit, no wind correction, about one lap per episode (H2 to H5)
   [IFH ch. 10].
8. `plane_sine` and `plane3d_figure8` have no real duty for this aircraft (C1, I1);
   `plane3d_circle`'s orbit is an autopilot loiter with no airliner duty found (G1).
