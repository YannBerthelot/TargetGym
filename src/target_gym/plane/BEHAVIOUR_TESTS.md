# Aircraft family: proposed behaviour tests

Proposed on 2026-11-08 under charter Amendment 1, step 3 "Match it". No test code is
written yet. Each test names its source in [REALITY.md](REALITY.md), the quantity, and a
tolerance fixed here, before any test runs. A failing test is a finding. Its fix is a
missing piece of physics or operation from a source, never a number moved until the test
passes.

Tolerances come from three places only: the source's own band, the spread of a rule of
thumb against exact physics, or the sampling error of a statistic. Each says which.

The cold review ([COLD_REVIEW.md](COLD_REVIEW.md)) predicts which tests fail as the
tasks stand: A3 and A4 (protected limits are trips today), A5 (turbulence is on for the
whole episode), B2 for `plane_sine`, B3 (no energy trade in the score), B5 (legs of
160 to 240 s against 60 s). Those predictions are not results; the tests decide.

## A. Plant physics (all seven tasks)

**A1. Turn rate against bank.** Hold a steady level coordinated turn at 25 deg and 30 deg
bank, at the task's cruise speed. Measure the heading rate.
- Source: kinematics of a coordinated level turn, rate = g tan(bank) / TAS; at 250 KTAS
  and 25 deg, about 2.0 deg/s. [IFH ch. 4] ties standard rate (3 deg/s) to airspeed/10 + 7.
- Tolerance: 3 % of the predicted rate (sideslip and speed loss in the turn).
- Second check: the bank that gives 3 deg/s at 150 KTAS must be within 2 deg of
  150/10 + 7 = 22 deg (the rule of thumb is 0.4 deg off the exact 22.4 deg there).

**A2. Load factor in a level turn.** Same runs as A1. Normal load factor = 1 / cos(bank):
1.10 g at 25 deg, 1.15 g at 30 deg.
- Source: physics of a level turn; the holding bank of 30 deg [IFH ch. 10].
- Tolerance: 0.02 g (the height loss allowed in the run, under 20 ft, adds less).

**A3. Load factor is protected, not tripped.** Drive the agent's action to full nose-up
and full bank in cruise.
- Source: 2.5 g / -1.0 g limit loads [CFR 25.337]; normal law protects load factor,
  attitude and AOA, and "the airplane cannot be stalled in normal law" [NTSB 1.6.3,
  BEA447].
- Pass: the episode does not end; normal load factor stays within [-1.0, 2.5] g plus
  0.1 g (overshoot of a protection loop, a tolerance stated here, not found in a source);
  AOA stays below its stated maximum.
- If the task's design keeps an unprotected aircraft on purpose, this test is replaced by
  a cost on the limit, and `PHYSICS.md` lists it as a known deviation.

**A4. Overspeed and altitude limits match the type.** Read the task's limits.
- Source: VMO 350 kt, MMO 0.82, maximum operating altitude 39 100 ft [TCDS].
- Tolerance: exact, after unit conversion (1 kt, 0.005 Mach, 100 ft).
- Pass also requires that crossing VMO/MMO is a warning or cost, not a trip [TCDS lists a
  warning, no source lists a trip].

**A5. Turbulence intensity in the "light" setting.** Fly the reference autopilot in
altitude hold for 30 min of simulated cruise at the task's gust setting labelled light,
or at its default if unlabelled. Measure normal load factor peak to peak in 1 min windows.
- Source: ICAO light turbulence is under 0.5 g peak to peak [BEA447]; recorded zones of
  0.1 to 0.5 g, mostly 0.2 or below [BEA447 Figure 24].
- Pass: every window under 0.5 g; median window between 0.1 and 0.3 g.
- Tolerance reason: the band is the figure's own range.

**A6. Turbulence scale length.** From the same run, estimate the integral scale of the
vertical gust along the flight path.
- Source: 1 750 ft (533 m) above 2 000 ft AGL [MADDEN, from MIL-F-8785C].
- Tolerance: 25 %, the sampling error of an integral-scale estimate from 30 min at about
  230 m/s (about 780 scale lengths); a tighter bound needs a longer run.

## B. Missions

**B1. Altitude hold in still air (`plane`, every plane3d task).** Gusts off, reference
autopilot, 10 min after capture.
- Source: +-65 ft (+-20 m) about the acquired altitude in straight and level flight,
  "nonturbulent, nongust conditions" [RVSM].
- Pass: every sample within 20 m.
- This is a fitness test of the plant and the reference together: an airliner autopilot
  meets it, so a plant on which a textbook TECS cannot is missing something.

**B2. The altitude target is an assigned level (`plane`, `plane_energy`).** Read the
target sequence over an episode.
- Source: altitude is assigned and held [RVSM, IFH ch. 10 "maintain the last assigned
  altitude"]; changes are discrete level changes, flown with a level-off lead of 10 % of
  vertical speed [IFH ch. 7].
- Pass: the target is piecewise constant. A target that moves continuously has no source
  (see `plane_sine` in the cold review).

**B3. Energy trade (`plane_energy`).** From trimmed cruise, step pitch demand only, then
throttle only, each for 10 s.
- Source: throttle sets total energy, pitch sets the split between potential and kinetic
  energy [TECS].
- Pass: after the pitch step, the rate of specific total energy (h + V^2 / 2g) changes by
  under 20 % of the change the throttle step gives.
- Tolerance reason: drag changes with speed during the pitch step; 20 % is set here.

**B4. Heading change duration (`plane3d_heading`).** A 90 deg heading change at 25 deg
bank and the task's speed.
- Source: rate from A1; roll-out lead of half the bank [IFH ch. 7].
- Pass: the time from roll-in to wings level is 90 / rate plus the roll-in and roll-out
  time, within 15 % (roll rate and its source: not found, hence the wider band).

**B5. Holding pattern geometry (`plane3d_racetrack`).** Fly the reference on the task's
pattern.
- Source: right turns, inbound leg 1 min at or below 14 000 ft MSL, 1.5 min above,
  turns at 3 deg/s or 30 deg or the flight director's bank, holding speed at most 200,
  230 or 265 KIAS by altitude [IFH ch. 10].
- Pass: the task's pattern, speed and altitude fit these rules; inbound leg within 5 s of
  the rule (the timing step of a pilot's clock, stated here).

**B6. Orbit radius (`plane3d_circle`).** The orbit's radius against the aircraft's
turn radius at its bank limit.
- Source: physics of A1; WP_LOITER_RAD is the circle's radius in ArduPlane [P409].
- Pass: the radius needs a bank within the task's bank limit at its speed, with the
  margin the charter gives the reference (limits at 80 % of the trip).
- No airliner duty was found for an orbit, so this test only checks feasibility.

**B7. Figure eight (`plane3d_figure8`).** No transport-aircraft source. The only source
[ACS] is a light-aircraft training manoeuvre with bank "not to exceed 40 deg" and altitude
moved on purpose. Proposed test: none until the owner decides whether the task stays.

## C. Reference controller configuration

**C1. TECS and L1 in their documented ranges.** Read the reference's printed values.
- Source: TECS_TIME_CONST 3.0 to 10.0 s, TECS_VERT_ACC default 7 m/s^2, NAVL1_DAMPING 0.6
  to 1.0 and advised 0.7 to 0.85, NAVL1_PERIOD larger "for less responsive airframes"
  [P409, TECS].
- Pass: each value in its range; any value outside is reported with its reason.

## Not testable yet (source not found)

- Sensor delays of the A320's air data and inertial references.
- A320 short period, phugoid and roll mode time constants.
- The A320 autopilot's bank limit per mode and its ALT* capture law (FCOM, owner).
- A tolerance on altitude keeping in turbulence.
