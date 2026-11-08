# Aircraft family: the real aircraft and its missions

The record required by charter Amendment 1 for `plane`, `plane_sine`, `plane_energy`,
`plane3d_heading`, `plane3d_circle`, `plane3d_racetrack` and `plane3d_figure8`. It was
written on 2026-11-08 from primary sources only, before the tasks' code was read. The
tasks name the Airbus A320 as their airframe, so the record is about an A320 and its
autopilot, with ArduPlane 4.0.9 where the charter names it as the reference controller.

Where no primary source was found, the record says "not found". Values that only appear
in secondary sources (training sites, simulator projects, forums) are listed at the end as
unconfirmed and are not used above.

## Sources

Each was opened, and the quoted lines were checked a second time.

- **[TCDS]** EASA, Type-Certificate Data Sheet EASA.A.064, Airbus A318/A319/A320/A321,
  issue 62, 26 June 2026. A320 section, items 10 and 11.
  <https://www.easa.europa.eu/en/downloads/16507/en>
- **[CFR]** 14 CFR 25.337, "Limit maneuvering load factors", via Cornell LII.
  <https://www.law.cornell.edu/cfr/text/14/25.337>
- **[RVSM]** FAA AC 91-85B, Authorization of Aircraft and Operators for Flight in Reduced
  Vertical Separation Minimum Airspace, appendix A, A.4.1.4.
  <https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_91-85B.pdf>
- **[IFH]** FAA, Instrument Flying Handbook, FAA-H-8083-15B: chapter 4 (standard rate
  turn), chapter 5 (VSI lag), chapter 6 (trend vectors), chapter 7 (level off, turns to a
  heading), chapter 10 (holding).
  <https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/FAA-H-8083-15B.pdf>
- **[ACS]** FAA, Commercial Pilot Airplane Airman Certification Standards, FAA-S-ACS-7B,
  Area V Task E, Eights on Pylons.
  <https://www.faa.gov/training_testing/testing/acs/commercial_airplane_acs_7.pdf>
- **[NTSB]** NTSB AAR-10/03, US Airways 1549, A320, section 1.6.3 "Flight Envelope
  Protections" (quotes the Airbus FCOM and FCTM).
  <https://www.ntsb.gov/investigations/AccidentReports/Reports/AAR1003.pdf>
- **[BEA447]** BEA, final report on AF447, A330 F-GZCP, 1 June 2009 (FAA-hosted copy):
  section on flight control laws, and Figure 24, the turbulence recorded in cruise with
  the autopilot in ALT CRZ. The A330 shares the A320's control-law family; that is an
  inference, and the report itself is about an A330.
  <https://www.faa.gov/sites/faa.gov/files/f-cp090601.en_.pdf>
- **[TECS]** ArduPilot, "TECS (Total Energy Control System) for Speed and Height Tuning
  Guide". <https://ardupilot.org/plane/docs/tecs-total-energy-control-system-for-speed-height-tuning-guide.html>
- **[P409]** ArduPilot, Complete Parameter List, Plane stable V4.0.9.
  <https://ardupilot.org/plane/docs/parameters-Plane-stable-V4.0.9.html>
- **[L1]** ArduPilot, "Navigation Tuning" (L1 controller, after Park, Deyst and How,
  AIAA-2004-4900). <https://ardupilot.org/plane/docs/navigation-tuning.html>
- **[MADDEN]** M. Madden, NASA, verification of the Dryden turbulence model in LaSRS++
  (NTRS 20190000875), which restates the MIL-F-8785 scale lengths.
  <https://ntrs.nasa.gov/api/citations/20190000875/downloads/20190000875.pdf>

**For the owner to open by hand** (blocked to automated readers, or not public):
- Airbus A320 FCOM and FCTM, flight controls (DSC-27) and auto flight (DSC-22-30). These
  hold the normal-law bank and load factor numbers, the AP bank limits per mode and the
  ALT* capture law. Not public.
- BEA report on D-AXLA (A320, Perpignan, 2008), section 1.6.8 on normal-law
  protections. bea.aero refused the connection.
- MIL-F-8785C and MIL-HDBK-1797: everyspec.com serves a page, not the PDF.
- FAA AIM 5-3-8 (holding): faa.gov returned 403. [IFH] chapter 10 carries the same rules.
- eCFR: redirects automated readers. [CFR] was read on Cornell LII instead.

## What the pilot or autopilot reads, and its delay

- **Pilot:** altitude, vertical speed, airspeed and attitude on the primary flight
  display. A conventional VSI "lags a few seconds behind the actual change in pressure";
  an instantaneous VSI shows the rate "with little or no lag" [IFH ch. 5 and 6]. Glass
  displays add trend vectors that show airspeed and altitude "in 6 seconds" [IFH ch. 6].
- **A320 autopilot:** air data (ADR) and inertial references; the airspeed on each PFD is
  "generally derived from ADR1" [BEA447, speed parameters]. Sensor and computer
  latencies: not found.
- **TECS:** height and height rate from a complementary filter of barometric height and
  vertical acceleration (TECS_HGT_OMEGA, range 1 to 5 rad/s), and airspeed fused with
  longitudinal acceleration (TECS_SPD_OMEGA) [TECS, P409]. Without an airspeed sensor
  a synthetic estimate "is usually not recommended since the synthetic airspeed estimate
  can be very wrong on occasion" [TECS].
- **Wind:** neither the pilot nor TECS reads the gust directly. Gusts are seen through
  airspeed, acceleration and the height filter. A direct gust measurement: not found.

## What it moves, and in which mode

- **A320 normal law:** "the longitudinal orders from the sidesticks command a load factor
  ... and the lateral orders command a rate of roll" [BEA447, for the A330 normal law].
  The flight computers then move elevator, ailerons and spoilers [NTSB 1.6.3]. The pilot
  and the autopilot do not move surfaces directly.
- **Autothrust** sets thrust. In cruise the A330 of [BEA447] flew "ALT CRZ / NAV": an
  altitude hold and a lateral navigation mode, both on the autopilot.
- **TECS** "coordinates throttle and pitch angle demands to control the aircraft's height
  and airspeed" [TECS]. Throttle holds total energy; pitch sets the split between
  potential and kinetic energy [TECS]. TECS_SPDWEIGHT 1 blends, 0 makes pitch hold height
  and throttle speed, 2 the reverse [P409].
- **L1** turns cross-track error into a lateral acceleration, hence a bank demand,
  limited by LIM_ROLL_CD [L1, P409]. ArduPlane's loiter circles use "a modified PD control
  law rather than the L1 control law" [L1].
- **Modes for the missions:** altitude hold; altitude change with speed held; heading
  select; lateral navigation; holding. The A320's own mode logic and bank limits per
  mode: not found (FCOM, owner).

## Limits, and whether each is a trip, a timed limit or a cost

- **Load factor.** Limit manoeuvring load factor not less than 2.5 g for this weight
  class, negative not less than -1.0 g up to VC [CFR 25.337(b), (c)]. A limit load is a
  design load, not a trip. The A320's flight controls limit it in normal law ("flight
  envelope protections", [NTSB 1.6.3]; "load factor" protection, [BEA447]). Exceeding it
  is not a trip in any source; what follows an exceedance (inspection): not found.
  *Classification: a protected limit, enforced by the aircraft, not a trip.*
- **Bank.** Normal law limits pitch and bank attitude [BEA447]. Values: not found in a
  primary source (see unconfirmed list). Holding turns use 3 deg/s, 30 deg bank, or the
  flight director's bank, whichever applies [IFH ch. 10]. ArduPlane bank limit is
  LIM_ROLL_CD [P409]. *Classification: a protected limit, not a trip.*
- **Stall.** "The airplane cannot be stalled in normal law." Above a threshold the AOA is
  proportional to stick and "will not exceed the allowable maximum" [NTSB 1.6.3].
  *Classification: protected; a stall is not a trip but a loss of control, possible only
  outside normal law.*
- **Overspeed.** VMO 350 kt, MMO 0.82 [TCDS item 10]. The TCDS lists a "VMO/MMO warning"
  [TCDS]. What the flight controls do above VMO: not found in a primary source.
  *Classification: an operating limit with a warning, not a trip.*
- **Altitude.** Maximum operating altitude 39 100 ft, 39 800 ft with modification 30748
  [TCDS item 11].
- **Altitude keeping.** RVSM: the altitude control "should be capable of controlling
  aircraft height within a tolerance band of +-65 ft (+-20 m) about the acquired
  altitude when the aircraft is operated in straight and level flight under
  nonturbulent, nongust conditions" [RVSM A.4.1.4]. *Classification: a performance
  requirement, not a trip.* A tolerance in turbulence: not found.
- **Holding speeds.** Maximum 200 KIAS up to 6 000 ft, 230 KIAS from 6 001 to 14 000 ft,
  265 KIAS above [IFH ch. 10].
- **TECS limits:** TECS_VERT_ACC, maximum vertical acceleration used to correct errors,
  default "7 m/s/s (equivalent to +- 0.7 g)" [TECS]; climb and sink rate limits
  TECS_CLMB_MAX and TECS_SINK_MAX [P409].

No source describes a trip that ends a flight. Every limit found is either protected by
the flight controls, warned, or a performance requirement.

## Time scales

- **Turns:** standard rate 3 deg/s; bank for it is about airspeed/10 + 7 [IFH ch. 4]. At
  A320 speeds that rate needs more than 30 deg, so turns are bank-limited. From the
  physics of a level coordinated turn (not a source value): at 250 KTAS and 25 deg bank
  the rate is about 2.0 deg/s and the radius about 3.6 km; at 30 deg, 2.5 deg/s.
- **Holding:** 1 min inbound leg at or below 14 000 ft MSL, 1.5 min above [IFH ch. 10]. A
  180 deg turn at 25 to 30 deg bank takes 70 to 90 s at these speeds (physics).
- **Level off:** lead the altitude by 10 % of the vertical speed: 1 000 fpm, 100 ft
  lead [IFH ch. 7]. At 1 000 fpm that is 6 s.
- **Heading capture:** lead the heading by half the bank angle [IFH ch. 7].
- **Turbulence:** zones lasting 3 to 30 min, separated by calm of over an hour, in a
  Rio-Paris cruise [BEA447 Figure 24]. Scale length 1 750 ft above 2 000 ft AGL
  [MADDEN, quoting MIL-F-8785C and MIL-HDBK-1797]; at 230 m/s that is about 2.3 s per
  scale length.
- **TECS:** TECS_TIME_CONST range 3.0 to 10.0 s [P409]; raise it in steps of 1, "not
  going above 10" [TECS].
- **L1:** NAVL1_PERIOD default 20 s, "needs to be larger for less responsive airframes"
  [P409]; NAVL1_DAMPING 0.7 to 0.85 [P409].
- **Aircraft modes (short period, phugoid, roll mode) of the A320:** not found.

## Disturbances

- **Turbulence intensity:** ICAO light turbulence is "changes in the normal load factor
  at the centre of gravity of less than 0.5 g peak to peak" [BEA447]. Recorded in cruise
  on autopilot: zones of at most 0.1 to 0.5 g peak to peak, mostly 0.2 or below
  [BEA447 Figure 24, read from the figure].
- **Spectrum:** Dryden or von Karman, scale length 1 750 ft at altitude [MADDEN]. Intensity
  at altitude comes from a probability-of-exceedance chart in MIL-F-8785C (owner).
- **Steady wind:** holding legs are timed to correct for wind, and drift is corrected on
  both legs [IFH ch. 10]; on eights on pylons wind changes the pivotal altitude [ACS].

## The missions, and whether each is a real duty

- **Altitude hold in gusts (`plane`).** Real. Cruise in ALT CRZ on the autopilot, through
  light turbulence, holding +-65 ft in still air [RVSM, BEA447]. A bad day: turbulence of
  up to 0.5 g peak to peak for minutes [BEA447 Figure 24]; in AF447 the autopilot then
  disconnected for an unrelated reason (iced pitot probes).
- **Sine altitude tracking (`plane_sine`).** No real duty found. Neither ATC, the RVSM
  rules nor the autopilot documents describe following a periodic altitude profile.
- **Energy management (`plane_energy`).** Real. TECS's job: hold height and speed by
  trading pitch against throttle, throttle on total energy, pitch on the split [TECS].
  Real cases: altitude changes with speed held, and speed changes at constant altitude.
  A published profile of such a mission for an A320: not found.
- **Heading capture (`plane3d_heading`).** Real: heading select, roll out with a lead of
  half the bank [IFH]. The A320's HDG mode bank limit: not found (FCOM, owner).
- **Orbit (`plane3d_circle`).** Real for an autopilot: ArduPlane LOITER circles a point
  at WP_LOITER_RAD [P409]. For an airliner, a constant-radius orbit as a routine duty:
  not found.
- **Racetrack (`plane3d_racetrack`).** Real: the holding pattern. Right turns, 1 min
  legs at or below 14 000 ft, turns at 3 deg/s or 30 deg or the flight director's bank,
  speed limits by altitude, wind corrected on the legs [IFH ch. 10].
- **Figure eight (`plane3d_figure8`).** Real only as a training manoeuvre in light
  aircraft: eights on pylons, bank "not to exceed 40 deg", altitude moved on purpose to
  keep the pivotal altitude [ACS]. For a transport aircraft: no duty found. A figure-eight
  loiter for unmanned aircraft appears in patent titles, but no text was read: not
  confirmed.

## The job, in one sentence

Hold the assigned altitude and speed, and follow the assigned heading or track, inside
the protected envelope and smoothly, while turbulence and wind push the aircraft around;
the flight controls, not the pilot, keep load factor, bank and AOA inside their limits.

## Unconfirmed (secondary sources only, not used above)

- A320 normal law: bank 67 deg at full stick, returning to 33 deg when released; load
  factor +2.5 to -1 g clean; AP disconnect above 45 deg bank (simulator project docs and
  training summaries).
- A320 AP bank limit 25 deg in HDG, 15 to 25 deg by speed in NAV (forum quote of a
  Honeywell FMGS excerpt).
- FAA AC 120-5 (1963): turn altitude hold off in severe turbulence (search snippet).
- ICAO Doc 8168: 25 deg bank or 3 deg/s, whichever is less (search snippet of an AIP).
- Thrust-limited maximum altitude: where residual climb at max climb thrust falls to 100
  to 300 fpm (training slides and a forum post, per the 2026-11-08 review; not found in
  an Airbus manual).
- Holding-area design wind: an omnidirectional 2h + 47 kt at h thousand feet, after
  ICAO PANS-OPS Doc 8168 (PPRuNe posts quoting it; a 2003 post gives 3h + 47; the
  primary text was not reached). Used only as the top of the scenarios' wind range.
- Cost index: operators trade fuel against time; long-range cruise is the speed at 99 %
  of the best specific range, the curve being flat there (secondary accounts of Airbus,
  "Getting to grips with the cost index"; the document was not reached).
