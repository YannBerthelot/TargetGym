# Drum boiler: plant reality record

This record describes real drum boilers (utility and industrial) and how their
drum level is read, moved and protected. It comes only from outside sources,
listed at the end as [S1] to [S19] (no S18). Every source was fetched twice unless
marked otherwise. Numbers read off a figure are labelled "read from Fig. N".

## 1. What operators and the controller read

- **Drum level, differential pressure (DP) transmitter.** The high side taps
  steam pressure plus a reference leg; the low side senses drum pressure, the
  water column and the steam column above the level tap. Drum pressure cancels
  out, and an optional drum pressure signal corrects the level via gain and
  bias [S4, p. 3].
- **Pressure compensation.** Calibration holds at one drum pressure, so the
  level is corrected for pressure. The constant k "is determined empirically":
  with control in manual, the drum pressure setpoint is changed slightly and
  level is plotted against pressure [S16].
- **Filtering.** "filtering level measurements is generally counter-productive"
  because it adds lag to a process that has almost none [S16]. In the Lund
  plant tests the signals "were filtered and sampled at a rate of 0.1 Hz"
  [S1, printed p. 22].
- **Measurement noise.** At Abbott Power Plant (Champaign, IL) the measured
  outputs carry noise from "boiling effect on water surface, pressure waves"
  and from sensor faults such as pneumatic line leaks, vibration and electrical
  noise; the authors fit first-order coloured noise models to it [S3, p. 63].
  The paper's level output equation (31) carries a delay tau6 = 10 (listed with
  tau1..tau7 = 2, 2, 3, 3, 4, 10, 2) [S3, p. 62 to 63]. Level is reported in
  inches, pressure in psi, steam flow in kg/s [S3, eq. 16 to 19].
- **Gauge glass and remote indicators (ASME Section I, PG-60, as reported).**
  Up to 400 psi (26 bar) MAWP: one gauge glass in continuous service. Above
  400 psi: two glasses, or two independent remote level indicators on
  continuous display plus one glass that may be isolated but kept serviceable.
  If the operator in the control area cannot see a glass, two independent
  remote indicators must be continuously displayed [S6, slides "ASME Water Gage
  Code Minimum Requirements"; S7].
- **Gauge limits.** Tubular glass to 250 psi, reflex to 350 psi, flat
  transparent to 2,000 psi, ported bi-colour to 3,000 psi [S6, slide 2; S7].
  Magnetic gauges: indication only, no trip switches, up to 900 psi [S6; S7].
  The lowest edge of the indication range must be at least 2 inches above the
  lowest safe operating level [S7]. Isolation valves between water column and
  drum, where fitted, must be locked open [S7].
- **European practice (EN 12952-7).** At least one water gauge glass on all
  power boilers combined with 2 remote level indicators; the lowest
  permissible level on each glass is marked "LWL"; ball checks are mandatory
  [S6, slides "PED Requirements" and "EN Requirements"].
- **Photos.** Real gauge glasses and a drum with 2 glasses, 1 remote indicator
  and 2 DP transmitters [S6, slides "Code Compliant Installation"].
- **Steam flow and feedwater flow.** Three-element control reads drum level,
  steam flow and feedwater flow [S5; S12]. Steam flow is commonly measured in
  tonnes per hour and water in m3/h outside the US [S16]. Coriolis meters are
  recommended for range and signal-to-noise [S15].
- **Pressure measured for control.** Most operating plants base pressure
  control on throttle valve pressure (TSVP) or first stage turbine pressure
  [S2, sec. 2.3]. At Abbott the controlled pressure is the header pressure,
  with several boilers feeding one common header [S3, p. 58].
- Transmitter damping times used in the field: not found.

## 2. What they move, and in which mode

- **Feedwater valve.** Three-element control manipulates the feedwater control
  valve; the level controller output sets the feedwater flow controller's
  setpoint (cascade) [S4, p. 3; S5]. Single-element: "Level alone drives makeup
  water valve via a P+I loop"; two-element adds steam flow feedforward [S4, p. 3].
- **Feedwater pump speed** as the manipulated variable: not found.
- **Startup and low load.** At startup "both the feedwater and main steam flow
  rates remain below the measurable ranges", so single-element control is used
  [S5]. A GE patent states the change from single to three-element happens at
  about 20 to 30 percent of the flow range [S12, Background]. At very low loads
  "control instability can occur, requiring the control system to be switched
  to simple level feedback control" [S2, sec. 1].
- **Shrink/swell compensation.** A lead-lag relay in the feedforward
  ("shrink/swell relay"); its lag "will typically fall in the range of one
  minute to three minutes" [S14].
- **Auto and manual.** The controller in S4 includes a "hard manual" backup if
  digital control fails [S4, p. 3]. Runback tests: "elevate the drum level a
  few inches before your test" [S10].
- **Startup level.** Hitachi patent: the normal control level (NWL) "is
  lowered in advance of the start-up operation" so that the highest level from
  swell cannot exceed the maximum allowed [S19].
- **Firing rate and who holds pressure.** Drum units run in base,
  boiler-following, turbine-following or coordinated mode [S10]. In
  boiler-following "the boiler master is in automatic and the turbine is not";
  the boiler controls throttle pressure by modulating firing rate. In
  turbine-following the turbine master controls throttle pressure with the
  governor valves. In coordinated mode the boiler master looks mostly at
  pressure error and the turbine mostly at MW error (rules of thumb 10:1 and
  10:4) [S10]. Lu et al. simulate in boiler-follow mode with sliding pressure
  [S2, sec. 3.2].

## 3. Limits: trip, timed limit or cost

- **Low-low level, trip.** Training slides for an unnamed utility unit list the
  master fuel trip cause "Drum Level Low-low (-330 mm)", and boiler trips when
  the drum level is "low" or high "for more than ten seconds" (timed trips)
  [S9]. A 95 MW unit had its "low drum level trip ... set at 7.7 inches"
  [S10, Fig. 7 text].
- **High-high level, trip.** The extremely high level trip protects the steam
  turbine from water carry over; in multi-shaft combined cycle the steam
  turbine trips, in single-shaft the whole unit. The low level trip protects
  the boiler "from damage from high temperature", tripping the heat source
  [S12, Background].
- **Marine boiler alarm table** [S13]: low level, alarm only; low-low, alarm
  and burner stop, with "two independent level switches connected electrically
  in series"; high level, alarm only; high-high, alarm and feedwater pump stop;
  high steam pressure, alarm and burner stop.
- **UK shell boilers** [S8]: first low alarm shuts the burner but lets it
  re-fire when level recovers; second low alarm ("lockout") needs a manual
  reset. High level actions depend "on the individual plant requirements".
- **High-high value and delays, added 2026-11-08 by the fix session.** Hub
  Power Station's FSSS logic lists "Drum Level HH (+254 mm) (5 seconds)" and
  "Drum Level LL (-254 mm) (5 seconds)" as master fuel trip inputs [S20,
  slides 9, 11, 12]. A BHEL 500 MW boiler trips on "Drum level very low for
  more than 5 seconds" and "Drum level very high for more than 10 seconds"
  (unit protection slides, slide 27, SlideShare deck 29657473; fetched once).
  A Babcock and Wilcox patent says instantaneous level trips would cause
  excessive false trips, and that setting the delay "involves compromises"
  [S21].
- **EN limiters.** EN 12952-11 (water-tube) and EN 12953-9 (shell) set
  requirements for water level limiters; the text itself is paywalled (see
  owner list). Values: not found.
- **Alarm distances from normal water level** (other than the trips above):
  not found in a source I could open.
- **Pressure and safety valves.** High steam pressure stops the burner on the
  marine boiler [S13]. ASME safety valve set and accumulation limits: not
  confirmed from a primary text (see owner list).
- **Costs.** Too much water risks carryover into the steam header; too little
  threatens dry-out and tube damage [S4, p. 1]; deviations can cause "boiler
  tube overheating or water carryover into turbomachinery" [S15]. Poor control
  "can result in a costly plant trip" [S2, sec. 1].
- **Load ramp limits (cost and wear).** "Typical values used by the industry are
  1% or 2% per minute unit load rate of change"; the author tuned boilers up to
  5% per minute but nobody uses it because of wear [S10].

## 4. Time scales

- **Plant data, P16-G16 (Oresundsverket, Malmo; 160 MW, oil fired, natural
  circulation).** Open-loop tests "with the normal regulators removed",
  PRBS-like steps in fuel, feedwater and steam flow, at medium and high load,
  sampled at 0.1 Hz [S1, printed p. 22].
  - Medium load, steam valve steps (Fig. 12, printed p. 25): steam flow steps
    between about 52 and 67 kg/s; drum pressure moves 8.2 to 9.1 MPa in
    roughly linear ramps of about 0.6 MPa per 250 s; level stays within about
    plus or minus 0.12 m (read from Fig. 12).
  - High load, steam steps (Fig. 15, printed p. 27): steam about 100 to 112
    kg/s, pressure 10.2 to 10.9 MPa, level about minus 0.1 to plus 0.13 m
    (read from Fig. 15).
  - Swell shape (Fig. 16, printed p. 27): each steam step gives a fast level
    jump of about 0.05 to 0.07 m within one or two 10 s samples, the opposite
    of the slow mass trend that follows, which drifts about 0.1 m over 250 to
    300 s (read from Fig. 16). The authors: "the shrink and swell effect is
    larger at medium load" [S1, printed p. 26].
  - Feedwater steps of about 52 to 68 kg/s at medium load move level within
    about plus or minus 0.12 m (read from Fig. 11, printed p. 24); the right
    half plane zero "is approximately 4 times slower at medium load" (model
    step, Fig. 8) [S1, printed p. 20].
  - Fuel steps show an overshoot the authors call "rapid swell"
    [S1, printed p. 23, Fig. 10].
- **Stored energy as time constants** (Table 1, S1 printed p. 4), in J/W = s:
  P16-G16 at 160 MW: metal 320, water 333, steam 37, total 690 s; at 80 MW
  total 1444 s. Eraring (Australia) at 660 MW: metal 587, water 137, steam 35,
  total 759 s. Less stored water in the larger plant gives "larger variations
  in water level" [S1, printed p. 4].
- **Circulation loop.** Its time constant "is about a second" [S1, printed
  p. 10].
- **Abbott boiler no. 2** (22.1 kg/s steam at 2.24 MPa, 325 psi, oil and gas,
  cogeneration): PRBS test over about 5000 s moved drum pressure about 290 to
  340 psi and level about minus 3 to plus 3 inches (read from Figs. 3 and 5,
  S3 p. 63 to 64).
- **Load ramp rates.** Industry 1 to 2 %/min, up to 5 %/min possible [S10].
  Hard coal plants: most common 1.5 to 4 % of nominal per minute,
  state of the art 3 to 6 %/min; old South African units "usually do not
  exceed 1 percent per minute" [S17, Table 1 and p. 13]. S17 covers coal units
  in general, not only drum boilers. Industrial steam flow ramps "up to 20-25%
  per minute" [S15].
- **Start-up times** (S17 Table 1, p. 47, read 2026-11-08 by the fix session):
  hard coal, most commonly used, hot start 2.5 to 3 h, cold 5 to 10 h; state
  of the art hot 80 min to 2.5 h. Minimum load 25 to 40 %.
- **Pump restart.** Feedwater pumps "may take more than 30 seconds to start,
  come up to speed and pressure" [S11].
- **Shrink versus swell.** The level drop on a load decrease "is typically much
  smaller" and "of shorter duration" than swell [S14]. Saturated water volume
  "can expand by as much as 30%" during start-up [S14].
- **Simulation only (500 MW).** A 10 %/min load ramp gave "almost
  halved level deviation of 40mm at low load" versus a step [S2, sec. 5.2].

## 5. A normal day and a bad day

- **Normal day.** At Abbott, setpoints "change several times during every 24 h
  period", and unmeasured user steam demand is the main load disturbance
  [S3, p. 62]. Under steady operation drum level control "is usually not a
  problem"; frequent load changes make it one [S2, sec. 1]. Utility units
  change load at 1 to 2 %/min [S10]. Turndown can be 4:1 [S16].
- **Bad day, feed pump or fan runback.** On a 95 MW unit an ID fan trip
  caused a runback lasting "just over two minutes". In a feedwater pump
  runback test from 75% load, "The drum level dropped about 6.5 inches" against
  a trip at 7.7 inches, "a little too close for comfort" [S10, Figs. 5 to 7].
  If the standby pump is slow, the >30 s pump start time eats the drum
  inventory [S11]. All boiler feed pumps off is a boiler trip; if the level
  falls out of sight on feedwater failure, the operator trips fuel at once
  [S9].
- **Bad day, swell at low load.** "most trips occur due to high-level trip
  events", because fixed setpoints at low flow and pressure leave too little
  room for swell [S11]. Swell is larger at low load [S1, printed p. 20;
  S2, sec. 5.2].
- **Bad day, nuclear steam generators.** Up to 13% of all reactor trips in
  France in 1983 were attributed to steam generator control problems
  [S1, printed p. 2, citing its ref. 45].
- **Sootblowing, turbine trip and load rejection effects on level:** not found
  in a primary source.

## 6. The operator's job in one sentence

Keep drum water between the low trip that starves the tubes and the high trip
that floods the turbine while load, pressure and feedwater supply change,
using feedwater, the boiler and turbine masters, and manual control when the
automatics cannot cope [S4, p. 1; S10; S12].

## How often each window occurs

Added 2026-11-08 for the frequency-weighted headline. No source in this record
gives how often a unit runs back or trips a feed pump, or how its hours split
between full and part load; [S3] has setpoints "change several times during
every 24 h", and [S10] calls runback tests necessary after any change to the
gas pass, fans or mills. The weights are therefore ours (PHYSICS.md).

## For the owner to open by hand

- https://www.control.lth.se/fileadmin/control/Education/DoctorateProgram/PhysicalModeling/Lectures/Astrom-Bell.pdf
  (Astrom and Bell, Automatica 36, 2000; HTTP 404). The 1998 report S1 was used.
- https://core.ac.uk/download/83772844.pdf (HTTP 403).
- https://www.controleng.com/articles/optimizing-strategy-for-boiler-drum-level-control/ (HTTP 403).
- https://forbesmarshall.com/Knowledge/Energy-Conservation-Tips/Steam-Generation/Tip-Safety-Avoiding-low-water-level-condition-in-the-boiler (certificate error).
- https://docs.legis.wisconsin.gov/code/admin_code/sps/safety_and_buildings_and_environment/326_360/341_a/_267?down=1
  (ASME PG-67 excerpt on safety valves; DNS failure).
- Scribd copies not opened (search snippets give alarm and trip numbers):
  https://www.scribd.com/doc/204599123/Boiler-Interlock-and-Protection ,
  https://www.scribd.com/doc/94643170/03-ID-FD-Logic-System ,
  https://www.scribd.com/doc/262048419/00840-0100-4360 (Rosemount drum level note).
- Paywalled standards: NFPA 85 (drum level and master fuel trip clauses),
  EN 12952-11 and EN 12953-9 (water level limiters), ASME BPVC Section I
  PG-60 and PG-67 (read here only through S6 and S7).

## Sources

- [S1] K. J. Astrom, R. D. Bell, "Drum-Boiler Dynamics", Tech. Report
  TFRT-7577, Lund Institute of Technology, 1998, scanned PDF (Table 1, Figs.
  5 to 18, sec. 6). https://lup.lub.lu.se/search/files/4879107/8727216.pdf
  (second fetch by direct download; key pages re-read).
- [S2] C. X. Lu, N. W. Rees, S. C. Donaldson, "The use of the Astrom-Bell model
  for the design of drum level controllers in power plant boilers", IFAC 2005.
  https://skoge.folk.ntnu.no/prost/proceedings/ifac2005/Fullpapers/02665.pdf
- [S3] G. Pellegrinetti, J. Bentsman, "Nonlinear control oriented boiler
  modeling: a benchmark problem for controller design", IEEE TCST 4(1), 1996,
  pp. 57 to 64. https://www.dia.uned.es/~fmorilla/benchmark09_10/Pellegrinetti_Bentsman96.pdf
- [S4] Yokogawa, "Advanced Steam Drum Level Control in Boilers", application
  note AN-P-20250822-01.
  https://web-material3.yokogawa.com/2/4723/files/AN-P-20250822_Steam_Drum_Level_Control_Monitoring_AppNote.pdf
- [S5] Yokogawa, "Boiler Drum Level Control", application note web page.
  https://www.yokogawa.com/eu/library/resources/application-notes/boiler-drum-level-control
- [S6] J. Kolbus (Clark-Reliance), "Drum Level Instrumentation and Compliance
  with PED and ASME Requirements", slides, Oct. 2021.
  https://www.ccj-online.com/wp-content/uploads/2022/06/Drum-Level-Instrumentation-and-PED-ASME-Requirements-2021-Oct-21.pdf
- [S7] J. W. Kolbus, "Drum Level Instrumentation Update", POWER, 1 Feb. 2013.
  https://www.powermag.com/drum-level-instrumentation-update/
- [S8] Spirax Sarco, "Water level alarms", Learn about steam.
  https://www.spiraxsarco.com/learn-about-steam/the-boiler-house/water-level-alarms
- [S9] "Boiler and turbine emergencies", operator training slides, unnamed
  plant, SlideShare. https://www.slideshare.net/slideshow/boiler-turbine-emergencies/34808540
- [S10] T. Leopold, "Boiler-Tuning Basics, Part II", POWER, 1 May 2009.
  https://www.powermag.com/boiler-tuning-basics-part-ii/
- [S11] US 10323547 B2, "Steam drum level control system...", General
  Electric Co. https://patents.google.com/patent/US10323547B2/en
- [S12] US 7053341 B2, "Method and apparatus for drum level control for
  drum-type boilers", General Electric Co.
  https://patents.google.com/patent/US7053341B2/en
- [S13] L. Josefsson, "Marine Steam Boilers Alarm and Trip Functions".
  https://steamesteem.com/steam-boilers/boilers-alarm-trip.html
- [S14] A. D. Houtz, "Dynamic Shrink/Swell and Boiler Level Control", Control
  Guru. https://controlguru.com/dynamic-shrinkswell-and-boiler-level-control/
- [S15] J. Cahill, G. McMillan (Emerson), "Why Boiler Drum Level Control Is So
  Challenging", 3 Sept. 2025. https://www.emersonautomationexperts.com/?p=62109
- [S16] M. King, "Practical Process Control Part 11: Controlling Steam Drum
  Level", The Chemical Engineer, 29 Feb. 2024.
  https://www.thechemicalengineer.com/features/practical-process-control-part-11-controlling-steam-drum-level
- [S17] Agora Energiewende, "Flexibility in thermal power plants", 2017 (Table
  1; Fig. S3). https://www.agora-energiewende.org/fileadmin/Projekte/2017/Flexibility_in_thermal_plants/115_flexibility-report-WEB.pdf
- [S20] "FSSS logic presentation", Hub Power Station (CFB boiler), SlideShare.
  https://www.slideshare.net/MuhammadAsif771197/fssslogic-presentation-modifiedv11pptx
  (fetched once, 2026-11-08).
- [S21] US 4433646 A, "Boiler water trip system", Babcock and Wilcox Co., 1984.
  https://patents.google.com/patent/US4433646A/en (fetched once, 2026-11-08).
- [S22] Elexon, "Elexon insights: system prices 2018": "in 59% of Settlement
  Periods the System Price was between £30/MWh and £60/MWh"; average System
  Price £41.39/MWh when long, £78.84/MWh when short.
  https://www.elexon.co.uk/bsc/data/system-prices-analysis-report/elexon-insights-system-prices-2018/
  (fetched once, 2026-11-08).
- [S22b] Swansea University paper (cronfa 68151, 2024), citing Elexon: average
  annual imbalance price £44.61/MWh in 2017 to 2020 and £156.20/MWh in 2021 to
  2022. https://cronfa.swan.ac.uk/Record/cronfa68151 (read from a search
  summary only, not fetched).
- [S23] FOSSEE, Engineering Thermodynamics, chapter 12 (vapour power cycles),
  examples 12.4 to 12.6: ideal-cycle steam rates of 2.18, 3.85 and 4.51 kg/kWh.
  https://tbc-python.fossee.in/convert-notebook/Engineering_Thermodynamics/ch12.ipynb
  (fetched once, 2026-11-08). Textbook cycles, not plant data.
- [S24] N. Kumar, P. Besuner, S. Lefton, D. Agan, D. Hilleman (Intertek
  APTECH), "Power Plant Cycling Costs", NREL/SR-5500-55433, April 2012, Table
  1-1 (pp. 12 to 13) and p. 11: lower-bound C&M cost per significant load
  follow (more than 15 to 20 % of capacity), median $1.92/MW for gas-fired
  steam units and $3.34/MW for small subcritical coal (2011 $); faster-ramp
  multipliers preliminary (footnote 5).
  https://docs.nlr.gov/docs/fy12osti/55433.pdf (fetched once, 2026-11-09).
- [S19] US 4353204 A, "Method of water level control for a combined cycle
  power plant", Hitachi Ltd., 1982. https://patents.google.com/patent/US4353204A/en

Confirmation: S2, S3, S4, S6, S17 re-downloaded byte-identical; S1 re-downloaded and its cited pages re-read by OCR; all others re-fetched and matched.
