# Wind turbine: reality record

Plant: a utility-scale, variable-speed, pitch-regulated, three-bladed upwind horizontal axis wind
turbine, with the focus on control above rated wind speed (collective pitch holds rated rotor speed
and power) and on tower motion. Every item names its source; tags such as [S1] point to the
Sources list at the end. Where no source we could read says something, the item says "not found".
Page numbers are the printed page numbers of each document unless marked "PDF page".

## 1. What the controller and the remote operators read

- **Generator speed.** The NREL 5 MW baseline pitch and torque controllers use measured generator
  speed as their only feedback input [S1, Sec. 7.1, p. 17]. It is filtered by a recursive
  single-pole low-pass filter, corner frequency 0.25 Hz, set to about one quarter of the blade's
  first edgewise frequency (about 1.08 Hz, Table 9-1), shown for a 0.0125 s step in the Bode plot
  of Figure 7-1 [S1, p. 18; Table 7-2, p. 27]. ROSCO keeps the same rule (first or second order
  low-pass at one quarter of the first edgewise frequency) [S2, Sect. 2.1].
- **Speed and position sensors on a real research turbine.** The NREL CART (600 kW, two-bladed)
  has absolute position encoders on pitch, yaw, teeter, low-speed and high-speed shaft; all data
  are sampled at 100 Hz and the controller runs at 100 Hz [S3, Sect. II, PDF page 5]. A GE overspeed
  patent lists the rotor speed sensors in use on the main shaft: pendulum or gravity sensor,
  accelerometer, dynamo, optical detector [S4, FIG. 3 description].
- **Nacelle accelerometer.** On CART, accelerometers in the nacelle measure tower fore-aft and
  side-side motion [S3, PDF page 5]. On the Areva M5000 (5 MW), tower-top acceleration is measured and
  fed back to collective pitch; the authors note acceleration sensors and collective pitch are
  "already there" on most turbines [S5, Sect. 3, p. 7].
- **Nacelle anemometer and vane.** Each turbine traditionally carries an anemometer and vane on
  the nacelle, behind the hub, used for yaw control and cut-in/cut-out [S6, PDF page 3].
  Known biases: the rotor and nacelle distort the flow, so a nacelle transfer function (IEC
  61400-12-2) is needed [S6, PDF page 3]. On a GE 1.5sle, the nacelle anemometer underestimates the
  upwind speed more during high turbulence and the transfer function changes with stability
  [S6, abstract]. Earlier field work found a linear correction overestimates at 6 to 11 m/s and
  underestimates below 4 m/s and above 16 m/s (Hunter et al. 2001, as cited) [S6, PDF page 4]. After an
  empirical transfer function, the residual fit error was 0.36 m/s RMSE [S6, PDF page 10].
  ROSCO does not rely on the anemometer: it estimates rotor-effective wind speed with a Kalman
  filter, RMSE 0.48 m/s in a 10 min simulation at 11 m/s [S2, Sect. 5.1, Fig. 6].
- **Pitch angle and power.** Pitch from absolute encoders [S3, PDF page 5]; previous pitch command is
  used for gain scheduling and to switch the torque law [S1, p. 20 and p. 25]. Power sensor
  location, sample rate and filtering on a commercial turbine: not found.
- **Loads.** CART carries blade root flap and edge strain gauges, tower bending gauges and shaft
  torque transducers [S3, PDF page 5]; the M5000 field test added strain gauges at the tower top [S5,
  Sect. 2.3, p. 6].
- **Remote operators (SCADA).** Industry practice is 10 minute aggregated signals; one operating
  wind farm also stored 4 s data [S7, Sect. 3.1, PDF page 6]. The IEC 61400-25-2 information model
  names logical nodes WTUR (turbine), WROT (rotor), WTRM (transmission), WGEN (generator), WCNV
  (converter), WNAC (nacelle), WMET (met), WALM (alarms), WAPC (plant active power control), and an
  annex for statistical data [S8, contents pages]. Photos or videos of real SCADA screens: not
  found.
- **Measurement delay of commercial turbine sensors.** Not found.

## 2. What they move, and in which mode

- **Actuators.** Collective blade pitch and generator torque [S1, Sec. 7.2 to 7.3]. CART pitches
  each blade with its own electromechanical servo; its converter can command torque from minus to
  plus rating at any speed [S3, PDF page 5].
- **Regions (NREL 5 MW)** [S1, Sec. 7.2, p. 19 and Table 7-2]:
  Region 1: below cut-in, zero torque, wind accelerates the rotor for start-up.
  Region 1.5: linear torque ramp from 670 to 871 rpm generator speed (minimum rotor speed 6.9 rpm,
  as on the REpower 5M).
  Region 2: torque = 0.0255764 N m/rpm^2 times speed squared, tracking peak Cp 0.482 at tip speed
  ratio 7.55 and 0 deg pitch.
  Region 2.5: linear transition with 10 % slip, to limit tip speed (noise), up to 1161.963 rpm
  (99 % of rated).
  Region 3: constant power (torque inversely proportional to filtered speed); gain-scheduled PI
  pitch on the error to rated generator speed 1173.7 rpm [S1, p. 20]. The torque law switches to
  Region 3 whenever the previous pitch command is 1 deg or more [S1, p. 20].
  ROSCO offers constant torque or constant power in Region 3, notes constant power gives more power
  fluctuation, and uses constant torque for the IEA 15 MW [S2, Sect. 3.2, Eq. 21].
- **Near rated.** ROSCO adds set-point smoothing between torque and pitch loops and peak shaving
  (a wind-speed-scheduled minimum pitch); the default toolbox routine cuts maximum rotor thrust by
  25 % [S2, Sects. 5.2 and 5.3]; against the NREL 5 MW controller, over 10 % lower peak thrust and
  1.7 % lower AEP [S2, abstract and Sect. 6.1].
- **Tower damping.** Fore-aft: tower-top acceleration to collective pitch through a band-pass with a
  3p notch, scheduled for 4 to 25 m/s [S5, Sect. 3.1, Eq. 6 and 7]. CART designs damp fore-aft with
  pitch and side-side with generator torque [S3, abstract and Sect. I].
- **Start-up and shutdown.** ROSCO start-up: blades parked at 90 deg, free-wheel, then rating raised
  in torque steps (demo over 400 s) [S9, Ex. 29 and 32]. ROSCO shutdown triggers: filtered pitch
  above a threshold (demo 30 deg), yaw error (demo 25 deg), generator speed (demo threshold 8.5
  rpm), or a set time; blades go to feather in staged pitch and torque rates [S9, Ex. 30; S2,
  Sect. 5.4]. Commercial start/stop sequences: not found.
- **Curtailment and wind farm set points.** EU rules require an active power set point the module
  follows within a TSO-set time and tolerance, an input port to reduce output, and a logic port to
  cease output within 5 s [S10, Art. 13(6), 14(2)(a), 15(2)(a)]. CART3 field tests followed a
  de-rate step, then primary frequency response to an under-frequency event (Fig. ES-8), and an AGC
  signal built from real ACE data of a Western Interconnection balancing area (Fig. ES-9) [S11,
  p. xx to xxi]. ROSCO takes a power rating R as speed, torque and minimum pitch references [S9,
  Ex. 29].
- **Storm control.** ROSCO soft cut-out demo ramps rating from 1.0 at 20 m/s to 0.5 at 30 m/s and
  0 at 40 m/s, past the normal 25 m/s cut-out [S9, Ex. 29]. Manufacturer storm-control numbers
  (for example Siemens Gamesa High Wind Ride Through): not found (see blocked list).

## 3. Limits, and whether each is a trip, a timed limit or a cost

| Limit | Value | Kind | Source |
|---|---|---|---|
| Pitch rate | 8 deg/s absolute (NREL 5 MW, guessed from GE long-blade tests) | hard saturation | [S1, p. 26, Table 7-2] |
| Pitch rate on CART | up to 18 deg/s; acceleration up to 150 (printed as "degrees per second") | actuator capability | [S3, PDF page 5] |
| Pitch angle | 0 deg (Region 2 optimum) to 90 deg (feather) | hard saturation | [S1, p. 26] |
| Generator torque | 47,402.91 N m, 10 % above rated 43,093.55 N m | hard saturation | [S1, p. 20, Table 7-2] |
| Torque rate | 15,000 N m/s | hard saturation | [S1, p. 20] |
| Rated power | 5 MW electrical, 5.297 MW mechanical, generator efficiency 94.4 % | regulated set point | [S1, p. 19] |
| Rotor speed | 6.9 to 12.1 rpm, rated tip speed 80 m/s | operating range | [S1, Table 1-1, p. 2] |
| Cut-out wind | 25 m/s (cut-in 3, rated 11.4 m/s) | shutdown | [S1, Table 1-1] |
| Overspeed | above a stored "permissible" speed the pitch system feathers on battery power, independent of the main controller | trip (safety system) | [S4] |
| Overspeed trip level in % of rated | not found (GL and IEC values are behind paywalls) | | |
| Tower fatigue | bending moments carry over 99 % of tower bottom fatigue damage | cost (DEL) | [S5, Sect. 1, p. 3] |
| Grid frequency | 49.0 to 51.0 Hz unlimited operation; 51.0 to 51.5 Hz at least 30 min (Continental Europe) | timed limit | [S10, Art. 13(1), Table 2] |
| High-frequency trip | 51.5 Hz (Northern Ireland setting, 2006) | trip | [S12, Sect. 3] |
| Ramp rate | 1 to 10 MW/min controller range; set to max 5 MW/min averaged over 10 min, 1 min average at most 3 times that; no up-ramp at 50.2 Hz or above; at most one start-up per 10 min | timed limit | [S12, Sects. 1 and 2] |

## 4. Time scales

- **Controller step.** NREL 5 MW simulations use 0.0125 s (80 Hz) [S1, Sec. 8, p. 29]; CART runs at
  100 Hz [S3, PDF page 5]. Commercial controller rates: not found.
- **Speed loop.** Pitch PI designed for a rotor-speed response of 0.6 rad/s (about 0.1 Hz),
  damping 0.6 to 0.7 (0.7 used), KP(0) = 0.01882681 s, KI(0) = 0.008068634; gains fall with
  pitch, power sensitivity doubles at 6.302336 deg [S1, pp. 22 to 25, Fig. 7-4, Table 7-2].
  ROSCO's floating example uses 0.2 rad/s bandwidth [S2, Sect. 6.2].
- **Tower.** NREL 5 MW first tower fore-aft 0.3240 Hz, side-side 0.3120 Hz, drivetrain torsion
  0.6205 Hz, blade collective flap 0.6993 Hz, second tower fore-aft 2.9003 Hz (FAST) [S1, Table
  9-1, p. 30]. The M5000 spectra mark tower, 1p and 3p frequencies but their values were removed
  for confidentiality [S5, p. 3].
- **Actuators.** CART torque loop bandwidth 500 rad/s [S3, PDF page 5]. NREL 5 MW ADAMS pitch actuator 30
  Hz, 2 % damping, chosen to be negligible, not as a real actuator [S1, Sec. 7.4, p. 26]. Measured
  pitch actuator bandwidth of a commercial turbine: not found.
- **Wind.** IEC turbulence: sigma1 = Iref (0.75 Vhub + 5.6); length scale 42 m above 60 m hub height;
  extreme operating gust lasts 10.5 s; direction change 6 s; coherent gust 10 s; shear 12 s [S13,
  quoting IEC 61400-1 clause 6].
- **Statistics.** SCADA 10 minute aggregates are standard [S7]; the nacelle transfer function study
  used 1 Hz met data averaged to 10 min [S6, PDF page 5]. Simulated load cases use 6 seeds per wind speed,
  3 to 25 m/s in 2 m/s steps [S2, Table 2].
- **Grid.** EU frequency response: initial delay over 2 s must be justified, full activation within
  30 s, sustained 15 to 30 min [S10, Art. 13(2)(e), 15(2)(d), Table 5]. Pitch balancing on the M5000
  reached steady state within about 60 s in the field [S5, Sect. 2.3].

## 5. A normal day and a bad day

- **Normal day.** Above rated, pitch holds rated generator speed and torque holds rated power [S1,
  Sec. 7]. The M5000 field tests were run "in partial load" (the paper's words); the field pitch
  trace (Fig. 4, axis 11.5 to 15 deg) shows the three blades converging within about 60 s once
  pitch balancing starts [S5, pp. 4 and 6]. Measured above-rated rotor speed and pitch traces from a
  utility turbine: not found. With active tower damping on, tower-top acceleration around
  the tower frequency was "fully suppressed" in the measured spectra while 3p remained (Figs. 7
  and 8), with pitch modulation usually below 0.5 deg [S5, Sect. 3.3 and 4]. Simulated DLC 1.2 gain:
  2.1 % lower tower bottom DEL for 1.26 % more blade root DEL [S5, Sect. 3.2]. On CART3, a de-rated
  turbine tracked AGC with high-frequency power fluctuations the authors expect to smooth out at
  plant level [S11, p. xxi]; continuous AGC at 10 % de-rate had very little effect on DELs in
  simulation [S11, p. xix].
- **Turbulent wind and extreme gust.** IEC load cases DLC 1.1 (normal turbulence) and 1.3 (extreme
  turbulence) span 3 to 25 m/s [S2, Table 2]; the extreme operating gust magnitude is the smaller of
  1.35 (Ve1 - Vhub) and 3.3 sigma1 / (1 + 0.1 D / 42 m), with Ve1 = 0.8 x 1.4 Vref [S13]. Measured
  rotor speed overshoot in a gust on a real utility turbine: not found.
- **Grid loss.** Simulation only: an emergency stop after grid loss is about 3 times as damaging to
  drivetrain bearings as a normal stop [S14, abstract]. A plant that becomes islanded must detect it
  and shut down at once (Northern Ireland) [S12, Sect. 3]. Measured grid-loss traces: not found.
- **Overspeed event, real accident.** Bouin, France: three days before the collapse, in wind above
  40 m/s, the pitch system failed and the turbine stopped; it was diagnosed remotely because no one
  could climb; a data misreading led a technician to move the turbine to a position that drove the
  rotor past its overspeed limit; the overspeed protection acted but worn pitch-system brake blocks
  prevented braking; the 60 m tower took loads beyond design and broke [S15]. The operator stopped
  the seven other turbines and replaced their brake blocks [S15].
- **Frequency events.** Above 50.2 Hz the Northern Ireland setting blocks up-ramps; droop 4 % with a
  49.75 to 50.25 Hz deadband (normal running) [S12, Sects. 2 and 3]. EU droop 2 to 12 %, deadband 0
  to 500 mHz, response range 1.5 to 10 % of capacity [S10, Tables 4 and 5]. US ERCOT requires
  frequency response only from plants already curtailed with headroom [S11, p. 10].

## 6. The operator's job in one sentence

- **Turbine controller:** below rated, track the best tip speed ratio with torque; above rated, pitch
  to hold rated speed and power without exceeding pitch, torque and speed limits, while damping
  tower and drivetrain motion [S1, Sec. 7; S5].
- **Human remote operator:** watch 10 minute SCADA data and alarms, follow the grid operator's set
  points, ramp limits and stop signals, and decide on remote resets and stops, which is the step
  that went wrong at Bouin [S7; S10; S12; S15]. A source stating this job in one sentence: not found.

## For the owner to open by hand

- IEC 61400-1 (turbulence classes, DLC table, overspeed cases): https://webstore.iec.ch (paywalled;
  values here come from S13, which quotes it).
- IEC 61400-25-2 full text: https://webstore.iec.ch (only the free contents sample S8 was read).
- IEC 61400-12-2 nacelle anemometry: https://www.dinmedia.de/en/standard/din-en-61400-12-2/195226576
  (paywalled).
- Bossanyi, Wright, Fleming, Controller field tests on the NREL CART2 turbine, NREL/TP-5000-49085:
  https://core.ac.uk/download/71274298.pdf (HTTP 403).
- Bossanyi, Fleming, Wright, Validation of individual pitch control by field tests, IEEE TCST 21(4),
  2013: https://www.osti.gov/biblio/1087779 (connection timed out).
- Field Test Results with IPC on the NREL CART3, AIAA 2012-1019:
  https://research-hub.nrel.gov/en/publications/field-test-results-with-individual-pitch-control-on-the-nrel-cart-2/
  (abstract only).
- Hansen and Henriksen, Basic DTU Wind Energy controller, E-0028 (overspeed and storm control):
  https://core.ac.uk/download/13801994.pdf and gitlab.windenergy.dtu.dk (bot wall).
- Siemens Gamesa High Wind Ride Through sheet:
  https://www.siemensgamesa.com/-/media/siemensgamesa/downloads/en/products-and-services/services/asset-optimization/high-wind-ride-through-feature.pdf
  (returned no content).
- GL Guideline 2010 (speed limits n1, n3, n4, nA): not freely available.
- www.nrel.gov did not resolve; S1 and S11 were read from mirrors (docs.nlr.gov, www1.eere.energy.gov).

## Sources

All were fetched twice and the cited phrases found both times, except where marked.

- [S1] Jonkman, Butterfield, Musial, Scott, Definition of a 5-MW Reference Wind Turbine for Offshore
  System Development, NREL/TP-500-38060, Feb. 2009. https://docs.nlr.gov/docs/fy09osti/38060.pdf
  (original https://www.nrel.gov/docs/fy09osti/38060.pdf). Sec. 7, Tables 1-1, 7-2, 9-1, Figs. 7-1,
  7-2, 7-4.
- [S2] Abbas et al., A reference open-source controller for fixed and floating offshore wind
  turbines, Wind Energ. Sci. 7, 53 to 73, 2022. https://wes.copernicus.org/articles/7/53/2022/
- [S3] Wright and Stol, Designing and Testing Controls to Mitigate Dynamic Loads in the Controls
  Advanced Research Turbine, NREL/CP-500-42490, 2008.
  https://mapcruzin.com/wind-power-publications/research-development/42490.pdf
- [S4] US 2011/0142634 A1, Overspeed protection system and method (GE).
  https://patents.google.com/patent/US20110142634A1/en
- [S5] Duckwitz and Shan, Active tower damping and pitch balancing: design, simulation and field
  test, J. Phys. Conf. Ser. 555, 012030, 2014 (Areva M5000 field test).
  https://iopscience.iop.org/article/10.1088/1742-6596/555/1/012030/pdf
- [S6] St. Martin et al., Atmospheric turbulence affects wind turbine nacelle transfer functions,
  Wind Energ. Sci. Discuss. wes-2016-45, manuscript version 3.
  https://wes.copernicus.org/preprints/wes-2016-45/wes-2016-45-manuscript-version3.pdf
- [S7] Gonzalez, Stephen, Infield, Melero, On the use of high-frequency SCADA data for improved wind
  turbine performance monitoring, J. Phys. Conf. Ser. 926, 012009, 2017.
  https://strathprints.strath.ac.uk/62619/1/Gonzalez_etal_JPCS_2017_high_frequency_SCADA_data_for_improved_wind_turbine_performance_monitoring.pdf
- [S8] IEC 61400-25-2:2015 free sample (contents only).
  https://cdn.standards.iteh.ai/samples/19182/0d3a7b85bb594badafc87a3b5510d32b/IEC-61400-25-2-2015.pdf
- [S9] ROSCO documentation, Examples 29 (power control), 30 (shutdown), 32 (startup).
  https://rosco.readthedocs.io/en/latest/source/examples.html
- [S10] Commission Regulation (EU) 2016/631, requirements for grid connection of generators (RfG),
  Arts. 13 to 15, Tables 2, 4, 5. https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0631
- [S11] Ela et al., Active Power Controls from Wind Power: Bridging the Gaps, NREL/TP-5D00-60574,
  Jan. 2014, executive summary Figs. ES-7 to ES-9 and Sect. 1. https://www1.eere.energy.gov/wind/pdfs/60574.pdf
- [S12] SONI, Wind Farm Power Station Settings Schedule, dated 15.03.06 (Northern Ireland Grid Code).
  https://cms.soni.ltd.uk/sites/default/files/media/documents/Archive/WIND%20FARM%20POWER%20STATION%20SETTING%20SCHEDULE.pdf
- [S13] pyiecwind theory page, quoting IEC 61400-1 clause 6 (secondary; edition-dependent numbering
  not given). https://pyiecwind.readthedocs.io/en/stable/theory.html
- [S14] Scott et al., Effects of extreme and transient loads on wind turbine drive trains, AIAA 2012
  (abstract only, simulation). https://strathprints.strath.ac.uk/48223/ (not re-confirmed: one fetch).
- [S15] French ARIA database, incident 50913, wind turbine collapse, Bouin (Vendee).
  https://www.aria.developpement-durable.gouv.fr/fiche_detaillee/50913_en/?lang=en
