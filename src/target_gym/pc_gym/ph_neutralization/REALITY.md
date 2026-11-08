# Plant reality record: ph_neutralization

The real plant is a continuous stirred-tank pH neutralization: a strong acid
neutralized by a strong base with a carbonate buffer, as in the Hall and
Seborg, Henson and Seborg bench rigs, and the industrial effluent
neutralization units they stand for. Every statement below comes from a
cited source. "Not found" means no source we could read says it. This
record was written without reading the task's code or docs.

## Sources

Each source was fetched twice: once to draft, once to check. Figures were
read as page images where the source is a PDF.

- [P1] R.C. Hall, D.E. Seborg, "Modelling and self-tuning control of a
  multivariable pH neutralization process, Part I", American Control
  Conference, Pittsburgh, 1989, pp. 1822-1827. Abstract record only:
  https://manuscript.isc.ac/Inventory/49/487932.htm Checked twice.
- [P2] M.A. Henson, D.E. Seborg, "Adaptive nonlinear control of a pH
  neutralization process", IEEE TCST 2(3), 1994, pp. 169-182,
  https://doi.org/10.1109/87.317975. Abstract record only:
  https://manuscript.isc.ac/Inventory/49/1168872.htm Checked twice.
- [P3] G.K. McMillan (Emerson), M.S. Sowell (Solutia), "Advances in pH
  modeling and control", ISA 54th International Instrumentation Symposium,
  2008. Field case of a plant waste treatment pH system.
  https://www.emersonautomationexperts.com/papers/AdvancespHModelingControl.pdf
  Read: pp. 3 to 10, Figure 2 (p. 5). Checked twice.
- [P4] Hach Company, "pH Control Systems", application note LIT2862, 2015.
  https://cdn.hach.com/7FYZVWYB/at/9z8zqtksnx6n7634rvhhsxh/LIT2862_pH_Control_Systems.pdf
  Read: all 4 pages, Figures 1 to 4. Checked twice.
- [P5] South Fork Instruments / REFEX Sensors, "pH Neutralization in
  Industrial Wastewater", application note, 2017.
  https://www.southforkinst.com/wp-content/uploads/pH-Neutralization-in-Industrial-Wastewater-1.pdf
  Read: all 4 pages, system figure on p. 1. Checked twice.
- [P6] Emerson, "pH measurement in continuous neutralization of industrial
  wastewater", application web page (Rosemount 396P).
  https://www.emerson.com/en/measurement-instrumentation/industries/water-and-wastewater/ph-measurement-in-continuous-neutralization-of-industrial-wastewater
  Checked twice.
- [P7] A. Faanes, S. Skogestad, "pH-neutralization: integrated process and
  control design", IFAC ADCHEM, Pisa, 2000.
  https://folk.ntnu.no/skoge/publications/2000/pH_pisa_adchem2000/pH_pisa_adchem2000.pdf
  Read: pp. 1 to 3, Figure 1, Table 1. Checked twice.
- [P8] US 40 CFR 401.17, "pH effluent limitations under continuous
  monitoring". https://www.law.cornell.edu/cfr/text/40/401.17 Checked twice.
- [P9] US 40 CFR 403.5(b)(2), pretreatment prohibited discharges.
  https://www.law.cornell.edu/cfr/text/40/403.5 Checked twice.
- [P10] Yokogawa, "Industrial pH/ORP electrodes" product page, batch
  neutralization application note summary.
  https://yokogawa.com/solutions/products-and-services/measurement/analyzers/liquid-analyzers/ph-sensors/industrial-ph-orp-electrodes
  Checked twice.

## The plant

Lab rig: Hall and Seborg built a demonstration unit of two stirred tank
reactors, each running a strong acid and strong base neutralization, with
pH and liquid level controlled in each tank [P1]. Henson and Seborg ran a
bench-scale pH neutralization system [P2]. Rig volumes, flows and reagent
concentrations: not found in an open source (see the owner list).

Industrial unit: the waste stream enters a stirred tank reactor, acid or base
is added continuously to hold a target pH, and treated waste leaves the
reactor [P6]. A simple system has pH instruments, an effluent holding tank,
reagent storage tanks with addition pumps, and agitators [P5, p. 1]. Plants
run batch or continuous; continuous plants flow through a series of tanks or
basins [P5, p. 2]. Strong acid or base neutralization is usually done in
several steps [P7, p. 1]. One field plant used two in-line stages, each a
static mixer, then an unagitated attenuation tank [P3, pp. 4-6, Figure 2].
Size example: for a flow of 0.01 m3/s, a design analysis finds a total
volume of 250,000 m3 with one tank, 316 m3 with two, 40.7 m3 with three,
9.51 m3 with four [P7, Table 1, p. 3]. Typical real tank volumes: not found.

## What operators read

- pH in the tank: probe on a dip tube in the holding tank [P5, figure p. 1;
  p. 3]. Emerson: a coating-resistant sensor installed in the stirred tank
  reactor [P6]. For closed tanks, through top or side ports [P5, p. 3].
- The probe must not sit where chemical is added, because mixing and
  reaction take time [P5, p. 3, item 3].
- pH at discharge: a dedicated sensor at the point of discharge to record
  effluent pH for inspection [P5, p. 2]; shown in a retractable housing at
  the outflow [P5, figure p. 1].
- In-line: inject at the static mixer inlet and place the sensor at the mixer
  outlet [P4, p. 3, Figure 4].
- Field case: three pH transmitters with middle signal selection at each
  mixer outlet and on the tank, and flow transmitters on waste and reagent
  [P3, pp. 4-5, Figure 2].
- Level sensing to signal tank full or empty in batch systems [P4, p. 1].
- Electrode response time or time constant: not found. Sampling period of the
  pH signal: not found.

## What they move, and in which operating mode

- Reagent pumps or solenoid valves switched on and off by a relay controller
  in batch systems [P4, System A, p. 1]; with deadband or latching for
  continuous input [P4, System B, p. 1].
- Two final elements with different setpoints, for example 1 GPM of caustic
  below pH 3 and 0.1 GPM below pH 4 [P4, p. 2].
- Time-proportioning: a cycle timer of a few seconds to a few minutes gives
  "chemical shots" [P4, System C, p. 2]. Yokogawa: pH measurement with a
  timer-controlled chemical feed for batch neutralization [P10].
- Continuous: proportional or two-mode (PI) controller with 4-20 mA output to
  an I/P and a pneumatic valve, or an electric pump or valve [P4, Systems D
  and E, pp. 3-4, Figure 4].
- On/off valves preferred for abrasive or clogging reagent such as lime
  slurry [P4, System F, p. 4].
- Field case: each stage had multiple reagent valves behind a splitter so one
  plugged valve does not disable the loop; pH loops in cascade (RCAS) under
  a fuzzy logic set point optimizer; a "kicker" stepped open the second stage
  valves on low pH [P3, pp. 4-5, Figure 2]. Split range points can cause
  limit cycles [P3, p. 4]. High resolution valves respond to signal changes
  under 0.2 % [P3, p. 4]. Feedforward is named as a loop enhancement [P3,
  p. 3].
- Manual override of dosing is common where pH measurement is poor [P5, p. 2].
- Valve sizes in the lab rig: not found.

## Limits

- Discharge pH range under continuous monitoring: total time outside the
  permitted range at most 7 h 26 min per calendar month, and no single
  excursion over 60 minutes [P8]. This is a timed limit. The range itself is
  set by the applicable guideline, not by this section [P8].
- Discharge to a public sewer works: no discharge with pH lower than 5.0,
  unless the works is designed for it [P9]. This is a hard limit.
- Field case: any tank discharge below the low environmental pH limit, however
  brief, could be a recordable violation; effluent was diverted if tank pH got
  too low [P3, pp. 4-5]. This acts as an interlock.
- Batch systems: lock out the mixer and pH control when the level is wrong
  [P4, p. 1]. Tank outlet sized to make overflow unlikely [P4, p. 2].
- Cost: reagent overdosing is costly [P5, p. 2]; reagent cost is cut by set
  points closer to the pH limits [P3, p. 4]. Operating valves too near the
  seat raises stick-slip, plugging and trim erosion [P3, p. 5].
- Reagent pump saturation limits: not found.

## Time scales

- Retention time (volume over inflow) above 10 minutes for a continuous-input
  tank under on/off control [P4, p. 2].
- Static mixer residence 1 to 6 s, mostly dead time [P3, p. 3].
- Mixing delay about 7 s, nearly independent of tank size, after Walsh
  (1993); the design analysis assumes 10 s effective delay per tank [P7,
  p. 2].
- In-line control: delay between addition and sensing should not exceed a few
  seconds [P4, p. 3]. Long tank, trough or tank series: at least one minute
  [P4, p. 2]. Two-mode controller with sample/hold: up to eight minutes [P4,
  p. 4].
- Field case: the second stage loop took almost a minute to open reagent
  valves to 50 %; a kicker ramp of about 1 % per second was proposed [P3,
  p. 10].
- Electrode response, valve stroke, sampling: not found. Lab rig residence
  time: not found.

## A normal day and a bad day

Normal day: waste flows in, the controller compares pH with its set point and
runs chemical pumps to correct it, agitators keep the probe reading a
representative sample [P5, p. 1]. Field case: shift metrics of reagent to
influent flow ratio track performance [P3, p. 9].

Disturbances and failures:
- Large unknown changes in strong acid concentration of the influent from
  production units [P3, p. 4]; loops went from calm to "rocking and
  reeling" during one lunch break [P3, p. 8].
- Buffering changes: unmeasured buffering changes on the bench rig [P2];
  unpredictable buffer changes shift the titration curve [P1].
- Coating of pH glass and reference junction by undissolved material [P6].
  Oily or solid build-up, bulb breakage by abrasives, junction plugging,
  reference poisoning by heavy metals, sulfides, proteins [P5, pp. 2-3].
- Junction blockage gives sluggish response and leads to reagent overdosing
  [P5, p. 2]. Electrolyte dilution gives drift [P5, p. 2].
- Electrode failure of any type: covered by middle-of-three selection; valve
  plugging: covered by multiple valves [P3, pp. 4-5].
- Kicker over-active and overdosing until its pH filter was increased [P3,
  p. 8].
- Oversized final element gives overshoot [P4, p. 1]; long delay makes pH
  cycle through the set point [P4, p. 3].
- Maintenance: cleaning by soaking 5 minutes in 2 to 5 % HCl [P5, p. 4];
  automated spray cleaning for heavy coating [P5, p. 3].
- Lab rig experiments: step responses over a wide range of conditions and
  multiloop PID tests [P1]; comparison of adaptive nonlinear, nonlinear and
  PI controllers [P2].

## The operator's job in one sentence

Keep the effluent pH at its set point and inside the discharge permit by
adding acid or caustic [P5, p. 1], without overdosing reagent [P5, p. 2].

## Proposed behaviour tests source list

| Source and figure | Quantity | Value as printed |
|---|---|---|
| [P8] 40 CFR 401.17 | Allowed excursion time | at most 7 h 26 min per month, at most 60 min per excursion |
| [P9] 40 CFR 403.5(b)(2) | Hard floor to sewer | pH not lower than 5.0 |
| [P4] p. 2 | Tank retention time for on/off control | above 10 minutes |
| [P4] pp. 2-4 | Dead time classes | a few s (in-line), at least 1 min (trough, tank series), up to 8 min (two-mode with sample/hold) |
| [P3] p. 3 | Static mixer dead time | 1 to 6 s |
| [P3] p. 10 | Second stage valve opening time | almost a minute to 50 % |
| [P7] p. 2 | Mixing delay per tank | about 7 s; 10 s assumed |
| [P7] Table 1, p. 3 | Volume versus number of tanks at 0.01 m3/s, kd = 2.5e6 | 250,000; 316; 40.7; 9.51; 6.96 m3 for 1 to 5 tanks |
| [P7] p. 2 | Example product spec | pH 7 plus or minus 1 |
| [P1] abstract | Lab rig step responses | figures not available openly; see owner list |
| [P2] abstract | Lab rig titration and closed-loop runs | figures not available openly; see owner list |

## For the owner to open by hand

- Henson and Seborg 1994, https://doi.org/10.1109/87.317975 (IEEE paywall).
  Check: tank volume, acid, buffer and base stream flows and concentrations,
  electrode location and time constant, sampling period, titration curve
  figure, buffer step experiments and their pH traces.
- Hall and Seborg 1989, ACC pp. 1822-1827 (IEEE paywall). Check: two-tank
  rig sizes, pumps, step response figures.
- Emerson application note "pH measurement in industrial waste
  neutralization" (68368), https://www.emerson.com/documents/automation/application-note-ph-measurement-in-industrial-waste-neutralization-en-68368.pdf
  (returned an empty page to the reader). Check: hold-up time rule for on/off
  control, staging, timer limit on reagent additions, level and discharge
  interlock.
- G.K. McMillan, "What are the top 10 mistakes made every day in pH system
  design", https://www.automation.com/article/what-are-the-top-10-mistakes-made-every-day-in-ph-system-design
  (403). Check: reagent injection and valve location delays, vessel geometry,
  number of stages.
- eCFR text of 40 CFR 401.17 and 403.5 at https://www.ecfr.gov (bot check);
  the Cornell copy was read instead.

## How often each window occurs (2026-11-08)

How often a normal day and each bad day occur: not found in the sources read.
