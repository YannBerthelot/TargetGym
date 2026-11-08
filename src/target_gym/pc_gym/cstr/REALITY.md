# Plant reality record: `cstr`

Claim: a non-isothermal continuous stirred-tank reactor with an irreversible,
first-order, exothermic liquid reaction A to B, cooled through a jacket, with
the coolant as the manipulated variable.

This record was built from primary sources only. Nothing in this repository
(code, PHYSICS.md, docs, notes) was read to write it. "Not found" means no
primary source found in this pass says it. PDF page numbers are the page index
of the PDF file; the printed page number follows in brackets where it differs.

## Sources

Each source was fetched twice. "Checked twice" means the second fetch confirmed
every statement cited from it.

- [S1] Luís, Astolfi, Kershenbaum, "Application of a Lyapunov-based nonlinear
  controller to a reactor with highly exothermic reactions", AIChE Annual
  Meeting 2005, paper 557c. Pilot CSTR (PARSEX) at Imperial College London.
  https://skoge.folk.ntnu.no/prost/proceedings/aiche-2005/non-topical/Non%20topical/papers/557c.pdf
  Pages 3 to 6, Figure 1, Tables 1 to 4, Figure 2. Checked twice.
- [S2] US CSB, "T2 Laboratories, Inc. Runaway Reaction", Report 2008-3-I-FL,
  Board voting copy, 15 Sept 2009.
  https://www.csb.gov/assets/1/20/T2_Final_Copy_9_17_09.pdf?13900
  Figures 8 and 9, PDF pages 12, 25, 27 to 33. Checked twice.
- [S3] US CSB, "Bayer CropScience LP, Pesticide Chemical Runaway Reaction
  Pressure Vessel Explosion", Report 2008-08-I-WV, 2011.
  https://www.csb.gov/assets/1/20/bayer_report_final.pdf
  Figure 11, PDF pages 36 to 48. Checked twice.
- [S4] US CSB, "Morton International, Inc. Runaway Chemical Reaction", Report
  1998-06-I-NJ, 2000. https://www.csb.gov/assets/1/20/morton_report.pdf?13798
  Figures 4, 5 and 14, PDF pages 20 to 36 and 48. Checked twice.
- [S5] UK HSE, "Chemical reaction hazards and the risk of thermal runaway",
  INDG254(rev1), 2014. https://www.hse.gov.uk/pubns/indg254.pdf
  Pages 1, 2 and 4. Checked twice.
- [S6] Lipták (with Langford, Wade), "Jacket temp: thermocouple multiplexers"
  (Ask the Experts), Control Global, 14 Oct 2006.
  https://www.controlglobal.com/control/loop-control/article/11383011/jacket-temp-thermocouple-multiplexers
  Checked twice.
- [S7] Lipták (with Langford, Caro), "Ask the Experts: Cascade reactor
  temperature control", Control Global, 5 Oct 2018.
  https://www.controlglobal.com/articles/2018/ask-the-experts-cascade-reactor-temperature-control
  Checked twice.
- [S8] Pearson and Cordino (Herzog-Hart Corp), US Patent 4,744,408,
  "Temperature control method and apparatus", 1988.
  https://patents.google.com/patent/US4744408A/en  Figures 2, 3 and 6.
  Checked twice.
- [S9] Grady and Gregorovich (DuPont), US Patent 5,447,998, "Clear coat resins
  by continuous polymerization", 1995.
  https://patents.google.com/patent/US5447998A/en  Detailed description,
  claims 1 and 3, Examples II and V. Checked twice.
- [S10] Delancey (WIKA), "Thermowells: what to know about response times in
  electrical temperature measurements", white paper WP007.
  https://www.wika.com/media/Others/EN-US/wp007_thermowells_en_us.pdf
  Figures 2 and 3, PDF pages 4 and 6. Checked twice.
- [S11] Buchelli, Haynie, Wicker (Equistar), US Patent 7,786,227, "Monomer
  concentration prediction and control in a polymerization process", 2010.
  https://patents.google.com/patent/US7786227B2/en  Background and Figure 1
  description. Checked twice.
- [S12] Cahill (Emerson), "Improving jacket heating and cooling temperature
  control", Emerson Automation Experts blog, 30 Jan 2019.
  https://www.emersonautomationexperts.com/2019/services-consulting-training/improving-jacket-heating-and-cooling-temperature-control/
  Checked twice.

## The plant

The task claims a continuous jacketed reactor. The one continuous stirred tank
with plant data found here is the PARSEX pilot reactor [S1, p.3, Fig. 1]: a
tank "reactor" whose contents recirculate through an external shell-and-tube
cooler with water on the shell side, standing in for a jacket. The reaction is
simulated: a computer computes the heat a first-order A to B reaction would
release and injects that much steam each second or so. Feed is 0 to 110 g/s,
residence time about 34 min [S1, Table 1, p.5]. An industrial continuous
counterpart is the DuPont acrylic resin CSTR, run at 80 to 180 °C with
residence times of 15 to 120 min, 20 to 60 min preferred, cooled by a
circulating fluid in an outer jacket, with optional internal coils "to
effectively prevent any run-away" [S9]. A continuously fed, level-controlled
industrial vessel with a cooled recirculation loop is the 4,500-gallon Bayer
residue treater, held near 135 °C with level at about 50 percent [S3, PDF
p.36 to 38 (printed 28 to 30)]. Jacketed vessel sizes run from "several
gallons to several hundred thousand gallons" [S8]. Industrial propylene glycol
hydration in a CSTR: not found.

## What operators read

- Reactor temperature. One probe in the liquid. At Morton (2,000-gallon batch
  kettle) it was one thermocouple on a digital readout and a chart recorder
  that stopped at 150 °C [S4, PDF p.21 (17), Fig. 5]. T2's control screen
  shows reactor temperature, pressure, hot-oil and cooling-water loop
  temperatures [S2, Fig. 9, PDF p.29 (21)].
- Thermowell lag. In WIKA's flowing-water test (over 0.2 m/s) one sensor
  reaches 63 percent of a step in roughly 4 s bare, 13 s in a thin-walled
  well and 45 s in a thick-walled well, read from the curves [S10, Fig. 2,
  PDF p.4]. Example t0.5, t0.63, t0.9 of about 1.5, 2 and 6 s [S10, Fig. 3,
  PDF p.6]. These are lab values in water, not in a reactor.
- Jacket inlet and outlet temperatures. Both have transmitters in the patent
  loop (41 inlet, 42 outlet), plus cooling-water in and out at the cooler
  [S8, Fig. 6]. One plant measured jacket temperature at the outlet, which
  "adds approximately two minutes of dead time" (reader's statement) [S6].
  Lipták: the slave sensor sits on the jacket outlet, or the inlet when
  utility upsets dominate [S6].
- Coolant flow. PARSEX logs fresh cooling-water flow and steam flow [S1,
  Fig. 2]. Morton had no flow indicator and no jacket inlet or outlet
  temperature; operators listened for the water [S4, PDF p.25, p.34 (21, 30)].
- Level. Bayer: automatic level control near 50 percent, often run in manual
  [S3, PDF p.36, p.40 (28, 32)].
- Concentration analyser. Bayer sampled for the lab once every 24 hours [S3,
  PDF p.40 (32)]. A loop polyethylene reactor's gas chromatograph has "a 7 to
  10 minute dead time", against densitometer readings every 2 to 5 s [S11].
  An online analyser on a jacketed liquid CSTR: not found.

## What they move, and in which mode

- Cascade: reactor temperature master sets the jacket temperature slave. The
  slave must be at least four times faster; it is often P-only, proportional
  band about 15 percent [S6], or 10 to 20 percent [S7]. The coolant is
  recirculated to cut dead time [S6].
- Split range heating and cooling on one slave output: 4 mA full cooling,
  12 mA crossover, 20 mA full heating [S7]. A batch plant with this scheme
  showed large overshoot and undershoot, cured by Lambda tuning, cascade mode
  on the jacket loop and feedforward [S12]. Bayer's split range was found
  misprogrammed: in changeover both heater and cooler valves closed [S3,
  footnote 29, PDF p.47 (39)].
- Recirculating loop with heating and cooling exchangers in series, pumped
  [S8, Figs. 3 and 6].
- Coolant flow directly. In PARSEX's usual SISO scheme the fresh cooling water
  controls reactor temperature; in [S1] a MIMO scheme moves feed
  concentration (for T) and fresh water (for cooler temperature) [S1, p.4].
- Manual. Morton operators ran steam and water valves by hand-wheel [S4, PDF
  p.20 (16)]. T2's cooling program injected water intermittently on the rate
  of temperature rise [S2, PDF p.30 (22)].
- Feed rate as a handle: [S1] uses feed concentration; feed-rate moves on a
  jacketed industrial CSTR: not found.

## Limits

- High temperature, trip or interlock. Bayer had safety interlocks (minimum
  temperature, maximum pressure) on the feed valve, password-protected, plus
  an "operating" low-flow interlock; operators bypassed them [S3, PDF p.38,
  p.46 (30, 38)]. Safe operating limit 155 °C [S3, PDF p.47 (39)]. Morton had
  no alarms and no automatic shutdown [S4, PDF p.21 (17)].
- Pressure relief. T2: rupture disk set at 400 psig on a 600 psig MAWP vessel
  [S2, PDF p.27 (19)]. Morton: two 6-inch disks at 10 psig [S4, PDF p.26 (22)].
- Emergency measures listed by HSE: crash cooling outside set limits,
  inhibitor, dump into quench fluid, relief vents, containment [S5, p.4].
- Slave setpoint high and low limits "if the process requires it" [S7].
- Cooling capacity is a hard ceiling. PARSEX fresh water range 40 to 110 g/s
  [S1, Table 1], though Figure 2 shows 120 to 136 g/s; the paper does not
  explain the mismatch.
- Cost limits (energy, product quality): not found as numbers.
- IEC 61511 trip design: not found in an open copy.

## Time scales

- Residence time: about 34 min (PARSEX) [S1, p.5]; 15 to 120 min, 20 to 60
  preferred (DuPont CSTR) [S9].
- Closed-loop settling: a 2 °C setpoint step on PARSEX settled after about
  110 min [S1, Fig. 2, p.5].
- Jacket dead time: about 2 min at the outlet in one plant [S6].
- Thermowell: seconds to under a minute in water [S10].
- Analyser: 7 to 10 min (GC, loop reactor) [S11]; 24 h lab sample [S3].
- Heat-up during a normal batch: 1 to 2 °C per minute (Morton) [S4, PDF p.25
  (21)].
- Runaway time to maximum rate (TMRad) for a liquid CSTR: not found. Rate
  rule of thumb: rate doubles every 10 °C [S5, p.1].
- Jacket time constant and reactor thermal time constant as measured numbers:
  not found.

## A normal day and a bad day

Normal day. PARSEX at about 58 °C reactor, 47 °C cooler, feed 90 g/s at 23
to 25 °C, CA,in 1.5 kmol/m3, CA about 0.33 kmol/m3 [S1, Tables 3 and 4].

Bad day, by disturbance:

- Feed temperature: feed at room temperature, 23.1 and 24.9 °C on two days
  [S1, Tables 3, 4]. Its effect on a trace: not found.
- Feed concentration: Bayer's cooler could hold temperature only while
  methomyl stayed below about 0.5 percent, and the relief was sized for 1
  percent [S3, PDF p.36 to 37 (28 to 29)]. Feeding started without preheat
  and with interlocks bypassed, so methomyl built up [S3, PDF p.46 (38)].
- Catalyst activity drop of 40 percent: PARSEX reactor fell 1.7 °C below
  setpoint and overshot 0.7 °C before settling [S1, p.5 to 6, Fig. 3].
- Coolant temperature: Bayer cooler water at a constant 80 °C [S3, PDF p.37
  (29)]. Seasonal or supply-temperature swings on a reactor: not found.
- Fouling: CSB lists "formation of mineral scale inside the jacket" as a
  credible cause of lost heat removal at T2; components were run to failure
  [S2, PDF p.32 (24)]. Bayer's vent line plugged with deposits, raising
  pressure [S3, PDF p.36 (28)]. A measured fall of U with time on a
  continuous unit: not found.
- Agitator or circulation failure: Bayer lost recirculation at 22:21; in under
  3 min the temperature hit 141 °C and climbed over 2 °C per minute [S3,
  Fig. 11, PDF p.47 (39)].
- Cooling failure: T2 reported a cooling problem at 1:23 pm; the reactor
  burst at 1:33 pm [S2, PDF p.17 (9)]. Morton switched to full cooling at
  100 °C and the disks blew at about 180 to 190 °C [S4, PDF p.25 to 26].
- Repeated near misses at Morton: charts off scale above 150 °C for about 30
  to 75 min on earlier batches, "cooling water of no use" [S4, PDF p.48 (44)].

## The operator's job in one sentence

Hold reactor temperature at its setpoint by moving the jacket coolant, through
a cascade on the jacket temperature, while keeping heat removal ahead of heat
release so the reactor never crosses into runaway [S5, p.1 to 2; S6; S4, PDF
p.34 (30)].

## Proposed behaviour tests source list

- [S1, Fig. 2]: closed-loop 2 °C setpoint step on a pilot CSTR. Reactor 58 to
  about 60 °C, peak about 60.5 °C near 60 min, settled after about 110 min;
  cooler 47 to about 48.5 °C; residence time about 34 min.
- [S1, Fig. 3 text]: 40 percent drop in k0. Dip 1.7 °C, overshoot 0.7 °C.
- [S1, p.4]: design point T about 55 °C, Tj about 47 °C, about 74 percent
  conversion, k0 2.0e11 1/s, E/R 10700 K, dH -200 kJ/mol, CA,in 1.5 kmol/m3.
- [S3, Fig. 11]: loss of circulation. Rise over 2 °C/min, 141 °C within
  3 min, rupture 12 min after flow loss at 160.3 °C (figure label).
- [S2, PDF p.17]: cooling loss to rupture, about 10 min (batch, not CSTR).
- [S4, PDF p.25]: normal heat-up 1 to 2 °C/min; abnormal rise read as
  "moving very fast" (batch).
- [S5, p.1]: rate doubles per 10 °C rise, as a rough check on kinetics.
- [S6]: jacket outlet dead time about 2 min (one plant, reader's statement).
- [S10, Fig. 2]: thermowell 63 percent time about 4 to 45 s.
- [S11]: analyser dead time 7 to 10 min.

## For the owner to open by hand

- Emerson, "Improve batch reactor temperature control" (doc 2690866). The
  host d1-live.emerson.com did not resolve and emerson.com redirects to a
  search page. https://d1-live.emerson.com/documents/automation/improve-batch-reactor-temperature-control-en-2690866.pdf
  Check: 800 L reactor trends, oscillation period, jacket dead time, the
  "integrating" jacket-to-reactor response.
- Chylla and Haase, "Temperature control of semibatch polymerization
  reactors", Comput. Chem. Eng. 17 (1993) 257. Paywalled. Check: industrial
  jacket recirculation loop dead time, fouling factor drift batch to batch.
- Pinheiro and Kershenbaum, Comput. Chem. Eng. 23 (1999) S859, DOI
  10.1016/S0098-1354(99)80211-7. Paywalled. Check: PARSEX traces and feed-flow
  disturbances.
- Kershenbaum and Kittisupakorn, Trans IChemE 72A (1994) 55. Paywalled.
  Check: PARSEX volume, cooler time constant, valve and sensor lags.
- CCPS, "Guidelines for Process Safety in Batch Reaction Systems" and
  "Guidelines for Safe Automation of Chemical Processes"; IEC 61511. Paid.
  Check: high-temperature trip practice and response times.
