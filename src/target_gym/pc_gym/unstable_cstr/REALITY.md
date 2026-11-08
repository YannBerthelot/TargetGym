# Plant reality record: `unstable_cstr`

Claim: a jacketed exothermic CSTR held at its open-loop unstable middle steady
state, with the coolant command reaching the reactor through a valve and a
jacket lag.

This record was built from primary sources only. Nothing in this repository
(code, PHYSICS.md, docs, notes) was read to write it. "Not found" means no
primary source found in this pass says it. The plant, instruments and jacket
loop are shared with `cstr`; see `../cstr/REALITY.md` for those. This file
covers what changes at an unstable point. PDF page numbers are the page index
of the PDF file; the printed page number follows in brackets where it differs.

## Sources

Each source was fetched twice. "Checked twice" means the second fetch confirmed
every statement cited from it. Tags S1 to S12 match `../cstr/REALITY.md`.

- [S1] Luís, Astolfi, Kershenbaum, AIChE Annual Meeting 2005, paper 557c,
  PARSEX pilot CSTR. https://skoge.folk.ntnu.no/prost/proceedings/aiche-2005/non-topical/Non%20topical/papers/557c.pdf
  Pages 3 to 6, Fig. 1, Tables 1 to 4, Fig. 2. Checked twice.
- [S2] US CSB, T2 Laboratories report 2008-3-I-FL, 2009.
  https://www.csb.gov/assets/1/20/T2_Final_Copy_9_17_09.pdf?13900
  Fig. 9, PDF pages 17, 27, 30, 32, 33. Checked twice.
- [S3] US CSB, Bayer CropScience report 2008-08-I-WV, 2011.
  https://www.csb.gov/assets/1/20/bayer_report_final.pdf
  Fig. 11, PDF pages 36 to 48. Checked twice.
- [S4] US CSB, Morton International report 1998-06-I-NJ, 2000.
  https://www.csb.gov/assets/1/20/morton_report.pdf?13798
  Fig. 14, PDF pages 21, 25, 26, 34 to 36, 48. Checked twice.
- [S5] UK HSE, INDG254(rev1), 2014. https://www.hse.gov.uk/pubns/indg254.pdf
  Pages 1, 2, 4. Checked twice.
- [S6] Lipták et al., Control Global Ask the Experts, 14 Oct 2006.
  https://www.controlglobal.com/control/loop-control/article/11383011/jacket-temp-thermocouple-multiplexers
  Checked twice.
- [S7] Lipták et al., Control Global Ask the Experts, 5 Oct 2018.
  https://www.controlglobal.com/articles/2018/ask-the-experts-cascade-reactor-temperature-control
  Checked twice.
- [S8] US Patent 4,744,408, Herzog-Hart, 1988.
  https://patents.google.com/patent/US4744408A/en  Checked twice.
- [S10] WIKA white paper WP007, thermowell response times.
  https://www.wika.com/media/Others/EN-US/wp007_thermowells_en_us.pdf
  Figs. 2 and 3. Checked twice.
- [U1] McMillan, "Solutions for unstable industrial processes", ISA Mentor
  Program, republished by ISA Sarnia.
  https://isasarnia.com/solutions-for-unstable-industrial-processes/
  Checked twice. Practitioner article with no numbers.
- [U2] Go, Gobaco, Ha, "Temperature control of a non-isothermal CSTR at the
  unstable region using PID mode", BSc thesis, De La Salle University, 1995.
  Abstract only. https://animorepository.dlsu.edu.ph/etd_bachelors/9284
  Checked twice.

## The plant

Same unit as `cstr`. Documented experiments at an unstable middle steady state
come from laboratory and pilot reactors, not production units. A 1995 lab CSTR
with sodium thiosulfate and hydrogen peroxide was run "at the unstable region
of the multiple steady-state behavior", with cooling-water flow moved by a
proportional valve on an electronic actuator [U2]. The PARSEX pilot CSTR at
Imperial College was built to test controllers on exothermic reactions
[S1, p.3]. A paper title says PARSEX was run at an unstable steady state under
nonlinear MPC (Pinheiro and Kershenbaum 1999), but that paper was not opened
(see the owner list). A production reactor run deliberately at an open-loop
unstable point, with plant data: not found. The propylene glycol (propylene
oxide hydration) pilot CSTR named in the task: not found in an open source.

## What operators read

As in `cstr`: reactor temperature probe, jacket inlet and outlet temperatures,
coolant flow, level [S8, Fig. 6; S6]. What changes at an unstable point:

- Rate of rise. T2's cooling program acted on "the rate of reaction
  temperature increase" [S2, PDF p.30 (22)]. Bayer's board saw a steady rise
  of about 0.6 °C/min for hours, then over 2 °C/min after circulation loss
  [S3, PDF p.47 (39), Fig. 11].
- Pressure. At Bayer the first alarm the operator acted on was high pressure,
  at about 22:25 [S3, PDF p.48 (40)].
- Circulation flow. Bayer logged recirculation flow; it dropped to zero at
  22:21 [S3, Fig. 11].
- Sensor range. Morton's chart recorder saturated at 150 °C and was "of no
  use"; only the digital readout kept reading [S4, PDF p.26 (22)].
- Lags that matter more here. McMillan: a window of allowable PID gains
  closes as heat-transfer and thermowell time constants and loop dead time
  approach the positive-feedback time constant [U1]. Thermowell 63 percent
  times of roughly 4 to 45 s in water [S10, Fig. 2]. Jacket outlet dead time
  about 2 min in one plant [S6].

## What they move, and in which operating mode

- The controller must stay in automatic. McMillan: "The point of
  acceleration cannot be measured in practice because it is unsafe to have
  the controller in manual" [U1].
- Gain floor. "A PID gain too low will allow a reactor to runaway" [U1].
- Cascade reactor to jacket, slave at least four times faster than the
  master [S6, S7]; coolant recirculated to cut dead time [S6].
- Coolant valve: cooling-water flow through a proportional valve [U2]; fresh
  cooling water on PARSEX [S1, p.4].
- Second handle: feed concentration (or catalyst) on PARSEX [S1, p.4].
- Emergency: switch to full cooling. Bayer and Morton operators did this; it
  did not stop either runaway [S3, PDF p.48 (40); S4, PDF p.25 (21)].

## Limits

Protection layers, from HSE: inherent safety first (less unreacted material,
for example a continuous process), then process control with "sensors,
alarms, trips", then protective measures: containment, relief vents, crash
cooling outside set limits, inhibitor, dump into quench fluid [S5, p.4].

- High temperature or pressure interlock on the feed valve (trip). Bayer had
  them, password-protected, and operators bypassed them on start-up [S3, PDF
  p.38, p.46 (30, 38)].
- Safe operating limit as an alarm level: 155 °C at Bayer [S3, PDF p.47 (39)].
- Relief: at T2 the disk was set at 400 psig; CSB found no relief of any size
  at 400 psig could have saved the vessel once the second reaction began,
  while 75 psig likely would have relieved the first [S2, PDF p.33 (25)].
  At Bayer the relief valve opened at 44 psig at 22:27 [S3, Fig. 11].
- Secondary reaction above a threshold temperature: T2 above 390 °F (199 °C)
  [S2, PDF p.33 (25)]; Morton decomposition onset 195 °C [S4, PDF p.36 (32)].
- Emergency cooling source: T2 had none; backup water was not available to
  the operator [S2, PDF p.30 (22)].
- Feed cutoff, quench or dump on a continuous unstable reactor, with trip
  setpoints and response times: not found. IEC 61511 text: not opened.

## Time scales

- Runaway from cooling loss to rupture: about 10 min at T2 (1:23 to 1:33 pm,
  batch) [S2, PDF p.17 (9)].
- From loss of circulation to rupture: 12 min at Bayer (22:21 to 22:33); in
  under 3 min temperature reached 141 °C and was rising over 2 °C/min
  [S3, PDF p.47 (39), Fig. 11].
- Final stage of a secondary runaway in a test cell: 2340 °F per minute and
  32,000 psig per minute [S2, PDF p.33 (25)].
- Rate rule of thumb: doubles per 10 °C [S5, p.1].
- How long a lab PID held an unstable set point: "a maximum of 18 minutes and
  16.93 seconds", within +2 and -3 °C [U2].
- Residence time at PARSEX: about 34 min [S1, p.5].
- Positive-feedback time constant of a real reactor, as a number: not found.
- Time to maximum rate (TMRad) for the task's chemistry: not found.
- Open-loop divergence time from the middle steady state, measured: not found.

## A normal day and a bad day

Normal day: the controller holds the middle state with small swings. In the
lab case, +2 and -3 °C around set point [U2]. On PARSEX (stability not stated)
a 2 °C step settled in about 110 min [S1, Fig. 2].

Bad day:

- Controller in manual or saturated. Not allowed for positive-feedback
  processes [U1]. Lab control was lost after about 18 min [U2].
- Cooling failure: T2, 10 min to rupture [S2].
- Circulation (mixing) failure: Bayer, 12 min to rupture; Bayer review found
  the split-range logic closed both heater and cooler valves at changeover
  [S3, footnote 29, PDF p.47 (39)].
- Feed or start-up error: Bayer fed without preheat, with interlocks
  bypassed, so unreacted material built up before the temperature rose
  [S3, PDF p.46 (38)].
- Starting above the intended temperature: Morton batch 32 started heating
  from above the required 40 to 42 °C, leaving less time to react [S4, PDF
  p.35 (31)].
- Fouling: mineral scale inside the jacket was a credible cause of lost
  cooling at T2 [S2, PDF p.32 (24)].
- Earlier warnings: Morton batches went off chart (over 150 °C) for 30 to
  75 min before the accident [S4, PDF p.48 (44)].

## The operator's job in one sentence

Keep the temperature controller in automatic with enough gain to hold the
reactor at a point it would leave on its own, watch the rate of rise,
pressure and circulation, and trip feed or apply emergency cooling before
heat release outruns the jacket [U1; S5, p.4; S3, PDF p.48].

## Proposed behaviour tests source list

- [S3, Fig. 11]: after loss of circulation, over 2 °C/min, 141 °C within
  3 min, rupture at 12 min, 160.3 °C (figure label), relief at 44 psig 6 min
  after flow loss.
- [S2, PDF p.17]: cooling loss to rupture about 10 min (batch).
- [S4, PDF p.25]: normal heat-up 1 to 2 °C/min against a runaway that moved
  "very fast"; disks vent at 180 to 190 °C (batch).
- [U2]: PID at an unstable state holds +2 and -3 °C, at most about 18 min.
- [S1, Fig. 2]: settling after about 110 min with residence time about
  34 min (stability of the point not stated).
- [U1]: qualitative. Too low a gain runs away, and a window of gains closes
  as lags approach the positive-feedback time constant.
- Published open-loop trace leaving a middle steady state, or a measured
  limit-cycle period: not found in open sources (see owner list).

## For the owner to open by hand

- Pinheiro and Kershenbaum, "Model predictive control of reactor temperature
  in a CSTR pilot plant operating at an unstable steady-state", Comput. Chem.
  Eng. 23 (1999) S859, DOI 10.1016/S0098-1354(99)80211-7. Paywalled. Check:
  open- and closed-loop traces at the unstable point, feed-flow steps.
- Vleeschhouwer and Fortuin, "Theory and experiments concerning the
  stability of a reacting system in a CSTR", AIChE J. 36 (1990). Scribd copy
  blocks automated readers: https://www.scribd.com/document/58126463/orig-artikel
  Check: measured multiple steady states, limit cycles and runaways
  (oxiranemethanol hydration), periods and amplitudes.
- Vermeulen and Fortuin, Chem. Eng. Sci. 41 (1986) 1291. Paywalled. Check:
  measured large temperature oscillations in a liquid-phase CSTR.
- Heemskerk (1980) and Heemskerk and Fortuin (1983), cited by Vleeschhouwer
  and Fortuin as the source experiments. Full titles not confirmed. Check:
  limit-cycle period and amplitude.
- Chang and Schmitz, "Feedback control of unstable states in a laboratory
  reactor", Chem. Eng. Sci. (1975). Paywalled. Check: controller gains needed,
  effect of delay.
- Root and Schmitz, AIChE J. 16 (1970) 356 (loop reactor). Paywalled. Check:
  report that a three-mode controller failed to stabilise.
- core.ac.uk PDF 32319716 (returned 403). https://core.ac.uk/download/32319716.pdf
  Check: a table of experimental CSTR systems with thermal oscillations.
- McMillan, "101 Tips for a Successful Automation Career", Appendix F. Book.
  Check: positive-feedback time constant formula and example values.
- CCPS guidelines and IEC 61511. Paid. Check: trip response time against
  runaway time.

## Amendment 3: the LDPE autoclave family (2026-11-08)

The owner's review found industrial operation at an unstable point in the LDPE
autoclave family, and the task is rebuilt as a member of it. These sources
were fetched once on 2026-11-08 and read in full (L1, L2, L3) or by their
abstract (L4). Facts used:

- [L1] E. Caliani, M. Cavalcanti, F. A. N. Fernandes, L. M. F. Lona,
  "Modelling and simulation of high pressure industrial autoclave polyethylene
  reactor", ICHEAP-8, 2007, with Politeno plant data,
  https://skoge.folk.ntnu.no/prost/proceedings/icheap8-pres07/icheap8webpapers/191%20Caliani.pdf .
  The reactor is "essentially adiabatic and cooling is provided by the inflow
  of cold monomer"; initiator fed at several points "provides control of the
  temperature" (p. 1). PID controllers manipulate the initiator feed on
  measured segment temperatures (p. 2). The model needs temperature
  controllers "because the industrial reactor normally operates at an unstable
  steady state in which the operation can either cause the temperature to rise
  or cool down the reactor until no polymerization occurs" (p. 2). 1600 atm,
  ethylene feed at 353 K, zone temperatures held at 513, 513, 537, 513 and
  531 K (p. 3); control "within a few residence time".
- [L2] A. M. Moutinho de Jesus, "Modeling an industrial LDPE plant with Aspen
  Plus and Aspen Polymers", MSc thesis, Instituto Superior Tecnico (Repsol
  Sines autoclave),
  https://scholar.tecnico.ulisboa.pt/api/records/TalL7DXsQdaOukAly8w-K0N1DI3adWwoNttF/file/a5d30e27d29cacd9c06971b51f4821d9c3e5662ebb889ab7070d3f05fa2ef17b.pdf .
  A 5-zone autoclave at 1250 to 1950 bar and 170 to 300 C (Table 1, Fig. 11);
  40 grades, each with its zone temperatures (section 3.2); residence time 20
  to 90 s and initiator half-lives of 0.02 to 1 s (p. 21); initiator
  decomposition constants from supplier sheets (Table 3); kinetic constants
  pooled from six literature sets, among them Ham et al.'s autoclave set
  (Table 4).
- [L3] RSU International Research Conference 2018, paper G2-IN18-006, a CFD
  study of an LDPE autoclave with ethylene decomposition,
  https://rsucon.rsu.ac.th/files/proceedings/inter2018/G2-IN18-006.pdf . A
  498 L autoclave fed 8 kg/s of mixture at 499 kg/m3 (Tables 3, 4); the
  initiator premixed at a mass fraction near 1.2e-4; ethylene decomposition
  "can produce a large amount of heat" and the runaway is "a problem in LDPE
  autoclave reactor plant".
- [L4] L. Marini, C. Georgakis, "Low-density polyethylene vessel reactors:
  Part I: Steady state and dynamic modelling", AIChE J. 30(3), 1984, 401-408,
  https://doi.org/10.1002/aic.690300308 , by its abstract: a perfectly mixed
  and an imperfectly mixed model; the reactor is open-loop unstable at low
  temperatures and open-loop stable at high polymerization temperatures.

How often each operating window occurs (normal day, grade changes, feed
upsets): not found in these sources.
