# Plant reality record: distillation

This record says what the real plant behind `distillation` looks like, from primary sources only.
It was written without reading the task's code or design notes, so it can be used to check the task.
Each statement carries a source tag such as [D1 p.15 Fig. 5]. The sources are listed at the end.
"Not found" means no primary source read for this record says it. Nothing was filled from memory.

## The plant

Column A is a binary column with 41 stages counting the reboiler and the total condenser, fed on stage 21 from the bottom [D2].
Relative volatility is 1.5, the feed is F = 1 kmol/min of saturated liquid with zF = 0.5, and the products are yD = 0.99 and xB = 0.01 [D2].
Nominal reflux is L = 2.706 kmol/min and boilup V = 3.206 kmol/min, so D = B = 0.5 kmol/min [D2].
Nominal holdup is 0.5 kmol on every stage; for more realistic studies the page suggests 10 kmol in the reboiler and 32.1 kmol in the condenser [D2].
Column A is a model, not a plant. Its author built it as "a more realistic example" for analysis [D1 p.13, section 4].
A real counterpart with plant data is an industrial propylene/propane (PP) splitter in São Paulo, Brazil, designed for 99.5% propylene [D4 p.2].
That splitter has a heat pump: the top vapour is recompressed and condensed in the bottom reboilers [D4 p.2].
Its reflux runs between 200 and 320 t/h and its feed between 10 and 45 t/h (MPC input bounds) [D4 p.5 Table 2].
Tray count, diameter and pressure of a real high-purity column: not found in the sources read.

## What operators read

- Tray temperatures. A thermocouple sits on a selected tray and is used to correct the analyser between samples [D5].
  Column pressure "significantly affects" tray temperature, and a linear pressure correction is usually enough [D3].
  Inferential temperature control is not feasible below relative volatility 1.4 and usually works above 2.0 [D3].
  Column A (alpha 1.5) sits between those two limits [D2, D3].
  Skogestad recommends at least three temperature sensors in each column section [D1 p.34, section 7].
  Thermocouple response time: not found.
- Online analysers (process GC). Total analyser delay on distillation columns is "about three minutes to about thirty minutes" [D5].
  It is transport time plus analysis time; the patent's example is 2 min transport plus 4 min analysis [D5].
  On a PP splitter, an analyser cycle of 5 or 10 min makes little difference to feedback control [D3].
  The real PP splitter has two analyser controllers: AC 1 (propane in the propylene product) and AC 2 (propylene in the propane product) [D4 p.5 Fig. 1].
  A specific vendor cycle time for a C3 splitter GC: not found (vendor note blocked, see the last section).
- Pressure. Column pressure is one of five controlled outputs in Skogestad's control problem [D1 p.25, section 6.2].
  The real PP splitter shows pressure controllers PC 1 and PC 2 on the overhead [D4 p.5 Fig. 1].
- Levels. Condenser holdup MD and reboiler holdup MB are controlled outputs [D1 p.25, section 6.2].
  The real PP splitter controls the main reboiler level (LC 5), which sets the exposed reboiler area [D4 p.2, p.5].
- Flows. Feed (FC 1), reflux (FC 2) and heat pump flow (FC 3) are flow controllers on the real PP splitter [D4 p.5].
- Differential pressure across a packed section. Its fluctuation grows as the column nears flooding [D6].

## What they move, and in which mode

- Skogestad lists five manipulated flows: L, V, D, B and the overhead vapour VT [D1 p.25, section 6.2].
  The five controlled outputs are xD, xB, MD, MB and p [D1 p.25, section 6.2].
- LV configuration: L and V control composition, D and B hold the levels [D1 p.14, section 4.3].
  For Column A its steady-state RGA element is 35.94; DV gives 0.45 and (L/D)(V/B) gives 3.29 [D1 p.9, eqs. 31 to 33].
  RGA elements above about 10 imply serious control problems [D1 p.9].
- Cascade: fast temperature loops with an outer composition loop are recommended to make the quality loops fast [D1 p.33].
- One-point control: holding xB alone by an LV scheme "explains the popularity of this scheme in industry for one-point control" [D1 section 6.4].
- No composition control ("open loop") has been common in industrial practice [D1 section 6.2].
- MPC: the real PP splitter study manipulates heat pump flow, feed flow and reflux flow with an MPC on a 1 min sample [D4 p.5].
  Outputs are held in zones: reboiler level 4 to 80%, propane in propylene 0 to 0.45 mol%, propylene in propane 0 to 2 mol% [D4 p.5 Table 1].
  Skogestad wrote in 1997 that no industrial 5x5 or 4x4 MPC had been reported [D1 p.33].
- Feedforward from feed rate: not found as a measured plant practice in the sources read.
- Analyser correction: the delayed tray temperature is aligned with the analyser delay to correct the analyser value [D5].

## Limits

- Product specifications. On the real PP splitter: propane in propylene at most 0.45 mol%, propylene in propane at most 2 mol% [D4 p.5 Table 1].
  These are control zones, so a cost or quality limit, not a trip.
- Input bounds and move limits on the real PP splitter: reflux 200 to 320 t/h, move at most 0.13 t/h per minute [D4 p.5 Table 2].
  Feed 10 to 45 t/h (0.02 t/h per minute); heat pump flow 220 to 350 t/h (0.15 t/h per minute) [D4 p.5 Table 2].
- Flooding. Approaching flood shows as rising DP fluctuation; at incipient flooding the filtered DP standard deviation is "significantly greater" than normal [D6].
  Whether flooding is a trip, an alarm or a cost: not found.
- Weeping: not found.
- Levels: controlled outputs [D1 p.25]; level trip settings: not found.
- Pressure relief or high-pressure trip settings: not found.

## Time scales

- Column A dominant time constant: about 194 min, for any small change in external flows, including F and zF [D1 p.14 to 15].
  The three slowest modes of the linear model are 193.9, 12.0 and 3.5 min [D1 p.15, section 4.4].
- Internal flows (L and V up together): close to first order with about 15 min if flow dynamics are neglected [D1 p.15].
- Hydraulic lag: 0.063 min per tray; reflux takes about 39 x 0.063 = 2.46 min to reach the reboiler [D1 p.13, Fig. 4].
  The same paper says "about 2.54 min" when discussing Figure 6 [D1 p.15].
- Real PP splitter: composition time constant of the primary product about 2 hours [D3].
- Real PP splitter, plant step tests (model G2): first-order terms of 118.6 to 118.7 min for the level [D4 p.5].
  Composition terms are second order with denominator s^2 + 0.01688 s + 9.857e-5 (time in minutes) [D4 p.5].
  Reflux to propylene in the propane product carries a dead time of 111 min [D4 p.5, G2 element (2,3)].
  The paper notes the plant and simulator models differ mainly in gains and dead times [D4 p.5].
- Ethane/propane splitter (a fast, low-reflux column): time constant under 5 min [D3].
- Analyser total delay: 3 to 30 min [D5]. Skogestad's closed-loop simulations use a 1 min measurement delay [D1 p.25, p.33 Fig. 18].
- Flooding indicator: a 5 min rolling window, so it needs at least 5 min to settle after a duty step [D6].

## A normal day and a bad day

Normal day. Feed rate and composition move, and the controls hold the products.
On the real PP splitter the MPC holds both impurities inside their zones while pushing feed up and heat pump and reflux flow down [D4 p.5].
Bad day, feed rate. In Column A with LV two-point PI control and a 1 min analyser delay, a 20% feed step is shown [D1 p.33 Fig. 18].
In that figure xD drops by about 4.3e-3 and xB rises by about 3.5e-3, and both return within about 40 min.
Bad day, feed composition. A step of zF from 0.5 to 0.6 gives a much smaller bump, about 1e-3, in that figure [D1 p.33 Fig. 18].
A boilup disturbance from steam pressure is the case of Figure 12 [D1 p.24].
Flooding. As reboiler duty rises toward flood, DP stops being steady and its high-frequency noise grows [D6].
Analyser failure: with DV control, a failed top measurement leaves D constant and performance is poor [D1 section 6.5].
With DB control, performance is very poor if a failure leaves D or B constant [D1 p.33].
The LV configuration "performs poorly with large dead-times" [D1 section 6.5].
Real plant trends of a feed upset on a high-purity column: not found (one candidate is listed for the owner).

## The operator's job in one sentence

Keep both product impurities inside their zones, at the least reflux and boilup, without flooding the column [D4 p.5, D6].

## Proposed behaviour tests: sources list

| Source and figure | Quantity | Value as printed |
|---|---|---|
| D1 p.15 Fig. 5 | xD and xB after a 0.1% step in L at constant V (Column A) | xD 0.99 to about 0.992; xB 0.01 to about 0.0135; time constant about 194 min |
| D1 p.15 Fig. 5 | same step in V at constant L | almost the same size, opposite sign |
| D1 p.15 Fig. 6 | L and V both up 10%, D constant | gain about 100 times smaller than Fig. 5; both products purer; faster |
| D1 p.15 Fig. 6 | xB initial response | sharp drop for about 2.5 min until the reflux reaches the bottom |
| D1 p.13 Fig. 4 | delay from reflux change to reboiler | about 2.46 min (39 x 0.063) |
| D1 p.15 section 4.4 | slowest linear modes | 193.9, 12.0, 3.5 min |
| D1 p.9 eq. 31 | LV gain matrix and RGA | [[0.8754, -0.8618], [1.0846, -1.0982]], lambda11 = 35.94 |
| D1 p.33 Fig. 18 | 20% feed step, LV PI control, 1 min delay | xD dip about -4.3e-3, xB peak about +3.5e-3, back within about 40 min |
| D2 | nominal steady state | L 2.706, V 3.206, D = B = 0.5 kmol/min, yD 0.99, xB 0.01 |
| D3 | real PP splitter composition time constant | about 2 hours |
| D4 p.5 G2 | real PP splitter reflux to propylene in propane | dead time 111 min, second order |
| D5 | analyser total delay on columns | about 3 to 30 min |
| D6 | DP noise near flood | standard deviation rises with each duty step |

## Sources

- [D1] S. Skogestad, "Dynamics and control of distillation columns: a tutorial introduction", Trans IChemE 75 Part A, 1997, 539-562. Submitted version read: https://skoge.folk.ntnu.no/publications/1997/dist_plenary/submitted_version/dist_plenary.pdf . Figures 5, 6 (p.15) and 18 (p.33) were looked at as images. Checked twice.
- [D2] S. Skogestad, "MATLAB Distillation column model (Column A)", NTNU web page, https://skoge.folk.ntnu.no/book/matlab_m/cola/cola.html . Checked twice.
- [D3] J. Riggs, "Distillation: Inferential Temperature Control & Single-Ended Control", Control Guru (practitioner training material by a Texas Tech professor and industrial consultant), https://controlguru.com/distillation-inferential-temperature-control-single-ended-control/ . Checked twice.
- [D4] A. I. Hinojosa, D. Odloak, "Using Dynsim to study the implementation of advanced control in a Propylene/Propane Splitter", IFAC DYCOPS 2013, Mumbai, pp. 33-38, https://skoge.folk.ntnu.no/prost/proceedings/dycops2013-and-cab2013/media/DYCOPS/files/0039.pdf . Page 5 (Fig. 1, G2 matrix, Tables 1 and 2) looked at as an image. Only G2 comes from plant step tests; the closed-loop runs are simulations. Checked twice.
- [D5] M. H. Beauford (Phillips Petroleum), "Inferential correction of composition for a distillation analyzer", US Patent 5,464,504, 1995, https://patents.google.com/patent/US5464504 . Checked twice.
- [D6] J. Cahill (Emerson), "Early Detection of Distillation Column Flooding Conditions", 1 Aug 2012, on R. Pihlaja's field test with the University of Texas Separations Research Program, https://www.emersonautomationexperts.com/2012/measurement/pressure/early-detection-of-distillation-column-flooding-conditions . Vendor blog. Checked twice.

- [D7] Distillation column design chapter, Taylor and Francis, book 9780429107290, chapter 10.1201/b10875-7, https://www.taylorfrancis.com/books/9780429107290/chapters/10.1201/b10875-7 . Fetched once, 2026-11-08, as a search-engine extract; the page itself returns 403 to the reader, so the re-fetch check could not be made (for the owner to open). The extract: with an upper limit of 1,000 ppm heavy key in the distillate and a standard deviation of 100 ppm, the desired impurity may be 700 ppm, 3 standard deviations below the limit; a second example puts a 32 % limit with a 0.2 % standard deviation.

- [D8] M. G. Jacobsen, S. Skogestad, "Active constraint regions for optimal operation of distillation columns", Ind. Eng. Chem. Res. 51 (2012) 2963-2973, https://skoge.folk.ntnu.no/publications/2012/jacobsen_active-constraints-part2-distillation/ie201220m.pdf . Fetched twice on 2026-11-09, byte-identical. Column A (41 stages, feed on 21, alpha 1.5, zF 0.5): J = pF F + pV V - pD D - pB B with pF = pB = 1 $/mol, pD = 2 $/mol and pV 0.01 to 0.02 $/mol (Table 1, p. 2965); for hydrocarbons "pV/pF should be about 0.01" since combustion heat is about 100 times vaporization heat (p. 2965). With cheap energy the bottoms' spec is inactive and the bottoms are over-purified, moving light key to the distillate; past its spec the distillate's purity is give-away (p. 2966).

## For the owner to open by hand

- J. Ford (Maverick), "Distillation columns: Product composition control, process identification models", Control Engineering, https://www.controleng.com/distillation-columns-product-composition-control-process-identification-models/ (403). Check the depropanizer trend figure and the reported dead time 34.5 min and lag 14 min from reflux to tray temperature or GC (seen only in a search snippet, not cited above).
- Emerson Rosemount application note "Improving fractionator tower performance with process gas chromatographs" (00800-..., en-71678), https://www.emerson.com/documents/automation/application-note-improving-fractionator-tower-performance-process-gas-chromatographs-rosemount-en-71678.pdf (blocked). Check GC cycle times, tap location, and the overhead accumulator delay.
- Emerson Rosemount application note on ethylene fractionation trains, en-71778, https://s1-live.emerson.com/documents/automation/application-note-improving-ethylene-plant-fractionation-train-performance-process-gas-chromatographs-rosemount-en-71778.pdf (host unreachable). Check the C3 splitter analyser and its cycle time.
- R. Pihlaja, "Distillation Column Flooding Diagnostics with Intelligent Differential Pressure Transmitter", Emerson whitepaper, https://www2.emersonprocess.com/siteadmincenter/PM%20Rosemount%20Documents/3051S_ASP_Distillation_Column_Flooding.pdf . Check the DP trend figures and how far ahead of flood the warning came.
- S. Skogestad, M. Morari, "Understanding the dynamic behavior of distillation columns", Ind. Eng. Chem. Res. 27, 1988 (paywalled). The original Column A paper.
- Fortuna, Graziani, Xibilia, "Soft sensors for product quality monitoring in debutanizer distillation columns", Control Engineering Practice 13, 2005 (paywalled). A search snippet says the GC delay is 30 to 75 min; check it in the paper.

## How often each window occurs (2026-11-08)

How often a normal day and each bad day occur: not found in the sources read.

### Stream numbers, second search (2026-11-11)

A second search looked for plant records, operator surveys and standards on
how long a plant runs between upsets and how long each upset lasts. Every
figure below was read once and confirmed on a second fetch.

- **A normal day.** The ASM consortium's visits to large process plants found
  that abnormal situations "occur almost on a daily basis in most large
  process" plants; most are small, and they are more likely during startup,
  feed changes, product grade changes and shutdown [ST1, p. 3]. The stream's
  normal regime keeps its day, now sourced for a large plant.
- **A shift.** Most US refineries have staffed process operations with two
  12-hour shifts since the 1990s, and 91.7 % of 351 refinery workers surveyed
  worked 12-hour shifts [ST2, pp. 118, 120]. Across 400 shift operations in
  all industries, 37 % ran 12-hour and 34 % 8-hour shifts [ST3]. Family range
  8 to 12 h; the stream's 8 h is inside it. That an upset regime lasts a
  shift, and brings one event a visit, is ours.
- **Feed changes.** A refinery crude unit (Marifu No. 4) runs different crude
  slates "with crude switches occurring every 2-5 days" [ST4]. A crude unit
  is not this task's family (high-purity splitters), so the 8 h hold is kept.
  For comparison, the stream's two composition regimes (weights 0.1 each, one
  event per 8 h visit) start a composition step about every 40 h, each held
  8 h; on a crude unit a new crude holds until the next switch, 2 to 5 days.
- **Not found.** How often an upstream unit changes the feed rate.

- [ST1] P. Bullemer, D. Reising, "Managing Human Reliability: An Abnormal
  Situation Management Historical Perspective", Mary Kay O'Connor Process
  Safety Center Symposium, 2015, p. 3.
  https://process.honeywell.com/content/dam/process/en/documents/document-lists/doc_asm-consortium/white-papers/October%2031%202015%20-%20Managing%20Human%20Reliability%20An%20Abnormal%20Situation%20Management%20Historical%20Perspective.pdf
- [ST2] McNamara, Robbins, "Shift Work and Sleep Disturbance in the Oil
  Industry", Workplace Health & Safety 71(3), 2023, pp. 118, 120 (journal
  pages). https://stacks.cdc.gov/view/cdc/230000/cdc_230000_DS1.pdf
- [ST3] Circadian, "8-Hour vs. 12-Hour Shift Schedules" (survey year not
  given). https://circadian.com/managing-247-article-1-0712
- [ST4] Singh, Hiroshima, Kakinoki, Williams, Friedman, "Japan refiner
  implements crude unit multivariable controller", Oil & Gas Journal,
  4 Nov 2002.
  https://www.ogj.com/general-interest/companies/article/17233746/japan-refiner-implements-crude-unit-multivariable-controller
