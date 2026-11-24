# Glass melting furnace: plant reality record

This record describes real regenerative glass melting furnaces, float first and
container where float sources are silent, and how they are read, moved and protected.
It pools sources across the family's plants (charter Amendment 3, rule 19) and comes only
from outside sources, listed at the end as [S1] to [S44]. Each was read twice unless
marked "seen once". Where nothing was found, it says "not found". Nothing here is a
modelling choice.

## 1. What operators and the control system read

- **Crown thermocouples**, in the crown, often in platinum thimbles [S28]. One 650 t/d
  float furnace has one on the centreline about 12 ft from the charging end wall [S5].
  Their delay and accuracy: not found.
- **Bottom thermocouples**, in pockets in the bottom blocks, platinum-thimbled where they
  touch the glass [S27]. The same float furnace has two, left and right, about 12 ft down
  the tank from its boost burners [S5]. Delay and accuracy: not found.
- **Throat or waist temperature** is a measured point and a controlled output of one
  supervisory system [S2, p. 5].
- **Canal temperature**, by a radiation pyrometer viewing the glass in the canal [S6]; the
  glass must be at least 1100 C before the tin bath [S16]. Feedback from it is suspended
  during a firing reversal [S6].
- **Glass level**, held by the batch charger [S1, slide 17; S7, Fig. 7].
- **Batch line.** Operators inspect the batch blanket visually and interpret it [S24]. A
  camera system reads the batch line position; one study reports it in statistics
  normalised to the range, on a non-float furnace [S23]. Reading cadence: see section 7.
- **Cullet ratio** is held by the batch plant's own loops by feed rate [S30].
- **Defects in the product** are read after forming, at the cold end: see section 7.
- **Gas quality.** Natural gas composition "is usually weekly and sometimes even daily"
  [S19]; a stabiliser measures the Wobbe index to correct it [S19]. Network figures:
  entry points vary within 4 MJ/m3, mixed exit points by more than 6 MJ/m3 over a month,
  LNG and biomethane by more than 3 MJ/m3 within a day [S20].

## 2. What they move, and in which mode

- **Crown and bottom setpoints, by hand.** Operators set the crown and glass-bottom
  setpoints themselves; tuning them "can take several weeks to several months", and must
  be repeated after a pull change [S4]. Below those setpoints, firing follows the crown.
- **Many large furnaces are run by hand** at the melter, sometimes partly automatic [S7,
  sec. 4.1]. Supervisory model predictive control takes over daily firing where installed
  [S1; S2].
- **Firing profile.** The fuel split between the side ports sets where the hot spot sits
  and how large it is [S15]. Per-port shares on a float furnace: not found.
- **Boosting** (oxy-fuel or electric) where fitted, trimmed by bottom temperatures [S5].
- **Pull and its changes** come from the forming line; pull change is an input to float
  furnace control [S1, slide 10]. See section 7 for sizes.
- **Working-end and waist cooling.** Refiners are cooled by submerged coolers and
  under-floor cooling [S17b], or by air blown across the glass in the working end [S29,
  seen once]. Waist coolers set how much glass comes forward, 1.2 to 2.5 times the pull
  [S17].
- **Tweel** meters the flow into the tin bath from the canal temperature [S6].

## 3. Limits, and what kind each is

| Limit | Value | Kind | Source |
|---|---|---|---|
| Silica crown, hot side | Visible degradation above 1620 C; theoretical limit 1630 C | Wear, a cost over the campaign. No automatic protection found | [S13; S14] |
| Silica crown, cold side | Sodium silicate forms below 1400 C | Wear | [S13] |
| Real crown operation | 1550 to 1580 C on one float furnace, std 2.21 K under operators, 1.2 K under MPC | Where operators keep it | [S1, slide 13] |
| Glass quality: bubbles | Bubbles leave only when fining is complete; sulfate fining is strong in oxidised melts from 1420 to 1500 C; fining is set by "the time-temperature integral of melting and refining"; the shortest path's residence is 15 to 25 % of the mean; front-end firing in the refiner can raise reboil bubbles | A zone: glass with defects is lost to cullet | [S12; S31; S45; S46] |
| Canal | At least 1100 C before the tin bath | A zone for forming | [S16] |
| Price of glass lost to cullet | not found | | |
| Crown wear rate against temperature | see section 7 | | |

## 4. Time scales

- Lags between action and reaction in the melting tank: 1 to 16 h [S7, sec. 4.1].
- A furnace takes "ten-odd hours" to change its thermal state; the bottom temperature
  "shows typical dead time behavior"; one crown point near the charging side was
  approximated with about 5 h of dead time [S8].
- Residence time, 600 t/d float, CFD: shortest 12.3 h (8.0 h with boosting), mean 68.4 h
  [S11]. Shortest over mean, float, container and TV furnaces: 0.15 to 0.20 [S12].
- A glass temperature loop in a forehearth answers in 20 to 40 min; after a pull change
  the glass took 4 to 6 h to settle, up to 10 h [S9, forehearth, container].
- Setpoint tuning by hand: weeks to months [S4].

## 5. A normal day and a bad day

- **Normal day.** Crown within about 1550 to 1580 C, std 2.21 K under operators [S1, slide
  13]; canal std 0.92 under operators, 0.32 under MPC [S1, slide 15]; pull wandering by
  about 2 % on a 650 t/d float furnace [S5]; gas composition drifting weekly or daily
  [S19]. On one plant (type not stated), operators held the controlled temperature within
  2 K only 19.6 % of the time, an MPC 64.8 to 74.4 % [S3, slide 9].
- **Bad day.** A pull or job change: the whole setpoint tuning "has to be repeated" [S4];
  PID loops kept cycling for more than 8 h after a pull change, and impatient operators
  switching to manual prolonged it [S9]; MPC reduces losses during product changes by 20
  to 50 % [S2, p. 6]. A gas quality jump of several MJ/m3 within a day [S20]. Foam over
  the melt shields it from the flames and returns heat to the crown [S32].

## 6. The job, in a sentence

Keep the glass melted, fined and at forming temperature at the canal, through gas, batch
and pull changes, at the least fuel, without wearing the crown [S1, slide 8: stability,
fuel optimisation, consistency; S2; S13].

## 7. The named property

**Lower variance lets the setpoint drop toward the defect limit, and the saving is the
lower setpoint** [S1, slide 18; S3, slide 10]. Its size in the family is 2 to 4 % of fuel:
"up to around 2-3 %" on one float furnace against manual control [S2, p. 7], 2 to 4 % over
425 installations [S3, slide 11], 1 to 3 % expected [S2, p. 6]. What makes the variance
hard to reduce is in sections 1 to 5: lags of hours, unmeasured gas and batch changes, and
quality read only after forming.

**Crown wear.** A patent's model of a silica crown over an oxy-fuel furnace gives
corrosion of 5.14 and 10.32 cm per year at a 1450 C crown surface [S34]. Its dependence on
temperature is "very sensitive" and given in no number [S35]. A campaign runs "up to 16
years and more" before the rebuild [S36]; one float rebuild cost 35 MEUR in 2008, for a
furnace expected to last 18 years [S37]. Wear against crown temperature in air-fired
furnaces, an activation energy, and campaign life against crown temperature: not found.
So crown wear has no family price.

**Quality, read late.** Float lines inspect the cut glass at the cold end: scanners
detect core bubbles, stones and scratches [S38; S39, seen once]. Their unit (per area or
per kg) and bubble resolution: not found. The ribbon leaves the bath at 10.4 m/min for
2.5 mm glass [S40], 7.5 m/min in another example [S41, seen once]; one producer's bath is
about 50 m and its lehr about 100 m [S42]. A delay from the furnace to the scanner: not
found as a figure.

**Pull changes.** A 650 t/d float furnace's pull wanders within about 2 % [S5]. A
container plant's most frequent pulls are 144 and 204 t/d [S3, slide 8]. Float pull-change
sizes, frequency, notice and ramp rate: not found.

**Cullet and batch.** The cullet ratio is the cullet fed over the glass made, held from
hopper load cells weighed about every 2 min [S43, seen once for the cadence]. A batch
camera takes images "at regular time intervals", no number given [S44]; 30 min on one
plant [S23, as reported by a reviewer, not re-read].

## 8. What the plant looks like

Saint-Gobain's float line, in its own 3D tour and an inside view: six cross-fired ports
per side with regenerators along both sides; a charger across the back wall; past the last
port the tank keeps its full width for a long unfired stretch, then narrows to a waist,
widens into the working end, and narrows to the canal onto the tin [S25]. Inside, flames
come from one side at a time, and batch islands cover much of the surface [S25]. One
455 t/d float furnace measured: melter and refining zone 42 m long and 11 m wide, six
ports, batch islands covering 85 % of the island zone [S10]. A 1300 t/d float furnace: a
14 m wide melter, a waist 7.5 m long and 4.5 m wide, a working end 20 m long and 10 m wide
[S18].

## 9. The stream: how long regimes last and how often events come (2026-11-11)

Searched for the continuing furnace (docs/proposals/long-strata.md, S1). A figure
counts when it was read twice on separate fetches; "seen once" marks the rest.

- **Gas swings at a consumer.** At one French site in June 2016, LNG raised the
  Wobbe index from 50.4 to 52.4 MJ/m3 between 18:59 and 20:59 on 6 June, and the
  index went back to pipe gas and up again three times in about two days; each
  LNG episode spent about 8 h above half its rise, and the third (to 52.7 MJ/m3)
  was still on at the chart's end (read from the chart) [ST1]. The source behind
  the slide is a glass-melting paper. The stream's gas regimes deliver 1.5 swings a
  day, each held 8 h (PHYSICS.md, the stream). The swing there is 4 % in 1 to 2 h, against the task's 6 %
  over 6 h (ours, section 7); the window keeps its own.
- **Seen once.** ENTSOG: at mixed exit points no daily change above 5.5 MJ/m3 was
  recorded, though a month can exceed it; 2 of 29 points with 15 min data changed
  by more than 3 MJ/m3 within 15 min. JRC: maximum hourly rates of change of 2.7
  to 3.5 MJ/m3 an hour in four countries. A container-glass vendor guide: "40+"
  job changes a line a year, 20 to 80 in its examples.
- **Not found.** How often a float line changes pull, by how much, with how much
  notice and how fast; how long a changed pull holds; how long an LNG-affected
  period of a network lasts. Searched: float pull change, thickness change, job
  change (also in Chinese), GRTgaz, Gasunie, National Gas and DVGW Wobbe series.
  The regime dwells (a week), the pull change rate (one a pull regime's visit) and
  its hold (3 days) are ours.

### Sources for section 9

- [ST1] J. Leicher (GWI), "Gas Quality and Hydrogen from the Perspective of
  Industrial End Users", Swedish Gas Days 2019, slide 7, citing M. Ourliac, "Deal
  with gas quality variations and melt glass with syngas from gasification",
  IFRF/GWI TOTeM 44, Essen, 2017. https://www.energigas.se/media/2515/1_2-joerg-leicher_gwi-s.pdf

## 10. Transients: what a change of temperature or pull costs (2026-11-15)

Searched for what makes glass defective when the furnace moves, as opposed to when it
holds still. A figure counts when it was read twice on separate fetches; "seen once"
marks the rest. Nothing here is a modelling choice.

**The steps the glass passes, in order.** Batch melts to a rough melt near 1200 C; the
sand grains dissolve, leaving a seedy melt near 1400 C; fining clears it near 1500 C;
refining and conditioning bring it to about 1250 C [TR1, slide 5]. Fresh melt holds on the
order of 100,000 bubbles per kg [TR1, slide 36]. In the blanket more than half the sand
dissolves within 10 min once it is above 1200 C, and convection in the melt speeds
dissolution up to five times over still glass [TR2, seen once].

**Fining moves with the batch's redox.** In oxidised melts sulfate decomposes at 1430 to
1480 C, the fining onset; in reduced melts sulfate and sulfide react between 1140 and
1350 C. Organic contamination makes the fining range fluctuate, and bubbles and seeds
follow [TR1, slides 54 to 56].

**A second fining on the way down.** Bubbles under about 100 micrometres hardly rise. They
dissolve back into the glass during controlled cooling, in a narrow temperature window,
about 1350 to 1275 C for float glass, and only if the melt was left lean in dissolved gas
[TR1, slides 43, 49, 51].

**Heating glass again makes bubbles.**
- A melt nearly saturated with gas forms new bubbles, or grows very small ones, when its
  temperature, its chemistry or its stirring changes [TR1, slide 36].
- SO3 dissolves less as the melt gets hotter: about 0.3 wt % at 1400 C in float glass, so
  a glass holding 0.3 wt % starts to give it off at about 1400 C when heated [TR3]. Float
  glass normally keeps about 0.2 wt % [S45], 0.25 wt % in one calculation [TR3].
- Firing the front of the refiner too hard raises reboil bubbles [S45]. On one float
  furnace, lowering the waist temperature from about 2920 to 2850 F cut the top bubbles
  sharply; they were about 0.25 mm and mostly carbon dioxide [S45].
- Under vacuum, reboil at the glass and refractory contact starts once the SO2 partial
  pressure passes the absolute pressure [TR12, seen once, a vacuum degasser].
- How many kelvin of reheating start reboil in a float refiner, how soon the bubbles
  come, how long they last and how many: not found.

**The tank blocks give off defects when their temperature rises.**
- Fused-cast AZS exudes more glassy phase under temperature cycles than held steady. Held
  at 1550 C it gives off 1.4 wt % over 80 h and then stops. Five swings between 1450 and
  1550 C give 2.1 wt %, and five between 1500 and 1550 C give 1.8 wt %; swings of 100 K
  or less act more slowly and for longer [TR4, section 2].
- The cause is oxygen released in the block as it heats, which pushes the glassy reaction
  layer and its zirconia seam into the melt as bubbles, knots and stones, from new and
  used blocks alike, in high-alkali silicate glass. After 96 h at 1550 C, a dip of only
  50 K and the return released them again [TR4, sections 4 and 5].
- The paper's advice for a tank in operation is a constant process temperature [TR4,
  section 6].
- A control supplier says the same of working furnaces: a temperature change at the
  contact between melt and refractory strips the corrosion layer into the glass as seeds,
  stones and knots, and instability in the melter makes the bubble field from the batch
  oscillate, raising the seed count. Disturbances from pull, batch or combustion that
  are left alone travel through the waist and working end into forming [S7, section 3].

**The crown wears faster when its temperature cycles.** On a furnace whose silica crown
corroded about twice as fast as expected, with bricks of the best grade, the cause given
is the repeated cycling of the crown's temperature: 1405 to 1430 C in the first zone and
1575 to 1605 C in the hottest, swings of 25 to 30 K [TR5; an electrovacuum-glass furnace].
After a process change that lowered the melter's temperature, one plant saw more crown
corrosion and dripping [TR11, seen once]. A limit on how fast a silica crown may change
temperature in operation, in K per hour: not found.

**Foam.** Foam can cut the radiation reaching batch and melt by as much as 60 %, which
cools the bath, slows refining and leaves more bubbles and undissolved sand in the
product. A higher pull favours foaming, which caps the pull. A reducing atmosphere
destabilises foam [S32, chapter introduction and section on atmosphere].

**The bottom glass answers late.** Heat added at the surface takes several hours to show
in the bottom glass, and the delay moves with the pull and with the batch cover, neither
of them monitored [TR8, seen once]. Proportional-integral loops are still the usual
temperature control in furnace sections; identification tests take several weeks because
the furnace is slow, and their models hold only near the point tested [TR9, seen once].

**Flow.** In a float working end, a side-to-side difference of more than 24 K in bottom
temperature sends part of the returning bottom flow out with the product, which hurts
uniformity [TR10, seen once, abstract only]. How the tank's currents answer a change of
pull or firing, and with which defects: not found.

**The batch blanket.** On one 455 t/d float furnace batch islands cover 85 % of the island
zone [S10]. A claim that the blanket breaks up near the third or fourth port was not
confirmed on reading its patent (US4329165), which says only that it reaches a
considerable distance.
How the batch line moves with pull, and how fast: not found.

**How plants run a change.** Operators keep tables of the furnace's readings for each
configuration and reproduce them, and they keep a safety factor from the optimum so as
not to risk the glass; hand control is weakest in the transient phases, the changes of
tonnage, glass type or colour [TR13]. A supervisory controller is not allowed its full
correction in one step, because a very large step disturbs the furnace; a stable firing
profile is what lets it absorb pull changes [S7, sections 3 and 4.1]. Float lines no
longer run at steady pull, thickness and colour [TR14, seen once]. Ramp rates for a pull
change, how far ahead temperatures are moved, the time to be steady again, any rule on
setpoint steps, and how far ahead a plant knows its changes: not found.

**What defects are worth.** With supervisory control, suppliers report energy savings of
about 2 to 3 %, yield up to about 8 % better, and time and production lost in product
changes cut by up to half [S7, section 4.4]. On one float furnace, dispersing the foam
with a different flame cut the bubble density and raised the yield of 4 mm glass from
50 % to over 90 % [TR7]. Hours or tonnes rejected after a pull change, defect counts
against time after a change, and the share of a float line's losses due to melting
defects: not found.

### Sources for section 10

- [TR1] M. Hubert, IMI-NFG Course on Processing of Glass, Lecture 2, "Industrial glass melting and fining processes", Lehigh University, https://www.lehigh.edu/imi/teched/GlassProcess/Lectures/Lecture02_Hubert_Melting%20and%20fining%20processes.pdf
- [TR2] Beerkens, Muijsenberg, van der Heijden, Glastech. Ber. Glass Sci. Technol. 67 (1994) 179-188, abstract, https://oa.tib.eu/renate/items/5a415375-9e39-4a20-b756-faf0691ff903
- [TR3] US5922097, Praxair (Kobayashi, Beerkens), 1999, https://patents.google.com/patent/US5922097A/en
- [TR4] M. Dunkl, Glastech. Ber. 62 (1989) 389-395, https://oa.tib.eu/renate/bitstreams/4c2c2d07-cca4-4f29-a379-7664eec2cc96/download
- [TR5] Balandis, Nizeviciene, Science of Sintering 43 (2011) 295-303, https://epubl.ktu.edu/object/elaba:3184407/3184407.pdf
- [TR7] US6715319B2, Pilkington, 2004, https://patents.google.com/patent/US6715319B2/en
- [TR8] P. J. Firth, "Bottom temperature control of a glass furnace using fuzzy logic", IEE Colloquium Digest 1994/018, abstract, https://manuscript.isc.ac/Inventory/49/323698.htm
- [TR9] Auchet, Riedinger, Iung, Malasse, "Simplified first-principles modelling of glass furnaces for control purpose", IFAC, https://skoge.folk.ntnu.no/prost/proceedings/dycops04/pdffiles/papers/172.pdf
- [TR10] Xu, Li, Feng, Yang, Liu, "3D fluid flow regularity in a float glass furnace, Part 1", Glass Technology, abstract, https://katalog.lib.cas.cz/KNAV/EdsRecord/edb,117206450
- [TR11] R. Bei, ICG TC11, October 2024, https://icglass.org/wp-content/uploads/2024/11/3-Refractory-behavior-in-case-of-glass-melting-process-change-Rongxing-Bei.pdf
- [TR12] US9352994B2, https://patents.google.com/patent/US9352994B2/en
- [TR13] US6796147B2, Stein Heurtey, filed 1999, https://patents.google.com/patent/US6796147B2/en
- [TR14] Muller, Chmelar, Glass Worldwide, Sept-Oct 2012, pp. 52-53, as summarised by British Glass, https://britglass.org.uk/knowledge-base/digital-library-and-information-services/advances-float-glass-furnace-control

## Sources

- [S1] G. Noboa, Glass Service, Glassman Sao Paulo 2019 slides, https://glassmanevents.com/content-images/main/01-Gabriel-Noboa-Glassman-Sao-Paulo-05122019.pdf
- [S2] Schill, Glass Service, ES III paper, https://www.ksr.tul.cz/glassman/download/09_05_16-10-Schill.pdf
- [S3] Cruise Control for Glass Furnaces deck (AIGMF), https://aigmf.com/docs/Cruise%20Control%20for%20Glass%20Furnaces%20automated%20furnace%20control%20and%20batch%20monitoring%20systems.pdf
- [S4] US6454562B1, oxy-boost control in furnaces, https://patents.google.com/patent/US6454562B1/en
- [S5] US11261117B2, synchronized oxy-fuel boosting of a regenerative float furnace, https://patents.google.com/patent/US11261117B2/en
- [S6] US5055122A, canal pyrometer and tweel control, https://patents.google.com/patent/US5055122A/en
- [S7] Chmelar, Bodi, Muysenberg, Glastech. Ber. Glass Sci. Technol. 73 (2000) 276, https://oa.tib.eu/renate/bitstreams/14a42da5-abd3-43b4-8142-c87961478664/download
- [S8] US5272621A, fuzzy control of a process with dead time, https://patents.google.com/patent/US5272621A/en
- [S9] Gough, Andritz, predictive-adaptive control of molten glass (forehearth), https://www.andritz.com/resource/blob/14714/d3ee3da793d5914c788eceeb102f7921/aa-automation-adaptive-control-molten-glass-data.pdf
- [S10] Wang, Brewster, McQuay, Webb, Glastech. Ber. 73 (2000) no. 10, https://oa.tib.eu/renate/bitstreams/dfbd7f19-5c22-4bb8-b1bf-e19159e41180/download
- [S11] Materials Transactions 60 (2019) 1034-1043, https://www.jstage.jst.go.jp/article/matertrans/60/6/60_M2019044/_html/-char/en
- [S12] Beerkens, TNO slides, Cairns 2011, https://publications.tno.nl/publication/100347/EAHeqQ/beerkens-2012-new.pdf
- [S13] SEFPRO, Glass Machinery Plants & Accessories 4/2026, https://www.glassonline.com/wp-content/uploads/2026/05/046_049_SEFPRO-GM-4-2026.pdf
- [S14] British Glass, spinel crown abstract, https://britglass.org.uk/knowledge-base/digital-library-and-information-services/advanced-spinel-materials-melter-crown-and
- [S15] Hubert, Lehigh IMI-NFG course, lecture 3, https://www.lehigh.edu/imi/teched/GlassProcess/Lectures/Lecture03_Hubert_industglassmeltfurnaces.pdf
- [S16] Pyrometer vendor page, canal and working tank, https://www.pyrosales.com.au/?p=1464
- [S17] US4798616, waist cooler, https://patents.google.com/patent/US4798616 ; [S17b] US4091156, https://patents.google.com/patent/US4091156
- [S18] Materials 17 (2024) 4989, https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11509868/fullTextXML
- [S19] Glass Service, Glass Machinery Plants & Accessories 3/2021, p. 41, https://www.glassonline.com/wp-content/uploads/2025/01/GLASS-SERVICE-GM3_2021.pdf
- [S20] ENTSOG, Wobbe index and GCV in European networks (2017), https://www.entsog.eu/sites/default/files/entsog-migration/publications/Events/2017/INT1124-170918%20WI%20and%20GCV%20in%20European%20networks.rev%204.pdf
- [S21] Eurostat, non-household gas prices, H2 2025, https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20260508-3
- [S22] AGC, cullet and energy, https://www.agc-glass.eu/cs/node/186
- [S23] Klemiato, Rotter, Skowiniak, Production Engineering 15 (2021), batch line position
- [S24] Glass Service 2017 camera deck, https://www.gsl.cz/wp-content/uploads/2017/08/paper-muijsenberg-erik-gs-camera.pdf
- [S25] Saint-Gobain, Float Process (YouTube, 2021), https://www.youtube.com/watch?v=1HDWJgFLCfA, stills shared by the owner
- [S26] ICG TC, bottom lining considerations (2023), https://icglass.org/wp-content/uploads/2024/07/5-Bottom-Lining-Considerations-2023-12-20.pdf
- [S27] Tempsens, temperature in glass manufacturing, https://tempsens.com/blog/temperature-most-critical-factor-for-glass-manufacturing/
- [S28] Eurotherm, crown and bottom thermocouples, https://www.eurotherm.com/it/glass-manufacture/crown-bottom-thermocouples/ (seen once)
- [S29] US9016094, working end air cooling (seen once)
- [S30] Batch plant patent on cullet ratio and blanket loops, glass-sources-3a (session notes)
- [S31] Beerkens, Fining of glass melts, British Glass abstract, https://www.britglass.org.uk/knowledge-base/digital-library-and-information-services/fining-glass-melts-what-we-know-about
- [S32] Pilon et al., Foams in glass manufacturing (2012), https://www.seas.ucla.edu/~pilon/Publications/FoamsInGlassManufacturing2012.pdf
- [S33] Gas prices and cullet: [S21], [S22]
- [S34] US8640500B2, crown corrosion with an added burner, https://patents.google.com/patent/US8640500B2/en
- [S35] Crown corrosion modelling abstract, TIB, https://oa.tib.eu/renate/items/1fa3b1da-2bc8-492a-91b0-4aa7ed870c8b
- [S36] Glass melting furnace, Wikipedia, https://en.wikipedia.org/wiki/Glass_melting_furnace
- [S37] British Glass digest, Saint-Gobain Aviles furnace (2008), https://www.britglass.org.uk/knowledge-base/digital-library-and-information-services/spain-saint-gobain-furnace-upgraded-back
- [S38] ISRA FloatScan, https://isravision.com/en-en/products/floatscan-premium
- [S39] Grenzebach float inspection (seen once)
- [S40] US4217125A, float ribbon speed, https://patents.google.com/patent/US4217125A/en
- [S41] US3853523, float ribbon speed (seen once), https://patentimages.storage.googleapis.com/45/4c/c8/0ca4901e1d425f/US3853523.pdf
- [S42] Producer page, tin bath and lehr lengths, https://www.uzsm.uz/en/activities/production/production-secial-building-materials/glassware/
- [S43] US12410087B2, cullet ratio, https://patents.google.com/patent/US12410087B2/en
- [S44] US9103799B2, batch camera, https://patents.google.com/patent/US9103799B2/en
- [S45] US6673730, float glass fining by the time-temperature integral, reboil from refiner firing, https://patents.google.com/patent/US6673730B1/en
- [S46] US6871514B2, critical path residence, https://patents.google.com/patent/US6871514B2/en
