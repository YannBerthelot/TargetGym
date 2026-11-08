# Reality record: building space heating with slow heat delivery

Plant: a room or zone heated by water pipes cast in concrete (thermally
activated building systems, TABS, and embedded radiant floors), with
radiators, baseboards and air systems for contrast. Each item names its
source; the numbered list is at the end. Where a source is a simulation or a
design guide rather than a measured building, the item says so. "Not found"
means no source we could read states it.

## 1. What occupants, operators and the controller read

- **Zone (room) air temperature, wall sensor.** In a US survey of radiant
  buildings, 43% used zone temperature as the sole control variable and 26%
  used slab temperature [6, p. 2, citing Higgins et al. 2015]. A Madrid TABS
  office measures room and floor temperature with two sensors per floor
  (Honeywell H7012B room sensors), and the BMS uses the average of each
  floor's sensors [14, pp. 12 and 15, Figs. 6, 13].
- **Air sensor versus operative temperature.** CBE notes that in radiant
  buildings mean radiant and air temperatures sit closer together than in
  forced air buildings, so an air sensor is closer to operative temperature
  there [7, p. 13, footnote 2].
- **Slab (core) temperature sensor.** All eleven interviewed North American
  TABS designers measure slab temperature; most put the sensor between tubes
  at tube depth, three put it 1 to 2 inches below the surface, and large
  zones average several sensors [7, p. 8]. Rehau specifies the core probe in
  a protection pipe, at pipe level, midway between two pipes [12, p. 12].
- **Return water temperature.** Rehau accepts a return water probe instead of
  a core probe, insulated, at least 10 cm from the manifold [12, p. 12]. The
  Madrid BMS starts the heat pump compressor on return water temperature
  falling below setpoint plus 1 K [14, p. 15, Fig. 16].
- **Outdoor temperature, filtered.** The Siemens TABS strategy uses the mean
  outdoor temperature of the last 24 hours [9, part 1]. The Münster office
  (14 000 m2) feeds 24 h means of outdoor temperature and solar radiation to
  its TABS supply setpoint [13, "Control for tabs"]. Guideline 36 recommends
  a global outdoor sensor for warm-up timing because unit-mounted sensors
  read high when the unit is off [2, p. 5, note to 5.3.7.3]. SwissEnergy warns
  that the outdoor reading on heating controllers "is often wrong" (faulty
  sensor or sun on it) [11, p. 1].
- **Room feedback filtering.** Siemens TABS room feedback uses the minimum
  and maximum room temperature of the last 24 hours as the controlled
  variable [9, part 2]; the CBE controller uses the previous day's occupied
  minimum or maximum, applied once per day [6, pp. 4-5].
- **Occupants** read comfort, not a sensor: in 500 surveys across eight
  radiant buildings, the top two complaints were "lack of thermostat
  control" and "slow system response" [8, p. 15].
- Sensor lag or filter constants of the wall thermostat itself: not found.

## 2. What they move, and in which mode

- **Supply water temperature via a mixing valve, by heating curve.** The
  heating curve maps outdoor temperature to flow temperature [11, p. 2].
  SwissEnergy guidance flow temperatures at -8 °C outdoor: underfloor
  heating 30 to 35 °C (built after 2010), 35 to 50 °C (before 1990);
  radiators 35 to 40 °C (after 2010) up to 60 to 70 °C (before 1980)
  [11, p. 3]. Madrid: maximum heating supply 49 °C, minimum cooling supply
  18 °C, by linear interpolation between two outdoor temperatures
  [14, pp. 12 and 15, Table 1, Fig. 15]. Münster: TABS supply 16 to 30 °C
  [13, "Technical concept"]. Rehau computes the heat curve from design flow
  and return temperatures at a design outdoor temperature of -20 to -5 °C
  [12, p. 7].
- **Weather compensation versus room feedback.** Siemens RVS61 controllers
  offer pure weather compensation, weather compensation with 1 to 99% room
  influence (about 60% for a good reference room, about 20% for a poor one),
  and 100% pure room compensation; thermostatic valves in the reference room
  must be fully open [16, p. 80]. Rehau TABS control combines a PI room
  controller and a P core controller, weighted 1:1 by default [12, p. 7].
- **Pump on/off.** Siemens TABS control can run the zone circulation pump
  intermittently by pulse width modulation over a period, placing on-times in
  cheap or efficient hours; in simulation this cut pump on-time to 47% but
  raised heating and cooling energy [9, pp. 3 and 7, Fig. 6]. The CBE
  controller is an on/off valve on slab temperature with a lockout window,
  for example off 14:00 to 02:00 (simulation) [6, pp. 4-5].
- **Zone valves and room-by-room control.** German GEG §63 requires
  room-by-room control; Rehau states that concrete core TABS alone cannot do
  it and needs a faster system added; near-surface TABS can, for heating
  loads up to 30 W/m2 [12, p. 6]. Münster adds radiators "for additional room
  by room heating in the morning hours" [13, "Introduction"].
- **Heat source.** Reversible ground source heat pumps in Münster (530 kW
  heating) and Madrid [13], [14, p. 15]. The Ithaca homes use condensing
  boilers on outdoor reset ("modulating" is not stated in the source) [15].
- **Modes (Guideline 36, air systems).** Occupied, warm-up, cool-down,
  setback, setup, unoccupied [1], [2, 5.3.2.3]. Default zone setpoints (VAV):
  occupied heating 21 °C (70 °F), cooling 24 °C (75 °F); unoccupied heating
  16 °C (60 °F), cooling 32 °C (90 °F) [3, p. 3, Table 3.1.1.1-1].
- **Modes (TABS practice).** Most TABS buildings hold a near constant slab
  setpoint round the clock, reset slowly by season or recent outdoor
  temperature; only a few allow setback, and small [7, p. 7]. Madrid runs a
  day setpoint and a night setpoint (00:00 to 07:00); in heating the night
  setpoint is higher, to store heat for the morning on the night tariff
  [14, p. 14, Figs. 11 and 12]. Rehau "CCT charge control" charges the core
  outside use hours [12, p. 7].
- **Heating limit (summer cut-off).** SwissEnergy guidance values for a 20 °C
  room: 15 to 17 °C outdoor for uninsulated pre-1977 buildings, 12 to 15 °C
  for 1995 to 2010, 8 to 10 °C for passive houses [11, p. 3]. Siemens TABS
  simulation design: heating limit 13.1 °C [9, p. 5].
- **Frost protection.** Guideline 36 (as implemented by LBL): any zone below
  4.4 °C (40 °F) puts the zone group in setback until all zones exceed
  7.2 °C (45 °F), with a level 3 alarm [1]; Addendum e removes the separate
  freeze protection mode because setback already covers cold zones
  [2, p. 3, item 4]. Rehau industrial floors hold a base of about +15 °C in
  halls [12, p. 6]. Pipe or water freeze limits: not found.

## 3. Limits, and what kind each is

| Limit | Value | Kind | Source |
|---|---|---|---|
| Floor surface, occupied area | 29 °C | design limit on heat flux (EN 1264) | [5, p. 23] |
| Floor surface, edge zone | 35 °C | design limit | [5, p. 23] |
| Floor surface, bathrooms | 33 °C | design limit | [5, p. 23] |
| Floor surface, comfort, feet on floor | 19 to 29 °C | comfort requirement (ASHRAE 55) | [4] |
| Supply water at heating elements, cement or calcium sulphate screed | 55 °C (45 °C asphalt) | hard limit, safety device must cut hot water even without power | [17] |
| Design flow temperature for one Uponor floor build-up | above 55.5 °C the 29 °C limit is exceeded | design limit | [5, p. 27] |
| Cooling water flow | not below 15 to 16 °C | condensation limit | [5, p. 31] |
| Cooling floor surface | about 20 °C minimum | comfort limit | [5, p. 31] |
| Condensation (cooling) | dew point detector cuts cold water | trip | [17]; moisture switches on the supply pipe disable chilled water [7, p. 8] |
| Operative temperature change, no occupant control | 1.1 °C in 15 min; 2.2 °C in 1 h | comfort limit, timed (ASHRAE 55) | [4] |
| Vertical air gradient, ankle to head | 3 °C seated, 4 °C standing | comfort limit | [4] |
| Room setpoints (TABS design, simulation) | 21 °C heating and 26 °C cooling, used all year; the paper says room temperatures stay in the comfort range | comfort band | [9, p. 5] |
| Comfort band (CBE simulation) | 22.3 to 26 °C operative (PMV -0.5 to +0.5) | comfort band | [6, p. 3] |
| Field deadbands, 8 radiant buildings | 2.8 to 5.6 °C between heating and cooling air setpoints; one at 2.2 °C had poor comfort | operating band | [8, pp. 20-21] |
| Heating to cooling changeover | slab lockout 2 to 24 h | timed limit | [7, p. 9] |
| Warm-up | optimal start no earlier than 3 h before occupancy, required warm-up time under 3 h | timed limit | [1], [2, p. 5] |
| Setback entry | zone below unoccupied heating setpoint for 5 min | timed trigger | [2, p. 5, 5.3.7.4] |
| Energy | about 1% heating saved per 1 °F of 8 h night setback | cost | [15] |

The ASHRAE 55 values at 0.5, 2 and 4 h: not confirmed from a readable source
(see the list for the owner).

## 4. Time scales

- **Slab time constant.** Concrete core TABS (type E): 145 to 462 min in
  heating, 146 to 419 min in cooling; embedded floor systems (types A, B, C,
  G): 16 to 161 min heating; near-surface TABS (type F): 7.5 to 22 min;
  radiant ceiling panels 38 to 91 s in heating (CFD, 63.2% step response,
  66 cases) [10, p. 7, Fig. 3]. Earlier work cited there: 30 min to 2 h for
  light floors, 3 to 6 h for heavy floors [10, p. 3]. CBE: TABS time constant
  "typically 4 to 8 hours" [6, p. 2].
- **Room response to a setpoint change.** In TABS and embedded systems it can
  take one hour to several hours before a setpoint change is felt [8, p. 15].
- **Correction horizon.** "Since TABS react slowly, only a day-to-day
  compensation is promising" [9, p. 3]; room feedback output is
  limited to about -5 to +5 K [9, part 2]. The CBE controller spreads its
  daily slab setpoint change over the next 12 h [6, pp. 4-5].
- **Heating curve tuning.** After a curve change, observe room temperature for
  two weeks [11, p. 1]; to check the setpoints, log room temperature for 2 to 3
  days with thermostat valves fully open [11, p. 4].
  Rule of thumb: lowering the curve 2 °C lowers the room 2 °C for underfloor
  heating, while lowering it 5 °C lowers the room 2.5 °C for radiators
  [11, p. 2].
- **Setback recovery (contrast, baseboard homes, Ithaca NY).** Recovery times
  in the case study table: 1.0 h (old, uninsulated, 6600 Btu/°F), 3.4 h
  (2009 IECC, 8131 Btu/°F), 6.9 h (super insulated, 10 685 Btu/°F), with
  boilers sized at twice the design load [15]. The same study advises no
  setback for radiant floors, uninsulated interior concrete, or occupants
  expecting recovery under 1 hour [15].
- **Daily cycle.** Office occupancy 8:00 to 18:00 in the CBE model [6, p. 3];
  Madrid night setpoint 00:00 to 07:00 [14, p. 14]; the CBE controller allows
  one mode per day [6, pp. 4-5].
- **Seasonal.** TABS slab setpoints change slowly over the seasons, tuned over
  the first few seasons of operation [7, p. 7]; changeover can happen in days
  or hours depending on weather [7, p. 9].
- Measured slab and room temperature step response from a real building:
  not found in a readable source.

## 5. A normal day and a bad day

- **Normal day (TABS office).** The slab holds a near constant setpoint set by
  the 24 h outdoor mean; internal and solar gains during occupancy are absorbed
  by self-regulation, because room and slab temperatures sit close and heat
  flow falls as the room warms [7, p. 7], [9, p. 7]. Madrid charges the floors
  at night on the cheap tariff and lets them release heat by day [14, p. 14].
  Münster adds radiators for morning heating [13].
- **Bad day: morning start and mode change.** Rehau: high thermal mass "could
  lead to overshooting of room temperature, especially when the system is
  started or there is a change from reduced to normal mode" [12, p. 6]. One
  designer abandoned load shifting "to avoid the risk of morning discomfort"
  [7, p. 7].
- **Bad day: low gains.** Siemens TABS control lowers the cooling setpoint
  correction on weekends and other days with fewer internal gains
  [9, part 2]; comfort holds only while real gains stay inside the bounds
  assumed at design (unknown but bounded method) [9, part 1].
- **Bad day: sun and gains above design.** The CBE study left out shading on
  purpose to produce intra-day load swings (simulation) [6, pp. 4-5]; field
  buildings that perform well use sun shading [8, pp. 20-21]. A building with a
  small 2.2 °C deadband and supplemental air cooling appeared to overcool
  [8, pp. 20-21].
- **Bad day: cold snap.** A boiler sized to the load on outdoor reset "may
  never recover from setback", especially at design conditions [15].
  SwissEnergy: rooms facing out or north, top floors and corners are the hard
  rooms; tune the curve below 0 °C and again above 10 °C [11, p. 1].
- Measured overheating on sunny days in a slab building with numbers:
  not found.

## 6. The operator's job in one sentence

Choose the water temperature (or slab setpoint) about a day ahead from the
weather, so the room stays inside its comfort band through the coming
occupied hours without heating and cooling on the same day, because the slab
cannot correct within the day [9, p. 3], [7, p. 7].

Added 2026-11-08 for the frequency-weighted headline: no source in this
record gives how many weeks of a heating season are cold snaps, sunny spells
or low-occupancy weeks; the weights are ours (PHYSICS.md).

## For the owner to open by hand

- ASHRAE Standard 55-2023, section on temperature variation with time (full
  drift and ramp table): https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy
- ASHRAE Guideline 36-2021 full text (only Addenda e and v were readable):
  https://www.ashrae.org/technical-resources/bookstore/guideline-36
- EN 1264-1 to -5 and ISO 11855 (surface and water limits, quoted here only
  through Uponor and Wavin): https://www.iso.org/standard/77627.html
- EN 15232-1 / ISO 52120-1 (control function classes): not readable,
  https://www.iso.org/standard/65883.html
- Gwerder et al., Applied Energy 85 (2008) 565-581 and 86 (2009) 1606-1616:
  https://doi.org/10.1016/j.apenergy.2007.08.001
- Olesen, "Radiant floor heating in theory and practice", ASHRAE Journal
  2002 (copy at scispace returned no text).
- Favero et al., Building and Environment (Table 1 drift limits), 403 Forbidden:
  https://irinsubria.uninsubria.it/retrieve/55d117d7-5d2e-48d5-adf7-63e9134f7784/
- Ning et al. 2017, Energy and Buildings 137 (published τ95 scheme), 403 at
  structurae; the 2015 preprint [10] was used instead.

## Sources

1. LBNL Modelica Buildings library, Buildings.Controls.OBC.ASHRAE.G36_PR1.Generic.SetPoints (OperationMode), implementing G36 Part 5.C.6. https://simulationresearch.lbl.gov/modelica/releases/v7.0.0/help/Buildings_Controls_OBC_ASHRAE_G36_PR1_Generic_SetPoints.html
2. ASHRAE Addendum e to Guideline 36-2021 (2024), sections 5.3.2.3, 5.3.7, 5.4.6. https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/g36_2021_e_20240229.pdf
3. ASHRAE Addendum v to Guideline 36-2021 (2024), Table 3.1.1.1-1. https://www.ashrae.org/file%20library/technical%20resources/standards%20and%20guidelines/standards%20addenda/g36_2021_v_20240229.pdf (re-confirmed 2026-11-08)
4. Wikipedia, "ASHRAE 55" (summary of Standard 55-2013). https://en.wikipedia.org/wiki/ASHRAE_55
5. Uponor, "Planning principles, underfloor heating and cooling", TI 1186660 v1, pp. 23, 27, 31. https://brandportal.uponor.com/m/7187327a41365ace/original/TI-planning-principles-UFHC-EN-1186660-v1.pdf (re-confirmed 2026-11-08; 55.5 °C is the source's "ϑV,des > 55,5 °C" footnote on p. 27)
6. Raftery, Duarte, Schiavon, Bauman, "A new control strategy for high thermal mass radiant systems", Building Simulation 2017, CBE (simulation). https://escholarship.org/uc/item/5tz4n92b
7. Paliaga, Farahmand, Woolley, "Current practice for design and control of high thermal mass radiant cooling systems", ACEEE Summer Study 2018 (11 designer interviews). https://escholarship.org/uc/item/4qg9276c (re-confirmed 2026-11-08)
8. Dawe, Karmann, Schiavon et al., "Field evaluation of thermal and acoustical comfort in eight North-American buildings using embedded radiant systems", PLOS ONE 16(10), 2021. https://escholarship.org/uc/item/24k6q5zg (re-confirmed 2026-11-08)
9. Gwerder, Tödtli, Lehmann, Renggli, Dorer, "Control of thermally activated building systems", CLIMA 2007, Helsinki (Siemens, Empa; simulation). https://sid.siemens.com/api/khub/documents/nqXZtSuC0od3uV1aZJ5VhA/content (re-confirmed 2026-11-08, with two wording fixes in the body: the quote on day-to-day compensation, and 21/26 °C are room setpoints rather than a stated comfort band)
10. Ning, Schiavon, Bauman, "A classification scheme for radiant systems based on thermal time constant", ISHVAC-COBEE 2015. https://escholarship.org/uc/item/1sx88662 (re-confirmed 2026-11-08)
11. SwissEnergy (Swiss Federal Office of Energy), "Operational optimisation, heating: set the heating curve". https://pubdb.bfe.admin.ch/en/publication/download/9982 (re-confirmed 2026-11-08, except that the 2 to 3 day logging on p. 4 is for checking room temperature with thermostat valves open, not specifically night setback; body corrected)
12. Rehau, "NEA SMART 2.0 control system, application TABS", 04/2023, pp. 6, 7, 12. https://www.rehau.com/downloads/1084944/954663-en-06-2022.pdf (re-confirmed 2026-11-08)
13. IEA ECBCS Annex 48, German case study 1, LVM office building, Münster. https://nex75.iea-ebc.org/Data/publications/EBC_Annex_48_Case_Study_G1.pdf (re-confirmed 2026-11-08)
14. Laera et al., "Control strategies, monitoring and management for the efficient behavior of TABS", Building & Management 3(1) 2019, pp. 7-23 (Apolonio Morales 29, Madrid, BMS screens). https://polired.upm.es/index.php/building_management/article/download/3873/3966/14454 (re-confirmed 2026-11-08)
15. US DOE Building America case study, "Hydronic systems: designing for setback operation", Ithaca NY, DOE/GO-102014-4370, 2014. https://www.energy.gov/sites/prod/files/2014/06/f16/ba_case_study_hydronic_setback.pdf (re-confirmed 2026-11-08, except that "modulating" boilers is not stated; body corrected)
16. Siemens RVS61.843 / Albatros2 user manual, "Room influence", p. 80. https://manualslib.com/manual/3049546/Siemens-Albatros2.html?page=80
17. Wavin, "Underfloor radiant systems, product guide", p. 15. https://pinaxo.com/public/e6d311ee955bda8cd8ef2d7b748cea6f/wavin-underfloor-radiant-systems/15

Added by the Amendment 3 fix session, 2026-11-08 (each fetched or read once):

18. Brine-to-water heat pump datasheets, EN 14511: Dimplex SI 90TU, COP 5.00
    at B0/W35, 3.80 at B0/W45, 3.10 at B0/W55 (compressor 1); Wolf BWS-1, 4.6
    to 4.7 at B0/W35 and 2.8 to 2.9 at B0/W55; Vaillant geoTHERM 22 to 46 kW,
    4.4 to 4.5 and 3.1 to 3.2; CTA Optiheat Inverta, 4.7 / 3.5 / 2.7. Read from
    a search summary of the datasheets, not each PDF.
    https://www.dimplex.co.uk/sites/g/files/emiian551/files/2024-03/Technical_data_SI90TU_Issue_0.pdf
19. Spanish 2.0TD electricity offer (retailer listing on roams.es and
    comparadorluz.com): punta 0.263, llano 0.184, valle 0.139 EUR/kWh; valle
    00:00 to 08:00 on weekdays, punta 10:00 to 14:00 and 18:00 to 22:00. Read
    from a search summary. https://roams.es/energia/luz/serosense/tarifa-dh-2-0-td-serosense/
20. LBNL Indoor Air Quality Scientific Findings, "Temperature and office work
    performance": performance peaks near 71 F (21.7 C) and falls above and
    below it (fetched once). Seppanen and Fisk, ASHRAE Transactions 2005: a 2 %
    decrement in work performance per degree C above 25 C (read from a search
    summary). https://iaqscience.lbl.gov/temperature-and-office-work-performance
21. Eurostat news release, 27 March 2024: EU average hourly labour cost EUR
    31.8 in 2023. https://ec.europa.eu/eurostat/web/products-eurostat-news/w/ddn-20240327-2
    (read from a search summary).

Re-confirmation (2026-11-08): every source above was fetched twice. Sources
1, 4, 16 and 17 (web pages) were re-read and confirmed on the second fetch.
Sources 2 to 15 are PDFs read by text extraction; the second fetch of each
again contained the cited passages, but for 3, 5, 7 to 15 that second fetch
may have been served from the fetch tool's 15 minute cache. Those eleven were
then fetched a third time with a query string and the PDFs read by text
extraction, with the results noted beside each source. Sources 2 and
6 were downloaded fresh both times.
