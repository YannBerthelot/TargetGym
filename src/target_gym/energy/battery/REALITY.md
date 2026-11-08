# Grid battery (BESS): plant reality record

Plant: a grid-scale lithium-ion battery energy storage system dispatched against
grid signals or prices. The property of interest is a state of charge (SoC) that
nobody measures directly: the battery management system (BMS) estimates it.

Every item below names its source, with the tag used in the Sources list at the
end. All sources were fetched twice on 2026-11-08 and said the same thing both
times, except where marked otherwise. Numbers are the sources' own.

## 1. What the operator, plant controller or EMS reads

- **Dispatch or regulation signal, every 2 s (PJM).** PJM sends a single fleet
  regulation signal (RegA, in +/- MW) "on a two-second scan rate". It sends the
  assigned regulation MW (AReg) every 10 s. [PJM-M12, sec. 4.3.2, p. 39]
- **PJM no longer has a separate fast signal (RegD).** Manual 12 revision 56
  (1 Oct 2025) "Removed reference to two types of Regulation signals and RegD"
  and removed the accuracy, delay and precision score terms. [PJM-M12, Revision
  History, p. 113]
- **Feedback the plant sends back to PJM, every 2 s.** The owner sends Total
  Regulation (TReg, MW of capability) and Current Regulation (CReg, MW position).
  Both are "calculated every two seconds and sent on a two-second scan rate".
  TReg must drop as soon as capability falls. [PJM-M12, sec. 4.3.2, p. 40]
- **AGC set-points, every 4 s (Australia).** AEMO's AGC can send Hornsdale "a
  new MW set-point ... at a rate of up to once every four seconds". [AEMO-HPR,
  sec. 2.1, p. 5]
- **AGC instructions, every 4 s (California).** In CAISO, a storage resource with
  a regulation award receives "4-second automatic generator control (AGC)
  instructions". The exact SoC effect of this is "uncertain". [CAISO-SOC, sec. 3,
  p. 4-5]
- **What a GB provider must log for NESO.** Data is logged at 20 Hz, or at 2 Hz
  for Dynamic Regulation when it is not stacked, and sent hourly. It holds the
  time stamp, input frequency, availability, metered active power, the
  Performance Baseline, armed state, and, for energy-limited units, "its State of
  Energy in MWh". [NESO-ST, para. 15.3-15.4, p. 16-17]
- **NESO checks SoC from that same data.** It uses the reported state of energy
  columns (soe_export_mwh, soe_import_mwh) at the start of each settlement
  period. [NESO-SOE, p. 4 and 6]
- **Test data rates.** The Sandia and PNNL protocol finds data "every second"
  necessary for frequency regulation and every 10 s enough for peak shaving or
  arbitrage. It says sampling should be "at least double the rate of the signal".
  [PNNL-ESS, Foreword, p. iii-iv]
- **Grid frequency.** NESO keeps frequency within 50 Hz +/- 1% and sets a service
  deadband of +/- 0.015 Hz. [NESO-WEB; NESO-ST, Schedule 1 table, p. 33]
- **Cell voltage and temperature limits.** The EVE LF280K grid LFP cell
  (3.2 V, 280 Ah) works between 2.5 V and 3.65 V. It may discharge from -20 to
  55 °C and charge only from 0 to 55 °C. [EVE-LF280K, items 4 and 8, p. 3-4]
- **How the SoC estimate is made, and why it drifts.**
  - Current integration (coulomb counting) "cannot correct initial estimation
    errors and tends to accumulate errors over time". [ARX-LFP, sec. I, p. 1]
  - LFP has a flat plateau over roughly SOC 20% to 80%. There, "a 1 mV OSC
    [open circuit voltage curve] error" can lead to "nearly a 5% SOC error".
    [ARX-LFP, sec. I and eq. 5, p. 1-2]
  - Temperature moves the curve. On the second plateau, the same OCV maps to SoC
    values 12.89% apart between 50 °C and 25 °C, and 17.42% apart between 25 °C
    and -10 °C. With the 25 °C curve, the estimate drifts by about +0.51% per °C
    of heating. [ARX-LFP, Fig. 5 and text, sec. IV]
- **Measured SoC error at real sites (vendor data).**
  - ACCURE reports BMS SoC errors "up to ±15%" across more than 100 operating
    BESS sites, with some outliers above 40%.
  - Its Figure 1 ("50 MWh+ LFP BESS") shows a daily gap of over 12% between the
    BMS SoC and the corrected SoC.
  - Plain Ah counting is illustrated as reaching "10% SOC error after 14 days".
  - The author works for ACCURE and the piece promotes its product. [MPS-ACCURE]
- **Delay.** PJM's performance score "allows a 10 second latency for signal
  propagation delay". [PJM-M12, sec. 4.4.5, p. 48]
- **Delay of the SoC telemetry itself:** not found.

## 2. What they move, and in which mode

- **Megapack control modes.** Megapack lists Direct Real Power, Direct Reactive
  Power, Ramp Rate Control, Site Control, Frequency Support, Power Factor
  Control, Virtual Inertia and Voltage Control. It talks Modbus TCP, DNP3 and
  REST API. [TESLA-MP2XL, Controls and Communications]
- **Regulation under AGC.** Hornsdale normally runs under AEMO's AGC. Up to
  30 MW of it is used for regulation FCAS, and 70 MW of discharge is reserved for
  the South Australian government. [AEMO-HPR, sec. 1-2.1, p. 3-5]
- **Contingency response by droop.** Hornsdale provides contingency response "at
  all times, irrespective of FCAS market outcomes". [AEMO-HPR, sec. 2.2, p. 7]
  It uses a deadband of 50 +/- 0.15 Hz and a 1.72% droop, with a lab response
  time of about 100 ms. [AUR-Y1, p. 5 and the modelling notes]
- **GB dynamic services (DC, DM, DR).** Power follows a frequency-response curve
  around an Operational Baseline that the provider declares.
  - SoC is managed only "by submission of updated Operational Baselines for
    either charging or discharging". [NESO-ST, para. 6.11(v), p. 10]
  - Baselines may not change while frequency is inside the deadband. [NESO-ST,
    para. 6.11(vii), p. 10]
  - SoC "shall not be managed by over and/or under delivery including
    'spilling'". [NESO-ST, para. 6.13, p. 10]
- **Capacity held back for SoC recovery (GB).** In the opposite direction a unit
  must reserve 10% of the offered quantity for DC, 20% for DM and 40% for DR.
  [NESO-SOE, sec. 3, p. 8-11]
- **Continental Europe FCR.** A limited-reservoir provider must keep, "at any
  point during normal state", enough energy to stay available for the minimum
  time once the alert state starts. [ENTSOE-LER, definitions and sec. 1, p. 3-5]
- **Arbitrage and market scheduling (CAISO).** The market models
  SoC(t) = SoC(t-1) - (P_dis + eta * P_ch), with a "typical round trip efficiency
  might be around 85%". [CAISO-SOC, eq. 1, p. 4] CAISO proposes that only a
  fraction mu of a regulation award enters the next interval's SoC. [CAISO-SOC,
  sec. 3, p. 5]
- **Peak shaving duty cycle.** Defined in PNNL-ESS, Appendix A. Not read in
  detail here.

## 3. Limits: trip, timed limit or cost

- **Power and energy ratings.**
  - Megapack 2 XL: the 2-hour unit is 1927 kW / 3854 kWh, the 4-hour unit is
    979 kW / 3916 kWh, at 25 °C. [TESLA-MP2XL, Power and Energy]
  - Hornsdale: 100 MW discharge, 80 MW charge, 129 MWh, "approximately 75 minutes
    at full discharge". Of that, 119 MWh was usable for Neoen's market operation.
    [AEMO-HPR, sec. 1, p. 3]
- **Cell voltage, 2.5 to 3.65 V.** A hard limit. The datasheet gives the range
  but not the BMS trip action. [EVE-LF280K]
- **Cell temperature.** Charging is allowed only from 0 to 55 °C. A hard limit.
  [EVE-LF280K]
- **Container ambient.** Megapack operates at -30 to 50 °C ambient. The datasheet
  does not say what derating applies near the edges. [TESLA-MP2XL]
- **SoC window, GB: a timed limit with a payment cost.**
  - At the start of each 4-hour contract block (EFA), the unit must hold the
    Contracted Response Energy Volume: the contracted MW times 15 min (DC),
    30 min (DM) or 60 min (DR).
  - Energy Recovery is 0.2 x that volume per 30-min settlement period.
    [NESO-SOE, Table 2, p. 7; NESO-GUIDE, p. 5]
  - A unit whose reported SoC falls below the minimum may be "deemed
    unavailable" until NESO is satisfied. [NESO-ST, para. 6.12, p. 10]
- **Unavailability costs the whole settlement period (GB).** The availability
  factor is zero if a unit is unavailable for 0.1% or more of a settlement
  period. Payment is also scaled by a K factor from the worst performance score
  in the block. [NESO-ST, Schedule 3, p. 49-53]
- **Minimum energy time, Continental Europe (a timed requirement).** TSOs
  proposed Option D: "TminLER = 30 minutes for all LER", with an interim period
  for existing 15-minute units. [ENTSOE-LER, sec. 6, p. 41-42]
- **PJM performance: a qualification limit and a cost.**
  - To qualify, a unit needs two consecutive tests of 75% or more. Each test runs
    40 minutes. [PJM-M12, sec. 4.4.1-4.4.2, p. 43-44]
  - If the rolling score over 200 half-hour intervals falls below 40%, the unit
    is barred from the market. [PJM-M12, sec. 4.4.4, p. 47]
  - A self de-assignment scores zero for the rest of the interval. [PJM-M12,
    sec. 4.4.9]
- **Ramp rates.**
  - GB services must start responding within 0.5 s (DC and DM) or 2 s (DR), and
    reach full delivery within 1 s (DC and DM) or 10 s (DR). [NESO-WEB, technical
    requirements table]
  - Megapack offers a ramp-rate control mode but gives no number. [TESLA-MP2XL]
- **Degradation.**
  - The EVE LF280K is rated for 6000 cycles or more at 0.5C/0.5C and 25 °C, to
    80% of nominal capacity. [EVE-LF280K, item 10]
  - NREL's 2025 ATB assumes 85% round-trip efficiency, "approximately one cycle
    per day" and a 15-year life. Degradation "is a function of the usage rate",
    and augmentation is paid from fixed O&M (4% of capital cost per year).
    [NREL-ATB-2025]
  - Megapack round-trip efficiency is 92.0% (2-hour) and 93.7% (4-hour), for a
    full-depth cycle including auxiliaries, at 25 °C. [TESLA-MP2XL]
- **Degradation cost in $ per cycle:** not found.

## 4. Time scales

- **Signal updates.** 2 s at PJM [PJM-M12, sec. 4.3.2]. 4 s for AGC at AEMO
  [AEMO-HPR, sec. 2.1] and at CAISO [CAISO-SOC, p. 4]. 20 Hz logging in GB
  [NESO-ST, para. 15.4].
- **PJM scoring.** Each sample covers 10 s, and the score is averaged per
  interval. The rolling history is the last 200 intervals of 30 minutes.
  [PJM-M12, sec. 4.4.4-4.4.5]
- **GB settlement.** A 30-minute settlement period inside a 4-hour EFA block.
  [NESO-SOE, Table 1; NESO-GUIDE, p. 5]
- **Response speed.** Hornsdale responds in about 100 ms, against the 6 s
  contingency requirement. [AUR-Y1, p. 5] In AEMO's Figure 3, measured on 18 Dec
  2017, Hornsdale went from about 5 MW to about 26 MW within a few seconds as
  frequency fell from about 49.97 Hz to about 49.77 Hz. [AEMO-HPR, Fig. 3, p. 7]
- **Storage duration.**
  - Megapack comes in 2-hour and 4-hour configurations. [TESLA-MP2XL]
  - The ATB covers 2, 4, 6, 8 and 10 hours, with 4 hours as the default.
    [NREL-ATB-2025]
  - Hornsdale holds about 75 minutes at full power. [AEMO-HPR]
- **Energy the services require.** DC 15 min, DM 30 min, DR 60 min [NESO-WEB].
  Continental Europe FCR 15 to 30 min [ENTSOE-LER].
- **Energy neutrality of regulation.** In PJM's 2011-2012 signal, "24-hour
  signals were energy neutral". [PNNL-ESS, Appendix B, p. B.1] The PNNL 24-hour
  test cycle in Figure 5.3.2 swings between -1 and +1 p.u. throughout. At the end
  of the test the battery "shall be brought back to its initial SOC". [PNNL-ESS,
  sec. 5.3.2, p. 20]

## 5. A normal day and a bad day

- **Normal day: regulation.** In AEMO's Figure 2, Hornsdale's measured MW lies
  almost on top of the AGC set-point for one hour, moving between about -18 MW
  and +30 MW. In Figure 1, a large steam unit misses several set-point swings.
  [AEMO-HPR, Fig. 1-2, p. 6]
- **Normal day: GB energy accounting.**
  - Example from the service terms: a unit with a 50 MWh contracted volume that
    exports 2 MWh in settlement period 1 must still hold 48 MWh at the start of
    period 2.
  - The 2 MWh is credited back in later periods. Each period may recover no more
    than 20% of the contracted volume.
  - [NESO-ST, para. 6.11(iii), p. 9]
- **Bad day: frequency stays off nominal for a long time.**
  - Long-lasting one-way deviations (average of 50 mHz or more) are "the most
    impacting elements leading to the possibility of LER exhaustion".
    [ENTSOE-LER, sec. 1, p. 5]
  - In GB, NESO may waive a failed SoC check after "extended periods" of
    deviation beyond 0.1 Hz, or after several events in a row. [NESO-ST,
    para. 6.11(vi), p. 10]
  - NESO's guidance says the provider may declare itself unavailable, without
    penalty, once the minimum drops to zero beyond 0.2 Hz. [NESO-SOE, p. 4]
- **Bad day: the grid splits.** On 16 Nov 2019, South Australia was cut off from
  the rest of the grid for about 5 hours. The study's case study and Appendix B
  describe Hornsdale's response. [AUR-Y2, p. 10-13]
- **Bad day: wrong SoC estimate.**
  - At Dogfish (ERCOT, 75 MW, 1 hour), 3.8 MWh of tradable energy sat idle
    during a $267/MWh price peak.
  - An unnamed UK operator lost 7-10% of revenue over two months to frequency
    service penalties, which the article attributes to inaccurate energy
    estimation.
  - Both are vendor reports. [MPS-ACCURE]
- **Bad day: heat and safety systems.**
  - At Moss Landing Phase I (300 MW / 1,200 MWh), on 4 Sep 2021, the smoke
    detector triggered water release at below-design smoke levels, after "an
    apparent programming error".
  - Failed hoses sprayed water onto the racks, and about 7% of modules and other
    systems were damaged.
  - Module temperatures were "within established temperature limits" when the
    water was first released.
  - This is a trade-press summary of Vistra's findings. [PE-VISTRA]
- **Measured SoC traces over a day at a named grid site, from a primary
  source:** not found.

## 6. The operator's job in one sentence

The operator delivers the contracted power profile to the second and keeps the
estimated stored energy inside the window the market rules require for the
whole contract block. The only tool for this is declared baseline or schedule
changes, never under-delivery. [NESO-ST, para. 6.11-6.13; PJM-M12, sec. 4.4.5]

## For the owner to open by hand

- **ERCOT:** the PUCT remand of NPRR1186 (SoC awareness, accounting and
  monitoring). ercot.com returned a bot-check page.
  https://www.ercot.com/files/docs/2024/02/19/6.1%20PUCT%20Remand%20of%20NPRR1186_Improvements%20Prior%20to%20the%20RTC%20B%20Project%20for%20Better%20ESR%20State%20of%20Charge%20Awareness_Accounting%20and%20Monitoring.pdf
- **CAISO:** the Regulation Energy Management straw proposal. The download came
  back as an empty PDF.
  https://www.caiso.com/Documents/Presentation-RegulationEnergyManagementStrawProposal-ISOPresentation.pdf
- **NREL:** Cole et al., Cost Projections for Utility-Scale Battery Storage,
  2025 update. docs.nrel.gov did not resolve.
  https://docs.nrel.gov/docs/fy25osti/93281.pdf
- **Photos or videos of real BESS control screens:** none found to cite.

## 7. Prices (added for Amendment 3, fetched once, 2026-11-08)

- **GB day-ahead prices.** Mean GBP 74.0/MWh in 2024 and GBP 81.2 in 2025. The
  highest hourly price was GBP 491.0 (12 Dec 2024, 16:00) and GBP 958.5 (22 Jan
  2025, 17:00); the lowest GBP -28.2 (13 Apr 2024, 13:00) and GBP -30.5 (25 May
  2025, 11:00). 141 hours were negative in 2024 and 190 in 2025.
  [EMAPS-GB]
- **Dynamic Regulation's availability price.** Averaged GBP 4.36/MW/h in
  January 2024; Dynamic Regulation High fell from GBP -4.19 to -5.87/MW/h in
  the same month. A vendor's market review. [MODO-JAN24]
- **Price spike at a named site.** Dogfish's $267/MWh peak (sec. 5,
  MPS-ACCURE).

Sources for this section:

- **EMAPS-GB:** Electricity Maps, Electricity Grid Review 2025: Great Britain,
  day-ahead price table. https://www.electricitymaps.com/grid-in-review-2025/great-britain
- **MODO-JAN24:** Modo Energy, Balancing Mechanism revenues overtake frequency
  response in January 2024, 4 Jan 2024.
  https://modoenergy.com/research/battery-energy-storage-revenue-janauary-2024-balancing-mechanism-wholesale-frequency-response-stack-capacity-market-triads

## Added by the fix session (fetched once, 2026-11-08)

These were read once, by text extraction of the PDFs, and not re-fetched.

- **Gate closure (GB).** Physical Notifications serve as the Operational
  Baseline "for an upcoming Settlement Period". Gate Closure is "the time 60
  minutes prior to the start of the Settlement Period", after which the PN
  "can no longer be submitted or changed". A PN has at lowest 1 minute
  resolution and may change within the period. [NESO-GUIDE, p. 36]
- **SoE management guidance (GB).** Begin the EFA block with stored energy
  adequate for the contracted quantity, review it at the start of each
  settlement period, and "return the stored energy to an appropriate level by
  following the submitted operational baselines". [NESO-GUIDE, p. 37, items
  1 to 4]
- **DR response curve.** DR "provides a constant power response across the
  operational frequency range, reaching full delivery at a ±0.2 Hz frequency
  deviation". [NESO-GUIDE, p. 4]
- **Cell resistance.** EVE LF280K AC internal resistance at most 0.25 mOhm at
  1 kHz. [EVE-LF280K, item 5]
- **Cycle-life test conditions.** 25 +- 2 degC, 0.5C constant current and
  voltage charge to 3.65 V with a 0.05C cut-off, 30 min rest, 0.5C discharge to
  2.5 V, 30 min rest, repeated. [EVE-LF280K, item 10]

## Sources

- **PJM-M12:** PJM Manual 12, Balancing Operations, Rev. 57, effective
  22 Apr 2026. https://www.pjm.com/-/media/documents/manuals/m12.ashx
- **NESO-ST:** NESO, Response Services Service Terms, v6, effective 31 Jul 2026.
  https://www.neso.energy/document/384606/download
- **NESO-SOE:** NESO, SOE Monitoring Guidance for Energy Limited DC/DM/DR
  Providers, v2, Mar 2025. https://www.neso.energy/document/347241/download
- **NESO-GUIDE:** NESO, Dynamic Response Services Provider Guidance, Jun 2026.
  https://www.neso.energy/document/276606/download
- **NESO-WEB:** NESO, Dynamic Services (DC/DM/DR) web page, technical
  requirements table.
  https://www.neso.energy/industry-information/balancing-services/frequency-response-services/dynamic-services-dcdmdr
- **ENTSOE-LER:** ENTSO-E, Explanatory document to all CE TSOs' proposal for the
  minimum activation time period for LER (SOGL Art. 156(11)), Jul 2021.
  https://consultations.entsoe.eu/system-operations/rgce-fcr-with-ler-sogl-156-11/supporting_documents/210723%20%20Explanatory%20document%20to%20all%20TSOs%20proposal%20for%20the%20definition%20of%20Time%20Period%20%20Clean.pdf
- **AEMO-HPR:** AEMO, Initial operation of the Hornsdale Power Reserve Battery
  Energy Storage System, Apr 2018, Fig. 1-3.
  https://www.aemo.com.au/-/media/Files/Media_Centre/2018/Initial-operation-of-the-Hornsdale-Power-Reserve.pdf
- **AUR-Y1:** Aurecon, Hornsdale Power Reserve Year 1 Technical and Market
  Impact Case Study, 2018.
  https://www.aurecongroup.com/-/media/files/downloads-library/thought-leadership/aurecon-hornsdale-power-reserve-impact-study-2018.pdf
- **AUR-Y2:** Aurecon, Hornsdale Power Reserve Impact Study Year 2.
  https://blythbattery.com.au/wp-content/uploads/2023/01/Aurecon-Hornsdale-Power-Reserve-Impact-Study-year-2.pdf
- **CAISO-SOC:** CAISO, Energy Storage Enhancements, State of Charge
  Implementation Update workshop paper, 2 Oct 2023.
  https://stakeholdercenter.caiso.com/InitiativeDocuments/Workshop-Paper-Energy-Storage-Enhancements-State-of-Charge-Implementation-Update-Oct-2-2023.pdf
- **PNNL-ESS:** PNNL and Sandia, Protocol for Uniformly Measuring and Expressing
  the Performance of Energy Storage Systems, PNNL-22010 Rev. 2, Apr 2016,
  Fig. 5.3.2 and App. B.
  https://www.sandia.gov/app/uploads/sites/163/2023/03/PNNL-22010Rev2.pdf
- **TESLA-MP2XL:** Tesla, Megapack 2 XL Datasheet, Rev. 1.5.1, 10 Feb 2023
  (copy filed with the Connecticut Siting Council).
  https://portal.ct.gov/-/media/csc/3_petitions-medialibrary/petitions_medialibrary/mediapetitionnos1601-1700/pe1607/petitionersubmissions/supplement-attachment-a---megapack_2_xl_datasheet.pdf
- **EVE-LF280K:** EVE Energy, LF280K Product Specification (B sample), copy
  hosted by a distributor.
  https://rcscomponents.kiev.ua/datasheets/LF280K_with_M6_threaded.pdf
- **ARX-LFP:** Paizulamu et al., LiFePO4 Battery SOC Estimation under OCV-SOC
  Curve Error Based on Adaptive Multi-Model Kalman Filter, arXiv:2410.23646,
  2024. https://arxiv.org/pdf/2410.23646
- **MPS-ACCURE:** D. Rüwald (ACCURE), SOC errors in LFP batteries: the $1 million
  per GWh challenge for BESS, Modern Power Systems, 6 Aug 2026.
  https://www.modernpowersystems.com/analysis/energy-storage/soc-errors-in-lfp-batteries-the-1-million-per-gwh-challenge-for-bess/
- **NREL-ATB-2025:** NREL Annual Technology Baseline 2025, Utility-Scale Battery
  Storage. https://atb.nlr.gov/electricity/2025/utility-scale_battery_storage
- **PE-VISTRA:** K. Clark, Vistra releases Moss Landing energy storage incident
  findings, Power Engineering, 25 Jan 2022. https://www.power-eng.com/?p=115477
