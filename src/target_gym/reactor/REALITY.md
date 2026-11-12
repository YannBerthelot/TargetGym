# PWR load follow: plant reality record

This record describes how pressurised-water reactors of the large Western family
(EDF's 900 to 1450 MWe units, Westinghouse 2 to 4-loop plants and their licensees)
follow load: what operators read, what they move, which limits they keep and what a
day of load follow looks like. It pools sources across the family (charter rule 19),
listed at the end as [S1] to [S17] (no S5). EDF practice is the frame; a Westinghouse figure
is used where EDF's was not found, and says so. Numbers read off a figure are
labelled "read from fig. N".

## 1. The job

Follow the grid's load plan through the day, keeping the average coolant temperature
on its program and the axial offset inside its band, at the least lost output and
wear. Rods are fast but tilt the power; boron barely tilts it but arrives late
[S1, annex II-3; S2, section 2.2].

## 2. What operators read

- **Power and delta I.** Ex-core detectors in two sections give power and delta I,
  the top minus bottom power over rated; the axial offset is delta I over power
  [S3, eqs. 2 and 3; S11].
- **Tavg and its program.** The rods' controller compares Tavg with a program in
  load [S6; S2, section 2.2].
- **Boron.** An on-line boronmeter, "provided after a significant delay, of the
  order of some twenty minutes", with noise "of the order of 7%" [S12]. A newer
  vendor meter claims under a minute [S13].
- **The plan.** The dispatch's load plan is known in advance [S2, section 1].
- **Xenon.** Operators plan dilution with a xenon tool run on the power history
  [S2, section 1]; the history is theirs to read.

## 3. What they move, and in which mode

- **Mode A (EDF, and the Westinghouse procedure).** One black group on Tavg; the
  operator "must manually adjust the boron concentration" to keep the rods from
  moving too far [S2, section 2.2]. The black rods move only "if |Δθ| > 1.6°C"
  [S1, annex II-5.1]. Westinghouse: rods on Tavg, "Flux difference limits should be
  maintained by boration" [S7].
- **Rod speed program (Westinghouse).** 1.5 F dead band, 8 to 72 steps/min [S6].
- **Boron.** Dilution about 10 kg/s and boration about 3 kg/s; 5 to 15 min from the
  command to the core [S2, section 2.2].
- **Mode G** (48 of EDF's 56 units) uses grey and black power-shimming groups plus an
  R group in a 27-step band [S14]. Not modelled; recorded in
  docs/proposals/reactor-coupling.md.

## 4. Limits

| limit | kind | form | sources |
|---|---|---|---|
| axial offset | penalty clock | within +-0.05 of the reference during load follow (EDF) [S1, annex II-4.3]; 1 h outside in 24 h, then a power reduction to under 50 % (Westinghouse: D.C. Cook and RAOC specifications, from search summaries) | S1, S4 |
| Tavg minimum for criticality | timed | 15 to 30 min to restore in the family's specifications (older custom specs 15 min, STS 3.4.2 30 min, from the reviewer's memory, not opened) | review |
| rod insertion limit | timed | straight lines in power; KAIST's lead gives 22 % of the bank in at full power to 78 % at half; 2 h to restore (STS 3.1.6, from memory) | S4, review |
| power, fuel, hot leg | trip | reactor protection | general |

## 5. Costs

- Lost production at $80 to $100/MWh on 1 GWe [owner's range].
- A scram's restart: Doel-3 ran 18 h at reduced power after a 6 h scram [S15];
  APR1400 restart windows run 22 to 70 h with dilution rate and burnup [S16].
- Rod wear, effluent treatment: no family figure found; the task's prices are ours
  and their weight on the room is reported (docs/proposals/reactor-axial.md).

## 6. A day of load follow

| shape | ramp | hold | source |
|---|---|---|---|
| 100-50-100 | 2 %/min (Mode A 1 to 2 %/min at the start of a cycle) | plateaus of at least 2 h, at most two deep moves a day | S1, section 2.2.4 and annex II-4.3; S2, section 2.2 |
| 100-50-100 | about 8 %/h, no hold | none | S3, section 3 (day 1) |
| 100-75-100 | about 4 %/h, no hold | none | S3, section 3 (day 2) |
| return from 50 % | 3 to 5 %/min (EUR); 5 %/min from 50 % | | S3 quoting the EUR; S8 |
| baseload with frequency control | none | | S1, section 2 |

Through a day of 100-50-100 at 8 %/h, bank D travels from 225 to about 190 steps
while delta I stays inside +-5 % (read from figs. 3 and 6), and xenon reads 110
against 96 at full power at the 50 % point (read from fig. 5) [S3]. Xenon peaks
about 6 h (S9) to 7 to 8 h (S10) after a reduction.

## 7. How often each day comes

EDF's units spend, per unit and year, 1400 h in frequency regulation, 250 h in load
following (7.1 TWh) and 41 h in short hot shutdowns; about 4 % of available
generation goes unproduced for modulation [S17, slide 7]. An EDF load-follow day is
off full power for about 6.3 h, so 250 h is about 40 days a year, 11 % of days. The
task weighs baseload days (with frequency control) at 0.892 and the five load-follow
days at 0.0216 each, an equal split that no source gives (ours).

## 8. The stream: regimes, restarts and the cycle (2026-11-11)

Searched for the continuing plant (docs/proposals/long-strata.md, S1 and R8). A
figure counts when it was read twice on separate fetches; "seen once" marks the
rest.

- **The offset's allowance is a rolling day.** Westinghouse's CAOC bases: between
  50 % and 90 % of rated power the flux difference may sit outside its target band
  for "a 1 hour cumulative penalty deviation time limit, cumulative during the
  preceding 24 hours", alarmed past "1 hour in the previous 24 hours"; below 50 %
  the time counts half [ST1, B 3.2.3A]. The task's clock now looks back 24 h; it
  still counts in full at every power, EDF's scope (section 4).
- **How load-follow days come.** EDF's units load-follow 250 h a year [S17]; units
  stood at their minimum on weekend nights "and sometimes longer", and one made two
  variations a day down to 20 % [S17, slides 4 and 5]. How many days in a row a unit
  load-follows, and the gaps between: not found. Seen once: downward modulation now
  comes more at midday in spring and summer, historically more at night and at
  weekends (RTE, 2025); day-ahead programmes are due the day before at 16:30 (CRE
  deliberation 2024-45), which would make a day's notice hours rather than the
  task's 0.5 to 2 h. The stream keeps the day as a 24 h regime and the window's
  notice, ours.
- **Load following through the cycle.** EDF's units keep the full range, an 80 %
  amplitude down to 20 % of rated power, up to 90 % of the normal cycle, and only
  the floor rises after [S17, slides 6 and 12]. Seen once: past about 90 % of the
  cycle units take part only in frequency control (World Nuclear Association); EDF's
  1300 MWe campaigns went from 12 to 18 months; below about 100 ppm boron, removal is
  too slow to follow xenon (US 4,432,930). The stream's refuelling every 30 days is
  ours: it keeps the core at the start of cycle the coefficients describe.
- **Restart after a trip.** Restart times as in section 5 [S15; S16]. After a
  refuelling outage most plants ramp at 10 to 30 % an hour below about 20 % of rated
  power and 3 % an hour above it [ST2]; Exelon's plants about 15 % an hour below the
  threshold and at most 3 % above, with 0.5 % an hour recommended from 75 to 100 %
  [ST3]. Seen once: for EDF, time shut down leaves the fuel's damage margin where it
  was, so a mid-cycle trip does not restart the conditioning (IRSN 2023-00018). The
  post-trip state operators restart from: not found. The restart's xenon is the
  plant's own zero-power decay over the lump's time (PHYSICS.md section 4).
- **The return to power after a trip** (re-record pass, step 2, 2026-11-11). Doel-3's
  return after its 6 h scram was held at 15 % of rated power, because the xenon
  built while it was down put the axial offset off its band, and the unit ran 18 h
  at reduced power before it reached full power again [S15]. APR1400's restart time
  of 22 to 70 h runs from the scram to the withdrawal of the rods for the restart:
  the earliest restart that keeps the core critical over the next five days at a
  given dilution rate, 69.6 h at 0.1 ppm/h [S16]. KEPCO NF's fuel conditioning
  guideline ramps at 3 %/h above a 50 % threshold (2 to 4 %/h tested) and finds
  that the rate below the threshold does not matter to the fuel (5 to 50 %/h
  tested) [ST4]. All three were read twice. Seen once: typical limits of 3 to 4 %/h
  above a threshold of 20 to 40 % (EPRI TR-112140-V2, a search summary). Ramp
  limits written for a trip rather than a startup: still not found, so the task
  uses the startup limits.
- **Not found.** EDF's critical boron at the start and end of a cycle; post-trip
  ramp limits specific to a trip. NRC, OECD-NEA, IAEA and some EDF documents could
  not be opened by an automated reader.

### Sources for section 8

- [ST1] US NRC, NUREG-1431 Rev. 3.0, Standard Technical Specifications,
  Westinghouse Plants, Bases B 3.2.3A, pp. B 3.2.3A-1, -3, -5 (copy at
  https://naygn.org/wp-content/uploads/2013/09/PWR-TS-Basis.pdf).
- [ST2] EPRI 1000847 (2001), report summary: power ascension after refuelling.
- [ST3] EPRI 1012915 (2006), summary and section 1: fuel reliability and power
  ascension rates.
- [ST4] Lee et al. (KEPCO Nuclear Fuel), *FALCON code simulation for verification of
  fuel preconditioning guideline*, KNS Autumn 2015, 15A-200, Table 3 and section 2.6.
  <http://www.kns.org/files/pre_paper/34/15A-200%ec%9d%b4%ed%9d%ac%ed%9b%88.pdf>

## Sources

- [S1] IAEA Nuclear Energy Series NP-T-3.23 (2018), *Non-baseload operation in
  nuclear power plants*. <https://www-pub.iaea.org/MTCD/Publications/PDF/P1756_web.pdf>
- [S2] Barbier et al., OAPS, ICAPP 2025. <https://arxiv.org/pdf/2605.20169>
- [S3] Krsko, NENE 2022 paper 911.
  <https://www.djs.si/nene2022/proceedings/htm/pdf/NENE2022_911.pdf>
- [S4] Moon et al., KNS Autumn 2018, 18A-534.
  <https://www.kns.org/files/pre_paper/40/18A-534문상래.pdf>
- [S6] NRC Westinghouse Technology manual, HRTD 8.1 and 5.0 (rod control, Tavg
  program), as read by an earlier session.
- [S7] US 4,057,463 (Westinghouse). <https://patents.google.com/patent/US4057463A/en>
- [S8] US 4,222,822 (Westinghouse). <https://patents.google.com/patent/US4222822A/en>
- [S9] US 4,582,669 (Westinghouse), xenon suppression.
- [S10] NRC Generic Fundamentals Examination, PWR item 192006.
- [S11] nuclear-power.com, axial flux difference (definition only).
- [S12] US 8,489,338 (Areva NP). <https://patents.google.com/patent/US8489338B2/en>
- [S13] Framatome COMBO boron meter datasheet (vendor).
- [S14] Muniglia et al., ICAPP 2016, hal-01355257; arXiv 2107.11201.
- [S15] Delayed return to full power at Doel-3, INIS record 1ypr9-dk595.
  <https://inis.iaea.org/records/1ypr9-dk595>
- [S17] P. Mercel (EDF), *For a low carbon electricity generation: flexibility of the
  French nuclear fleet*, IEA workshop on nuclear power, 25 February 2019.
  <https://iea.blob.core.windows.net/assets/imports/events/260/Session3.3_Mercel_Philippe.pdf>
- [S16] Castro Monteiro et al., KNS Spring 2020, 20A-065, APR1400 restart.
  <https://www.kns.org/files/pre_paper/44/20A-065-Rafael-Castro-Monteiro.pdf>

Not opened, and not bypassed: NRC technical specifications and course chapters on
nrc.gov, EDF's HAL documents and ScienceDirect, which block automated readers.
