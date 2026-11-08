# Plant reality record: compressor_surge

This record says what the real plant behind `compressor_surge` looks like, from primary sources only.
It was written without reading the task's code or design notes, so it can be used to check the task.
Each statement carries a source tag such as [C3 p.2]. The sources are listed at the end.
"Not found" means no primary source read for this record says it. Nothing was filled from memory.
Several sources are vendor documents or practitioner trade articles; they are labelled as such.

## The plant

A centrifugal compressor driven by a motor or turbine pushes gas into a discharge system, and an anti-surge valve recycles gas back to suction when flow gets too low [C3 p.1 to 2, C6].
Surge is "an uncontrolled flow event" with large pressure swings and cyclic flow reversal; repeated surges bring high vibration, thermal loading and damage to bearings, seals and impellers [C3 p.1].
A semi-industrial test machine shows the scale: a single-stage 400 mm shrouded impeller, 450 kW motor, up to 11 000 rpm, built for subsea gas compression research [C8 p.4 to 5].
Its discharge (surge) volume was 0.18 m3 in one layout and 3.32 m3 with an added manifold and tank [C8 p.8].
Typical power, pressure ratio and flow of a process or pipeline machine: not found in the sources read.
Consumers drawing from a header through their own valve: not found as a described plant layout in the sources read.

## What operators read

- Flow element (orifice dP) on the suction. Flow dP shows "the most telling surge signature in almost all applications" [C5].
  dP transmitters should have a 63% response time under 0.2 s, preferably about 100 ms (CCC's CTO, quoted) [C1].
  Where the meter sits matters: an orifice 27.5 m upstream of the inlet shows a damped, clean oscillation; one 1.75 m upstream shows a noisy one [C8 p.10 Fig. 7].
- Suction and discharge pressure. At the start of a surge cycle flow dP and discharge pressure drop, while suction pressure rises [C4, Table 1].
  Suction pressure can show surge oscillations [C5].
- Temperatures. Suction temperature rises at surge onset [C4, Table 1]; inlet temperature can serve as a detection signal [C5].
  In the NTNU rig, a large surge volume clearly changed the inlet temperature transient during sustained surge [C8 p.10 to 11, Fig. 8].
  Temperature sensor response time on a plant: not found. The rig logged temperatures at 0.1 to 2 Hz [C8 p.6, instrumentation table].
- Speed. Rotational speed falls at surge onset in the Woodward signal table [C4, Table 1].
- Motor current is one of up to four gradients a surge detector may watch [C3 p.3].
- Vibration: high vibration follows excessive surge [C3 p.1]. Probe types and response times (API 670): not found, see the owner list.
- Event recording: surge detectors should record at 20 ms or faster, at least 10 s before and 10 s after each event [C5].

## What they move, and in which mode

- Anti-surge (recycle) valve, PI control on a control line set to the right of the surge line [C3 p.2].
- Backup line between control line and surge line: crossing it overrides PI and steps the valve open in discrete increments (open-loop step) [C3 p.2].
- Setpoint shift: if the backup line is crossed a set number of times in a set period, the control line moves further from the surge line [C3 p.2].
- Rate action: the valve opens early when flow falls fast, before the control line is reached [C3 p.2].
- Rate-of-change detection can stop surge "during the first cycle" by opening the recycle valve [C7].
- Limit control: a pressure-ratio limiter and, where needed, a choke limit control [C3 p.3].
- Inlet guide vanes: temperature compensation is mostly used on adjustable IGV compressors [C3 p.3].
- Speed and IGV as process (performance) control: Siemens names constant suction or discharge pressure as the usual target [C3 p.4].
  Their dynamics: not found.
- Emergency shutdown: the surge valve opens essentially as the driver is tripped; the valve must be fast and large enough for the system volume [C6].
- Load sharing between parallel machines: named as a feature, details not read [C3 p.3].

## Limits

- Surge limit line: the measured or predicted onset of surge on the compressor map [C3 p.2].
- Surge control line and margin. Values differ by source:
  - 10 to 15% between surge curve and control line (CCC's CTO) [C1].
  - Recycle valve opens at a preset turndown, typically 10% [C6].
  - 3 to 5% of the pressure-ratio or dP span (Yokogawa) [C7].
  - The optimum control line is set at the on-site commissioning test (Siemens) [C3 p.2].
- Surge itself is a counted event, then a trip:
  - Centrifugal: excessive surge is typically 3 surge counts within 10 to 20 s, used as a trip condition [C5].
  - Axial: trip on the first or second surge, depending on the manufacturer [C5].
  - Siemens: warn or open the valve first; shutdown "in general, at the 3rd or 4th surge event" [C3 p.3].
  - CCC's CTO: always prevent three or more successive surges within about 10 s; some OEMs use windows up to half an hour [C1].
  - Woodward patent: example threshold of three consecutive surge cycles [C4].
- Each surge also costs: service life and efficiency drop even without obvious damage [C3 p.1].
- Recycling costs energy: the margin trades protection against efficiency [C3 p.1, C1].
- Choke and pressure-ratio limits: limiters exist [C3 p.3]; their settings: not found.

## Time scales

- Flow reversal: under 0.02 s [C1]; 0.03 s from large positive to large negative flow [C2]. Trade articles by Emerson practitioners.
- Surge cycle period: about 0.1 s on laboratory benchtop units, about 1 to 3 s on large industrial installations [C1].
- Laboratory rig with 110 mm impeller and 0.203 m3 plenum: mild surge at 4.0 Hz; system surge below 20 Hz [C9 p.2 to 3, Fig. 4].
- NTNU rig: flow oscillation amplitude about +/-0.15 kg/s (small volume) and +/-0.2 kg/s (large volume) at closed valve [C8 p.7, p.9].
  Its Figure 7 shows about five cycles between t = 6.24 and 6.36 s on the printed axis (my reading). The text gives no frequency.
- Surge frequency rises when the discharge volume falls [C8 p.4, citing earlier work].
- Valve stroke: T95 for a full-scale signal under 2.5 s; solenoid trip from closed to fully open under 1 s (CCC's CTO) [C1].
- Positioner and actuator lags: "within 150-300ms" [C7].
- Laboratory fast valve: 19 ms to open, 16 ms to close after the signal [C9 p.2].
- Whole loop budget: valve response plus transmitter damping plus half of update, scan and execution intervals under 2 s, or under 1 s to run closer to surge [C2].
- Controller scan: 20 ms and 40 ms task cycles tested on a PLC; response 36 to 132 ms [C7].
- Surge detector window: phases of one cycle within an example 2 s window [C4].
- Speed and IGV loop time constants: not found.
- Driver run-down after a trip: not found in figures read (see Kurz et al. in the owner list).

## A normal day and a bad day

Normal day. The operating point sits right of the control line and the recycle valve stays shut or nearly so [C3 p.2].
The live compressor map with the control line, plus trends, is the operator's view [C3 p.4].
Bad day, slow approach. Demand falls; the point reaches the control line and PI opens the recycle valve to hold forward flow [C3 p.2].
Bad day, fast approach. Flow falls fast; rate action or the backup line steps the valve open before surge [C3 p.2].
Bad day, surge. On trends, flow dP drops sharply, discharge pressure drops, suction pressure and temperature rise, and this repeats each cycle [C4 Table 1, C5].
On a rig held at closed throttle, flow swings between forward and slightly reverse values for as long as the valve stays shut [C8 p.8 Fig. 5].
Bad day, trip. An emergency shutdown cuts the driver; the recycle valve must open at once and the piping volume sets the response [C6].
A repeated-surge event raises the control margin, and after 3 or 4 surges the machine is shut down [C3 p.2 to 3].
Real plant trend of a surge event with time axis: not found (candidates in the owner list).

## The operator's job in one sentence

Meet the header pressure or flow demand while keeping the operating point right of the control line, with as little recycle as possible and no counted surge [C3 p.1 to 2, C5].

## Proposed behaviour tests: sources list

| Source and figure | Quantity | Value as printed |
|---|---|---|
| C1 | surge cycle period, large industrial machine | about 1 to 3 s |
| C1 | surge cycle period, laboratory benchtop | about 0.1 s |
| C1, C2 | flow reversal time | under 0.02 s; 0.03 s |
| C1 | control line margin | 10 to 15% |
| C6 | recycle valve opening point | preset turndown, typically 10% |
| C7 | control line margin | 3 to 5% of span |
| C1 | anti-surge valve T95, solenoid full open | under 2.5 s; under 1 s |
| C2 | total loop delay budget | under 2 s (under 1 s near surge) |
| C5 | excessive surge trip, centrifugal | 3 counts within 10 to 20 s |
| C3 p.3 | shutdown on repeated surge | 3rd or 4th surge event |
| C4 Table 1 | signal directions at surge onset | flow dP and discharge pressure fall; suction pressure, suction temperature rise; speed falls |
| C8 p.8 Fig. 5 | flow during sustained surge at closed throttle (rig) | oscillates around zero, about +/-0.15 to +/-0.2 kg/s |
| C9 Fig. 4 | mild surge frequency (lab rig, 0.203 m3) | 4.0 Hz |

## Sources

- [C1] G. McMillan, S. Weiner, "Fine points of compressor surge control", Control Global, 12 May 2016, quoting Serge Staroselsky (CTO, Compressor Controls Corp.), https://www.controlglobal.com/articles/2016/fine-points-of-compressor-surge-control . Trade article. The 0.1 s and 1 to 3 s periods and the 0.02 s reversal are McMillan's words; the margin, valve and transmitter figures are Staroselsky's. Checked twice.
- [C2] C. Stuart, G. McMillan, T. Hildebrand (Emerson), "Compressor surge control: deeper understanding, simulation can eliminate instabilities", Control Global, 25 Oct 2018, https://controlglobal.com/articles/2018/compressor-surge-control-deeper-understanding-simulation-can-eliminate-instabilities . Trade article. Its Figure 2 ("severe surge") seems to come from their digital twin, not a plant, so it is not used as plant evidence. Checked twice.
- [C3] Siemens Energy, "Compressor Train One Control: Anti-Surge Control", brochure, 2024, https://assets.siemens-energy.com/dam/de680584-da86-4302-9be2-b25d00a7b288/Compressor-Train-One-Control-2024-pdf_Original%20file.pdf , pp. 1 to 4. Vendor document. Checked twice.
- [C4] A. Benim (Woodward), "Detection and counting of surge cycles in a compressor", US Patent 9,695,831, 2017, https://patents.google.com/patent/US9695831 , Table 1 and FIGS. 2A, 2B (figures not viewed as images). Checked twice.
- [C5] W. Jacobson, M. Zaghloul, M. Tolmatsky, S. Staroselsky (Compressor Controls Corp.), J. McWhirter, "Compressor load sharing control and surge detection techniques", excerpted as "Surge detection systems", Turbomachinery Magazine, 18 Sep 2019, https://www.turbomachinerymag.com/surge-detection-systems/ . Checked twice.
- [C6] R. Kurz, R. Zwerver, S. Simons, A. M. Alvarado, K. Brun, "Dynamic simulation and testing to assess rundown speed of a compressor", Asia Turbomachinery & Pump Symposium 2018, excerpted as "Five requirements for successful surge avoidance", Turbomachinery Magazine, 18 Feb 2019, https://www.turbomachinerymag.com/view/five-requirements-for-successful-surge-avoidance . Checked twice.
- [C7] Yokogawa, "Compressor Surge Prevention Control implemented on Yokogawa Stardom", application note, June 2012, https://yokogawa.com/au/library/resources/application-notes/compressor-surge-prevention-control-implemented-on-yokogawa-stardom . Vendor document. Checked twice.
- [C8] A. Serena, L. E. Bakken (NTNU), "System influence on the surge behaviour of a centrifugal compressor", ETC15, Budapest, 2023, paper ETC2023-136, https://www.euroturbo.eu/paper/ETC2023-136.pdf . Figures 5 (p.8) and 7 (p.10) looked at as images. A test rig on air, not a plant. Checked twice.
- [C9] N. Hagino, K. Uda, Y. Kashiwabara, "Prediction and Active Control of Surge Inception of a Centrifugal Compressor", IGTC2003 Tokyo, TS-038, https://nippon.zaidan.info/seikabutsu/2003/00916/pdf/igtc2003tokyo_ts038.pdf . A laboratory rig, not a plant. Checked twice.

## For the owner to open by hand

- E. M. Greitzer, "Surge and Rotating Stall in Axial Flow Compressors, Part II: Experimental Results and Comparison With Theory", J. Eng. Power 98(2), 1976, 199-211, DOI 10.1115/1.3446139 (ASME, paywalled). Check the measured surge cycle traces, period versus the B parameter, and the plenum sizes.
- API Standard 617 (axial and centrifugal compressors) and API Standard 670 (machinery protection systems), API (paywalled). Check what they say on surge protection, surge detection, vibration probe response, and whether surge is a trip.
- K. K. Botros, J. Grose, surge control design paper, Texas A&M Turbomachinery Symposium, https://oaktrust.library.tamu.edu/bitstream/1969.1/162142/1/T12.pdf (Cloudflare check). Check the field measurements of fast events such as ESD and the time to surge.
- J. J. Moore, A. Garcia-Hernandez, M. Blieske, R. Kurz, full-scale ESD surge tests in a hydrocarbon loop, https://core.ac.uk/download/87264778.pdf (Cloudflare check). Check the surge transient traces, valve sizes and opening times.
- Compressor Controls Corp. paper on field surge testing, https://core.ac.uk/download/79655542.pdf (Cloudflare check). Check surge signatures recorded in the field.
- R. Kurz, M. White, K. Brun, et al., tutorial on surge control and dynamic behaviour, https://core.ac.uk/download/187120819.pdf (Cloudflare check). Check valve timing in shutdown cases.
- Emerson, "Increased plant availability through proper anti-surge valve selection", white paper en-1653588, https://www.emerson.com/documents/automation/white-paper-increased-plant-availability-through-proper-anti-surge-valve-selection-en-1653588.pdf (redirects). A search snippet says "typically less than 2 seconds" stroke; confirm.
- Fisher, "Optimized antisurge control valves" brochure en-127170, https://s1-live.emerson.com/documents/automation/brochure-fisher-optimized-antisurge-control-valves-en-127170.pdf (host unreachable). A search snippet says the valves open in under 2 s; confirm.
- NOVA Gas Transmission field tests of a pipeline station, IPC1998 paper, https://asmedc.silverchair.com/IPC/proceedings-pdf/IPC1998/40238/1057/2507098/1057_1.pdf (blocked). Check station surge and valve dynamics data.

## How often each window occurs (2026-11-08)

How often a normal day and each bad day occur: not found in the sources read.
