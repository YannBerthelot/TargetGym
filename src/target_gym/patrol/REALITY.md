# Plant reality record: wake_surf

This record says what the real job behind `wake_surf` looks like: a transport aircraft
saving fuel by flying in the upwash of the aircraft ahead of it ("wake energy
retrieval", "wake surfing", extended formation flight). It is built from flight-test
papers and from the operators' own accounts. Each statement carries a source tag (for
the NASA papers, the section); the sources are at the end, with how each was read. "Not found" means no source read for
this record says it. The other tasks of this package (`patrol`, `patrol_range_bearing`)
are Extra and have no record.

Searched on 2026-11-23, in two passes. Every figure below was read a second time on
its page, except where its source is marked "read once".

## Property card

| | |
|---|---|
| Family | Transport aircraft in extended formation: the trail aircraft of a pair, in cruise |
| Named property | An economic trade next to a limit, around a hidden variable. The saving grows toward the vortex core, where the trim disturbance steepens, the ride degrades and a wake crossing ends the run. The wake's position is predicted, never measured |
| Structure | The lead's wingtip vortex pair, drifting with the wind and sinking behind it; the trail aircraft's wingtip in its upwash |

| Signature entry | Family | Task (`wake_surf-v3`) |
|---|---|---|
| Delay: time in trail, the wake's age at the trail aircraft | 5 to 15 s, 0.5 to 2 nm nose to tail [N3 I]; just under 6 s at 4,000 ft [N1]; 3 km or 1.2 to 1.5 nm behind for the A350 pair [A1, A2, A3] | 12 to 16 s (2.8 to 3.7 km at M 0.78) |
| Distance in trail, in spans | about 50, "the larger end of the presumed usable range" [N1]; about 20 at 4,000 ft for a large freighter [W1] | 78 to 103 |
| Gain spread: fuel-flow saving across the positions flown | 2 % to 8 % sustained over 80 to 140 ft of cross track [N1, figure 10]; nearly linear from 240 ft to 160 ft of cross track, to 6.8 and 7.8 % [N2] | 2 % at 1.7 spans from the lead's centreline to 16 % at the station, 0.87 span |
| Disturbance and notice | the wind drifts the wake; a 1 kt error in the cross-track wind is a 10 ft position error at 6 s in trail [N1]; an aircraft's own wind is good to 0.5 m/s at best and 2 to 3 m/s typically, as a vector [M1]. No notice: the wake is not seen | none: the wake stays on the lead's ground track |
| Distance to each limit | 10 inadvertent wake crossings in four flights [N4 III.D]; one test point in seven ended in one [N1]; cabin vibration is established by a 3.5 % fuel-flow saving [N4 V.A] | the core is 0.8 to 0.9 span inboard of the station; a 60 deg bank trips |
| What is measured | the lead's position and velocity by ADS-B twice a second, 5.1 m and 25 ft resolution; own wind estimate; the wake's place computed from these [N1] | the station's offset in the lead's frame, exact; the lead's fix with instrument noise |
| Actuators | lateral and vertical position by autopilot; along track by throttle, moved as seldom as possible, 60 s or more between moves [N1]; autopilot and autothrottle on the C-17 [N2, F1] | throttle, pitch and bank demands every second |
| Cost asymmetry | not found as a price: the sources trade fuel against ride and against a wake crossing, in words | fuel and time by the cost index; a trip priced by us |

Entries that depend on the reference (the limit distances in its own deviations) wait
for a reference that holds a station (charter J1).

## The plant

- The trail aircraft places its wingtip in the upwash part of the lead's wingtip vortex;
  the upwash lowers its induced drag, so it holds level flight on less thrust [N1 II].
- "Extended" formations are those more than 10 spans apart in the stream direction
  [N1 II]. Flight experiments on pairs of similar aircraft have shown fuel savings of
  9 % to 14 % in close formation and 7 % to 10 % in extended formation [N1 II].
- Airliners: two A350s flew Toulouse to Montreal on 9 November 2021, three kilometres
  apart [A1]; a later Airbus account gives about 1.2 nm (2.2 km) for that pair [A3], and
  the concept of operations gives 1.5 nm (3 km) [A2]. In July 2020 two A350s stayed 3 km
  apart for four hours, stabilised on autopilot [S1]. The A350-900 spans 64.75 m [A5],
  so 3 km is 46 of its spans and 2.2 km is 34 (arithmetic on the two figures).
- Business jets: a Gulfstream C-20A behind a Gulfstream III, 4,000 ft in trail, about 50
  spans, at M 0.7 and 35,000 ft, four flights in 2017 [N1 II, III.D].
- Military transports: two C-17s at 275 kt and 25,000 ft, the trail aircraft 1,000 and
  3,000 ft aft [N2 abstract]; later tests no closer than 4,000 ft, 6 September to 2 October
  2012 [F1].
- Freighters: a 777F held 4,000 ft in trail on automated position keeping [W2]. At
  20 spans, about 4,000 ft, its vortices "can drift significantly" [W1].
- The trajectories are parallel ground tracks at about the same altitude, the trail
  aircraft with as little as a span between it and the centre of the wake; the pair
  keeps 0.5 to 2 nm nose to tail, 5 to 15 s at cruise speeds [N3 I].

## What the crew and the automation read

- The lead's position and velocity, by a 1090 MHz ADS-B link, twice a second. Latitude
  and longitude have a resolution of 5.1 m and pressure altitude of 25 ft; their accuracy
  is that of the lead's navigation system [N1 III.B].
- The wake's place is computed, not sensed. Horizontally: the lead's reported position,
  the time in trail and the trail aircraft's own wind estimate. Vertically: the lead's
  reported altitude less an assumed descent rate times the time in trail [N1 III.B].
- The predicted descent rate is a function of the lead's fuel weight and span [N3 II].
  On the first flights high winds made large errors in the prediction [N3 VII.B].
- How well an aircraft knows its own wind: with a perfect navigation system the
  airspeed's uncertainty alone leaves up to 0.5 m/s (1 kt) on the wind vector; all
  sources together, 2 to 3 m/s (4 to 6 kt) is typical [M1 3.4.1]. Airline wind reports
  scatter 2 m/s against a weather model [M2 abstract], and about 1.8 m/s as a vector
  in a collocation study [M3]. A cruise wake study took 1.1 m/s on the wind
  speed, 1 deg on the heading and 2 m/s on the airspeed as its input uncertainties
  [M4 table 1]. The cross-track share of these: not found.
- No independent measurement of the wake existed in flight, and the accuracy of the
  estimate is unknown [N1 V.B]. On the C-17 there were no contrails and nothing showed
  where the core was [N2].
- The wake must be known "within a few percent of wingspan, around 10 ft", against the
  30 m normally assumed for unaugmented GPS [W1].
- The aircraft's own trim tells where it is in the wake: pitch attitude falls and roll
  trim rises with the saving [N1 V.C, V.D].
- Fuel flow: cockpit gauges with a 10 lb/h resolution [N1 III.C].

## What they move

- Lateral and vertical position, by an autopilot holding a parallel track at a set
  distance from the predicted wake. On the Gulfstream its commands went into the
  production autopilot as ILS localizer and glideslope signals [N1 III.B].
- Along-track distance, by throttle. On the Gulfstream the pilot moved the throttle by
  hand from a cue, as seldom as possible; the error was allowed to wander within
  +-250 ft "because the wake effects on the trail airplane change slowly in that
  dimension", and the throttle often stayed fixed for 60 s or more [N1 IV]. The
  tolerance began at +-100 ft and was widened after the first flights; following the
  throttle cue closely kept the distance but shook the height by more than 10 ft, and
  moving the throttle little let the distance leave +-250 ft with little effect on the
  hold across the wake [N3 IV.A, VII.A].
- On the C-17 the autoflight system held along-track distance, altitude and airspeed
  while the pilot flew the lateral axis by hand [N2]; in the 2012 tests automated
  control software flew it on autopilot [F1, read once].
- On the A350 a function places the follower behind the leader automatically and
  tracks its vortex [A3].
- Entry is from outside: the Gulfstream engaged its autopilot 4,000 ft aft and outside
  the wake's predicted influence, then stepped inward [N1 III.D]; Boeing's sequence
  stabilises about 1 nm behind and 500 ft to the side of the predicted wake, then moves
  to about 200 ft from the vortex, outboard of the core [W1].

## Where the saving is

- Gulfstream: more than 2 % at every stabilised point; two test points sustained 3.5 %
  to 8 % with excursions to 10 % [N1 V.A]. On its map the 8 % contour lies near 80 to
  90 ft of cross track from the wake's centreline and the 2 % contour near 140 ft, within
  20 ft of the predicted wake height [N1 figure 10, read off the text layer, so
  approximate].
- The saving's gradient across track was steeper in flight than a Burnham-Hallock
  vortex with a core of 1 % of the span predicts [N1 V.B].
- C-17: the saving rose nearly linearly as the cross-track distance, centre to centre,
  fell from 240 ft to 160 ft; the largest averages were 6.8 % and 7.8 % of fuel flow and
  9.2 % of thrust; vertical steps showed no clear trend [N2].
- C-17, 2012: almost 10 % at no closer than 4,000 ft [F1], and 10 % over more than
  90 min [W1]. 777F: up to 10 % [W2]. An A350 behind an A380: 12 % [W1].
- A350 pair: over 6 tons of CO2 saved on the Atlantic crossing; more than 5 % on long
  haul is given as the potential [A1]; 5 to 10 % per trip from the first tests [A2].
- Along track the wake's effect changes slowly [N1 IV]. The saving against in-trail
  distance beyond 50 spans: not found.

## Limits

- **Ride.** In the wake the cabin vibrates at 16 to 28 Hz, forward and mid cabin; the
  aft cabin does not, and the low-frequency motion that makes passengers sick stays
  near calm air's, a quarter of light turbulence's [N4 IV.A, V.C]. Discomfort sets in
  between no saving and a 3.5 % fuel-flow saving and does not grow past it; the
  3.5 % threshold is called arbitrary, and the levels are those of the "light chop"
  reference points flown outside the wake [N4 V.A, VII]. On the C-17 the crew went from
  "smooth ride, only 1.1 deg aileron" to "rougher ride quality, buffeting" as the
  cross track closed [N2]. Airbus reports turbulence like that of normal operations
  for its station [S1], and C-17 and NASA crews reported light chop [W1]. A ride limit
  that tightens toward the core, as a number: not found.
- **Wake crossing.** The trail aircraft passed through the vortex 10 times in the
  four Gulfstream flights, after a commanded move to a position that could not be
  held, a change in the winds or the navigation data, or a manoeuvre of the lead
  [N4 III.D, VI.D]. One test point of seven ended early with one [N1 V.A]. Those aboard
  rated crossings uncomfortable or very uncomfortable, and half said one would make
  them less likely to fly so again; the test team's view is that they "should be
  avoided" on passenger flights [N4 VI.D, VII]. The flight condition was chosen to
  leave structural margin on winglet and tail loads for one [N1 III.D]. A crossing's
  bank or load figures: not found.
- **Roll trim.** Steady aileron and spoiler are needed to hold cross-track position,
  more of it with more saving [N1 V.D]. C-17: about 7 +-3 deg of aileron at its closest
  point, almost none out of the wake [N2]. Gulfstream: "quite small" [N1 V.D].
- **Tracking near the core.** The Gulfstream's autopilot "did not consistently maintain
  zero tracking error": its trim response was slow against a trim disturbance whose
  gradient steepens as the wingtip nears the core, and three of the four flights went
  partly to adjusting gains for stability in the wake [N1 III.E, V.A]. The C-17 "is not
  stationary" at its closest point, where the upwash gradient is assumed to be high [N2].
- **Separation.** Oceanic rules keep aircraft 30 to 50 nm apart today; the pair joins
  at flight levels 1,000 ft apart, the follower 1.5 nm behind [A2]. A minimum distance
  in trail inside a pair: not found, beyond the test distances above.

## Time scales

- Time in trail: just under 6 s at 4,000 ft [N1 II].
- Stabilised points: 5 min or more of surfing, 3 min of reference outside the wake
  [N1 III.D]. The A350 tests held the station for hours [S1].
- Wake descent: the difference between the heaviest and lightest lead was 5.2 ft at the
  test's time in trail [N1 V.B]. The descent rate itself: not found in these sources.
- How fast the wind estimate's error moves: not found; it "tends to obscure
  performance trends" [N1 V.B].

## A normal day and a bad day

- Normal: the pair in cruise on an oceanic track, the follower stabilised in the
  updraft on autopilot for hours [S1, A2].
- Bad: a wake crossing, about ten in four research flights [N4]; a wind change that
  moves the wake against the prediction [N1, N3, W1]. How often either happens in service: not found. No commercial flight has
  flown inside a wake yet: the eight North Atlantic flights of September and October
  2025 tested the rendezvous only [A4].

## The job, in one sentence

Hold the trail aircraft where the lead's upwash pays, on as little thrust as level
flight needs, without degrading the cabin's ride and without ever crossing the wake.

## The stake

- 5 to 10 % of trip fuel for the follower [A2, A3]; up to 5 % on long haul in the
  latest statement [A4].
- For the pair, an 8 % saving on the trail aircraft is about 4 %, which may fall to
  3 % over a whole flight [W1].
- Airbus's aim across widebody operations: 3 to 4 million tons of CO2 a year [A2].
- US Air Force: "millions of gallons" a year [F1, read once].
- A fuel price and a per-flight saving in money: not found in these sources.

## The member this task represents

Two A320s on an oceanic track at FL330 to FL370. No source read here flew a
single-aisle airliner in a wake; the members flown are a widebody pair, a military
transport pair, a business jet pair and a freighter. The measured conditions are
M 0.7 or 0.75 at 35,000 ft [N1, N4] and 275 kt at 25,000 ft [N2]; the A350's are not
given.

## What stops each state the agent can drive

- Cross-track position toward the lead: a wake crossing (above). No protection system
  is described.
- Along-track distance: the pair's assigned spacing [A2]; a number for the closest
  approach is not found.
- Bank and load factor: the aircraft's own envelope protection (plane/REALITY.md).

## Sources

- **[A1]** Airbus press release, 9 November 2021.
  <https://www.airbus.com/en/newsroom/press-releases/2021-11-airbus-and-its-partners-demonstrate-how-sharing-the-skies-can-save>
- **[A2]** Airbus, "How a fello'fly flight will actually work", September 2020.
  <https://airbus.com/en/newsroom/stories/2020-09-how-a-fellofly-flight-will-actually-work>
- **[A3]** Airbus, wake energy retrieval with SESAR partners, September 2024.
  <https://www.airbus.com/en/newsroom/stories/2024-09-airbus-and-sesar-partners-are-taking-wake-energy-retrieval-to-the-next>
- **[A4]** Airbus press release on the GEESE trials, December 2025.
  <https://www.airbus.com/en/newsroom/press-releases/2025-12-airbus-and-partners-complete-successful-wake-energy-retrieval>
- **[A5]** Airbus aircraft pages, dimensions: A350-900 wing span 64.75 m, A320neo
  35.80 m. <https://www.aircraft.airbus.com/en/aircraft/a350/a350-900>,
  <https://www.aircraft.airbus.com/en/aircraft/a320-family/a320neo>
- **[S1]** Scientific American, 11 November 2020, quoting the Airbus project leader.
  <https://www.scientificamerican.com/article/birdlike-flight-formations-could-cut-airline-emissions/>
- **[N1]** Hanson, Pahle, Reynolds, Andrade and Brown, "Experimental Measurements of
  Fuel Savings During Aircraft Wake Surfing", AIAA 2018-3560. Full text read.
  <https://ntrs.nasa.gov/citations/20180004526>
- **[N2]** Pahle, Berger, Venti, Duggan, Faber and Cardinal, "An Initial Flight
  Investigation of Formation Flight for Drag Reduction on the C-17 Aircraft", AIAA
  2012-4802. Full text read. <https://ntrs.nasa.gov/citations/20120007201>
- **[N3]** Reynolds, Pahle and Hanson, "Design, Development, and Flight Evaluation of
  Pilot Displays and Long-Track Control for Wake Surfing Applications", 2018. Full text
  read. <https://ntrs.nasa.gov/citations/20180004464>
- **[N4]** Hanson, Andrade and Pahle, "Experimental Measurements of Passenger Ride
  Quality During Aircraft Wake Surfing", 2018. Full text read. It gives M 0.75 for the
  flights where [N1] gives M 0.7. <https://ntrs.nasa.gov/citations/20180004506>
- **[M1]** WMO Guide to Instruments and Methods of Observation (WMO-No. 8), 2018
  preliminary edition, aircraft-based observations, section 3.4.1. Full text read.
  <https://old.wmo.int/extranet/pages/prog/www/IMOP/publications/CIMO-Guide/Prelim_2018_ed/8_II_3_en_MR_clean.pdf>
- **[M2]** de Haan, KNMI Technical Report TR-338, 2013. Abstract and chapter 1 read.
  <https://cdn.knmi.nl/knmi/pdf/bibliotheek/knmipubTR/TR338.pdf>
- **[M3]** NOAA AMDAR pages, "What is the accuracy?", citing Benjamin, Schwartz and
  Cole 1999. <https://amdar.ncep.noaa.gov/faq.shtml>
- **[M4]** Brown and Holzapfel, "Case Studies in Modelling Cruise-generated Trailing
  Vortices", AIAA 2024-3773. Table 1 read. <https://elib.dlr.de/206671/1/AIAA-Paper-2024-3773.pdf>
- **[W1]** Aviation Week, 14 February 2020, reporting Boeing's Flanzer.
  <https://aviationweek.com/special-topics/sustainability/airbus-boeing-working-next-steps-fuel-saving-wake-surfing>
- **[W2]** Aviation Week on the Boeing and FedEx 777F test.
  <https://aviationweek.com/aerospace/boeingfedex-777-test-confirms-wake-fuel-burn-benefit>
- **[F1]** Air & Space Forces Magazine, 12 October 2012, relaying the AFRL release.
  <https://www.airandspaceforces.com/californiasurfingautomated-style>

Not opened: Bieniawski, Rosenzweig and Blake, AIAA 2014-1457 (the C-17 programme's
summary) and Flanzer, Bieniawski and Brown, AIAA 2020 (the 777F tests): no open copy
was found. The US Air Force's own releases returned an access error.
