# 3D airliner tasks: the hold and the vector

An addition to the aircraft family's record, [plane/REALITY.md](../plane/REALITY.md),
for `plane3d_heading` and `plane3d_racetrack`. That record holds the airframe, the
limits, the readings and the holding rules of [IFH ch. 10]. This one holds what it
lacks: what a hold costs, how an airliner flies one, and the holding rules of the AIM
that the handbook leaves out. Written on 2026-11-23, from sources only. Where no
source was found it says "not found". Arithmetic on these figures is in
[PHYSICS.md](PHYSICS.md), not here.

## Sources

- **[SF24]** Airbus, Safety First #24, July 2017, "Control your Speed... During
  Descent, Approach and Landing". Read as SKYbrary bookshelf item 4071; the PDF was
  fetched once and its text read twice.
  <https://skybrary.aero/sites/default/files/bookshelf/4071.pdf>
- **[AIM538]** FAA, Aeronautical Information Manual, 5-3-8, Holding. faa.gov refuses
  automated readers (as plane/REALITY.md notes), so it was read on a mirror that does
  not show the edition; the same quotes came back from a search and from a fetch of
  the page. For the owner to check against faa.gov.
  <https://www.faraim.org/faa/aim/chapter-5/section-5-3-8.html>
- **[NATS-HOLDS]** NATS, November 2016, on Heathrow's holds, as in plane/REALITY.md.
  Fetched again on 2026-11-23 on the same reprint, dated 24 November 2016.
  <https://www.atc-network.com/atc-news/nats/heathrow-holding-times-on-the-decline-thanks-to-new-technology>

## What a hold costs

- **Fuel, at the speed the aircraft chooses.** "GD speed is the engine-out operating
  speed in clean configuration. It provides an estimate of the speed for best
  lift-to-drag ratio." It is "the recommended speed ... for a holding in clean
  configuration" [SF24]. "In some phases of flight, GD is computed to minimize drag
  and thus, the fuel consumption (for example during the HOLD phase)" [SF24]. The
  auto flight system "computes GD speed using the aircraft weight ... and the
  pressure altitude" [SF24]. So an A320 holds at its green dot speed, which moves
  with its weight, to burn the least fuel per minute.
- **Time is not the crew's.** The hold lasts until ATC clears the aircraft on. At
  Heathrow, "Average holding times were about 8.5 minutes at the beginning of 2014,
  with that figure now just over 7.5 minutes", and "about 10,000 flights use the
  holds each month" [NATS-HOLDS].
- **What that fuel is worth.** The fall in holding is "a saving of more than 3,000
  hours that aircraft spend in the hold each year", and "NATS estimates that the fall
  in airborne holding has saved over 35,000 tonnes of CO2, the equivalent to more
  than 11,000 tonnes of fuel" [NATS-HOLDS]. The release does not say whether the
  tonnes are a year's or the total since 2014.
- **The speed is capped, not assigned.** The table gives maximum holding airspeeds:
  200 KIAS up to 6 000 ft, 230 KIAS from 6 001 to 14 000 ft, 265 KIAS above
  [AIM538], as [IFH ch. 10] does. A lowest holding speed: not found, other than the
  aircraft's own protection (plane/REALITY.md, limits).
- **The aircraft must stay inside the holding area.** The AIM's actions are
  "recommended to ensure that the aircraft remains within holding protected
  airspace" [AIM538]. The size of that airspace for a given hold: not found
  (FAA Order 8260.3 and ICAO Doc 8168 were not read).
- A price on a hold flown badly (a deviation report, a go-around, a separation
  loss): not found.
- **The lowest speed.** "VLS is the lowest selectable speed for the autopilot and
  the autothrust. Even if the selected target speed is below VLS, the A/THR will
  maintain VLS as a minimum" [SF24]. "For descent and approach flight phases, VLS of
  Fly-By-Wire aircraft is obtained using the Stall speed demonstrated during flight
  tests (VS1G) of the corresponding configuration, multiplied by a factor of 1.23.
  On A320 family aircraft, the factor may be increased for some Slats/Flaps
  configurations", so "A320 family: VLS = k x VS1G with 1.23 <= k <= 1.28" [SF24].
  Which k applies in clean configuration: not found. The PDF was fetched a second
  time on 2026-11-24 and its text is identical; Airbus's own page of the article
  returned no body text to an automated reader.
- A rule against holding under green dot in clean configuration, or a margin a
  crew must keep over VLS in a hold: not found. Green dot is "the recommended
  speed ... for a holding in clean configuration" [SF24], no more.

## How a hold is flown

- **Entry.** "the parallel, teardrop and direct entries are the procedures for entry
  and holding recommended by the FAA" [AIM538]. The entry is chosen by the heading on
  arrival, with "+/- 5 degrees in heading ... considered to be within allowable good
  operating limits for determining entry" [AIM538]. When "3 minutes or less from a
  clearance limit", "the pilot is expected to start a speed reduction" [AIM538].
- **Timing.** Inbound leg "At or below 14,000 feet MSL: 1 minute", 1.5 minutes above.
  "Outbound leg timing begins over/abeam the fix, whichever occurs later" [AIM538].
- **Wind.** "Compensate for wind effect primarily by drift correction on the inbound
  and outbound legs. When outbound, triple the inbound drift correction to avoid
  major turning adjustments" [AIM538].
- **Turns.** Without RNAV guidance, turns are made at "3 degrees per second, or 30
  degree bank angle, or 25 degree bank angle, provided a flight director system is
  used", whichever needs the least bank [AIM538]. With RNAV lateral guidance, "a
  minimum 25 degree bank angle should be selected regardless of altitude" [AIM538].
- **An RNAV hold is not the conventional one.** "There are differences between the
  holding entry and flight track assumed in conventional holding pattern design" and
  what an RNAV system flies. "RNAV systems apply a database coded or pilot-entered
  leg distance as a maximum length of the inbound leg." "RNAV systems may execute a
  'fly-by' turn when approaching the holding fix prior to entry." "Use of a shallower
  bank angle will expand both the width and length of the aircraft track", and "RNAV
  systems may fly up to and possibly beyond the limits of protected airspace before
  turning inbound" [AIM538]. "Pilots are expected to be familiar with the
  capabilities and limitations of the specific RNAV system used for holding"
  [AIM538].
- How the A320's flight management system sizes and flies its hold (turn radius,
  the wind it uses, the bank it commands): not found (FCOM, owner).
- A tolerance on the track flown in a hold, as a rule for operations: not found. The
  bands the task uses are a pilot's test standard (plane3d/PHYSICS.md, Amendment 2).

## What a vector costs

- A vector is a heading ATC assigns [IFH ch. 7]; its size and timing are ATC's
  (counts per aircraft and sector are in plane/REALITY.md, [PRINZO]).
- The rate of turn ATC expects on a vector, or a tolerance on how fast a heading must
  be reached: not found. The 3 deg/s, 30 deg and 25 deg figures above are holding
  rules [AIM538, IFH ch. 10]; [IFH ch. 4] gives the standard rate turn.
- A cost the crew or the autopilot can change while flying a vector, other than the
  fuel and time of plane/REALITY.md's cost index: not found.
- A source that makes a steady wind the crew's problem on a vector: not found. A
  vector is a heading, not a track [IFH ch. 7], and wind correction is described for
  tracks and holds [IFH ch. 10].

## Unconfirmed (secondary sources only, not used above)

- Green dot on the A320: 2 x weight in tonnes + 85 kt below FL200, plus 1 kt per
  1 000 ft above (training summaries); the default hold speed "approximately equal
  to Green Dot" (a forum quote of the FCOM).
- NATS on cross-border arrival management for Heathrow (XMAN): about 8 000 t of CO2
  and GBP 1.65 m of fuel a year, later "over 4,700 tonnes" of fuel and about
  GBP 2.5 m a year (search results; the nats.aero pages now return 404).
- An A320's holding fuel flow from Airbus's "Getting to grips with fuel economy":
  the document was not reached.
