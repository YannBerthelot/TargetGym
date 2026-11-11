"""``plane``'s scenario suite (charter Amendment 3, rule 20).

Each scenario is a window of cruise the family's sources show (plane/REALITY.md);
inside it the seed still draws the level, the zone's onset and length. Numbers
marked "ours" lie inside the family range given beside them (plane/PHYSICS.md).
"""

from target_gym.scenarios import Scenario

# Weights, each scenario's share of cruise time (Amendment 3, J5). Turbulence:
# zones of 3 to 30 min with calm of over an hour between [BEA447 Figure 24], so a
# zone is in force at most about a third of the time (30 / 90) and typically a
# fifth (16.5 / 76.5); a window with a zone in it is more likely than one without
# (ours: normal 0.40, calm 0.35, family range 0.2 to 0.6 each). The worst
# recorded zone and a zone over the whole window are rare (ours, 0.05 each,
# family range 0.01 to 0.1: Figure 24 shows most zones at 0.2 g or below). The
# mass and level corners (ours, 0.075 each, family range 0.05 to 0.2). The
# heavy corner stops at FL350: past it 78 t keeps under 1.3 g to the buffet
# line, the margin the recommended maximum altitude holds (pprune, secondary).
# Severe turbulence (ours, 1e-4: between MIL-F-8785C's 1e-5 exceedance of the
# severe intensity and its 1e-3 of the moderate one).

FT = 0.3048
#: The light zone's sigma, 0.2 g peak to peak a minute at FL330 (derived,
#: plane/PHYSICS.md); the worst recorded zone, 0.5 g, is 2.5 times it, the
#: gust load being linear in the gust. 1.71 m/s, MIL-F-8785C's moderate.
LIGHT = 0.684
WORST = 2.5 * LIGHT
#: MIL-F-8785C's severe intensity at FL330 (Figure 7, the 1e-5 exceedance
#: curve, as JSBSim tabulates it): 20.0 ft/s at 25 000 ft and 16.0 at 35 000,
#: so 16.8 ft/s, 5.12 m/s, three times the recorded worst.
SEVERE = 16.8 * FT

# ---- The stream (docs/proposals/long-strata.md, draft 4, S1 and S1b) ----
# Sources, second fetches and searches: plane/REALITY.md, "The stream". Numbers
# marked ours are set once, without tuning, and open for the owner.
#: An A320 sortie's mean length, h: the world A320 fleet's average flight,
#: 1.90 flight hours [AIRBUS-A320]. A new sortie draws its mass and level (S1b).
A320_SORTIE_H = 1.90
#: Turbulence zones start at this rate in a turbulent regime, per hour, set so
#: that over ``plane``'s suite the stream spends 7.9 % of its time in a zone,
#: the share of IATA reports above FL200 with a peak EDR of 0.1 or more
#: [LEE-SPRENGER], an upper bound since strong turbulence triggers extra
#: reports. With zones of 3 to 30 min [BEA447 Figure 24], the calm between them
#: averages 2.2 h, "over an hour" as the record has it.
ZONES_PER_H = 0.45
#: How long a weather regime lasts, h (ours: no source gives how long a flight
#: stays in one turbulence or wind climate; a turbulent area takes 12 to 35 min
#: to cross and lasts 30 min to a day [AC00-30C]).
REGIME_H = 2.0
#: The window's own zone never starts in a stream: only the stream starts zones.
_NO_WINDOW_ZONE = {"zone_onset_range": (1.0e9, 1.0e9)}
NOMINAL_MASS = 73_500.0


def stream(dwell_h, sortie_h=A320_SORTIE_H, zones=True, task_per_h=0.0, **overrides):
    """A scenario's stream fields: its dwell, its total event rate, and the
    shares of that rate (``event_mix``) a sortie, a zone and the task's own
    event take, each at its own rate (``plane.stream_events``)."""
    rates = (1.0 / sortie_h, ZONES_PER_H if zones else 0.0, task_per_h)
    total = sum(rates)
    return dict(
        dwell_h=dwell_h,
        events_per_h=total,
        stream_overrides={
            "stream_mode": True,
            "event_mix": tuple(r / total for r in rates),
            **_NO_WINDOW_ZONE,
            **overrides,
        },
    )


def _mass_stratum(mass):
    """A mass corner in a stream: the regime never moves the mass in flight
    (S1b); a sortie that starts in it flies at this mass."""
    return stream(
        A320_SORTIE_H, initial_mass=NOMINAL_MASS, sortie_mass_range=(mass, mass)
    )


def suite():
    return (
        Scenario(
            "normal",
            {},
            source="BEA447 Figure 24: cruise on autopilot, one zone of 3 to 30 min",
            description="one light zone, 0.2 g peak to peak, anywhere in 30 min",
            weight=0.4,
            **stream(REGIME_H),
        ),
        Scenario(
            "calm",
            {"turbulence_sigma": 0.0},
            source="RVSM A.4.1.4 (nonturbulent air); BEA447 Figure 24 (calm over an hour)",
            description="still air the whole window",
            weight=0.35,
            **stream(REGIME_H, zones=False),
        ),
        Scenario(
            "worst_zone",
            {"turbulence_sigma": WORST},
            source="BEA447 Figure 24 (zones up to 0.5 g peak to peak); ICAO light (< 0.5 g)",
            description="one zone at the recorded worst, 0.5 g peak to peak",
            weight=0.05,
            **stream(REGIME_H),
        ),
        Scenario(
            "long_zone",
            {"zone_onset_range": (0.0, 0.0), "zone_duration_range": (1800.0, 1800.0)},
            source="BEA447 Figure 24 (zones of up to 30 min)",
            description="a 30 min zone covering the whole window",
            weight=0.05,
            **stream(REGIME_H),
        ),
        Scenario(
            "heavy_high",
            {
                "initial_mass": 78_000.0,
                "target_altitude_range": (34_000.0 * FT, 35_000.0 * FT),
                "initial_altitude_range": (34_000.0 * FT, 35_000.0 * FT),
            },
            source="TCDS (MTOW 78 t); recommended maximum altitude, 0.3 g to buffet",
            description="78 t at FL340 to 350, the highest it keeps 1.3 g to buffet",
            weight=0.075,
            **_mass_stratum(78_000.0),
        ),
        Scenario(
            "severe_zone",
            {"turbulence_sigma": SEVERE},
            source="MIL-F-8785C Figure 7, severe (1e-5 exceedance) at FL330",
            description="one zone of severe turbulence, 5.1 m/s",
            weight=1e-4,
            **stream(REGIME_H),
        ),
        Scenario(
            "light_low",
            {
                "initial_mass": 60_000.0,
                "target_altitude_range": (29_000.0 * FT, 31_000.0 * FT),
                "initial_altitude_range": (29_000.0 * FT, 31_000.0 * FT),
            },
            source="ours: 60 t, family range 42.6 t (OEW) to 78 t (MTOW) [PHYSICS.md]",
            description="60 t at FL290 to 310, the lightest wing loading, most gust response",
            weight=0.075,
            **_mass_stratum(60_000.0),
        ),
    )


#: ``plane_energy``'s clearances in a stream, per hour: the window's one every
#: ``clearance_period`` (300 s) as the mean (ours: counts of clearances per
#: aircraft and sector are recorded [PRINZO], the time on frequency is not).
CLEARANCES_PER_H = 3600.0 / 300.0
#: A clearance's chance of moving the level rather than the speed: 2.9
#: altitude against 2.1 speed instructions per aircraft over a departure and
#: an approach sector [PRINZO].
LEVEL_SHARE = 2.9 / (2.9 + 2.1)


def energy_suite():
    """``plane_energy``'s one window, declared to carry its stream numbers: the
    clearances become events (S1b), with zones and sorties."""
    return (
        Scenario(
            "normal",
            {},
            source="TECS's job: level and speed clearances (plane/REALITY.md)",
            description="six clearances in 30 min, one light zone",
            weight=1.0,
            **stream(
                REGIME_H,
                task_per_h=CLEARANCES_PER_H,
                clearance_level_share=LEVEL_SHARE,
            ),
        ),
    )


def sine_suite():
    """``plane_sine``'s one window, declared to carry its stream numbers."""
    return (
        Scenario(
            "normal",
            {},
            source="a bandwidth probe (plane/EXTRA.md)",
            description="two periods of the sinusoid, one light zone",
            weight=1.0,
            **stream(REGIME_H),
        ),
    )
