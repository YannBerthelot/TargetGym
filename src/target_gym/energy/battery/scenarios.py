"""The battery's scenario suite (charter rule 20), each from ``REALITY.md``.

A normal block, then each documented upset. Inside a scenario the timing and
the size are drawn by the seed from the family's range (``env.initial_state``).
Each row holds ``target_gym.scenarios.Scenario``'s fields; the registry builds
them, so this package imports nothing outside its fingerprint.
"""

SPIKE = (150.0, 491.0)  # GBP/MWh: ours from twice the mean, to the GB 2024 high
NEGATIVE = (-28.2, 0.0)  # GBP/MWh: the GB 2024 low, to zero
DEVIATION = (0.19, 0.46)  # p.u.: DR's response to 50 to 100 mHz
WRONG = 0.20  # BMS error sd: MPS-ACCURE's outliers above 40 %, read as 2 sigma

#: The stream's numbers (docs/proposals/long-strata.md, draft 4, S1; owner's calls of
#: 2026-11-11). Each regime lasts one EFA block (4 h, NESO-SOE Table 1), the unit
#: its weight counts in, and brings the window's one event a visit, so each upset
#: keeps its weight's count a year: the long deviation its 20 (Q8, ours, family
#: range 5 to 50). How long each regime lasts beyond its block is ours, unconfirmed
#: by the owner. A stream starts with neither of the window's events, prices on a
#: ring, and the BMS recalibrating once the true charge leaves the LFP plateau at
#: 20 % or 80 % (ARX-LFP's "roughly SOC 20% to 80%"; that the BMS corrects right
#: there is ours, family range 5 to 20 % and 80 to 99 %). The wrong estimate acts
#: through those recalibrations: each in that regime draws its error at sd 20 %.
_BLOCK_H = 4.0
_STREAM = dict(
    dwell_h=_BLOCK_H,
    stream_overrides={
        "window_events": False,
        "price_ring": True,
        "knee_soc": (0.20, 0.80),
    },
)
_ONE_A_BLOCK = dict(_STREAM, events_per_h=1.0 / _BLOCK_H)


def battery_scenarios():
    return (
        dict(
            name="normal",
            overrides={},
            source="REALITY.md sec. 5, normal day: AEMO-HPR Fig. 2; PNNL-ESS App. B",
            description="flat day-ahead price, an energy-neutral signal",
            # 4 h blocks a year, of 2190. Ours, family range 1700 to 2100
            weight=1900.0,
            **_STREAM,
        ),
        dict(
            name="price spike",
            overrides={"price_event_range": SPIKE},
            source="REALITY.md sec. 5 and 7: MPS-ACCURE (Dogfish, $267/MWh peak); "
            "Electricity Maps (GB 2024 hourly high GBP 491)",
            description="one hourly day-ahead price between GBP 150 and 491/MWh",
            # blocks a year. Ours, family range 20 to 150: blocks holding an hour above GBP 150/MWh
            weight=60.0,
            **_ONE_A_BLOCK,
        ),
        dict(
            name="negative price",
            overrides={"price_event_range": NEGATIVE},
            source="REALITY.md sec. 7: Electricity Maps (GB 2024 low GBP -28.2, "
            "141 negative hours)",
            description="one hourly day-ahead price between GBP -28.2 and 0/MWh",
            # blocks a year. Ours, family range 30 to 200: blocks holding a negative hour
            weight=100.0,
            **_ONE_A_BLOCK,
        ),
        dict(
            name="long deviation",
            overrides={"bias_pu_range": DEVIATION},
            source="REALITY.md sec. 5, bad day: ENTSOE-LER sec. 1; NESO-ST para. "
            "6.11(vi); NESO-GUIDE p. 4",
            description="a one-way response of 0.19 to 0.46 p.u. for 30 to 90 min",
            # blocks a year. Ours, family range 5 to 50: deviations of 50 to 100 mHz for 30 to 90 min (ENTSOE-LER)
            weight=20.0,
            **_ONE_A_BLOCK,
        ),
        dict(
            name="wrong estimate",
            overrides={"soc_error_std": WRONG},
            source="REALITY.md sec. 1 and 5: MPS-ACCURE (errors up to 15 %, "
            "outliers above 40 %)",
            description="the BMS's error drawn with sd 20 %",
            # blocks a year. Ours, family range 20 to 200: blocks with a BMS error near 2 sigma (MPS-ACCURE)
            weight=100.0,
            **_STREAM,
        ),
        dict(
            name="long deviation, wrong estimate",
            overrides={"bias_pu_range": DEVIATION, "soc_error_std": WRONG},
            source="REALITY.md sec. 5: ENTSOE-LER sec. 1 with MPS-ACCURE",
            description="both: a long one-way draw on a badly estimated pack",
            # blocks a year. Ours, family range 0.1 to 5: the two together
            weight=1.0,
            **_ONE_A_BLOCK,
        ),
    )
