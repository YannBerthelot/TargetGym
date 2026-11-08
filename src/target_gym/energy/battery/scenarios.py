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


def battery_scenarios():
    return (
        dict(
            name="normal",
            overrides={},
            source="REALITY.md sec. 5, normal day: AEMO-HPR Fig. 2; PNNL-ESS App. B",
            description="flat day-ahead price, an energy-neutral signal",
        ),
        dict(
            name="price spike",
            overrides={"price_event_range": SPIKE},
            source="REALITY.md sec. 5 and 7: MPS-ACCURE (Dogfish, $267/MWh peak); "
            "Electricity Maps (GB 2024 hourly high GBP 491)",
            description="one hourly day-ahead price between GBP 150 and 491/MWh",
        ),
        dict(
            name="negative price",
            overrides={"price_event_range": NEGATIVE},
            source="REALITY.md sec. 7: Electricity Maps (GB 2024 low GBP -28.2, "
            "141 negative hours)",
            description="one hourly day-ahead price between GBP -28.2 and 0/MWh",
        ),
        dict(
            name="long deviation",
            overrides={"bias_pu_range": DEVIATION},
            source="REALITY.md sec. 5, bad day: ENTSOE-LER sec. 1; NESO-ST para. "
            "6.11(vi); NESO-GUIDE p. 4",
            description="a one-way response of 0.19 to 0.46 p.u. for 30 to 90 min",
        ),
        dict(
            name="wrong estimate",
            overrides={"soc_error_std": WRONG},
            source="REALITY.md sec. 1 and 5: MPS-ACCURE (errors up to 15 %, "
            "outliers above 40 %)",
            description="the BMS's error drawn with sd 20 %",
        ),
        dict(
            name="long deviation, wrong estimate",
            overrides={"bias_pu_range": DEVIATION, "soc_error_std": WRONG},
            source="REALITY.md sec. 5: ENTSOE-LER sec. 1 with MPS-ACCURE",
            description="both: a long one-way draw on a badly estimated pack",
        ),
    )
