"""The wind turbine's scenario suite (charter Amendment 3, rule 20).

A normal day and each upset the record documents, as overrides on the scored
configuration; inside each, the seed draws the mean wind, the turbulence and the
events' timing from their ranges. Sources are REALITY.md's. Each is a dict that
the registry turns into a ``target_gym.scenarios.Scenario``.
"""


def Scenario(name, overrides, source, description):
    """One scenario as the registry reads it (``target_gym.scenarios.Scenario``)."""
    return dict(name=name, overrides=overrides, source=source, description=description)


def wind_turbine_scenarios() -> tuple:
    return (
        Scenario(
            "normal",
            {},
            source="REALITY.md sec. 5, S1 Sec. 7, S2 Table 2, S13",
            description="above rated, 13 to 25 m/s, IEC class B normal turbulence",
        ),
        Scenario(
            "gusty_class_a",
            {"turbulence_iref": 0.16},
            source="S13 (Iref 0.16, class A)",
            description="a gusty day: class A turbulence intensity",
        ),
        Scenario(
            "extreme_gust",
            {"gust": 1.0},
            source="S13 (extreme operating gust, 10.5 s); REALITY.md sec. 5",
            description="one extreme operating gust at a time the seed draws",
        ),
        Scenario(
            "near_rated",
            {"v_mean_range": (11.4, 14.0)},
            source="S1 Table 1-1 (rated 11.4 m/s); S2 Sects. 5.2 and 5.3",
            description="the transition at rated wind, where set-point smoothing "
            "and peak shaving act",
        ),
        Scenario(
            "near_cut_out",
            {"v_mean_range": (21.0, 25.0)},
            source="S1 Table 1-1 (cut-out 25 m/s); S2 Table 2",
            description="high winds below cut-out, the pitch far toward feather",
        ),
        Scenario(
            "curtailment",
            {"derate_to": 0.9},
            source="S10 Art. 13(6); S11 Fig. ES-8 and p. xix; S9 Ex. 29",
            description="a 10 % de-rate set point at a time the seed draws",
        ),
    )
