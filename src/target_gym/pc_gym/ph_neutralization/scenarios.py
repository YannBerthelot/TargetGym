"""The pH neutralization's scenario suite (charter Amendment 3, rule 20).

Each scenario is an operating window the family's sources show; timing and
size inside it are drawn by the seed. Sources are the tags of this folder's
``REALITY.md``.

Weights (charter J4) are shares of operating time. No source says how often
each window occurs, so each is ours: the normal day in the family range 0.5 to
0.9, the rest split among the upsets.
"""


def _row(name, overrides, **kw):
    """One scenario as a plain row: the package imports nothing outside its
    fingerprint, and the registry builds the ``Scenario``."""
    return dict(name=name, overrides=overrides, **kw)


_QUIET = {"acid_upset_size": 0.0, "flow_swing": 0.0}


def ph_neutralization_scenarios() -> tuple[dict, ...]:
    return (
        _row(
            "normal",
            dict(_QUIET),
            source="[P5, p. 1]; [P2]",
            description="waste flows in, buffering drifts, reagent holds the pH",
            weight=0.7,
        ),
        _row(
            "acid_strength",
            {**_QUIET, "acid_upset_size": 0.2},
            source="[P3, p. 4]",
            description="unknown changes in the influent's strong acid",
            weight=0.1,
        ),
        _row(
            "flow_swing",
            {**_QUIET, "flow_swing": 0.2},
            source="[P3, Figure 2]; charter Amendment 3",
            description="the influent flow swings, metered",
            weight=0.1,
        ),
        _row(
            "buffer_loss",
            {**_QUIET, "q2_nominal": 0.0},
            source="[P1]; [P2]",
            description="buffering falls away and the curve steepens",
            weight=0.05,
        ),
        _row(
            "bad_day",
            {"acid_upset_size": 0.2, "flow_swing": 0.2},
            source="[P3, p. 8]",
            description="calm to rocking and reeling: acid and flow both move",
            weight=0.05,
        ),
    )
