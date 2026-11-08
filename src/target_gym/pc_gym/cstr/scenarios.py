"""The jacketed CSTR's scenario suite (charter Amendment 3, rule 20).

Each scenario is a window of operation the family's sources show; its event's
minute is drawn by the seed (``CSTRParams.event_window``). Sources are the
tags of this folder's ``REALITY.md``.

Weights (charter J4) are shares of operating time. No source says how often
each window occurs, so each is ours: the normal day in the family range 0.5 to
0.9, the rest split among the upsets.
"""

from target_gym.scenarios import Scenario


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "normal",
            {},
            source="[S1, Tables 3, 4]",
            description="the setpoint held; the feed at 23 to 25 C by the day",
            weight=0.7,
        ),
        Scenario(
            "setpoint_step",
            {"setpoint_step": 2.0},
            source="[S1, Fig. 2]",
            description="the reactor's setpoint moved up 2 K",
            weight=0.1,
        ),
        Scenario(
            "catalyst_loss",
            {"k0_drop": 0.4},
            source="[S1, Fig. 3]",
            description="a 40 % loss of activity",
            weight=0.1,
        ),
        Scenario(
            "feed_buildup",
            {"feed_rise": 0.1},
            source="[S3, PDF pp. 36-37, 46]",
            description="the feed's reactant rises 10 % (size ours)",
            weight=0.05,
        ),
        Scenario(
            "cooling_loss",
            {"cooling_loss": 5.0},
            source="[S2, PDF p. 17]; [S3, Fig. 11]",
            description="the cooling water fails for 5 min (length ours)",
            weight=0.05,
        ),
    )
