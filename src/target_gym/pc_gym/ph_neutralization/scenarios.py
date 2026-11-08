"""The pH neutralization's scenario suite (charter Amendment 3, rule 20).

Each scenario is an operating window the family's sources show; timing and
size inside it are drawn by the seed. Sources are the tags of this folder's
``REALITY.md``.
"""

from target_gym.scenarios import Scenario

_QUIET = {"acid_upset_size": 0.0, "flow_swing": 0.0}


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "normal",
            dict(_QUIET),
            source="[P5, p. 1]; [P2]",
            description="waste flows in, buffering drifts, reagent holds the pH",
        ),
        Scenario(
            "acid_strength",
            {**_QUIET, "acid_upset_size": 0.2},
            source="[P3, p. 4]",
            description="unknown changes in the influent's strong acid",
        ),
        Scenario(
            "flow_swing",
            {**_QUIET, "flow_swing": 0.2},
            source="[P3, Figure 2]; charter Amendment 3",
            description="the influent flow swings, metered",
        ),
        Scenario(
            "buffer_loss",
            {**_QUIET, "q2_nominal": 0.0},
            source="[P1]; [P2]",
            description="buffering falls away and the curve steepens",
        ),
        Scenario(
            "bad_day",
            {"acid_upset_size": 0.2, "flow_swing": 0.2},
            source="[P3, p. 8]",
            description="calm to rocking and reeling: acid and flow both move",
        ),
    )
