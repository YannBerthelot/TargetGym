"""The distillation column's scenario suite (charter Amendment 3, rule 20).

Each scenario is an operating window the family's sources show, with timing and
sign drawn by the seed (``DistillationParams.step_window``). Sources are the
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
            {"feed_step": 0.0, "zF_step": 0.0},
            source="[D4, p. 5]",
            description="the design point held in its zones at least boilup",
            weight=0.7,
        ),
        Scenario(
            "feed_rate_step",
            {"feed_step": 0.2, "zF_step": 0.0},
            source="[D1, p. 33, Fig. 18]",
            description="the feed rate steps 20 % up or down, metered",
            weight=0.1,
        ),
        Scenario(
            "feed_composition_step",
            {"feed_step": 0.0, "zF_step": 0.1},
            source="[D1, p. 33, Fig. 18]",
            description="the feed composition steps 0.1, unmeasured",
            weight=0.1,
        ),
        Scenario(
            "both_steps",
            {"feed_step": 0.2, "zF_step": 0.1},
            source="[D1, p. 33, Fig. 18]",
            description="both steps, at independent times, as Fig. 18 runs them",
            weight=0.1,
        ),
    )
