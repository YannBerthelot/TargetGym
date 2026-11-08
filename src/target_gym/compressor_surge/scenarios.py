"""The compressor's scenario suite (charter Amendment 3, rule 20).

Each scenario is a window of the consumers' demand the family's sources
describe; levels and timing inside it are drawn by the seed. Sources are the
tags of this folder's ``REALITY.md``. Sizes and rates are ours: the sources
name the cases, not their numbers.

Weights (charter J4) are shares of operating time. No source says how often
each window occurs, so each is ours: the normal day in the family range 0.5 to
0.9, the rest split among the upsets.
"""

from target_gym.scenarios import Scenario


def scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "normal",
            {"demand_range": (0.7, 0.95), "demand_ramp_s": 5.0},
            source="[C3, p. 2]",
            description="demand high, the point right of the control line, "
            "the recycle shut or nearly",
            weight=0.7,
        ),
        Scenario(
            "slow_approach",
            {"demand_range": (0.25, 0.6), "demand_ramp_s": 15.0},
            source="[C3, p. 2]",
            description="demand falls slowly toward the control line",
            weight=0.1,
        ),
        Scenario(
            "fast_approach",
            {"demand_range": (0.2, 0.6), "demand_ramp_s": 1.0},
            source="[C3, p. 2]",
            description="demand falls fast; rate action or the backup line",
            weight=0.05,
        ),
        Scenario(
            "load_swings",
            {"demand_range": (0.35, 0.95), "demand_ramp_s": 5.0},
            source="[C3, pp. 2-4]",
            description="demand swings across the map, both approaches",
            weight=0.15,
        ),
    )
