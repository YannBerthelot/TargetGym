"""Attempt 3's scenario suite (charter Amendment 3, rule 20), each from REALITY.md.

A float furnace runs continuously, so an episode is a window of it: seven days of
established operation. Inside each scenario the walks and the event's start time are
drawn by the seed, inside the window the parameters give. Every event is one the record
shows; sizes the record doesn't give are ours inside the family's range, said where they
are set (``physical_env.EnvParams``).
"""

from target_gym.scenarios import Scenario

# Shares of operating time (charter J4), ours inside the family's ranges, which the
# family sweep draws from. No source says how often a float line changes pull or meets a
# gas swing of this size:
# - gas swings: none at entry points that stay within 4 MJ/m3 over two years [S20], up to
#   weekly where LNG and biomethane blend in [S19; S20]: each direction 0 to 0.3;
# - planned pull changes: none on a float line holding its pull within 2 % [S5], up to
#   about 20 job changes a month on a container line [S9]: each direction 0 to 0.3;
# - the normal day: the rest, 0.4 to 1.
WEIGHTS = {"normal": 0.7, "pull_up": 0.05, "pull_down": 0.05}
WEIGHTS.update(gas_rich=0.1, gas_lean=0.1)

PULL_STEP = (
    0.10  # ours, inside the family's 2 % float wander to a container's 29 to 42 %
)
GAS_STEP = 0.06  # 3 MJ/m3 within a day on a Wobbe index of about 50 [S20], arithmetic


def suite() -> tuple:
    """The scenarios, normal day first."""
    return (
        Scenario(
            "normal",
            {},
            "[S5] pull wanders about 2 %; [S19] gas composition weekly or daily",
            "established operation: the walks only",
            weight=WEIGHTS["normal"],
        ),
        Scenario(
            "pull_up",
            {"pull_change": PULL_STEP},
            "[S1, slide 10] pull change in float control; [S3, slide 8]; [S4]",
            "a planned 10 % rise in pull, shown ahead by its notice",
            weight=WEIGHTS["pull_up"],
        ),
        Scenario(
            "pull_down",
            {"pull_change": -PULL_STEP},
            "[S1, slide 10]; [S3, slide 8]; [S4]",
            "a planned 10 % fall in pull, shown ahead by its notice",
            weight=WEIGHTS["pull_down"],
        ),
        Scenario(
            "gas_rich",
            {"gas_swing": GAS_STEP},
            "[S20] LNG and biomethane move the Wobbe index over 3 MJ/m3 within a day",
            "an unannounced 6 % rise in the gas's heating value",
            weight=WEIGHTS["gas_rich"],
        ),
        Scenario(
            "gas_lean",
            {"gas_swing": -GAS_STEP},
            "[S20]",
            "an unannounced 6 % fall in the gas's heating value",
            weight=WEIGHTS["gas_lean"],
        ),
    )
