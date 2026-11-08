"""Attempt 3's scenario suite (charter Amendment 3, rule 20), each from REALITY.md.

A float furnace runs continuously, so an episode is a window of it: seven days of
established operation. Inside each scenario the walks and the event's start time are
drawn by the seed, inside the window the parameters give. Every event is one the record
shows; sizes the record doesn't give are ours inside the family's range, said where they
are set (``physical_env.EnvParams``).
"""

from target_gym.scenarios import Scenario

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
        ),
        Scenario(
            "pull_up",
            {"pull_change": PULL_STEP},
            "[S1, slide 10] pull change in float control; [S3, slide 8]; [S4]",
            "a planned 10 % rise in pull, shown ahead by its notice",
        ),
        Scenario(
            "pull_down",
            {"pull_change": -PULL_STEP},
            "[S1, slide 10]; [S3, slide 8]; [S4]",
            "a planned 10 % fall in pull, shown ahead by its notice",
        ),
        Scenario(
            "gas_rich",
            {"gas_swing": GAS_STEP},
            "[S20] LNG and biomethane move the Wobbe index over 3 MJ/m3 within a day",
            "an unannounced 6 % rise in the gas's heating value",
        ),
        Scenario(
            "gas_lean",
            {"gas_swing": -GAS_STEP},
            "[S20]",
            "an unannounced 6 % fall in the gas's heating value",
        ),
    )
