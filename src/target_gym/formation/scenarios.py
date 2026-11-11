"""The formation's scenario suite: its one window, the normal formation, as the
benchmark has always scored it (one scenario, weight 1).

It is declared only to give the stream its numbers (docs/proposals/long-strata.md,
draft 4, S1b): a continuing formation is a sequence of the patrol's sorties, each
starting on a fresh route and slots, each aircraft at a fresh mass. Sources and
searches: plane/REALITY.md, "The stream".
"""

from target_gym.patrol.scenarios import HORNET_SORTIE_H
from target_gym.plane.scenarios import REGIME_H, stream


def formation_suite():
    return (
        dict(
            name="normal",
            overrides={},
            source="a lead and two wingmen on the patrol's route (formation/PHYSICS.md)",
            description="the formation held through the route's turns",
            weight=1.0,
            **stream(REGIME_H, sortie_h=HORNET_SORTIE_H, zones=False),
        ),
    )
