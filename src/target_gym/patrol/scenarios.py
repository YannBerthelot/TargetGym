"""The patrol tasks' scenario suites: each one window, the normal patrol, as the
benchmark has always scored it (one scenario, weight 1).

They are declared only to give the stream its numbers (docs/proposals/long-strata.md,
draft 4, S1b): a continuing patrol is a sequence of sorties, each starting with
the pair in cruise on a fresh route and station, each aircraft at a fresh mass.
Sources and searches: plane/REALITY.md, "The stream".
"""

from target_gym.plane.scenarios import A320_SORTIE_H, REGIME_H, stream

#: A Hornet patrol sortie's mean length, h: the RAAF's Hornets flew more than
#: 2 700 sorties and over 21 000 flying hours on Operation Okra, about 7.8 h a
#: sortie with tanker support [OKRA]. No training sortie length was found.
HORNET_SORTIE_H = 21_000.0 / 2_700.0


def patrol_suite():
    return (
        dict(
            name="normal",
            overrides={},
            source="a pair of Hornets on a patrol route (patrol/PHYSICS.md)",
            description="the wingman on its station through the route's turns",
            weight=1.0,
            **stream(REGIME_H, sortie_h=HORNET_SORTIE_H, zones=False),
        ),
    )


def wake_surf_suite():
    return (
        dict(
            name="normal",
            overrides={},
            source="fello'fly, two airliners 3 km in trail (plane3d/PHYSICS.md)",
            description="the follower on its station in the lead's upwash",
            weight=1.0,
            **stream(REGIME_H, sortie_h=A320_SORTIE_H, zones=False),
        ),
    )
