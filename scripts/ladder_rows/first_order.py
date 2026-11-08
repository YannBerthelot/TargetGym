"""The reference ladder's row for ``first_order`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score.
"""

from __future__ import annotations

from reference_ladder import (  # noqa: E402
    Cv,
    Row,
    corners,
)


def row() -> Row:
    def half_target(o, xp):
        return xp.reshape(o[1] / 2.0, (1,))  # the plant's unit gain per raw of 2

    x = Cv("x on u", 0, lambda o, xp: o[0], lambda o, xp: o[1], "x", 0.1, 60)
    return Row(
        150,
        half_target,
        (x,),
        balance=half_target,
        points=corners(target_x_range=(0.5, 1.5)),
        why={"cascade": "one state", "decoupler": "one loop", "override": "no limit"},
    )
