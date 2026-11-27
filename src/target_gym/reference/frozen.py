"""A kept rung as data, and the controller rebuilt from it with no bump test.

Charter J1 keeps one rung of the reference ladder per task
(``scripts/reference_ladder.py``). What the rung reads and how (its loops'
readings, the feedforward, the overrides' limits) is the task's ladder row. What
the bump tests and the zone placement found is a rung's law, a dict of numbers:

``rung``
    the rung's name (``"pi"``, ``"feedforward"``, ``"cascade"``, ...).
``loops``
    each of the row's loops, in order: its PI's ``kc``, ``ti`` and ``alpha``.
``mix``
    MVs by loops: the pairing, or the decoupler's matrix.
``bias``
    ``"balance"`` for the row's feedforward, else the constant raw action.
``setpoints``
    each zone loop's placed setpoint, by loop name. ``bound`` names those placed
    by the exit rule, whose row reads them while the rung steps.
``cascades``, ``overrides``, ``valve_positions``
    each one's numbers, in the row's order; a cascade's ``rest`` is None where
    the row computes it.
``start``
    whether the rung takes over bumplessly from the row's rest.
``dt``
    the control step, in seconds.

A published rung is ``{"rung": "published", "published": true}`` and a
practitioner's schedule ``{"rung": name, "schedule": setpoint}``.

:func:`assemble` turns a row and a law into the controller. The ladder builds
every rung through it, so a rung rebuilt from its frozen law runs the same
operations as the one the ladder kept.
"""

from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import numpy as np

from target_gym.experts.ladder import (
    Cascade,
    Override,
    Pi,
    Rung,
    ValvePosition,
    ladder_controller,
)
from target_gym.experts.pid import Controller

__all__ = ["FROZEN", "assemble", "bound", "changes", "constant", "load", "path"]
__all__ += ["write"]

#: One JSON per task: the kept rung's law and where it came from.
FROZEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "reference"


def path(task: str, root: pathlib.Path = FROZEN) -> pathlib.Path:
    return pathlib.Path(root) / f"{task}.json"


def load(task: str, root: pathlib.Path = FROZEN) -> dict | None:
    """Task ``task``'s frozen reference, None if it has none."""
    p = path(task, root)
    return json.loads(p.read_text()) if p.is_file() else None


def write(task: str, frozen: dict, root: pathlib.Path = FROZEN) -> pathlib.Path:
    p = path(task, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    return p


def changes(frozen: dict) -> dict:
    """The scored parameter changes a frozen reference's episodes run on, with
    the ranges JSON wrote as lists made tuples again."""
    return {
        k: tuple(v) if isinstance(v, list) else v for k, v in frozen["params"].items()
    }


def constant(value):
    """A reading that is ``value``, as float32."""
    value = float(value)
    return lambda o, xp: xp.asarray(value, dtype=xp.float32)


def bound(row, controller, placed: dict) -> Controller:
    """``controller`` with ``row.placed`` showing ``placed`` while it steps, also
    while it is traced: a row's own closures (a loop following a placed setpoint,
    a feedforward reading it) then see this rung's setpoints, not another
    rung's."""

    def step(params, memory, obs, xp=jnp):
        before = dict(row.placed)
        row.placed.update(placed)
        try:
            return controller.step(params, memory, obs, xp)
        finally:
            row.placed.clear()
            row.placed.update(before)

    return Controller(controller.params, controller.init_state, step)


def assemble(row, law: dict, spec=None) -> Controller:
    """Rung ``law`` of ``row`` as a pure controller. ``spec`` is the task's
    registry entry, which a published rung is built from."""
    if law.get("published"):
        return row.published(spec)
    if "schedule" in law:
        return row.schedules[law["rung"]](law["schedule"])

    def pi(loop, f):
        return Pi(loop, f["kc"], f["ti"], f["alpha"])

    loops = tuple(pi(i, f) for i, f in enumerate(law["loops"]))
    if len(loops) != len(row.loops):
        raise ValueError(
            f"the law has {len(loops)} loops and the row {len(row.loops)}: "
            "the row changed since the rung was frozen"
        )
    cascades = tuple(
        Cascade(
            c.read,
            c.rest if f["rest"] is None else _settled(f["rest"]),
            pi(c.loop, f),
        )
        for c, f in _named(row.cascades, law.get("cascades", ()))
    )
    overrides = tuple(
        Override(o.read, o.limit, o.floor, o.mv, f["gain_sign"], f["kc"], f["ti"])
        for o, f in _named(row.limits, law.get("overrides", ()))
    )
    valves = tuple(
        ValvePosition(v.read, v.setpoint, v.mv, f["kc"], f["ti"])
        for v, f in _named(row.valves, law.get("valve_positions", ()))
    )
    rung = Rung(
        law["rung"], loops, np.asarray(law["mix"], float), cascades, overrides, valves
    )
    setpoints = law.get("setpoints", {})
    targets = [
        constant(setpoints[c.name]) if c.name in setpoints else c.target
        for c in row.loops
    ]

    def read(o, xp):
        return xp.stack([c.read(o, xp) for c in row.loops])

    def target(o, xp):
        return xp.stack([t(o, xp) for t in targets])

    if law["bias"] == "balance":
        bias = row.balance
    else:
        const = np.asarray(law["bias"], float)

        def bias(o, xp):
            return xp.asarray(const, dtype=xp.float32)

    start = row.rest if law["start"] else None
    controller = ladder_controller(rung, read, target, bias, law["dt"], start)
    names = law.get("bound", ())
    if names:
        controller = bound(row, controller, {k: (None, setpoints[k]) for k in names})
    return controller


def _settled(value):
    """A cascade's rest where its reading settled in the bump test."""
    value = float(value)
    return lambda o, xp: value


def _named(parts, frozen) -> list:
    """The row's ``parts`` paired with their frozen numbers, which must name
    them in the row's order."""
    frozen = list(frozen)
    if frozen and [f["name"] for f in frozen] != [p.name for p in parts]:
        raise ValueError(
            f"the law names {[f['name'] for f in frozen]} and the row "
            f"{[p.name for p in parts]}: the row changed since the rung was frozen"
        )
    return list(zip(parts, frozen))
