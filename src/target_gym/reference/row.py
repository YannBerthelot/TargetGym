"""What a reference ladder row is made of (charter J1).

A task's row describes its reference ladder: the loops a rung closes, what each
reads and holds, the feedforward the plant's balances give, the cascades, limits
and valve position loops it may add, and how its bump tests run. The rows
themselves are ``target_gym.reference.rows``, one module per task;
``scripts/reference_ladder.py`` runs the bump tests on them, and
``target_gym.reference`` rebuilds a kept rung on its row from frozen numbers.
"""

from __future__ import annotations

import dataclasses

__all__ = ["Cv", "Inner", "Limit", "Row", "Valve", "corners"]


@dataclasses.dataclass(frozen=True)
class Cv:
    """A loop: ``read`` on MV ``mv``, to ``target``, each ``f(obs, xp)``. ``du`` and
    ``window`` size its bump test, on the reading named ``inst``. ``valve`` says the
    MV has a valve's dead band. ``room`` replaces the plant's own limit. ``zone`` is
    ``(low, high, cheap)`` for a loop kept inside a zone: ``target`` is then unused,
    and the loop holds the setpoint the module docstring describes. ``exit_terms``
    names the reward terms the score charges for leaving the zone: with any, the
    setpoint is the first of the zone's grid that never leaves, not 2 sigma.
    ``established(setpoint)`` gives the scored parameters that start an episode on
    that setpoint, for a plant whose reset depends on it."""

    name: str
    mv: int
    read: object
    target: object
    inst: str
    du: float
    window: int
    model: str = "fopdt"
    valve: bool = True
    room: float | None = None
    zone: tuple | None = None
    exit_terms: tuple = ()
    established: object = None


@dataclasses.dataclass(frozen=True)
class Inner:
    """A cascade under loop ``loop``: ``read`` rests at ``rest``, or where it settled
    in the test if None. ``du`` and ``window`` size the inner test, ``du_outer`` and ``window_outer`` the outer one, in the inner
    reading's units."""

    loop: int
    name: str
    read: object
    rest: object
    inst: str
    du: float
    window: int
    du_outer: float
    window_outer: int
    model: str = "fopdt"
    valve: bool = True


@dataclasses.dataclass(frozen=True)
class Limit:
    """An override keeping ``read`` past ``limit``. ``slope`` and ``theta`` are the
    datasheet's where a bump can't read it; otherwise ``du`` and ``window`` size one."""

    name: str
    read: object
    limit: float
    floor: bool
    mv: int
    inst: str = ""
    du: float = 0.0
    window: int = 0
    slope: float | None = None
    theta: float | None = None
    seeds: object = None


@dataclasses.dataclass(frozen=True)
class Valve:
    """A valve position loop (``ladder.ValvePosition``): hold ``read``, a loop's
    valve, at ``setpoint`` by MV ``mv``. Its bump test steps ``mv`` by ``du`` with
    every loop of the row closed and reads the valve over ``window`` steps; SIMC
    then takes ``VALVE_TAU_C`` dead times as its closed-loop time."""

    name: str
    read: object
    setpoint: float
    mv: int
    du: float
    window: int


@dataclasses.dataclass
class Row:
    """What the ladder needs of one task. ``rest`` holds the plant for its tests, on
    the ``test`` parameters. ``points`` are the other operating points the episode
    visits, each a change to ``test``: the ends of its drawn targets and loads. Every
    loop is bump-tested at each and tuned where the plant is most sensitive.
    ``balance`` is rung 2's bias, None without one. ``why`` gives the reason for each
    rung left out, and ``points`` for a task with one operating point. ``rest_at``
    gives the rest at a point's parameters where ``rest`` can't read it. ``build``
    fills ``placed``: each zone loop's measured deviation (None under the exit rule)
    and setpoint; ``scans``: under the exit rule, the seeds that left at each
    candidate it ran, by scenario; and ``unheld``: those no candidate holds, by
    scenario, which did not constrain the placement. Under the exit rule each rung
    is placed on its own: ``placements`` holds, per rung and loop, its ``setpoint``,
    ``exits``, ``unheld`` and whether it ``holds`` every seed it can, and ``starts``
    the scored parameters its episodes start with; ``placed``, ``scans``, ``unheld``
    and ``score`` then show the pi rung's until ``keep_rung`` shows the kept one's.
    ``schedules`` are the practitioner laws that are not loops of PIs, by rung name,
    each a function of its zone loop's set point giving a ``Controller``; each is
    placed by the exit rule like the PI rungs (owner, 2026-11-25).
    ``bumpless`` starts every rung from ``rest`` at the reset
    (``ladder_controller``'s ``start``), for a task whose reset stands on a steady
    state its balances can't compute, because something they need is hidden."""

    steps: int
    rest: object
    loops: tuple = ()
    test: dict = dataclasses.field(default_factory=dict)
    points: tuple = ()
    rest_at: object = None
    score: dict = dataclasses.field(default_factory=dict)
    balance: object = None
    cascades: tuple = ()
    limits: tuple = ()
    valves: tuple = ()
    decouple: bool = False
    holds: tuple = ()
    start: object = None
    published: object = None
    why: dict = dataclasses.field(default_factory=dict)
    placed: dict = dataclasses.field(default_factory=dict)
    scans: dict = dataclasses.field(default_factory=dict)
    unheld: dict = dataclasses.field(default_factory=dict)
    placements: dict = dataclasses.field(default_factory=dict)
    starts: dict = dataclasses.field(default_factory=dict)
    bumpless: bool = False
    schedules: dict = dataclasses.field(default_factory=dict)
    #: Each rung's law as ``build`` made it (``target_gym.reference.frozen``), by
    #: rung: what ``--freeze`` writes for the kept one.
    frozen: dict = dataclasses.field(default_factory=dict)


def corners(**ranges) -> tuple:
    """The operating points at the ends of drawn ranges, every combination of ends,
    each range pinned to one end: ``corners(a=(0, 1))`` is ``a=(0, 0)``, ``a=(1, 1)``.
    """
    points = [{}]
    for name, (low, high) in ranges.items():
        points = [{**q, name: (end, end)} for q in points for end in (low, high)]
    return tuple(points)
