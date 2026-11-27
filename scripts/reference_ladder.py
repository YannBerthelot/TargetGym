"""The reference controllers: one fixed ladder, run on every process, thermal and
energy task (docs/proposals/task-charter.md, J1).

Run with ``uv run python scripts/reference_ladder.py [--tasks ...] [--out DIR]``.
For each task it bump-tests the plant on nominal seeds 100 to 109, builds every rung
that applies, runs each on those seeds at sensor noise scale 1 over the task's scenario
suite, and keeps the cheapest rung that trips no seed. A rung's cost on a seed is the
task's ``J`` (charter J4): each scenario's mean cost per step, weighted by its share of
operating time. Nothing here reads a score to set a number: the gains come from the bump
tests, and the rows hold only what the plant's datasheet and balances say, plus each
test's step size and length.

Each task's row lives in its own file, ``scripts/ladder_rows/<task>.py``, defining
``row()``. A loop on a zone (``Cv.zone``) holds a setpoint ``k = 2`` standard deviations
inside the zone's cheap edge, never past its centre (``ladder.zone_setpoint``). The
deviation is the controlled variable's, with the pi rung holding it on the bump-test
seeds in the scored normal day, over the second half of each episode: first at the
zone's centre, then at each placed setpoint until the setpoint stops moving, since on
a nonlinear plant the spread depends on where the loop holds.
A zone whose exit the score prices as a violation is placed otherwise (J1, owner,
2026-11-23): its loop names the reward terms charged for leaving (``Cv.exit_terms``),
and each rung, on its own, holds the first setpoint of ``ladder.zone_grid``, from the
cheap edge, at which it leaves the zone on none of the seeds in any scenario
(``zone_exits``, ``ladder.first_without_exit``). A seed of a scenario that leaves
from every candidate does not constrain a rung's placement: the rung pays for its
exits there and it is listed. The ladder then keeps the cheapest rung that trips no
seed and leaves on none of the seeds it can hold (``keep_rung``). Where the plant's
reset depends on the setpoint (``Cv.established``), each candidate's episodes start on
it, and a rung's runs start on its own. A row whose reset no balance can compute
starts ``bumpless``, and so do its candidates. ``Row.placements`` and ``Row.starts``
keep each rung's; ``Row.placed``, ``Row.scans``, ``Row.unheld`` and the start in
``Row.score`` show the pi rung's until a rung is kept, then the kept one's.
Bumps are read against the same seed's unstepped run (``Plant(paired=True)``), so a
slow walk or drift the two runs share leaves the response.

A rung a task doesn't have is reported with the reason from its row.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym import evaluation, scenarios  # noqa: E402
from target_gym.benchmark import (  # noqa: E402
    episodes_of,
    run_policy,
    shipped_policy,
    task_info,
)
from target_gym.experts.ladder import (  # noqa: E402
    Pi,
    Rung,
    best_rung,
    cv_sigma,
    decoupler,
    first_without_exit,
    ladder_controller,
    leaves_only_unheld,
    most_sensitive,
    null_verdict,
    zone_grid,
    zone_setpoint,
)
from target_gym.experts.pid import Controller  # noqa: E402
from target_gym.identification import (  # noqa: E402
    DEAD_BAND,
    Instrument,
    Plant,
    noise_tune,
    pick_fit,
    simc,
    single_run,
)
from target_gym.reference.frozen import assemble, constant  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

SEEDS = tuple(range(100, 110))
#: The seeds a zone's setpoint is placed on (outside review M5, 2026-11-27): apart
#: from the ladder's 100 to 109 the J3 lines run on, the scoring seeds from 200 and
#: J2's hash seeds from 1000, so no line is read on the seeds its reference was
#: placed on.
PLACEMENT_SEEDS = tuple(range(110, 120))
NOISE = 1.0
RAW_BAND = DEAD_BAND * 2.0  # a valve's dead band on an action spanning 2
FREE = (-np.inf, np.inf)
#: A valve position loop's closed-loop time, in dead times. Its source tuned it at one
#: dead time, as every other loop here, found "an aggressive behavior and oscillations"
#: and gives the remedy: it "should be increased to at least" 30 (Reyes-Lua, Zotica and
#: Skogestad, IFAC ADCHEM 2018, p. 438). On the cstr production prototype one dead time
#: asks a gain of 8.2 and six of ten normal-day seeds leave the band at every set point.
VALVE_TAU_C = 30.0


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


def scheduled(name, row, spec, label, law):
    """The scheduled rung ``label``, ``law(setpoint)``, at the first point of its
    zone loop's grid, from the cheap edge, at which it leaves the zone on none of
    the seeds it can hold (J1, as ``build`` places the PI rungs)."""
    (c,) = [c for c in row.loops if c.zone and c.exit_terms]

    def exits_at(point):
        left = zone_exits(
            name,
            row,
            Reference("candidate", law(point)),
            c.exit_terms,
            spec,
            seeds=PLACEMENT_SEEDS,
        )
        seeds = len({seed for _, seed in left})
        print(
            f"    {label}, {c.name}: setpoint {point:.4g}, {len(left)} episodes "
            f"leave, on {seeds} seeds",
            flush=True,
        )
        return left

    setpoint, scan, unheld = first_without_exit(zone_grid(*c.zone), exits_at)
    own = {
        "setpoint": setpoint,
        "exits": {k: _by_scenario(v) for k, v in scan.items()},
        "unheld": _by_scenario(unheld),
        "holds": leaves_only_unheld(setpoint, scan, unheld),
    }
    row.placements[label], row.starts[label] = {c.name: own}, {}
    row.frozen[label] = {"rung": label, "schedule": setpoint}
    print(f"    {label}, {c.name}: kept setpoint {setpoint:.4g}", flush=True)
    return assemble(row, row.frozen[label], spec)


def corners(**ranges) -> tuple:
    """The operating points at the ends of drawn ranges, every combination of ends,
    each range pinned to one end: ``corners(a=(0, 1))`` is ``a=(0, 0)``, ``a=(1, 1)``.
    """
    points = [{}]
    for name, (low, high) in ranges.items():
        points = [{**q, name: (end, end)} for q in points for end in (low, high)]
    return tuple(points)


def manual(hold):
    return Controller(None, 0.0, lambda _p, s, obs, xp=jnp: (hold(obs, xp), s))


def read_loop(
    plant, label, mv, cv, du, window, inst, room, model, loops=(), seeds=None
):
    """One loop's model from single-run bump tests on every seed."""
    seeds = range(plant.n_seeds) if seeds is None else seeds
    read = single_run(
        lambda d: [plant.bump(s, mv, cv, d, window, inst, loops) for s in seeds],
        du,
        room,
        inst,
        model,
        None,
    )
    best, _, _ = pick_fit(read.fits)
    integrating = model != "fopdt"
    out = {
        "k": float(best["k"]),
        "tau": None if integrating else float(best["tau"]),
        "theta": float(
            best["theta"] + inst.half_sample + (best["tau2"] if integrating else 0.0)
        ),
        "band": float(read.nb),
    }
    print(f"  {label}: {out}", flush=True)
    return out


def design(model, inst, db) -> dict:
    """SIMC with ``tc = theta``, with the reading's filter (N5, N6). The noise never
    lowers the gain: ``kc_min`` infinite stops N7."""
    return noise_tune(
        model["k"], model["tau"], model["theta"], model["band"], inst.h, db, math.inf
    )


def pi_of(i, out, dt) -> Pi:
    tf = out["Tf"]
    return Pi(i, out["Kc"], out["Ti"], -math.expm1(-dt / tf) if tf else 1.0)


def wrap(read):
    return lambda o, s, q: read(o, jnp)


def closed_inner(pis, mvs, n_cv, dt):
    """The inner PIs as the law runs them, for a test whose MV is their setpoint."""

    def actuate(u, y, m):
        m = [(y[n_cv + i], 0.0) for i in range(len(mvs))] if m is None else m
        u, out = np.array(u, float), []
        for i, (mv, p) in enumerate(zip(mvs, pis)):
            f, integral = m[i]
            f = f + p.alpha * (y[n_cv + i] - f)
            e = u[mv] - f
            v = p.kc * (e + (integral + e * dt) / p.ti)
            if -1.0 <= v <= 1.0:
                integral += e * dt
            u[mv] = np.clip(v, -1.0, 1.0)
            out.append((f, integral))
        return u, out

    return actuate


def closed_loops(pis, mvs, dt):
    """The loops' PIs as the law runs them, each holding its CV where the test
    found it and starting from its MV at rest, for a test of another MV."""

    def actuate(u, y, m):
        u = np.array(u, float)
        if m is None:
            m = [(y[p.loop], 0.0, y[p.loop], u[mv]) for p, mv in zip(pis, mvs)]
        out = []
        for (f, integral, sp, base), p, mv in zip(m, pis, mvs):
            f = f + p.alpha * (y[p.loop] - f)
            e = sp - f
            v = base + p.kc * (e + (integral + e * dt) / p.ti)
            if -1.0 <= v <= 1.0:
                integral += e * dt
            u[mv] = np.clip(v, -1.0, 1.0)
            out.append((f, integral, sp, base))
        return u, out

    return actuate


def gain_matrix(plant, row, inst, n):
    """Steady gains, CVs by MVs, each read at the end of the longest window."""
    window = max(c.window for c in row.loops)
    G = np.zeros((n, n))
    for i in range(n):
        for j, c in enumerate(row.loops):
            room = plant.room(c.mv)
            gains = []
            for s in range(plant.n_seeds):
                b = plant.bump(s, c.mv, i, min(c.du, room), window, inst)
                tail = b.dy[-max(len(b.dy) // 10, 1) :]
                gains.append(float(np.mean(tail)) / b.step)
            G[i, j] = np.median(gains)
    return G


def build(name: str, row: Row, spec=None) -> dict:
    """Every rung of ``row``, as ``{rung: Controller}``."""
    spec = spec or REGISTRY[name]
    env = spec.make_env()
    p = spec.make_test_params(**_test_params(row))
    rungs = {}
    row.placements, row.starts, row.frozen = {}, {}, {}  # this build's own, per rung
    if row.published:
        row.frozen["published"] = {"rung": "published", "published": True}
        rungs["published"] = assemble(row, row.frozen["published"], spec)
    if not row.loops:
        return rungs

    n, nc = len(row.loops), len(row.cascades)
    cvs = [wrap(c.read) for c in row.loops] + [wrap(c.read) for c in row.cascades]
    cvs += [wrap(o.read) for o in row.limits]
    cvs += [wrap(v.read) for v in row.valves]

    def params_at(change):
        return spec.make_test_params(**{**_test_params(row), **change})

    def rest_of(q):
        return row.rest_at(q) if row.rest_at else row.rest

    def plant_at(q):
        return Plant(
            spec,
            q,
            cvs,
            [(None, None)] * len(cvs),
            controller=manual(rest_of(q)),
            start=row.start,
            paired=True,
        )

    at_points = [params_at(change) for change in ({}, *row.points)]
    plants = [plant_at(q) for q in at_points]
    plant = plants[0]
    dt = plant.dt

    def inst_of(tag):
        return Instrument.of(env, p, tag)

    def tune(label, mv, cv, du, window, inst, db, model="fopdt", room=None, on=None):
        """SIMC at every operating point, kept where the plant is most sensitive."""
        designs = []
        for k, at in enumerate(on or plants):
            fit = read_loop(
                at,
                f"{label} at point {k}",
                mv,
                cv,
                du,
                window,
                inst,
                room or at.room(mv),
                model,
                row.holds,
            )
            designs.append(design(fit, inst, db))
        out = most_sensitive(designs)
        print(f"    {label}: {out}", flush=True)
        return out

    def valve(c):
        return RAW_BAND if c.valve else None

    pis = []
    for i, c in enumerate(row.loops):
        out = tune(
            c.name, c.mv, i, c.du, c.window, inst_of(c.inst), valve(c), c.model, c.room
        )
        pis.append(pi_of(i, out, dt))

    m = env.action_space(p).shape[0]
    mix = np.zeros((m, n))
    for i, c in enumerate(row.loops):
        mix[c.mv, i] = 1.0
    const = np.median(plant.u0, axis=0)
    flat = lambda o, xp: xp.asarray(const, dtype=xp.float32)  # noqa: E731

    def read(o, xp):
        return xp.stack([c.read(o, xp) for c in row.loops])

    def stacked(targets):
        return lambda o, xp: xp.stack([t(o, xp) for t in targets])

    targets = [c.target for c in row.loops]
    zoned = [i for i, c in enumerate(row.loops) if c.zone]
    #: Each zone's setpoint by name: the 2 sigma loops' once placed, and the
    #: centre of each loop priced as a violation until a rung places it.
    fixed = {}
    if zoned:
        centres = list(targets)
        for i in zoned:
            low, high, _ = row.loops[i].zone
            centres[i] = constant(0.5 * (low + high))
        for i in zoned:
            c = row.loops[i]
            if c.exit_terms:
                # Placed for each rung on its own, once the rung exists (``make``);
                # the centre meanwhile, for the row's 2 sigma loops' probes.
                fixed[c.name] = 0.5 * (c.zone[0] + c.zone[1])
                continue
            setpoint, at = None, list(centres)
            # The spread is the controlled variable's where its loop holds, which on
            # a nonlinear plant moves with the setpoint: start at the centre, then
            # re-read at each placed setpoint, with the row's other loops following
            # it (``row.placed``), until it moves less than PLACE_TOLERANCE sigma.
            for _ in range(PLACE_ROUNDS):
                probe = ladder_controller(
                    Rung("pi", tuple(pis), mix), read, stacked(at), flat, dt
                )
                held = cv_traces(name, row, probe, i, spec, PLACEMENT_SEEDS)
                sigma = cv_sigma([t[len(t) // 2 :] for t in held])
                last, setpoint = setpoint, zone_setpoint(*c.zone[:2], sigma, c.zone[2])
                row.placed[c.name] = (sigma, setpoint)
                print(
                    f"    {c.name}: sigma {sigma:.4g}, setpoint {setpoint:.4g}",
                    flush=True,
                )
                if last is not None and abs(setpoint - last) <= PLACE_TOLERANCE * sigma:
                    break
                at[i] = constant(setpoint)
            else:
                print(f"    {c.name}: placement still moving after {PLACE_ROUNDS}")
            fixed[c.name] = setpoint
    priced = [i for i in zoned if row.loops[i].exit_terms]

    def law(label, loops, bias, mix=mix, cascades=(), overrides=(), valves=()):
        """Rung ``label``'s numbers (``target_gym.reference.frozen``): ``bias``
        ``"flat"`` holds the bump tests' median rest, ``"balance"`` the row's."""
        return {
            "rung": label,
            "loops": [_numbers(c.name, q) for c, q in zip(row.loops, loops)],
            "mix": np.asarray(mix, float).tolist(),
            "bias": "balance" if bias == "balance" else [float(u) for u in const],
            "setpoints": dict(fixed),
            "bound": [],
            "cascades": list(cascades),
            "overrides": list(overrides),
            "valve_positions": list(valves),
            "start": bool(row.bumpless),
            "dt": float(dt),
        }

    def make(numbers):
        """The rung of law ``numbers`` (``law``) as a controller, its law kept in
        ``row.frozen``. Its loops on zones priced as a violation are placed for it
        alone (J1): each holds the first candidate, from the cheap edge, at which
        this rung leaves the zone on none of the seeds it can hold."""
        label = numbers["rung"]
        if not priced:
            row.frozen[label] = numbers
            return assemble(row, numbers, spec)
        kept, begun, record = {}, {}, {}

        def holding(setpoints):
            """The rung holding each loop of ``setpoints`` there, and its law."""
            placed = {row.loops[j].name: v for j, v in setpoints.items()}
            own = {**numbers["setpoints"], **placed}
            held = {**numbers, "setpoints": own, "bound": list(placed)}
            return assemble(row, held, spec), held

        for i in priced:
            c = row.loops[i]

            def exits_at(point, c=c, i=i):
                # The candidate's episodes start on it; the row's other loops
                # follow it while its controller steps (``frozen.bound``).
                extra = {**begun, **(c.established(point) if c.established else {})}
                probe = Reference("candidate", holding({**kept, i: point})[0])
                left = zone_exits(
                    name, row, probe, c.exit_terms, spec, extra, PLACEMENT_SEEDS
                )
                seeds = len({seed for _, seed in left})
                print(
                    f"    {label}, {c.name}: setpoint {point:.4g}, {len(left)} "
                    f"episodes leave, on {seeds} seeds",
                    flush=True,
                )
                return left

            setpoint, scan, unheld = first_without_exit(zone_grid(*c.zone), exits_at)
            kept[i] = setpoint
            if c.established:
                begun.update(c.established(setpoint))
            record[c.name] = {
                "setpoint": setpoint,
                "exits": {k: _by_scenario(v) for k, v in scan.items()},
                "unheld": _by_scenario(unheld),
                "holds": leaves_only_unheld(setpoint, scan, unheld),
            }
            if unheld:
                said = (
                    "no seed can be held, so the rule decided nothing and the "
                    "centre is kept"
                    if len(unheld) == _episodes(spec)
                    else "they do not constrain the placement"
                )
                print(
                    f"    {label}, {c.name}: no candidate holds "
                    f"{record[c.name]['unheld']}: {said}"
                )
            if not record[c.name]["holds"]:
                still = _by_scenario(set(scan[setpoint]) - set(unheld))
                print(
                    f"    {label}, {c.name}: no candidate holds every seed that "
                    f"some candidate holds; at the centre {still} leave"
                )
            print(f"    {label}, {c.name}: kept setpoint {setpoint:.4g}", flush=True)
        row.placements[label], row.starts[label] = record, begun
        controller, row.frozen[label] = holding(kept)
        return controller

    loops = tuple(pis)
    rungs["pi"] = make(law("pi", loops, "flat"))
    bias = "flat"
    if row.balance:
        bias = "balance"
        rungs["feedforward"] = make(law("feedforward", loops, bias))

    cascades = ()
    if row.cascades:
        inner = []
        for k, c in enumerate(row.cascades):
            mv = row.loops[c.loop].mv
            out = tune(
                c.name, mv, n + k, c.du, c.window, inst_of(c.inst), valve(c), c.model
            )
            inner.append(pi_of(c.loop, out, dt))
        outer = [
            Plant(
                spec,
                q,
                cvs,
                [(None, None)] * len(cvs),
                controller=manual(rest_of(q)),
                start=_inner_start(at, row, n),
                actuate=closed_inner(
                    inner, [row.loops[c.loop].mv for c in row.cascades], n, dt
                ),
                mv_range=FREE,
                paired=True,
            )
            for at, q in zip(plants, at_points)
        ]
        tuned = list(pis)
        for k, c in enumerate(row.cascades):
            base = row.loops[c.loop]
            label = f"{base.name} over {c.name}"
            room = 1.0 / abs(inner[k].kc)  # what the inner loop's output can answer
            db = RAW_BAND / abs(inner[k].kc)
            out = tune(
                label,
                base.mv,
                c.loop,
                c.du_outer,
                c.window_outer,
                inst_of(base.inst),
                db,
                base.model,
                room,
                outer,
            )
            tuned[c.loop] = pi_of(c.loop, out, dt)
        # Each inner loop rests where the row computes it, or where its
        # reading settled in the test.
        cascades = tuple(
            {
                **_numbers(c.name, q),
                "rest": None if c.rest else float(np.median(plant._y[:, n + k])),
            }
            for k, (c, q) in enumerate(zip(row.cascades, inner))
        )
        loops = tuple(tuned)
        rungs["cascade"] = make(law("cascade", loops, bias, cascades=cascades))

    for label, schedule in row.schedules.items():
        rungs[label] = scheduled(name, row, spec, label, schedule)

    if row.decouple:
        G = gain_matrix(plant, row, inst_of(row.loops[0].inst), n)
        print(f"    gains: {np.round(G, 4).tolist()}", flush=True)
        D = np.zeros((m, n))
        D[np.ix_([c.mv for c in row.loops], range(n))] = decoupler(G, range(n))
        rungs["decoupler"] = make(law("decoupler", tuple(pis), bias, mix=D))

    if row.limits:
        over = []
        for k, o in enumerate(row.limits):
            if o.slope is not None:
                model = {"k": o.slope, "tau": None, "theta": o.theta, "band": 0.0}
                out = design(model, Instrument(dt), None)
                print(f"    {o.name}: {out}", flush=True)
            else:
                designs = []
                for j, at in enumerate(plants):
                    model = read_loop(
                        at,
                        f"{o.name} at point {j}",
                        o.mv,
                        n + nc + k,
                        o.du,
                        o.window,
                        inst_of(o.inst),
                        at.room(o.mv),
                        "fopdt",
                        row.holds,
                        o.seeds,
                    )
                    designs.append((design(model, Instrument(dt), None), model))
                out, model = min(designs, key=lambda d: abs(d[0]["Kc"]))
                print(f"    {o.name}: {out}", flush=True)
            over.append(
                {
                    "name": o.name,
                    "gain_sign": float(np.sign(model["k"])),
                    "kc": float(out["Kc"]),
                    "ti": float(out["Ti"]),
                }
            )
        rungs["override"] = make(
            law("override", loops, bias, cascades=cascades, overrides=over)
        )
    if row.valves:
        # Each valve's answer to its MV with the loops closed, as the structure's
        # source tunes it, on plants whose loops hold their CVs at rest.
        closed = [
            Plant(
                spec,
                q,
                cvs,
                [(None, None)] * len(cvs),
                controller=manual(rest_of(q)),
                start=row.start,
                actuate=closed_loops(pis, [c.mv for c in row.loops], dt),
                paired=True,
            )
            for q in at_points
        ]
        first = n + nc + len(row.limits)
        held = []
        for k, v in enumerate(row.valves):
            designs = []
            for j, at in enumerate(closed):
                m = read_loop(
                    at, f"{v.name} at point {j}", v.mv, first + k, v.du, v.window,
                    Instrument(dt), at.room(v.mv), "fopdt", row.holds,
                )  # fmt: skip
                kc, ti = simc(m["k"], m["tau"], m["theta"], VALVE_TAU_C * m["theta"])
                designs.append({"Kc": kc, "Ti": ti})
            out = most_sensitive(designs)
            print(f"    {v.name}: {out}", flush=True)
            held.append(
                {"name": v.name, "kc": float(out["Kc"]), "ti": float(out["Ti"])}
            )
        # A row's overrides ride on its valve position rung too, selecting against
        # the valve position loop's MV: Shinskey's high-temperature override on the
        # feed [S13, p. 436].
        rungs["valve position"] = make(
            law(
                "valve position",
                tuple(pis),
                bias,
                overrides=over if row.limits else (),
                valves=held,
            )
        )
    _show(row, "pi")
    return rungs


def _show(row, label) -> None:
    """Show rung ``label``'s own placement as the row's: ``placed``, ``scans``,
    ``unheld``, and its start in ``score``."""
    for loop, own in row.placements.get(label, {}).items():
        row.placed[loop] = (None, own["setpoint"])
        row.scans[loop], row.unheld[loop] = own["exits"], own["unheld"]
    row.score = {**row.score, **row.starts.get(label, {})}


def keep_rung(row, results: dict):
    """The rung the ladder keeps of ``results`` (``run_rung``'s, by rung): the
    cheapest that trips no seed and leaves its zones on none of the seeds it can
    hold, None if every rung trips. If no rung holds what it can, the cheapest that
    trips none, and it says so. The row then shows the kept rung's placement and
    start (``_show``), for the runs that follow it: the nulls, the oracle, the
    ceiling."""
    order = [k for k in (*_ORDER, *row.schedules) if k in results]
    ranked = {k: results[k] for k in order}
    holds = {
        k: all(own["holds"] for own in row.placements.get(k, {}).values())
        for k in order
    }
    kept = best_rung(ranked, holds)
    if kept is None and not all(holds.values()):
        kept = best_rung(ranked)
        if kept:
            print(
                "  no rung holds every seed it can: kept the cheapest that trips none"
            )
    if kept:
        _show(row, kept)
    return kept


def _test_params(row) -> dict:
    """The bump tests' parameters: noise scale 1 and the row's episode, then the
    row's own changes, which may quiet a disturbance for clean bumps."""
    return {"sensor_noise_scale": NOISE, "max_steps_in_episode": row.steps, **row.test}


def _numbers(name, pi) -> dict:
    """A loop's PI as its frozen numbers."""
    return {
        "name": name,
        "kc": float(pi.kc),
        "ti": float(pi.ti),
        "alpha": float(pi.alpha),
    }


def _inner_start(plant, row, n):
    """The test's MVs at rest: each cascade's stands at where its reading settled."""

    def start(a, m):
        u = plant.u0.copy()
        for k, c in enumerate(row.cascades):
            u[:, row.loops[c.loop].mv] = plant._y[:, n + k]
        return u

    return start


class Reference:
    """A pure controller as a batched ``benchmark.Policy``."""

    oracle = False

    def __init__(self, name, controller):
        self.name, self.c = name, controller
        self._step = jax.jit(
            jax.vmap(lambda m, o: controller.step(controller.params, m, o, jnp))
        )

    def reset(self, key, obs, task):
        return jax.tree.map(
            lambda x: jnp.repeat(jnp.asarray(x)[None], task.n_episodes, 0),
            self.c.init_state,
        )

    def act(self, carry, obs, t):
        action, carry = self._step(carry, jnp.asarray(obs))
        return np.asarray(action, float), carry


class Recorder(Reference):
    """A ``Reference`` that also keeps ``read(obs)`` for every episode and step."""

    def __init__(self, name, controller, read):
        super().__init__(name, controller)
        self.read, self.values = read, []

    def act(self, carry, obs, t):
        rows = np.asarray(obs)
        self.values.append([float(self.read(o, np)) for o in rows])
        return super().act(carry, obs, t)


def cv_traces(name, row, controller, i, spec=None, seeds=SEEDS) -> list:
    """Loop ``i``'s reading under ``controller``, one array per seed of ``seeds``,
    on the scored normal day: the suite's first scenario at the scored parameters,
    so a disturbance a row quiets for its bump tests still moves it. The 2 sigma
    rule reads it on ``PLACEMENT_SEEDS``."""
    spec = spec or REGISTRY[name]
    env = spec.make_env()
    normal = scenarios.suite(spec)[0]
    params = scenarios.params_for(
        spec, normal, **{"sensor_noise_scale": NOISE, **row.score}
    )
    task = task_info(name, env, params, len(seeds))
    policy = Recorder(row.loops[i].name, controller, row.loops[i].read)
    rollout = run_policy(policy, env, params, task, seeds=seeds)
    values = np.asarray(policy.values)  # (steps, seeds)
    lengths = np.asarray(rollout["length"], int)
    return [values[:L, k] for k, L in enumerate(lengths)]


def run_rung(name, row, label, controller, spec=None) -> dict:
    """``controller`` on the ladder's seeds over the task's scenarios: each seed's
    ``J``, the seeds that trip in any scenario, and each scenario's own costs."""
    return run_suite(name, row, Reference(label, controller), spec)


def _suite_episodes(name, row, policy, spec, seeds=None, extra=None):
    """``(scenario, episodes)`` of ``policy`` over the suite at the row's scored
    parameters, then the start of the rung ``policy`` is named after, if it is one
    (``Row.starts``), then ``extra``."""
    env = spec.make_env()
    for s in scenarios.suite(spec):
        own = _own_seeds(seeds, s.name)
        own_start = row.starts.get(getattr(policy, "name", None), {})
        params = scenarios.params_for(
            spec,
            s,
            **{"sensor_noise_scale": NOISE, **row.score, **own_start, **(extra or {})},
        )
        task = task_info(name, env, params, len(own))
        if hasattr(policy, "bind"):  # a null that reads each episode's reset
            policy.bind(env, params, own)
        yield s.name, episodes_of(run_policy(policy, env, params, task, seeds=own))


def _own_seeds(seeds, scenario: str) -> tuple:
    """A scenario's seeds of ``seeds``: one tuple for every scenario, or
    ``{scenario: seeds}``; the ladder's where it names none."""
    own = seeds.get(scenario, SEEDS) if isinstance(seeds, dict) else seeds
    return tuple(own or SEEDS)


def zone_exits(name, row, policy, terms, spec=None, extra=None, seeds=None) -> tuple:
    """The seeds on which ``policy`` leaves a zone, each as ``(scenario, seed)``, in
    order: a step the task's evaluation scores (``evaluation.Evaluation``) charged
    one of the reward ``terms`` the score prices the exit with, or a trip. ``extra``
    joins the row's scored parameters. ``seeds`` is one tuple or ``{scenario:
    seeds}``, the ladder's by default; a placement runs on ``PLACEMENT_SEEDS``."""
    spec = spec or REGISTRY[name]
    left = []
    for scenario, episodes in _suite_episodes(name, row, policy, spec, seeds, extra):
        own = _own_seeds(seeds, scenario)
        for seed, e in zip(own, episodes, strict=True):
            unknown = sorted(set(terms) - set(e.terms))
            if unknown:
                raise KeyError(f"{name}: no reward term {unknown}")
            read = evaluation.scored_window(spec, len(e.cost))
            charged = any(bool(np.any(e.terms[t][read] > 0.0)) for t in terms)
            if charged or bool(np.any(e.failed)):
                left.append((scenario, seed))
    return tuple(sorted(left))


def _by_scenario(left) -> dict:
    """``(scenario, seed)`` pairs as ``{scenario: seeds}``, for a record."""
    out: dict = {}
    for scenario, seed in sorted(left):
        out.setdefault(scenario, []).append(seed)
    return out


def _keyed(scan: dict) -> dict:
    """A scan with its setpoints as text, for JSON."""
    return {f"{point:.6g}": left for point, left in scan.items()}


def _episodes(spec) -> int:
    """The episodes of one placement run over the suite."""
    return len(PLACEMENT_SEEDS) * len(scenarios.suite(spec))


def run_suite(name, row, policy, spec=None, seeds=None) -> dict:
    """``run_rung`` for any ``benchmark.Policy``, each episode's cost read over the
    task's scored window (``evaluation.episode_cost``). ``seeds`` replaces the ladder's,
    as one tuple or ``{scenario: seeds}``; with counts that differ by scenario,
    ``costs`` (per seed across scenarios) is None and ``trips`` the scenarios' sum."""
    spec = spec or REGISTRY[name]
    weights = scenarios.weights(spec)
    per = {}
    for scenario, episodes in _suite_episodes(name, row, policy, spec, seeds):
        per[scenario] = (
            np.array([evaluation.episode_cost(spec, e) for e in episodes]),
            np.array([bool(np.any(e.failed)) for e in episodes]),
        )
    if len({len(c) for c, _ in per.values()}) > 1:
        costs, tripped = None, np.array([t.sum() for _, t in per.values()])
    else:
        costs = [float(c) for c in sum(weights[n] * c for n, (c, _) in per.items())]
        tripped = np.any([t for _, t in per.values()], axis=0)
    return {
        "costs": costs,
        "trips": int(tripped.sum()),
        "scenarios": {
            n: {
                "costs": [float(x) for x in c],
                "trips": int(t.sum()),
                "tripped": [bool(x) for x in t],
            }
            for n, (c, t) in per.items()
        },
    }


def _climb(name: str, row=None) -> tuple:
    """``(row, rungs, results, kept)``: every rung of ``name``'s row (``row``, or
    its file's) built and run on the ladder's seeds, and the one kept."""
    print(name, flush=True)
    row = ROWS[name]() if row is None else row
    rungs, results = build(name, row), {}
    for label, controller in rungs.items():
        results[label] = run_rung(name, row, label, controller)
        r = results[label]
        print(f"  {label}: median {np.median(r['costs']):.4g}, {r['trips']} trips")
    kept = keep_rung(row, results)
    print(f"  kept: {kept}", flush=True)
    return row, rungs, results, kept


def climb(name: str, nulls: bool = False) -> dict:
    """The ladder on ``name``: each rung's result and the one kept, and with
    ``nulls`` the null policies against the kept rung (``check_nulls``)."""
    row, _, results, kept = _climb(name)
    placed = {k: {"sigma": v[0], "setpoint": v[1]} for k, v in row.placed.items()}
    for k, scan in row.scans.items():  # the exit rule's candidates, as run
        placed[k]["exits"] = _keyed(scan)
        placed[k]["unheld"] = row.unheld[k]
    out = {"rungs": results, "kept": kept, "left_out": row.why, "placed": placed}
    if row.placements:  # each rung's own placement; ``placed`` is the kept rung's
        out["rungs_placed"] = {
            rung: {
                loop: {**own, "exits": _keyed(own["exits"])} for loop, own in v.items()
            }
            for rung, v in row.placements.items()
        }
    if nulls and kept:
        out["nulls"] = check_nulls(name, row, results[kept])
    return out


def frozen_params(row, rung: str) -> dict:
    """The scored parameter changes rung ``rung``'s episodes run on, as
    ``_suite_episodes`` sets them: the noise scale, the row's, and the rung's own
    start (``Row.starts``)."""
    return {"sensor_noise_scale": NOISE, **row.score, **row.starts.get(rung, {})}


def exits_report(name, row, policy, seeds, spec=None) -> dict | None:
    """Where ``policy`` leaves the row's zones priced as a violation on ``seeds``,
    ``{scenario: seeds}``: the episodes that left, by scenario, out of how many.
    Run on J2's hash seeds, apart from those a zone was placed on, it is the
    reference's exits out of sample (outside review M5). None for a row with no
    such zone."""
    terms = sorted({t for c in row.loops if c.zone for t in c.exit_terms})
    if not terms:
        return None
    left = zone_exits(name, row, policy, tuple(terms), spec, seeds=seeds)
    return {
        "terms": terms,
        "seeds": {s: [int(x) for x in v] for s, v in seeds.items()},
        "left": _by_scenario(left),
        "episodes": sum(len(v) for v in seeds.values()),
    }


def _git() -> tuple[str, bool]:
    import subprocess

    def git(*a):
        root = pathlib.Path(__file__).resolve().parent.parent
        return subprocess.run(["git", *a], cwd=root, capture_output=True, text=True)

    status = git("status", "--porcelain", "--untracked-files=no").stdout
    return git("rev-parse", "HEAD").stdout.strip(), bool(status.strip())


def _provenance(name: str, kind: str, rung: str) -> dict:
    """What a frozen reference was frozen on: the plant's stamp, the charter's
    reference fingerprint (``charter.current``) and the commit."""
    import datetime

    from target_gym import charter
    from target_gym.provenance import environment_fingerprint

    spec = REGISTRY[name]
    commit, dirty = _git()
    return {
        "task": name,
        "format": 1,
        "kind": kind,
        "rung": rung,
        "env": f"{name}-v{spec.version}",
        "stamp": environment_fingerprint(spec),
        "fingerprint": charter.current(name)["fingerprints"]["reference"],
        "commit": commit,
        "dirty": dirty,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def placement_report(row, rung: str) -> dict:
    """Each zone loop's placement for rung ``rung``: the exit rule's setpoint, how
    many episodes left at each candidate run, the seeds no candidate held, by
    scenario, and whether the rung holds every seed it can; or the 2 sigma rule's
    sigma and setpoint."""
    out = {}
    for loop, own in row.placements.get(rung, {}).items():
        out[loop] = {
            "setpoint": own["setpoint"],
            "left": {
                f"{point:.6g}": sum(len(v) for v in by.values())
                for point, by in own["exits"].items()
            },
            "unheld": own["unheld"],
            "holds": own["holds"],
        }
    for c in row.loops:
        if c.zone and not c.exit_terms and c.name in row.placed:
            sigma, setpoint = row.placed[c.name]
            out[c.name] = {"sigma": sigma, "setpoint": setpoint}
    return out


def freeze(name: str, root=None, make_row=None) -> pathlib.Path:
    """Charter J1's kept rung of ``name`` as package data (``target_gym.reference``):
    its law, the parameters its episodes run on, and its provenance. Before it is
    written, the rung rebuilt from it on a fresh row must cost on the scoring
    seeds exactly what the ladder's own kept rung cost there; those costs are
    kept beside it, with its placement (``placement_report``) and its exits on
    J2's hash seeds (``exits_report``). ``make_row`` builds the row in place of
    its file's, for a test; the rung is then rebuilt on another row it builds."""
    import tempfile

    from target_gym import charter
    from target_gym import reference as R

    spec = REGISTRY[name]
    shipped = ROWS[name]
    row, rungs, _, kept = _climb(name, (make_row or shipped)())
    if kept is None:
        raise SystemExit(f"{name}: every rung trips, so there is no reference")
    frozen = _provenance(name, "ladder", kept)
    frozen["params"] = frozen_params(row, kept)
    frozen["law"] = row.frozen[kept]
    if any(c.zone and c.zone[2] is not None for c in row.loops):
        frozen["placed_on"] = list(PLACEMENT_SEEDS)  # a zone with a cheap side
        frozen["placement"] = placement_report(row, kept)
    ours = Reference(kept, rungs[kept])
    anchor = run_suite(name, row, ours, spec, seeds=charter.ANCHOR_SEEDS)
    frozen["anchor"] = {
        "seeds": list(charter.ANCHOR_SEEDS),
        "scenarios": {s: v["costs"] for s, v in anchor["scenarios"].items()},
        "trips": anchor["trips"],
    }
    weights = scenarios.weights(spec)
    seeds = charter.j2_seeds(frozen["commit"], weights)
    frozen["exits"] = exits_report(name, row, ours, seeds, spec)
    with tempfile.TemporaryDirectory() as tmp:
        R.write(name, frozen, pathlib.Path(tmp))
        if make_row is None:  # rebuilt as make_reference rebuilds it
            again, _ = R.build(name, pathlib.Path(tmp))
        else:
            law = R.load(name, pathlib.Path(tmp))["law"]
            again = R.assemble(make_row(), law, spec)
    check = run_suite(
        name,
        Row(0, None, score=R.changes(frozen)),
        Reference(kept, again),
        spec,
        seeds=charter.ANCHOR_SEEDS,
    )
    if check["scenarios"] != anchor["scenarios"]:
        raise SystemExit(
            f"{name}: the {kept} rung rebuilt from its law does not run as the "
            "ladder's: nothing written"
        )
    where = R.write(name, frozen, root or R.FROZEN)
    print(f"  froze {kept} in {where}", flush=True)
    return where


def freeze_expert(name: str, root=None) -> pathlib.Path:
    """A task whose reference is its shipped expert (charter.EXPERT_REFERENCE): the
    expert's fingerprint, so a retuned expert is a stale reference."""
    from target_gym import reference as R
    from target_gym.provenance import expert_fingerprint

    frozen = _provenance(name, "expert", "expert")
    frozen["expert"] = expert_fingerprint(REGISTRY[name])
    frozen["params"] = {"sensor_noise_scale": NOISE}
    where = R.write(name, frozen, root or R.FROZEN)
    print(f"{name}: froze its expert in {where}", flush=True)
    return where


#: A zone's placement stops once its setpoint moves under this many sigma, the 2-sigma
#: rule's own resolution at 1 %, or after PLACE_ROUNDS reads.
PLACE_TOLERANCE, PLACE_ROUNDS = 0.01, 8

_ORDER = (
    "pi",
    "feedforward",
    "cascade",
    "decoupler",
    "override",
    "valve position",
    "published",
)


class HoldReset:
    """The do-nothing null: each episode's actuator held where its reset put it,
    the plant's own starting move (rule 17), whatever the plant then does."""

    name, oracle = "hold reset", False

    def bind(self, env, params, seeds):
        import jax

        from target_gym.benchmark import episode_keys

        keys = [episode_keys(jax.random.PRNGKey(s), 1, 1)[0][0] for s in seeds]
        held = [env.reset_env(k, params)[1].actuator_held for k in keys]
        self.u = np.stack([np.asarray(h, float).reshape(-1) for h in held])

    def reset(self, key, obs, task):
        return np.zeros((task.n_episodes, task.n_act))

    def act(self, carry, obs, t):
        return self.u, carry


def null_set(name: str) -> list:
    """Charter J2's nulls on ``name``: the stored ones (``null_policies``) and,
    where the plant holds an actuator, the hold of its reset."""
    import jax

    from target_gym.null_policies import null_policies

    spec = REGISTRY[name]
    state = spec.make_env().reset_env(jax.random.PRNGKey(0), spec.make_test_params())[1]
    held = hasattr(state, "actuator_held")
    return list(null_policies(name)) + ([HoldReset()] if held else [])


def check_nulls(name: str, row, reference: dict) -> dict:
    """Charter J2: every null policy (``null_policies``) over the same scenarios and
    seeds as ``reference``, a ``run_suite`` result. A null whose stored table no
    longer fits the plant is reported with its error, not run. The tables and
    integrator gains come from the old plants' recordings until the re-record
    refits them, so a table null's result is marked stale."""
    runs, broken = {}, {}
    for null in null_set(name):
        try:
            runs[null.name] = run_suite(name, row, null)
        except Exception as err:  # a stale table's shape, or a missing field
            broken[null.name] = f"{type(err).__name__}: {err}"
    verdict = null_verdict(reference["costs"], {k: v["costs"] for k, v in runs.items()})
    stale = sorted(k for k in runs if k in ("table", "I-only"))
    for k, v in runs.items():
        print(
            f"  null {k}: median {np.median(v['costs']):.4g}, wins "
            f"{verdict['wins'][k]} of {verdict['seeds']}, {v['trips']} trips",
            flush=True,
        )
    return {**verdict, "runs": runs, "broken": broken, "stale": stale}


def expert_reference(name: str) -> dict:
    """A task whose reference is its shipped expert (aircraft, reactor): the expert
    over the suite, and the nulls against it."""
    print(name, flush=True)
    row = Row(0, None)
    reference = run_suite(name, row, shipped_policy("pid"))
    print(
        f"  expert: median {np.median(reference['costs']):.4g}, "
        f"{reference['trips']} trips",
        flush=True,
    )
    return {"reference": reference, "nulls": check_nulls(name, row, reference)}


#: The rows, one file per task in ``scripts/ladder_rows``, each defining ``row()``.
ROWS_DIR = pathlib.Path(__file__).resolve().parent / "ladder_rows"


def _row_of(path: pathlib.Path):
    def make() -> Row:
        # A row file imports this module by name, also when it runs as a script.
        sys.modules.setdefault("reference_ladder", sys.modules[__name__])
        spec = importlib.util.spec_from_file_location(f"ladder_rows.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.row()

    return make


ROWS = {
    p.stem: _row_of(p) for p in sorted(ROWS_DIR.glob("*.py")) if not p.stem[0] == "_"
}


def freeze_targets(tasks=None, experts=None) -> tuple[list, list]:
    """``(ladder tasks, expert tasks)`` that ``--freeze`` freezes: those named, or
    with none named the scored suite (``charter.FLAGSHIP``), each by its kind."""
    from target_gym import charter

    if tasks is None and experts is None:
        experts = [t for t in charter.FLAGSHIP if t in charter.EXPERT_REFERENCE]
        tasks = [t for t in charter.FLAGSHIP if t not in experts]
    return list(tasks or []), list(experts or [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--nulls", action="store_true", help="J2 against the kept rung")
    parser.add_argument(
        "--expert", nargs="*", default=[], help="tasks whose reference is the expert"
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="freeze each kept rung as package data (target_gym.reference)",
    )
    args = parser.parse_args()
    if args.freeze:
        named = args.tasks is not None or args.expert
        tasks, experts = freeze_targets(
            args.tasks if named else None, args.expert if named else None
        )
        for name in experts:
            freeze_expert(name)
        for name in tasks:
            freeze(name)
        return
    if args.tasks is None:
        args.tasks = list(ROWS)
    for name in args.expert:
        result = expert_reference(name)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
    for name in [] if args.expert and args.tasks == list(ROWS) else args.tasks:
        result = climb(name, args.nulls)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
