"""The reference controllers' one law, and how a rung is chosen.

docs/proposals/task-charter.md, rule 3: a fixed ladder of PI by SIMC with
``tc = theta``, then steady-state feedforward from the plant's balances, then cascade
on an available measurement, then static decoupler or override. Rungs differ only in
what they hand :func:`ladder_controller`, so a task's rung is data, not code.
``scripts/reference_ladder.py`` runs the bump tests and builds the rungs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np

from target_gym.experts.pid import Controller
from target_gym.identification import simc

__all__ = ["Cascade", "Override", "Pi", "Rung", "ValvePosition", "best_rung"]
__all__ += ["decoupler"]
__all__ += ["ladder_controller", "most_sensitive", "pi_gains"]
__all__ += ["K_BACKOFF", "cv_sigma", "null_verdict", "zone_setpoint"]
__all__ += ["ZONE_GRID_STEPS", "first_without_exit", "leaves_only_unheld", "zone_grid"]

#: Charter Amendment 3, rule 22: on a zone the reference sits this many standard
#: deviations inside the cheap edge, the usual two-sigma give-away, fixed before any
#: room is seen.
K_BACKOFF = 2.0

#: Charter J1 (owner, 2026-11-23), for a zone whose exit the score prices as a
#: violation: the candidate setpoints are the cheap edge and this many equal steps to
#: the centre. Ten, a tenth of the half-width each: set from the zone alone, before
#: any room is seen, and no finer, since ten seeds cannot tell closer neighbours
#: apart.
ZONE_GRID_STEPS = 10


def null_verdict(reference, nulls: dict) -> dict:
    """Charter J2 on paired per-seed costs: each null's ``wins``, the seeds it costs
    strictly less on, and ``fails``, true when any null wins more than half."""
    ref = np.asarray(reference, float)
    wins = {k: int(np.sum(np.asarray(v, float) < ref)) for k, v in nulls.items()}
    return {
        "wins": wins,
        "seeds": len(ref),
        "fails": any(w > len(ref) / 2 for w in wins.values()),
    }


def cv_sigma(traces, burn_in: int = 0) -> float:
    """The controlled variable's sample standard deviation, pooled over the seeds'
    ``traces`` after ``burn_in`` steps each."""
    samples = np.concatenate([np.asarray(t, float)[burn_in:] for t in traces])
    return float(np.std(samples, ddof=1))


def zone_setpoint(low: float, high: float, sigma: float, cheap, k: float = K_BACKOFF):
    """The reference's setpoint on the zone ``[low, high]``: ``k`` sigma inside the
    ``cheap`` edge (``"low"`` or ``"high"``), never past the centre; the centre when
    ``cheap`` is None. An infinite edge can't be backed off from, and a zone with an
    infinite edge has no centre."""
    bounded = bool(np.isfinite(low) and np.isfinite(high))
    centre = 0.5 * (low + high) if bounded else None
    if cheap is None:
        if not bounded:
            raise ValueError("a zone with an infinite edge has no centre")
        return float(centre)
    edge = {"low": low, "high": high}[cheap]
    if not np.isfinite(edge):
        raise ValueError(f"the cheap edge {cheap!r} is infinite")
    inward = 1.0 if cheap == "low" else -1.0
    point = edge + inward * k * sigma
    if bounded and (point - centre) * inward > 0:
        point = centre
    return float(point)


def zone_grid(low: float, high: float, cheap, steps: int = ZONE_GRID_STEPS) -> tuple:
    """The candidate setpoints on the zone ``[low, high]``: from the ``cheap`` edge
    (``"low"`` or ``"high"``) to the centre in ``steps`` equal steps, the edge first
    and the centre last. A zone with no cheap side holds its centre and has nothing to
    place; one with an infinite edge has no centre to stop at."""
    if cheap is None:
        raise ValueError("a zone with no cheap side holds its centre: no grid")
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError("a zone with an infinite edge has no centre")
    edge = {"low": low, "high": high}[cheap]
    centre = 0.5 * (low + high)
    return tuple(float(edge + (centre - edge) * i / steps) for i in range(steps + 1))


def first_without_exit(grid, exits_at: Callable) -> tuple:
    """Charter J1 on a zone priced as a violation: the first setpoint of ``grid``,
    which runs from the cheap edge, at which the reference leaves the zone on none of
    its seeds, as ``best_rung`` keeps a rung that trips none. ``exits_at(setpoint)``
    gives the seeds that leave there, as sortable ids (a seed, or a scenario and a
    seed); candidates past the kept one are not run.

    Seeds that can be held decide (owner, 2026-11-23): a seed that leaves from every
    candidate does not constrain the placement. The reference pays for its exits
    there, and it is returned as unheld for the record to state. Knowing that takes
    the whole grid, so the grid is run to its end whenever no candidate holds every
    seed.

    Returns ``(setpoint, scan, unheld)``: ``scan`` the seeds leaving at each
    candidate run, ``unheld`` those no candidate holds. Two cases keep the last
    candidate, the centre: no candidate holds every seed that some candidate holds
    (``scan`` then shows who leaves at the centre), and no seed can be held from any
    candidate, where the rule has nothing to decide with and ``unheld`` is every
    seed."""
    scan = {}
    for point in grid:
        scan[point] = tuple(sorted(set(exits_at(point))))
        if not scan[point]:
            return float(point), scan, ()
    leaving = [set(seeds) for seeds in scan.values()]
    unheld = set.intersection(*leaving)
    holdable = set.union(*leaving) - unheld
    if holdable:
        for point, seeds in scan.items():
            if not holdable & set(seeds):
                break
    return float(point), scan, tuple(sorted(unheld))


def leaves_only_unheld(setpoint, scan: dict, unheld) -> bool:
    """Whether a rung placed by :func:`first_without_exit` leaves its zone on none of
    the seeds it can hold: at its ``setpoint`` the seeds leaving are all ``unheld``.
    False where the centre was kept though a seed some candidate holds still leaves
    there. Takes what :func:`first_without_exit` returns."""
    return not set(scan[setpoint]) - set(unheld)


def pi_gains(k: float, tau, theta: float) -> tuple:
    """``(Kc, Ti)`` by SIMC with ``tc = theta``. ``tau`` None is an integrating loop,
    whose ``k`` is the slope."""
    return simc(k, tau, theta, theta)


def decoupler(G, pairs) -> np.ndarray:
    """``D``, loops to MVs, with ``G @ D`` diagonal and each loop keeping the gain it
    was tuned on. ``G`` is CVs by MVs, ``pairs[i]`` the MV of the loop on CV ``i``."""
    G = np.asarray(G, float)
    return np.linalg.inv(G) @ np.diag([G[i, j] for i, j in enumerate(pairs)])


def most_sensitive(designs: list) -> dict:
    """Of one loop's designs at several operating points, the one asking the smallest
    controller gain ``Kc``: where the plant is most sensitive, so the loop is stable
    at every point. With equal lags it is the point of largest plant gain, as C5 keeps
    the highest gain across seeds. A tie keeps the earlier point."""
    return min(designs, key=lambda d: abs(d["Kc"]))


def best_rung(results: dict, holds: dict | None = None):
    """The rung of lowest median cost that tripped no seed, None if all tripped.
    ``results`` runs up the ladder, so a tie goes to the simpler rung.

    ``holds`` (charter J1 on zones priced as a violation, each rung placed on its
    own) says which rungs leave their zones on none of the seeds they can hold
    (:func:`leaves_only_unheld`); a rung it marks false is not a reference and is
    passed over. A rung it does not name has no such zone."""
    best, cost = None, np.inf
    for name, row in results.items():
        if holds is not None and not holds.get(name, True):
            continue
        median = float(np.median(row["costs"]))
        if row["trips"] == 0 and median < cost:
            best, cost = name, median
    return best


@dataclass(frozen=True)
class Pi:
    """Loop ``loop``'s PI. ``alpha`` is how far its reading's filter moves a step."""

    loop: int
    kc: float
    ti: float
    alpha: float = 1.0


@dataclass(frozen=True)
class Cascade:
    """An inner loop on ``read(obs, xp)``. The outer loop ``pi.loop`` sets its
    setpoint, around ``rest(obs, xp)``, and it drives what the outer loop drove."""

    read: Callable
    rest: Callable
    pi: Pi


@dataclass(frozen=True)
class Override:
    """Keep ``read(obs, xp)`` above ``limit`` (``floor``) or below it, through MV
    ``mv``, on which its gain has sign ``gain_sign``. A P term on the limit's error
    over a reset that follows the applied move at ``ti``, select-style (Shinskey), so
    it takes over the moment the reading crosses."""

    read: Callable
    limit: float
    floor: bool
    mv: int
    gain_sign: float
    kc: float
    ti: float


@dataclass(frozen=True)
class ValvePosition:
    """Hold a loop's valve at ``setpoint`` by another MV, ``mv``, which no loop
    drives: Shinskey's valve position control, as Reyes-Lua, Zotica and Skogestad
    lay it out (IFAC ADCHEM 2018, section 2.3). ``read(obs, xp)`` is the watched
    valve's position, in the units of ``setpoint``. A PI moves ``mv`` from where the
    reset left it (the start hold's value, else the bias's), never past ``top``, the
    throughput asked for, and its integral holds while that limits it. ``kc`` has
    the sign of the watched valve's answer to ``mv`` with the loops closed."""

    read: Callable
    setpoint: float
    mv: int
    kc: float
    ti: float
    top: float = 1.0


@dataclass(frozen=True)
class Rung:
    """One step of the ladder. ``mix`` is MVs by loops, the pairing on the rungs
    without a decoupler."""

    name: str
    loops: tuple
    mix: np.ndarray
    cascades: tuple = ()
    overrides: tuple = ()
    valve_positions: tuple = ()


class Memory(NamedTuple):
    y: object
    integral: object
    inner_y: object
    inner_integral: object
    reset: object
    started: object
    #: A start hold's trims (``ladder_controller``'s ``start``): what the bias and
    #: each cascade's rest were off by at the reset. Zero without one.
    trim: object = ()
    inner_trim: object = ()
    #: Each valve position loop's integral, and the MV as it took it over.
    vp_integral: object = ()
    vp_base: object = ()


def ladder_controller(
    rung: Rung, read, target, bias, dt: float, start=None
) -> Controller:
    """``rung`` as a pure controller. ``read`` and ``target`` give the loops' CVs and
    setpoints, ``bias`` the raw action the loops act around, each ``f(obs, xp)``.

    ``start(obs, xp)`` is the raw action the actuators hold at the reset. With it
    the controller takes over as a loop switched to automatic on a running plant
    does, bumplessly: on its first step it re-bases ``bias`` on that action and each
    cascade's rest on its reading, and keeps those two trims for the episode. The
    loops still act on their errors from the first step. Without it nothing changes.
    """
    kc, ti, alpha = (np.array([getattr(p, f) for p in rung.loops]) for f in _PI)
    inner = [c.pi for c in rung.cascades]
    kc_i, ti_i, alpha_i = (np.array([getattr(p, f) for p in inner]) for f in _PI)
    paired = np.abs(rung.mix).T > 0  # loops by MVs

    def step(_, m: Memory, obs, xp=jnp):
        first = m.started > 0
        y = xp.where(first, m.y + alpha * (read(obs, xp) - m.y), read(obs, xp))
        e = target(obs, xp) - y
        integral = m.integral + e * dt
        out = [v for v in kc * (e + integral / ti)]

        y_i, i_i, trims = [], [], []
        for n, c in enumerate(rung.cascades):
            raw = c.read(obs, xp)
            f = xp.where(first, m.inner_y[n] + alpha_i[n] * (raw - m.inner_y[n]), raw)
            rest = c.rest(obs, xp)
            if start is not None:
                trims.append(xp.where(first, m.inner_trim[n], raw - rest))
                rest = rest + trims[n]
            e_i = rest + out[c.pi.loop] - f
            i_n = m.inner_integral[n] + e_i * dt
            out[c.pi.loop] = kc_i[n] * (e_i + i_n / ti_i[n])
            y_i.append(f)
            i_i.append(i_n)

        base, trim = bias(obs, xp), m.trim
        if start is not None:
            trim = xp.where(first, m.trim, start(obs, xp) - base)
            base = base + trim
        asked = base + xp.asarray(rung.mix) @ xp.stack(out)
        u = [v for v in xp.clip(asked, -1.0, 1.0)]
        resets = []
        for n, o in enumerate(rung.overrides):
            r = xp.where(first, m.reset[n], u[o.mv])
            wanted = r + o.kc * (o.limit - o.read(obs, xp))
            higher = o.gain_sign * (1.0 if o.floor else -1.0) > 0
            u[o.mv] = xp.clip(
                xp.maximum(u[o.mv], wanted) if higher else xp.minimum(u[o.mv], wanted),
                -1.0,
                1.0,
            )
            resets.append(r + dt / o.ti * (u[o.mv] - r))
        vp_i, vp_b = [], []
        for n, v in enumerate(rung.valve_positions):
            took = start(obs, xp)[v.mv] if start is not None else u[v.mv]
            base_v = xp.where(first, m.vp_base[n], took)
            e_v = v.setpoint - v.read(obs, xp)
            i_v = m.vp_integral[n] + e_v * dt
            wanted = base_v + v.kc * (e_v + i_v / v.ti)
            u[v.mv] = xp.clip(xp.minimum(wanted, v.top), -1.0, 1.0)
            vp_i.append(xp.where(u[v.mv] == wanted, i_v, m.vp_integral[n]))
            vp_b.append(base_v)
        action = xp.stack(u)

        moved = xp.abs(action - asked) > 0
        hold = xp.asarray(paired) @ moved > 0
        kept = xp.where(hold, m.integral, integral)
        kept_i = [
            xp.where(hold[c.pi.loop], m.inner_integral[n], i_i[n])
            for n, c in enumerate(rung.cascades)
        ]
        return action, Memory(
            y,
            kept,
            _stack(y_i, xp),
            _stack(kept_i, xp),
            _stack(resets, xp),
            xp.ones_like(m.started),
            trim,
            _stack(trims, xp) if start is not None else m.inner_trim,
            _stack(vp_i, xp),
            _stack(vp_b, xp),
        )

    n, c, o = len(rung.loops), len(inner), len(rung.overrides)
    zero = Memory(
        np.zeros(n),
        np.zeros(n),
        np.zeros(c),
        np.zeros(c),
        np.zeros(o),
        np.zeros(()),
        np.zeros(rung.mix.shape[0]),
        np.zeros(c),
        np.zeros(len(rung.valve_positions)),
        np.zeros(len(rung.valve_positions)),
    )
    return Controller(None, zero, step)


_PI = ("kc", "ti", "alpha")


def _stack(values, xp):
    return xp.stack(values) if values else xp.zeros(0)
