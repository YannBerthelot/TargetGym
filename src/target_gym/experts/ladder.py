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

__all__ = ["Cascade", "Override", "Pi", "Rung", "best_rung", "decoupler"]
__all__ += ["ladder_controller", "most_sensitive", "pi_gains"]
__all__ += ["K_BACKOFF", "cv_sigma", "zone_setpoint"]

#: Charter Amendment 3, rule 22: on a zone the reference sits this many standard
#: deviations inside the cheap edge, the usual two-sigma give-away, fixed before any
#: room is seen.
K_BACKOFF = 2.0


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


def best_rung(results: dict):
    """The rung of lowest median cost that tripped no seed, None if all tripped.
    ``results`` runs up the ladder, so a tie goes to the simpler rung."""
    best, cost = None, np.inf
    for name, row in results.items():
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
class Rung:
    """One step of the ladder. ``mix`` is MVs by loops, the pairing on the rungs
    without a decoupler."""

    name: str
    loops: tuple
    mix: np.ndarray
    cascades: tuple = ()
    overrides: tuple = ()


class Memory(NamedTuple):
    y: object
    integral: object
    inner_y: object
    inner_integral: object
    reset: object
    started: object


def ladder_controller(rung: Rung, read, target, bias, dt: float) -> Controller:
    """``rung`` as a pure controller. ``read`` and ``target`` give the loops' CVs and
    setpoints, ``bias`` the raw action the loops act around, each ``f(obs, xp)``."""
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

        y_i, i_i = [], []
        for n, c in enumerate(rung.cascades):
            raw = c.read(obs, xp)
            f = xp.where(first, m.inner_y[n] + alpha_i[n] * (raw - m.inner_y[n]), raw)
            e_i = c.rest(obs, xp) + out[c.pi.loop] - f
            i_n = m.inner_integral[n] + e_i * dt
            out[c.pi.loop] = kc_i[n] * (e_i + i_n / ti_i[n])
            y_i.append(f)
            i_i.append(i_n)

        asked = bias(obs, xp) + xp.asarray(rung.mix) @ xp.stack(out)
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
        )

    n, c, o = len(rung.loops), len(inner), len(rung.overrides)
    zero = Memory(
        np.zeros(n),
        np.zeros(n),
        np.zeros(c),
        np.zeros(c),
        np.zeros(o),
        np.zeros(()),
    )
    return Controller(None, zero, step)


_PI = ("kc", "ti", "alpha")


def _stack(values, xp):
    return xp.stack(values) if values else xp.zeros(0)
