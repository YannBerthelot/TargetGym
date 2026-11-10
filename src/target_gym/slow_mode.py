"""A task's slow mode ``t_s``: how long the plant takes to forget where it started.

docs/proposals/long-strata.md, rule R1. Two runs of the reference share one seed, so
every outside process (events, walks, prices, noise) is identical between them. Run A
is the reference throughout. Run B is first in manual mode, holding the reference's
first action moved by ``offset`` for ``pre`` steps, which puts its plant in another
state the plant can visit; then the reference takes over. From that hand-over the two
plants differ only by where they were, and ``t_s`` is the decay time of their gap.

The gap is taken over every element of the state, each scaled by its spread in run A,
and the largest element is kept, so the slowest state sets ``t_s`` whether or not it
is observed. Bookkeeping that only accumulates (a cumulative cost, a counter) never
forgets by construction, but neither does a plant integrator, and the data cannot
tell them apart. So nothing is dropped on its own: the caller names bookkeeping
fields in ``ignore``, with a reason, and every element whose gap never fell to
``e^-1`` is reported, so a missing name shows as ``t_s = inf``, never as a fast mode.

Nothing here knows a task's physics, so it sits outside the fingerprinted sources.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import numpy as np

#: The fit reads the median gap from ``e^-1`` to ``e^-5`` of its value at hand-over.
FIT_FROM, FIT_TO = np.exp(-1.0), np.exp(-5.0)


class Pair(NamedTuple):
    """One coupled pair after the hand-over: per state element, run A's values and
    the absolute gap ``(T, n)``; the element names; B's excess cost per step; and
    the trips of each run (a trip restarts the plant, so the pair is suspect)."""

    values: np.ndarray
    gap: np.ndarray
    names: tuple[str, ...]
    excess_cost: np.ndarray
    trips_a: int = 0
    trips_b: int = 0


class SlowMode(NamedTuple):
    t_s: float  # steps; ``inf`` when the gap never fell to ``e^-1``
    gap: np.ndarray  # median over pairs of the largest scaled gap, per step
    slowest: str  # the element that is largest at the end of the fit
    never_forgot: tuple[str, ...]  # elements whose gap stayed above e^-1 of its start
    start_excess: float  # median of B's summed excess cost after the hand-over
    tripped_pairs: int = 0  # pairs where either run tripped


def _leaves(state) -> tuple[list[str], np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    names, values = [], []
    for path, leaf in flat:
        a = np.asarray(leaf, dtype=float).ravel()
        if not np.issubdtype(np.asarray(leaf).dtype, np.number) or a.size == 0:
            continue
        base = jax.tree_util.keystr(path)
        names += [base] if a.size == 1 else [f"{base}[{i}]" for i in range(a.size)]
        values.append(a)
    return names, np.concatenate(values)


def coupled_pair(
    env,
    params,
    make_policy: Callable,
    seed: int,
    pre: int,
    length: int,
    offset: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    bias: bool = False,
) -> Pair:
    """Run A and B on ``PRNGKey(seed)``: ``pre`` steps with B in manual mode, then
    ``length`` steps with both under a fresh ``make_policy(params)``. A policy is
    called as ``policy(obs, state)`` and returns the action. With ``bias``, B's
    excursion is its own reference plus ``offset`` instead of a held action, so a
    plant that runs away open loop stays in its envelope."""
    reset_key, step_key = jax.random.split(jax.random.PRNGKey(seed))
    step = jax.jit(env.step_env)
    obs, state = env.reset_env(reset_key, params)
    obs_a, s_a, obs_b, s_b = obs, state, obs, state
    policy_a, policy_b, held = make_policy(params), None, None
    values, gaps, excess, names, trips = [], [], [], None, [0, 0]
    for t in range(pre + length):
        u_a = np.asarray(policy_a(np.asarray(obs_a), s_a), dtype=float)
        if held is None:
            held = np.clip(u_a + offset, low, high)
        if t < pre and not bias:
            u_b = held
        else:
            policy_b = make_policy(params) if policy_b is None else policy_b
            u_b = np.asarray(policy_b(np.asarray(obs_b), s_b), dtype=float)
            if t < pre:
                u_b = np.clip(u_b + offset, low, high)
        if t == pre and bias:
            policy_b = make_policy(params)  # the hand-over starts a fresh reference
            u_b = np.asarray(policy_b(np.asarray(obs_b), s_b), dtype=float)
        k = jax.random.fold_in(step_key, t)
        obs_a, s_a, r_a, _, info_a = step(k, s_a, u_a, params)
        obs_b, s_b, r_b, _, info_b = step(k, s_b, u_b, params)
        trips[0] += int(np.any(info_a.get("tripped", False)))
        trips[1] += int(np.any(info_b.get("tripped", False)))
        if t >= pre:
            names, v_a = _leaves(s_a)
            v_b = _leaves(s_b)[1]
            values.append(v_a)
            gaps.append(np.abs(v_a - v_b))
            excess.append(float(r_a) - float(r_b))  # reward is minus the cost
    return Pair(
        np.array(values), np.array(gaps), tuple(names), np.array(excess), *trips
    )


def slow_mode(pairs: list[Pair], ignore: tuple[str, ...] = ()) -> SlowMode:
    """``t_s`` from coupled pairs: ``-1 / slope`` of ``log`` of the median largest
    scaled gap, fitted where it lies between ``e^-1`` and ``e^-5`` of its start.
    ``ignore`` names bookkeeping fields: an element is skipped when its name, less
    any ``[i]`` index, ends with an entry (``".x"`` skips ``.lead.x``, not ``.x_dot``).
    """
    names = pairs[0].names
    base = [n.split("[")[0] for n in names]
    acc = np.array([any(b.endswith(i) for i in ignore) for b in base], bool)
    stuck = np.zeros(len(names), bool)
    largest, argmax = [], []
    for p in pairs:
        spread = p.values.std(axis=0)
        floor = 1e-9 * (1.0 + np.abs(p.values).mean(axis=0))
        scaled = p.gap / np.maximum(spread, floor)
        scaled[:, acc] = 0.0
        # An element counts once its start gap is visible next to the largest one.
        visible = scaled[0] >= 1e-3 * scaled[0].max()
        stuck |= visible & ~np.any(scaled <= scaled[0] * FIT_FROM, axis=0)
        largest.append(scaled.max(axis=1))
        argmax.append(scaled.argmax(axis=1))
    gap = np.median(np.array(largest), axis=0)
    start = float(np.median([p.excess_cost.sum() for p in pairs]))
    g0 = gap[0]
    t = np.arange(len(gap))
    fit = (gap <= g0 * FIT_FROM) & (gap >= g0 * FIT_TO)
    if g0 <= 0 or stuck.any() or not np.any(gap <= g0 * FIT_FROM):
        t_s, end = np.inf, len(gap) - 1
    else:
        last = np.argmax(gap < g0 * FIT_TO) if np.any(gap < g0 * FIT_TO) else len(gap)
        fit &= t < last
        if fit.sum() < 2:  # fell through the window in one step
            t_s, end = float(np.argmax(gap <= g0 * FIT_FROM)), 0
        else:
            slope = np.polyfit(t[fit], np.log(gap[fit]), 1)[0]
            t_s, end = -1.0 / slope, int(t[fit][-1])
    slowest = names[int(np.bincount([a[end] for a in argmax]).argmax())]
    never_forgot = tuple(n for n, a in zip(names, stuck) if a)
    tripped = sum(p.trips_a + p.trips_b > 0 for p in pairs)
    return SlowMode(float(t_s), gap, slowest, never_forgot, start, tripped)
