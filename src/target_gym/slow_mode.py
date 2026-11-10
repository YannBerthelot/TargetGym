"""A task's slow mode ``t_s``: how long the cost takes to forget where the plant started.

docs/proposals/long-strata.md, rule R1. Two runs of the reference share one seed, so
every outside process (events, walks, prices, noise) is identical between them. Run A
is the reference throughout. Run B is first in manual mode, holding the reference's
first action moved by ``offset`` for ``pre`` steps, which puts its plant in another
state the plant can visit; then the reference takes over. From that hand-over the two
plants differ only by where they were. ``t_s`` is the decay time of the extra cost
B still has to pay for its start, the bias a burn-in removes (owner, 2026-11-10);
the state's own decay is reported beside it.

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

#: An element has forgotten once its gap falls to ``e^-1`` of its start; the plant
#: has forgotten once the largest gap stays under ``e^-3`` (``FORGOT``), which an
#: exponential reaches at ``3 t_s``.
FIT_FROM, FORGOT = np.exp(-1.0), np.exp(-3.0)
#: A gap within this many float32 steps of the value is no gap: two runs that
#: agree to the arithmetic's resolution have forgotten.
ULPS = 16 * float(np.finfo(np.float32).eps)
#: The largest scaled gap at hand-over must reach this share of an element's spread
#: over the run; below it the excursion did not move the plant and nothing is fitted.
MIN_START = 0.05


def wrapped_gap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``|a - b|`` for angles in radians, on the circle: a whole turn is no gap."""
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _matches(names, entries) -> np.ndarray:
    """Whether each element's name, less any ``[i]`` index, ends with an entry."""
    base = [n.split("[")[0] for n in names]
    return np.array([any(b.endswith(e) for e in entries) for b in base], bool)


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
    #: Steps for the extra cost still to come from B's start to decay: the bias a
    #: burn-in removes. ``inf`` if it never does; ``nan`` if the start cost nothing
    #: above its own noise, or every pair tripped (a trip's cost swamps the bias).
    t_s: float
    gap: np.ndarray  # median over pairs of the largest scaled gap, per step
    slowest: str  # the element that forgets last
    never_forgot: tuple[str, ...]  # elements whose gap stayed above e^-1 of its start
    start_excess: float  # median of B's summed excess cost after the hand-over
    tripped_pairs: int = 0  # pairs where either run tripped
    floor: float = 0.0  # the slowest element's gap over the last third, over its start
    #: The state's own decay time, the slowest element's (a diagnostic: dead
    #: bands, limit cycles and neutral modes keep it long where the cost is not).
    state_t_s: float = np.nan


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
    angles: tuple[str, ...] = (),
    observe: Callable | None = None,
) -> Pair:
    """Run A and B on ``PRNGKey(seed)``: ``pre`` steps with B in manual mode, then
    ``length`` steps with both under a fresh ``make_policy(params)``. A policy is
    called as ``policy(obs, state)`` and returns the action. With ``bias``, B's
    excursion is its own reference plus ``offset`` instead of a held action, so a
    plant that runs away open loop stays in its envelope. Elements named in
    ``angles`` (as ``slow_mode``'s ``ignore``) are compared on the circle.
    ``observe(state, params)`` returns named scalars added as ``path.<name>``
    elements: a path task's errors relative to its path, which forget where a
    world-frame position along a closed path never does."""

    def elements(state):
        names, values = _leaves(state)
        if observe is None:
            return names, values
        extra = observe(state, params)
        names = names + [f"path.{k}" for k in extra]
        return names, np.concatenate([values, [float(v) for v in extra.values()]])

    reset_key, step_key = jax.random.split(jax.random.PRNGKey(seed))
    step = jax.jit(env.step_env)
    obs, state = env.reset_env(reset_key, params)
    obs_a, s_a, obs_b, s_b = obs, state, obs, state
    policy_a, policy_b, held = make_policy(params), None, None
    values, gaps, excess, names, trips, circle = [], [], [], None, [0, 0], None
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
            names, v_a = elements(s_a)
            v_b = elements(s_b)[1]
            circle = _matches(names, angles) if circle is None else circle
            values.append(v_a)
            gaps.append(np.where(circle, wrapped_gap(v_a, v_b), np.abs(v_a - v_b)))
            excess.append(float(r_a) - float(r_b))  # reward is minus the cost
    return Pair(
        np.array(values), np.array(gaps), tuple(names), np.array(excess), *trips
    )


def _crossing(g: np.ndarray, level: float) -> float:
    """When ``g`` last falls under ``level``, interpolated on the log scale
    (exact for an exponential); ``inf`` if it is still above at the end."""
    above = np.nonzero(g > level)[0]
    if not above.size:
        return 0.0
    last = int(above[-1])
    if last >= len(g) - 1:
        return np.inf
    hi, lo = np.log(g[last]), np.log(max(g[last + 1], 1e-300))
    return last + (hi - np.log(level)) / max(hi - lo, 1e-12)


def _decay_time(g: np.ndarray) -> float:
    """The time from staying under ``e^-1`` of the start to staying under
    ``e^-3``, over 2: an exponential's time constant, however it started."""
    first, third = _crossing(g, g[0] * FIT_FROM), _crossing(g, g[0] * FORGOT)
    return np.inf if third == np.inf else max(third - first, 0.0) / 2.0


def slow_mode(pairs: list[Pair], ignore: tuple[str, ...] = ()) -> SlowMode:
    """``t_s`` from coupled pairs. Each state element's decay time is half the time
    between its scaled gap staying under ``e^-1`` and under ``e^-3`` of its own
    start, exact for an exponential however it started and immune to a floor
    under ``e^-3``; ``inf`` for one that never stays under it (a neutral mode, a
    limit cycle's phase). ``t_s`` is the slowest element's median over pairs.
    ``ignore`` names bookkeeping fields: an element is skipped when its name, less
    any ``[i]`` index, ends with an entry (``".x"`` skips ``.lead.x``, not ``.x_dot``).
    """
    names = pairs[0].names
    skip = _matches(names, ignore)
    times, starts, largest = [], [], []
    for p in pairs:
        spread = p.values.std(axis=0)
        floor = 1e-9 * (1.0 + np.abs(p.values).mean(axis=0))
        resolved = np.where(p.gap > ULPS * np.abs(p.values), p.gap, 0.0)
        scaled = resolved / np.maximum(spread, floor)
        scaled[:, skip] = 0.0
        starts.append(scaled[0])
        largest.append(scaled.max(axis=1))
        # An element counts once its start gap is visible next to the largest one.
        visible = scaled[0] >= 1e-2 * scaled[0].max()
        times.append(
            [
                _decay_time(scaled[:, e]) if visible[e] else 0.0
                for e in range(len(names))
            ]
        )
    gap = np.median(np.array(largest), axis=0)
    per_element = np.median(np.array(times), axis=0)
    start = float(np.median([p.excess_cost.sum() for p in pairs]))
    tripped = sum(p.trips_a + p.trips_b > 0 for p in pairs)
    g0 = gap[0]
    slowest_i = int(np.argmax(per_element))
    state_t_s = per_element[slowest_i]
    if g0 < MIN_START:
        state_t_s = np.nan
    # The extra cost still to come from B's start, averaged over the pairs; for an
    # exponential it decays at the cost's own time constant.
    # A trip's lump cost would swamp the start's bias, so tripped pairs sit out.
    clean = [p for p in pairs if p.trips_a + p.trips_b == 0]
    excess = np.mean([p.excess_cost for p in clean], axis=0) if clean else None
    remaining = np.abs(np.cumsum(excess[::-1])[::-1]) if clean else np.zeros(1)
    # A lasting offset falls to zero only because the run ends; if the last third
    # still carries more than e^-3 of the bias, the cost never forgot.
    last_third = abs(float(excess[2 * len(excess) // 3 :].sum())) if clean else 0.0
    if not clean or remaining[0] <= FIT_FROM * np.abs(excess).sum():
        # No pair without a trip, or no net bias: a bias is a cost of one sign,
        # so it is most of the excess; flicker about zero is not.
        t_s = np.nan
    elif last_third > FORGOT * remaining[0]:
        t_s = np.inf
    else:
        t_s = _decay_time(remaining)
    never_forgot = tuple(n for n, t in zip(names, per_element) if t == np.inf)
    s0 = np.median([st[slowest_i] for st in starts])
    tail = np.median(
        [
            np.median(p.gap[2 * len(p.gap) // 3 :, slowest_i])
            / max(p.gap[0, slowest_i], 1e-300)
            for p in pairs
        ]
    )
    floor_ratio = float(tail) if s0 > 0 else 0.0
    return SlowMode(
        float(t_s),
        gap,
        names[slowest_i],
        never_forgot,
        start,
        tripped,
        floor_ratio,
        float(state_t_s),
    )
