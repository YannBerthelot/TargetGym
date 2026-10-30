"""Evaluation protocol for reach-and-hold tasks.

A reach-and-hold controller is judged on two numbers: the long-run average
cost while holding (the *gain*, what the plant pays forever) and the excess
cost of reaching a target after it moves (the *reach cost*, paid once per
target change). An episode return mixes the two in a proportion set by the
episode length, so the protocol reports them apart. docs/rl-protocol.md is the
user's guide.

Score a shipped controller::

    from target_gym.eval import evaluate_controller
    m = evaluate_controller("reactor", "pid", seeds=5)
    m["gain"], m["tracking"], m["running"], m["reach_cost"]

or a policy of your own: build ``Episode`` records with ``run_episode`` and
pass them to ``evaluate``. ``target_gym.scoring`` compares a policy with the
expert and the ceiling.

How a cycle is split. A target cycle runs from one target change to the next
(the first starts at the reset). ``cycle_segments`` splits each into

1. the *settle* after its own change, measured on each cycle so every
   controller is scored against its own transient (``_settle_of``);
2. the *hold*, the steps in between;
3. the *anticipation* of the next change: a controller that previews the
   schedule starts moving early, and that rising cost belongs to reaching
   the next target, not to the hold (``anticipation``, ``anticipations``).

``burn_in`` is per plant, three cost-bearing time constants, from
``src/target_gym/data/hold_measurements/`` (``scripts/measure_hold.py``,
which uses the same anticipation rule on its squared errors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from target_gym import records

# Absolute slack on "within twice the level", as the settle rule has always had.
_TOL = 1e-12
# The anticipation rule (``anticipation``).
_ANTICIPATION_MIN_CYCLE = 16  # shorter cycles are never searched
_ANTICIPATION_MIN_STEPS = 3  # a run is at least this long, unless it is huge
_ANTICIPATION_RATIO = 10.0  # its median is at least this many times the level
_ANTICIPATION_SHARE = 0.8  # share of its steps above twice the level
_ANTICIPATION_MAX_DIP = 2  # most steps in a row at or below twice the level
_ANTICIPATION_SHORT_RATIO = 100.0  # every step of a 1- or 2-step run is above


@dataclass
class Episode:
    cost: np.ndarray  # per-step cost, >= 0 (= -reward)
    in_band: np.ndarray  # bool per step
    target_change: np.ndarray  # bool per step: a new target became active
    failed: np.ndarray | None = None  # bool per step: the plant tripped on this step
    terms: dict = field(default_factory=dict)  # name -> per-step array


def _cycles(ep: Episode):
    idx = np.flatnonzero(ep.target_change)
    bounds = np.concatenate([[0], idx, [len(ep.cost)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")


class Segment(NamedTuple):
    """One target cycle split by ``cycle_segments``, as absolute step indices.

    ``start:hold_start`` is the settle after the cycle's own change,
    ``hold_start:hold_end`` the hold, and ``hold_end:end`` the anticipation of
    the next change (empty when none was detected)."""

    start: int
    hold_start: int
    hold_end: int
    end: int


def _longest_dip(above: np.ndarray) -> int:
    """Most consecutive ``False`` entries of a boolean array."""
    idx = np.flatnonzero(above)
    if len(idx) == 0:
        return len(above)
    return int((np.diff(np.concatenate([[-1], idx, [len(above)]])) - 1).max())


def anticipation(series, floor=0.0) -> int:
    """Steps at the end of one cycle that anticipate the change ending it.

    ``series`` is the cycle's per-step cost, or a ``(steps, outputs)`` array
    of per-output costs, in which case the longest run over the outputs is
    returned. The level is the mean over the second half of the cycle without
    the run, as the settle rule has it, or ``floor`` if that is higher
    (``floor`` may be one value per output). The run is the longest stretch at
    the end of the cycle that

    * starts with two steps costing more than twice the level, has at least
      four in five of its steps above twice the level, and never more than two
      in a row at or below it,
    * has a median of at least ten times the level,
    * is rising, its second half costing more on average than its first, and
    * is at least three steps long. A run of one or two steps counts only when
      every step costs a hundred times the level, and a two-step run rises.

    The run takes at most half the cycle, and a cycle shorter than 16 steps is
    not searched. 0 means no anticipation. The caller decides whether a change
    follows the cycle at all, and what the floor is (``anticipations``).

    The conditions keep stationary noise from reading as anticipation: on
    noise-only episodes the rule fires on at most 0.11% of the cycles
    ``anticipations`` searches (``tests/test_eval_protocol.py``).
    """
    x = np.asarray(series, float)
    if x.ndim == 2:
        floors = np.broadcast_to(np.asarray(floor, float), x.shape[1:])
        return max(
            (anticipation(x[:, i], floors[i]) for i in range(x.shape[1])), default=0
        )
    n = len(x)
    if n < _ANTICIPATION_MIN_CYCLE:
        return 0
    for j in range(n // 2, 0, -1):
        m = n - j
        level = max(float(x[m // 2 : m].mean()), float(floor))
        run = x[m:]
        above = run > 2.0 * level + _TOL
        if not above[:2].all():
            continue
        if j < _ANTICIPATION_MIN_STEPS:
            if (run > _ANTICIPATION_SHORT_RATIO * level + _TOL).all() and (
                j == 1 or run[1] > run[0]
            ):
                return j
            continue
        if above.mean() < _ANTICIPATION_SHARE:
            continue
        if _longest_dip(above) > _ANTICIPATION_MAX_DIP:
            continue
        if np.median(run) < _ANTICIPATION_RATIO * level + _TOL:
            continue
        if not run[j // 2 :].mean() > run[: j // 2].mean():
            continue
        return j
    return 0


def anticipations(series, cycles) -> list[int]:
    """``anticipation`` of every cycle ``(a, b)`` of ``series`` that another
    cycle follows, and 0 for the last cycle, which no change follows.
    ``cycles`` are consecutive, as ``_cycles`` returns them.

    Two searches. The first has no floor. The second uses as its floor the
    mean level of the cycles (the mean over each one's second half, with the
    run the first search found left out), so a cycle held at almost zero cost
    does not turn every small rise into a tenfold excursion. Only the second
    counts, and it never finds a longer run. The floor also hides an
    anticipation costing under ten times the episode's typical level, as on
    ``hvac``.

    A ``(steps, outputs)`` series is searched output by output, each with its
    own floor, and a cycle's anticipation is the longest over the outputs."""
    x = np.asarray(series, float)
    if x.ndim == 2:
        per_output = [anticipations(x[:, i], cycles) for i in range(x.shape[1])]
        return [max(js) for js in zip(*per_output)] if per_output else [0] * len(cycles)
    n = len(x)
    first = [anticipation(x[a:b]) if b < n else 0 for a, b in cycles]
    if not any(first):
        return first
    searched = [j > 0 for j in first]
    first = [j if b < n else anticipation(x[a:b]) for (a, b), j in zip(cycles, first)]
    levels = []
    for (a, b), j in zip(cycles, first):
        c = x[a : b - j]
        if len(c) >= 4:
            levels.append(c[len(c) // 2 :].mean())
    floor = float(np.mean(levels)) if levels else 0.0
    return [
        anticipation(x[a:b], floor) if s else 0 for (a, b), s in zip(cycles, searched)
    ]


def cycle_segments(cost, cycles, settle: int | None = None) -> list[Segment]:
    """Split each cycle ``(a, b)`` of a per-step ``cost`` into settle, hold
    and anticipation (module docstring). The anticipation comes from
    ``anticipations``; the settle is ``_settle_of`` measured on the cycle
    without its anticipation, so a cycle with none gets exactly the settle
    it would get without the rule."""
    cost = np.asarray(cost)
    out = []
    for (a, b), j in zip(cycles, anticipations(cost, cycles)):
        h = b - j
        out.append(Segment(a, a + _settle_of(cost[a:h], settle), h, b))
    return out


def _segments(ep: Episode, settle: int | None = None) -> list[Segment]:
    return cycle_segments(ep.cost, _cycles(ep), settle)


def _hold_level(cost: np.ndarray, s: Segment) -> float:
    """Mean cost over the hold steps (over the cycle without its anticipation
    when the hold is empty)."""
    if s.hold_end > s.hold_start:
        return cost[s.hold_start : s.hold_end].mean()
    return cost[s.start : s.hold_end].mean()


def gain(episodes, burn_in: int, key: str | None = None):
    """Mean per-step cost over every step after the burn-in, pooled per
    episode: the long-run average cost, transients included."""
    per_ep = []
    for ep in episodes:
        series = ep.cost if key is None else ep.terms[key]
        if len(series) > burn_in:
            per_ep.append(series[burn_in:].mean())
    return (
        (float(np.mean(per_ep)), _ci(per_ep))
        if per_ep
        else (float("nan"), float("nan"))
    )


def hold(episodes, burn_in: int, key: str | None = None, settle: int | None = None):
    """Mean per-step cost over the hold steps only (after the burn-in, after
    each cycle's settle and before its anticipation of the next change):
    ``rho_hold``."""
    per_ep = []
    for ep in episodes:
        series = ep.cost if key is None else ep.terms[key]
        m = hold_mask(ep, burn_in, settle)
        if m.sum() >= 1:
            per_ep.append(series[m].mean())
    return (
        (float(np.mean(per_ep)), _ci(per_ep))
        if per_ep
        else (float("nan"), float("nan"))
    )


def _settle_of(cost: np.ndarray, settle: int | None) -> int:
    """Steps of a cycle that belong to its transient.

    Measured on the cycle: the hold level is the mean cost over the second
    half, and the transient ends at the first step from which the cost stays
    within twice that level. At least one step, at most half the cycle, so
    the level is never taken over the window it defines.
    """
    n = len(cost)
    if settle is not None:
        return int(min(max(settle, 1), max(n // 2, 1)))
    if n < 4:
        return 1
    level = cost[n // 2 :].mean()
    ok = cost <= 2.0 * level + _TOL
    # last step that is still outside, plus one
    outside = np.flatnonzero(~ok[: n // 2])
    return int(min(max(outside[-1] + 1 if len(outside) else 1, 1), n // 2))


def reach_cost(episodes, settle: int | None = None):
    """Excess cost of each target change's transient over the hold level (the
    bias ``B`` of Theorem 4). The transient is the settle of the cycle the
    change starts, measured against that cycle's hold level, plus the
    anticipation at the end of the previous cycle, measured against the
    previous cycle's hold level. Negative when the cost is still rising at
    the end of the cycle, because the transient is then cheaper than the
    "hold" level and the controller reached no hold in the window.
    ``transient_cost`` and ``unsettled_fraction`` report that separately."""
    vals = []
    for ep in episodes:
        carry = None  # excess of the previous cycle's anticipation
        for s in _segments(ep, settle):
            level = _hold_level(ep.cost, s)
            v = (ep.cost[s.start : s.hold_start] - level).sum()
            if carry is not None:
                v = v + carry
            vals.append(v)
            carry = None
            if s.hold_end < s.end:
                carry = (ep.cost[s.hold_end : s.end] - level).sum()
    return (float(np.mean(vals)), _ci(vals)) if vals else (float("nan"), float("nan"))


def transient_cost(episodes, window: int):
    """Summed cost over the first ``window`` steps after each target change
    (up to the cycle's own anticipation, if that comes first), plus the cost
    of the anticipation steps among the last ``window`` before the change.

    Nothing is subtracted, so unlike ``reach_cost``, which is relative to each
    controller's own hold level, it compares two controllers whose holds
    differ. ``evaluate`` uses the plant's burn-in as the window."""
    vals = []
    w = max(int(window), 1)
    for ep in episodes:
        carry = None  # cost of the previous cycle's anticipation
        for s in _segments(ep):
            v = ep.cost[s.start : min(s.start + w, s.hold_end)].sum()
            if carry is not None:
                v = v + carry
            vals.append(v)
            carry = None
            if s.hold_end < s.end:
                carry = ep.cost[max(s.hold_end, s.end - w) : s.end].sum()
    return (float(np.mean(vals)), _ci(vals)) if vals else (float("nan"), float("nan"))


def settling_time(episodes, band: float):
    """Steps per move until the tracking cost enters ``band`` and stays there
    until the cycle's anticipation of the next change: 0 when it never leaves,
    the whole cycle when it ends outside. A move is the reset or a target
    change. Reads the ``tracking`` term, or the cost when there are no terms;
    ``scoring.settle_band`` is the plant's band."""
    vals = []
    for ep in episodes:
        series = ep.terms.get("tracking", ep.cost)
        for s in _segments(ep):
            outside = np.flatnonzero(series[s.start : s.hold_end] > band + _TOL)
            vals.append(int(outside[-1]) + 1 if len(outside) else 0)
    return (float(np.mean(vals)), _ci(vals)) if vals else (float("nan"), float("nan"))


def unsettled_fraction(episodes):
    """Share of cycles in which the cost over the second half exceeds the
    cost over the first half, with the anticipation of the next change left
    out. The controller is still moving away from a hold when the cycle ends,
    so its reach cost has no hold level to be measured against."""
    n_up = n_tot = 0
    for ep in episodes:
        for s in _segments(ep):
            c = ep.cost[s.start : s.hold_end]
            if len(c) < 4:
                continue
            n_tot += 1
            n_up += bool(c[len(c) // 2 :].mean() > c[: len(c) // 2].mean())
    return n_up / max(n_tot, 1)


def hold_mask(ep: Episode, burn_in: int, settle: int | None = None) -> np.ndarray:
    """Steps that count as hold: after the burn-in, after each cycle's settle
    and before its anticipation of the next change."""
    m = np.zeros(len(ep.cost), bool)
    for s in _segments(ep, settle):
        m[s.hold_start : s.hold_end] = True
    m[:burn_in] = False
    return m


def reach_fraction(episodes):
    n_reach = n_tot = 0
    for ep in episodes:
        for a, b in _cycles(ep):
            first = np.flatnonzero(ep.in_band[a:b])
            n_reach += int(first[0]) if len(first) else (b - a)
            n_tot += b - a
    return n_reach / max(n_tot, 1)


def failure_rate(episodes):
    n_fail = n_cyc = 0
    for ep in episodes:
        if ep.failed is None:
            continue
        for a, b in _cycles(ep):
            n_cyc += 1
            n_fail += bool(ep.failed[a:b].any())
    return n_fail / max(n_cyc, 1)


def time_in_band(episodes, burn_in: int):
    vals = [ep.in_band[burn_in:].mean() for ep in episodes if len(ep.cost) > burn_in]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate(
    episodes, burn_in: int, settle: int | None = None, band: float | None = None
):
    """The protocol's numbers for a list of ``Episode`` records, as a dict.

    - ``gain`` (and ``gain_ci``): mean cost per step after ``burn_in``,
      transients included. The headline; no segmentation enters it.
    - ``hold``: the same over the hold steps only.
    - ``reach_cost``: per target change, the cost above the hold level over
      the transient. Negative when the cycle never settled.
    - ``transient_cost``: per target change, the absolute cost over a
      ``burn_in``-step window either side; compare this across controllers.
    - ``unsettled_fraction``, ``reach_fraction``, ``failure_rate`` (trips per
      cycle) and ``time_in_band`` (a KPI, never a training signal).
    - One entry per reward term (``tracking``, ``running``, ...) and its
      ``hold_`` counterpart, when the episodes carry terms.
    - With ``settle``, ``move_cost``: per move (the reset or a target change),
      ``transient_cost`` over the ``settle`` steps after it. With ``band``,
      ``settling_time``. ``scoring`` compares both with the expert's.

    ``settle`` overrides the per-cycle settle with a fixed count.
    """
    g, gci = gain(episodes, burn_in)
    h, hci = hold(episodes, burn_in, settle=settle)
    rc, rcci = reach_cost(episodes, settle)
    tc, tcci = transient_cost(episodes, burn_in)
    out = dict(
        gain=g,
        gain_ci=gci,
        hold=h,
        hold_ci=hci,
        reach_cost=rc,
        reach_cost_ci=rcci,
        transient_cost=tc,
        transient_cost_ci=tcci,
        unsettled_fraction=unsettled_fraction(episodes),
        reach_fraction=reach_fraction(episodes),
        failure_rate=failure_rate(episodes),
        time_in_band=time_in_band(episodes, burn_in),
    )
    if settle is not None:
        out["move_cost"], out["move_cost_ci"] = transient_cost(episodes, settle)
    if band is not None:
        out["settling_time"], out["settling_time_ci"] = settling_time(episodes, band)
    for key in episodes[0].terms:
        out[key] = gain(episodes, burn_in, key)[0]
        out["hold_" + key] = hold(episodes, burn_in, key, settle=settle)[0]
    return out


# ---------------------------------------------------------------------------
# Running a registered environment through the protocol
# ---------------------------------------------------------------------------


def hold_settings(name: str) -> dict:
    """``burn_in`` for a plant, from the measurement file."""
    row = records.load_row("hold_measurements", name)
    return {"burn_in": int(row["burn_in"]) if row else 0}


def fixed_settle(name: str) -> int:
    """The settle every controller is scored with on ``name``: the slower of
    the shipped controllers' measured settles (``scripts/measure_hold.py``),
    rounded up. Fixed per plant, so a hold is the same window whoever flies
    it."""
    row = records.load_row("hold_measurements", name)
    measured = [row[k]["settle"] for k in ("pid", "mpc") if k in row]
    return int(math.ceil(max(measured, default=0)))


def scored_burn_in(name: str, params) -> int:
    """The burn-in ``evaluate_controller`` scores ``name`` with at ``params``:
    the hold row's, capped at half the episode. The oracles that weight the
    scored window read it too (``experts.mpc.protocol_burn_in``)."""
    burn_in = hold_settings(name)["burn_in"]
    return min(burn_in, int(params.max_steps_in_episode) // 2)


def run_episode(spec, params, policy, seed: int = 0, band=None) -> Episode:
    """One episode of a registered environment as an ``Episode`` record: the
    benchmark's one-episode run on ``PRNGKey(seed)`` (``runners.one_episode``).

    ``policy`` is ``policy(obs)`` or ``policy(obs, state)``. The cost is
    ``-reward``; the terms come from ``env.reward_terms`` when the environment
    provides one. ``band`` is a per-output error tolerance for the
    time-in-band KPI, in the observation's units (defaults to the task's own,
    ``benchmark.task_info``'s ``e_floor``).
    """
    import dataclasses

    from target_gym.benchmark import episodes_of, task_info
    from target_gym.runners.runners import one_episode

    task = task_info(spec.name, spec.make_env(), params, 1)
    if band is not None:
        e_floor = np.broadcast_to(np.asarray(band, float), task.e_floor.shape)
        task = dataclasses.replace(task, e_floor=e_floor.copy())
    return episodes_of(one_episode(spec, params, policy, seed, task=task))[0]


def evaluate_controller(name: str, kind: str = "pid", seeds: int = 3, params=None):
    """Score a shipped controller (``"pid"`` or ``"mpc"``) on a plant, with the
    plant's fixed settle and settle band."""
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy
    from target_gym.scoring import settle_band

    spec = REGISTRY[name]
    p = params or spec.make_test_params()
    episodes = [
        run_episode(spec, p, baseline_policy(spec, kind, p), seed=s)
        for s in range(seeds)
    ]
    return evaluate(
        episodes,
        burn_in=scored_burn_in(name, p),
        settle=fixed_settle(name),
        band=settle_band(name, p),
    )
