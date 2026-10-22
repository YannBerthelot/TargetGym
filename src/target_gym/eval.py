"""Evaluation protocol for reach-and-hold tasks.

A reach-and-hold controller is judged on two numbers. The first is the
long-run average cost while holding (the *gain*, what the plant pays forever).
The second is the excess cost of getting to the target after it moves (the
*reach cost*, paid once per target change). An episode return mixes the two in
a proportion set by the episode length, which is why the recorded baselines
are not the protocol.

A target cycle runs from one target change up to the next (the first cycle
starts at the reset). ``cycle_segments`` splits every cycle into three parts.

1. The *settle* after its own change. The level of the cycle is the mean cost
   over its second half, and the settle ends at the first step from which the
   cost stays within twice that level. It is at least one step and at most
   half the cycle, and it is measured on each cycle, so each controller is
   scored against its own transient. A first version applied the MPC's settle
   to the PID, which on the battery scored the PID's transient after every
   dispatch block as hold and manufactured an MPC advantage. With
   ``settle = 0`` the reach cost was zero by construction.
2. The *hold*, the steps in between.
3. The *anticipation* of the next change. A controller that previews the
   reference schedule (the shipped MPCs plan on the environment's own step,
   which carries the schedule) starts moving toward the next target before
   the change. That is good control, and it lowers the total cost. The rising
   cost it causes at the end of the cycle belongs to reaching the next target,
   and scored as hold it inflated the hold and deflated the reach cost.
   ``anticipation`` detects it from the cost alone, with no per-plant setting,
   as the longest run of steps at the end of the cycle that meets the
   conditions below. Each is stated against the level of the cycle without
   the run, the mean over its second half as the settle rule has it.

   * Its first two steps cost more than twice the level, and so do at least
     four in five of its steps, with never more than two in a row at or below
     it. An error that crosses zero, or passes through the dead band of the
     tracking cost, on its way to the next target makes such a dip, and one
     dip must not cut the run short. Opening on two steps keeps a lone noise
     spike just before the rise from pulling hold steps into the run.
   * Its median is at least ten times the level.
   * It is rising, its second half costing more on average than its first.
   * It is at least three steps long. A run of one or two steps counts only
     when every step of it costs a hundred times the level, and a two-step run
     must rise.

   Only a cycle of at least 16 steps that another cycle follows is searched,
   and the run takes at most half of it, so the last cycle of an episode
   never loses steps. The level is floored at the typical level of the
   episode, so that a cycle held inside the dead band at a cost of zero or
   nearly zero does not turn every small rise into a tenfold excursion.
   ``anticipations`` searches every cycle once with no floor (the last one
   too, for its level only), takes each cycle's level with the run it found
   left out, and searches again with the mean of those levels as the floor.
   Only the second search counts. The settle and the level are then measured
   on the cycle without the run. A cycle in which no run qualifies keeps
   exactly the settle, level and hold steps it had before the rule existed.

The conditions keep stationary per-step noise from reading as anticipation. A
cost of ``e**2`` with Gaussian ``e`` exceeds twice its mean on about 16% of
steps. In episodes of ten noise-only cycles the rule fires on at most 0.11% of
the cycles it searches, for Gaussian and heavy-tailed noise alike, and a
single cycle searched with no floor fires on at most 0.9% of 16-step cycles
and 0.3% of longer ones (``tests/test_eval_protocol.py`` measures the rates).
Noise correlated over tens of steps passes for anticipation more often, in up
to 0.22% of the cycles of such an episode and up to 7% of single cycles
searched with no floor. The rule then moves that excursion from the hold to
the next change's transient, which lowers the hold a little, and the gain is
unaffected either way.

The floor keeps a controller's own excursions at the edge of a dead band out
of the anticipation, and it also limits what the rule can see. On the
building (``hvac``) the PID's heating cost rises at the end of some night
cycles by more than the MPC's cost rises when it pre-heats or lets the zone
cool ahead of a change. From the cost alone the rule cannot take one and
leave the other, so it takes neither, and an anticipation costing less than
ten times the episode's typical level stays in the hold. A cycle that never
settles raises the floor of its episode, which makes the rule more
conservative. In a cycle too short for the controller to settle, where the
tail of the settle still dominates the second half, a modest anticipation
goes undetected in the same way and the cycle keeps its old numbers.

This module computes the following.

``gain``            mean per-step cost over every step after ``burn_in``,
                    pooled across episodes, with its 95% interval over
                    episodes. It is the long-run average cost of the natural
                    process, transients included, which is what Theorem 4
                    decomposes as ``rho = rho_hold + p * reach_cost``. It is
                    split into ``tracking`` and ``running`` (and ``failure``)
                    when the environment reports its reward terms. No
                    segmentation enters it, so two controllers are compared
                    on the same steps.
``hold``            the same over the hold steps only, ``rho_hold``. Split as
                    ``hold_tracking`` / ``hold_running``.
``reach_cost``      per target change, the summed cost above the cycle's hold
                    level over its settle, plus the summed cost above the
                    previous cycle's hold level over the anticipation that
                    prepared the change. The hold level is the mean cost over
                    the hold steps. ``rho = rho_hold + p * reach_cost`` at
                    change rate ``p``.
``transient_cost``  per target change, the summed cost over the first
                    ``burn_in`` steps after the change (up to the cycle's own
                    anticipation, if that comes first) plus the cost of the
                    anticipation steps among the last ``burn_in`` before the
                    change, so the window reaches as far on each side.
                    Nothing is subtracted, so it is what a target change
                    costs in absolute terms over a window the plant sets. Two
                    controllers' reach costs are each relative to their own
                    hold level and transient, so a controller holding far off
                    the target shows a small reach cost because its level
                    swallows its transient. The transient cost is the number
                    to compare across controllers.
``unsettled_fraction`` share of cycles whose cost, with the anticipation
                    removed, is higher over the second half than over the
                    first. Such a cycle reached no hold in the window, and its
                    reach cost (the transient measured against a "hold" level
                    above it) comes out negative. A negative reach cost marks
                    this case.
``reach_fraction``  share of steps spent before first entering the band.
``failure_rate``    share of cycles in which the plant trips (``info["tripped"]``).
``nea``             normalised expert advantage ``(PID - x) / (PID - floor)``,
                    which is 1 at the achievable floor, 0 at PID parity and
                    negative below PID. It is in cost units, so it needs
                    ``rho_floor``, which the floor-normalised reward makes 1
                    per tracked output when nothing avoidable is consumed.
``time_in_band``    a KPI only, never a training signal. The share of steps
                    with every tracked error inside ``band``.

``burn_in`` is per plant, from ``src/target_gym/data/hold_measurements.json``
(``scripts/measure_hold.py``), and is three cost-bearing time constants.
Passing ``settle`` overrides the per-cycle settle with a fixed count. The
anticipation is detected either way. ``scripts/measure_hold.py`` imports
``anticipations`` and runs it on the squared errors, so that the hold errors
behind the floors leave out the same steps, found with the same sensitivity
as on the quadratic tracking cost.

Usage::

    from target_gym.eval import evaluate_controller
    m = evaluate_controller("reactor", "pid", seeds=5)
    m["gain"], m["tracking"], m["running"], m["reach_cost"], m["nea"]

or, for a policy of your own, build ``Episode`` records with ``run_episode``
and call ``evaluate``.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "hold_measurements.json"

# Absolute slack on "within twice the level", as the settle rule has always had.
_TOL = 1e-12
# The anticipation rule (module docstring, ``anticipation``).
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
    cycle follows, and 0 for the cycle that runs to the end of the series,
    since no change follows it. ``cycles`` are consecutive, as ``_cycles``
    returns them.

    Two searches. The first has no floor, and once it finds a run anywhere it
    also searches the last cycle, whose level enters the floor too. The floor
    of the second is the mean, over every cycle of at least four steps, of the
    cycle's level (the mean over the second half) once the run the first
    search found is left out, so a long anticipation does not raise the floor
    it is measured against. A higher level only makes each condition harder,
    so the second search finds nothing where the first found nothing, and
    never a longer run.

    A ``(steps, outputs)`` series is searched output by output, each with its
    own floor, and a cycle's anticipation is the longest over the outputs.
    ``scripts/measure_hold.py`` calls this on its squared errors, so the hold
    errors behind the floors and the protocol's hold leave out the same kind
    of steps."""
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
    of the anticipation steps among the last ``window`` before the change, so
    the window reaches as far before the change as after it. Nothing is
    subtracted, so it is what the target change costs in absolute terms over
    a window the plant sets. Comparable between two controllers whose hold
    levels differ, which
    ``reach_cost`` is not, since it measures each controller's transient
    above that controller's own hold level, over a transient detected against
    that level. A controller holding far off shows a small reach cost because
    its level swallows its transient. ``evaluate`` uses the plant's burn-in
    (three cost-bearing time constants) as the window."""
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


def nea(rho_pi, rho_ref, rho_floor):
    d = rho_ref - rho_floor
    return (rho_ref - rho_pi) / d if d > 0 else float("nan")


def evaluate(
    episodes,
    burn_in: int,
    settle: int | None = None,
    rho_floor: float | None = None,
    rho_ref: float | None = None,
):
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
    for key in episodes[0].terms:
        out[key] = gain(episodes, burn_in, key)[0]
        out["hold_" + key] = hold(episodes, burn_in, key, settle=settle)[0]
    if rho_floor is not None:
        out["abs_gap"] = g - rho_floor
    if rho_floor is not None and rho_ref is not None:
        out["nea"] = nea(g, rho_ref, rho_floor)
    return out


# ---------------------------------------------------------------------------
# Running a registered environment through the protocol
# ---------------------------------------------------------------------------


def hold_settings(name: str) -> dict:
    """``burn_in`` for a plant, from the measurement file."""
    rows = json.loads(_DATA.read_text()) if _DATA.exists() else {}
    row = rows.get(name)
    return {"burn_in": int(row["burn_in"]) if row else 0}


def scored_burn_in(name: str, params) -> int:
    """The burn-in ``evaluate_controller`` scores ``name`` with at ``params``:
    the hold row's, capped at half the episode. The oracles that weight the
    scored window read it too (``experts.mpc.protocol_burn_in``)."""
    burn_in = hold_settings(name)["burn_in"]
    return min(burn_in, int(params.max_steps_in_episode) // 2)


def run_episode(spec, params, policy, seed: int = 0, band=None) -> Episode:
    """One episode of a registered environment as an ``Episode`` record.

    ``policy`` is ``policy(obs)`` or ``policy(obs, state)``. The cost is
    ``-reward``; the terms come from ``env.reward_terms`` when the environment
    provides one. ``band`` is a per-output error tolerance for the
    time-in-band KPI (defaults to each output's ``e_floor`` if the params carry
    one, else the reach segmentation uses the target changes alone).
    """
    import jax
    import jax.numpy as jnp

    from target_gym.runners.runners import _as_tuple, _wants_state

    env = spec.make_env()
    ti = list(_as_tuple(env.obs_target_index))
    vi = list(_as_tuple(env.obs_value_index))
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    terms_fn = getattr(env, "reward_terms", None)
    costs, bands, changes, failed, terms = [], [], [], [], {}
    prev_target = np.asarray(obs)[ti]
    if band is None:
        band = np.atleast_1d(np.asarray(getattr(params, "e_floor", np.inf), float))
    span = None
    while int(state.time) < int(params.max_steps_in_episode):
        o = np.asarray(obs)
        a = policy(o, state) if _wants_state(policy) else policy(o)
        obs, state, r, term, info = step(
            key, state, jnp.atleast_1d(jnp.asarray(a)), params
        )
        o = np.asarray(obs)
        costs.append(-float(r))
        err = np.abs(o[vi] - o[ti])
        bands.append(bool((err <= band).all()))
        tgt = o[ti]
        if span is None:
            span = np.maximum(np.abs(tgt), 1e-9)
        changes.append(bool((np.abs(tgt - prev_target) > 0.05 * span).any()))
        prev_target = tgt
        failed.append(
            bool(info.get("tripped", False)) if isinstance(info, dict) else False
        )
        # Exact per-step terms when the environment returns them (the
        # reactor sums ten sub-steps); otherwise the terms at the new state.
        step_terms = info.get("reward_terms") if isinstance(info, dict) else None
        if step_terms is None and terms_fn is not None:
            step_terms = terms_fn(state, params)
        if step_terms is not None:
            for k, v in step_terms.items():
                terms.setdefault(k, []).append(float(v))
    return Episode(
        cost=np.array(costs),
        in_band=np.array(bands),
        target_change=np.array(changes),
        failed=np.array(failed),
        terms={k: np.array(v) for k, v in terms.items()},
    )


def evaluate_controller(
    name: str, kind: str = "pid", seeds: int = 3, params=None, rho_ref=None
):
    """Score a shipped controller (``"pid"`` or ``"mpc"``) on a plant."""
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY[name]
    p = params or spec.make_test_params()
    episodes = [
        run_episode(spec, p, baseline_policy(spec, kind, p), seed=s)
        for s in range(seeds)
    ]
    return evaluate(
        episodes,
        burn_in=scored_burn_in(name, p),
        rho_floor=float(getattr(p, "rho_floor", float("nan"))),
        rho_ref=rho_ref,
    )


if __name__ == "__main__":
    # Smoke test against Theorem 4 of the note: constant hold cost 0.1 with reach
    # excursions of 5 at change rate p = 0.01 gives gain ~ 0.1 + p * 5 = 0.15,
    # hold ~ 0.1, reach_cost ~ 5, reach_fraction ~ p * 5.
    rng = np.random.default_rng(0)
    T, p = 20000, 0.01
    tc = rng.random(T) < p
    cost = np.full(T, 0.1)
    band = np.ones(T, bool)
    for i in np.flatnonzero(tc):
        cost[i : i + 5] += 1.0
        band[i : i + 5] = False
    ep = Episode(cost=cost, in_band=band, target_change=tc, failed=np.zeros(T, bool))
    m = evaluate([ep], burn_in=100, rho_floor=0.1, rho_ref=0.2)
    print({k: round(float(v), 4) for k, v in m.items()})
