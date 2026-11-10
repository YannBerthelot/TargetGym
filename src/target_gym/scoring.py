"""Scoring a policy against the expert and the ceiling.

Three numbers answer three questions (docs/rl-protocol.md, section 4). With
``b`` the expert's cost per step, ``x`` the policy's and ``c`` the ceiling's:

- ``ratio``: ``b / x`` over the whole episode, per seed, with ``win_rate``, the
  share of seeds where the policy costs less. Above 1 is better than the expert.
  ``beats_expert`` is the verdict: the ratio clears the 1% that separates two
  recordings of one row (``records.REL_TOL``) and the policy trips no more
  (``expert_trips``, ``policy_trips``, per episode). The ratio alone can hide a
  trip, because its interquartile mean trims the worst seeds.
- ``score``: ``log(b / x) / log(b / c)`` on the tracking cost while holding: 0
  at the expert, 1 at the ceiling (and capped there), negative below the
  expert. Being a ratio of logarithms it has no units and does not depend on
  the tracking exponent, so it compares across environments.
- ``headroom``: ``log10(b / c)``, the decades of room the task offers beyond
  the expert, and ``room_left``, the decades the policy has left.

The ceiling is as good as a plant can verify: every tracked quantity within
its instrument's resolution, at the reward's price (``ceiling``). Consumption
has no such bound, so ``consumption_ratio`` reports it beside the score, and
``s_max`` says how much of the score the task's best known hold allows.

An owner also asks what a win costs in wear and whether it matters.
``expert_travel`` and ``policy_travel`` are the full strokes moved per episode,
``expert_reversals`` and ``policy_reversals`` the changes of direction, and
``expert_off_spec`` and ``policy_off_spec`` the share of the hold outside the
task's tolerance (``spec_band``).

Moves have no ceiling either. A move is the reset or a target change, and
``transient_ratio`` is the expert's cost over the plant's fixed settle after
each move divided by the policy's, paired per seed like ``ratio``, with
``transient_win_rate`` and ``transient_ratio_ci``. ``settling_ratio`` compares
the steps until the tracking cost enters the plant's band and stays
(``settle_band``). ``score_policy`` adds the oracle's recorded ratios as the
best known (``best_known``), for information.

Score a policy of your own::

    from target_gym.scoring import score_policy
    m = score_policy("cstr", policy, seeds=10)
    m["ratio"], m["win_rate"], m["score"], m["headroom"]

``policy(obs)`` returns an action; one with a ``reset()`` method is reset
before each episode. The expert is the environment's shipped ``"pid"`` slot, run on
the same seeds.

The task charter (J4) adds the headline that replaces these as tasks are rebuilt.
``headline`` is the saving over the reference on a year of operation: each scenario
(``target_gym.scenarios``) weighted by its share of operating time, with the
per-scenario savings and the probability of improvement beside it. ``gap_closed``,
the share of the reference-to-ceiling gap a policy closes, is reported, never the
headline.

``seed_scores`` is a different number for a different job: one score per
training seed, on the whole episode with no burn-in, from costs measured
elsewhere, for a statistical test across trained agents.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np

from target_gym import eval as E
from target_gym import records
from target_gym.eval import fixed_settle

__all__ = [
    "CONSUMPTION",
    "JOB_SCORED",
    "SPEC_BAND",
    "TRACKED_TERMS",
    "ceiling",
    "compare",
]
__all__ += ["fixed_settle", "headroom", "best_known", "recorded", "s_max", "score"]
__all__ += ["score_policy", "seed_scores", "settle_band", "spec_band"]
__all__ += ["gap_closed", "headline"]

#: How far one episode can move a seed's score, either way (``seed_scores``).
_CLIP_DECADES = 3.0

_ALTITUDE = ("precision_floor", "e_floor_altitude", 1)
_HEADING = ("heading_precision_floor", "e_floor_heading", 1)
_PATH = ("position_precision_floor", "e_floor_path", 1)
_SLOT = ("slot_precision_floor", "e_floor_slot", 1)
_ONE = (("precision_floor", "e_floor", 1),)

#: Tasks scored on the operator's job (charter B5): keeping inside the plant's
#: limits at the source's economic cost, with no tracked setpoint. They have no
#: tracking ceiling or settle band (``ceiling`` and ``settle_band`` are NaN),
#: and they are in neither ``TRACKED_TERMS`` nor ``CONSUMPTION``. Their headline
#: is ``headline``; room comes from an oracle.
JOB_SCORED = frozenset(
    {
        "battery",
        "boiler_drum",
        "compressor_surge",
        "distillation",
        "hvac",
        "ph_neutralization",
        "wind_turbine",
    }
)

#: Per environment, the terms its reward tracks: ``(resolution field, e_floor
#: field, count)`` on its params. ``count`` is ``"wingmen"`` for a term charged
#: once per wingman. hvac prices comfort directly and has no ``e_floor``;
#: glass_furnace prices lost glass directly and has none either.
TRACKED_TERMS = {
    "plane": _ONE,
    "plane_energy": _ONE,
    "plane_sine": _ONE,
    "plane3d_heading": (_ALTITUDE, _HEADING),
    "plane3d_circle": (_ALTITUDE, _PATH),
    "plane3d_racetrack": (_ALTITUDE, _PATH, _HEADING),
    "plane3d_figure8": (_PATH,),
    "patrol": (_SLOT, _HEADING),
    "patrol_range_bearing": (_SLOT, _HEADING),
    "wake_surf": (_SLOT,),
    "formation": (
        _ALTITUDE,
        _HEADING,
        _SLOT[:2] + ("wingmen",),
        _HEADING[:2] + ("wingmen",),
    ),
    "first_order": _ONE,
    "cstr": _ONE,
    "unstable_cstr": _ONE,
    "four_tank": (("precision_floor", "e_floor", 2),),
    "cement_kiln": _ONE,
    "boiler_drum": (
        ("level_precision_floor", "e_floor_level", 1),
        ("pressure_precision_floor", "e_floor_pressure", 1),
    ),
    "reactor": _ONE,
    "battery": _ONE,
    "wind_turbine": _ONE,
    "hvac": (("precision_floor", None, 1),),
}


def _recycle_power(state, params, tripped):
    import jax.numpy as jnp

    from target_gym.compressor_surge.env import recycle_power

    return recycle_power(state, params, jnp)


#: Per environment, what its reward's ``running_cost`` charges, as ``fn(state,
#: params, tripped)``, so ``consumption_ratio`` reads the quantity itself. The
#: other running terms are not consumption. glass_furnace's running term is its
#: fuel bill, so the ratio reads that term itself.
CONSUMPTION = {
    "cement_kiln": lambda state, params, tripped: state.fuel,
}

_RVSM = 65.0 * 0.3048  # m
_RVSM_BAND = "+-65 ft about an acquired altitude, 14 CFR 91 App. G, 2(c)(2)(i)"
_ALTITUDE_BAND = "the autopilot's +-20 m altitude tolerance, plane/PHYSICS.md"
_SLOT_BAND = "AFF's 10 ft one sigma on the Hornet, NASA/TM-2002-210729"

#: Per environment, the tolerance on each tracked output, in the observation's
#: units, with where it comes from: a secondary source says so, and a figure no
#: second pass could re-open is labelled ours. It is a report: it does not enter
#: the reward. ``None`` is a task with no two-sided band to report.
#: Any other task gets ``SPEC_RULE_FLOORS`` floors, which is ours and no plant's
#: figure, until a sourced number replaces it
#: (docs/proposals/industrial-fixes.md).
SPEC_BAND = {
    "plane": ((_RVSM,), _RVSM_BAND),
    "plane_energy": ((_RVSM,), _RVSM_BAND),
    "plane_sine": ((_RVSM,), _RVSM_BAND),
    "plane3d_heading": ((_RVSM,), _RVSM_BAND),
    "plane3d_circle": ((_RVSM,), _RVSM_BAND),
    "plane3d_racetrack": ((_RVSM,), _RVSM_BAND),
    "patrol": ((3.048,), _SLOT_BAND),
    "patrol_range_bearing": ((3.048,), _SLOT_BAND),
    "wake_surf": ((30.0,), "ours: the station box, a span (plane3d/PHYSICS.md)"),
    "formation": ((20.0, 3.048, 3.048), _ALTITUDE_BAND + "; " + _SLOT_BAND),
    "ph_neutralization": (
        (1.0,),
        "the permit range pH 7 +-1, the example spec of Faanes and Skogestad "
        "2000 (ph_neutralization/REALITY.md [P7])",
    ),
    "glass_furnace": None,  # a zone, fining and canal, that the reward prices
    "reactor": (
        (1.5 / 1.8,),
        "ours: the rod controller's +-1.5 F dead band, as reactor/PHYSICS.md cites "
        "NRC HRTD 8.1, which no second pass could re-open",
    ),
    "hvac": None,  # one-sided at night, and the reward prices comfort itself
}
SPEC_RULE_FLOORS = 10.0


def spec_band(name: str, params=None, env=None) -> np.ndarray | None:
    """The tolerance on each tracked output, in the observation's units
    (``SPEC_BAND``), or ``None`` where a task has none."""
    from target_gym.benchmark import task_info
    from target_gym.registry import REGISTRY

    if name in SPEC_BAND:
        entry = SPEC_BAND[name]
        return None if entry is None else np.asarray(entry[0], float)
    spec = REGISTRY[name]
    p = params if params is not None else spec.make_test_params()
    e = env if env is not None else spec.make_env()
    return SPEC_RULE_FLOORS * task_info(name, e, p, 1).e_floor[0]


def _currency_per_floor(name: str, p, step_seconds: float) -> float:
    """What one floor-width of tracking error costs per control step on the
    plants that price it (``compute_reward_terms`` of each, summed over the
    step's sub-steps); 1 on the others."""
    if name == "reactor":
        per_unit_s = (
            p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW * 1000.0
        ) / 3600.0
        return per_unit_s * step_seconds * p.e_floor
    return 1.0


def _per_step(name: str, params, env, at_resolution: bool) -> float:
    """The reward's tracking charge per step with every tracked error one
    resolution (or one floor) beyond its tolerance."""
    from target_gym.registry import REGISTRY, control_step_seconds

    if name in JOB_SCORED:
        return float("nan")
    spec = REGISTRY[name]
    p = params if params is not None else spec.make_test_params()
    env = env if env is not None else spec.make_env()
    if name == "glass_furnace":
        # Any shortfall under the pass mark loses the step's whole pull to cullet.
        seconds = control_step_seconds(env, p)
        return float(p.lost_glass_price * p.pull * seconds / 1000.0)
    exponent = float(p.tracking_exponent)
    if name == "hvac":
        hours = p.delta_t / 3600.0
        return float(p.comfort_price * hours * p.precision_floor**exponent)
    total = 0.0
    for res, floor, count in TRACKED_TERMS[name]:
        if count == "wingmen":
            count = env.num_wingmen
        ratio = float(getattr(p, res)) / float(getattr(p, floor))
        total += count * (ratio**exponent if at_resolution else 1.0)
    return float(_currency_per_floor(name, p, control_step_seconds(env, p)) * total)


def ceiling(name: str, params=None, env=None) -> float:
    """The cost per step of tracking every quantity within its instrument's
    resolution: for each tracked term, the reward's charge for an error one
    resolution beyond its tolerance, ``(resolution / e_floor) ** p``, in the
    plant's currency where the reward prices tracking."""
    return _per_step(name, params, env, at_resolution=True)


def settle_band(name: str, params=None, env=None) -> float:
    """The tracking cost per step a move has settled within: every tracked
    error at its floor beyond its tolerance, 1 per term, in the plant's
    currency where the reward prices tracking. hvac has no floor, so its band
    is its ceiling, and glass_furnace's, whose reward is a step, too."""
    return _per_step(name, params, env, at_resolution=False)


def score(x: float, b: float, c: float) -> float:
    """``log(b / x) / log(b / c)``, capped at 1; NaN when the expert is already
    within the ceiling, since there is no room to score."""
    if not b > c:
        return float("nan")
    return float(math.log(b / max(x, c)) / math.log(b / c))


def s_max(b: float, c: float, bound: float) -> float:
    """The most of the score a task allows: ``score`` at ``bound``, its best
    known hold, or 1 where no bound is known or it sits under the ceiling. 0
    when the expert is already past the bound, NaN with no room to score."""
    if not b > c:
        return float("nan")
    known = bound if bound == bound and bound > 0.0 else c
    return max(score(known, b, c), 0.0)


def headroom(b: float, c: float) -> float:
    """Decades between a cost ``b`` and the ceiling ``c``; 0 within it."""
    return float(math.log10(b / c)) if b > c else 0.0


def gap_closed(policy_cost, reference_cost, ceiling_cost) -> np.ndarray:
    """Per seed, the share of the reference-to-ceiling gap the policy closes:
    ``(b - x) / (b - c)``. 0 at the reference, 1 at the ceiling, above 1 past it
    (a ceiling is a local optimum), negative above the reference. NaN where
    ``b <= c``, with no gap to close. Charter Amendment 3, rule 21."""
    x, b, c = np.broadcast_arrays(
        *(
            np.asarray(v, dtype=float)
            for v in (policy_cost, reference_cost, ceiling_cost)
        )
    )
    if np.isnan(x).any() or np.isnan(b).any() or np.isnan(c).any():
        raise ValueError("a cost is NaN")
    if (x < 0).any() or (b < 0).any() or (c < 0).any():
        raise ValueError("costs must be non-negative")
    gap = b - c
    has_gap = gap > 0
    with np.errstate(invalid="ignore"):
        return np.where(has_gap, (b - x) / np.where(has_gap, gap, 1.0), np.nan)


_N_BOOT = 2000


def _costs(values) -> np.ndarray:
    a = np.atleast_1d(np.asarray(values, dtype=float))
    if np.isnan(a).any() or (a < 0).any():
        raise ValueError("costs must be non-negative and not NaN")
    return a


def _saving(w, xs, bs) -> float:
    j_ref = sum(wi * bi.mean() for wi, bi in zip(w, bs))
    j_policy = sum(wi * xi.mean() for wi, xi in zip(w, xs))
    return 1.0 - j_policy / j_ref if j_ref > 0 else float("nan")


def headline(costs, weights) -> dict:
    """A task's headline (charter J4): the frequency-weighted saving over the
    reference.

    ``costs`` is ``{scenario: {"policy": ..., "reference": ...}}``, each a cost per
    step per seed, paired seed for seed. A scenario may add ``"ceiling"`` (per seed
    or one value) and ``"policy_trips"`` with ``"reference_trips"`` (per seed).
    ``weights`` is each scenario's share of operating time, at any positive scale.

    ``saving`` is ``1 - J_policy / J_reference``, with ``J = sum_s w_s * mean(cost |
    s)``; ``saving_ci`` its 95 % interval, resampling seeds within each scenario and
    keeping the pairs. Beside it: ``per_scenario`` savings, the ``worst`` scenario,
    and ``probability_of_improvement``, the weighted share of seeds won. With
    ceilings, ``gap_closed`` per scenario on its means; with trips,
    ``beats_reference``: the interval above 0 and no more trips in any scenario. A
    reference that costs nothing leaves the saving NaN, a solved task.
    """
    names = list(costs)
    if set(weights) != set(names):
        raise ValueError(f"weights {sorted(weights)} must cover {sorted(names)}")
    w = np.array([float(weights[n]) for n in names])
    if not (np.isfinite(w).all() and (w > 0).all()):
        raise ValueError(f"weights must be positive, got {w}")
    w = w / w.sum()
    xs = [_costs(costs[n]["policy"]) for n in names]
    bs = [_costs(costs[n]["reference"]) for n in names]
    if any(x.shape != b.shape for x, b in zip(xs, bs)):
        raise ValueError("policy and reference need the same seeds in each scenario")

    point = _saving(w, xs, bs)
    ci = (float("nan"), float("nan"))
    if math.isfinite(point):
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(_N_BOOT):
            picks = [rng.integers(0, len(b), len(b)) for b in bs]
            resampled_x = [x[i] for x, i in zip(xs, picks)]
            resampled_b = [b[i] for b, i in zip(bs, picks)]
            boots.append(_saving(w, resampled_x, resampled_b))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci = (float(lo), float(hi))
    per = {
        n: 1.0 - x.mean() / b.mean() if b.mean() > 0 else float("nan")
        for n, x, b in zip(names, xs, bs)
    }
    scored = {n: v for n, v in per.items() if v == v}
    out = {
        "saving": point,
        "saving_ci": ci,
        "per_scenario": per,
        "worst": min(scored, key=scored.get) if scored else None,
        "probability_of_improvement": float(
            sum(wi * np.mean(x < b) for wi, x, b in zip(w, xs, bs))
        ),
    }
    if all("ceiling" in costs[n] for n in names):
        out["gap_closed"] = {
            n: float(gap_closed(x.mean(), b.mean(), _costs(costs[n]["ceiling"]).mean()))
            for n, x, b in zip(names, xs, bs)
        }
    if all("policy_trips" in costs[n] and "reference_trips" in costs[n] for n in names):
        fewer = all(
            np.sum(costs[n]["policy_trips"]) <= np.sum(costs[n]["reference_trips"])
            for n in names
        )
        out["beats_reference"] = bool(ci[0] > 0 and fewer)
    return out


def _iqm(x: np.ndarray) -> float:
    x = np.sort(x)
    k = len(x) // 4
    return float(x[k : len(x) - k].mean())


def _ratio_ci(log_ratios: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """95% bootstrap interval on the interquartile mean, as a ratio."""
    rng = np.random.default_rng(0)
    boots = [_iqm(rng.choice(log_ratios, len(log_ratios))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(10**lo), float(10**hi)


def _paired(b: np.ndarray, x: np.ndarray) -> tuple[float, tuple, float]:
    """Per-seed ratios ``b / x`` as an interquartile mean, its interval and
    the share of seeds where ``x`` costs less."""
    logs = np.log10(b / np.maximum(x, 1e-300))
    return float(10 ** _iqm(logs)), _ratio_ci(logs), float(np.mean(x < b))


def _ratio(b: float, x: float) -> float:
    """``b / x``, 1 when both are 0 (neither ever left the band)."""
    if b == x:
        return 1.0
    return b / x if x > 0 else float("inf")


def compare(
    policy, expert, ceiling: float, burn_in: int, settle: int, band=None
) -> dict:
    """The three numbers and their companions for ``Episode`` lists run on the
    same seeds, in the same order (module docstring). ``band`` adds the
    settling ratio."""
    b = np.array([E.gain([ep], 0)[0] for ep in expert])
    x = np.array([E.gain([ep], 0)[0] for ep in policy])
    ratio, ratio_ci, win_rate = _paired(b, x)
    expert_trips, policy_trips = E.trip_counts(expert), E.trip_counts(policy)
    moves_b = np.array([E.transient_cost([ep], settle)[0] for ep in expert])
    moves_x = np.array([E.transient_cost([ep], settle)[0] for ep in policy])
    t_ratio, t_ci, t_win = _paired(moves_b, moves_x)
    b_hold = E.hold(expert, burn_in, "tracking", settle)[0]
    x_hold = E.hold(policy, burn_in, "tracking", settle)[0]
    out = dict(
        ratio=ratio,
        ratio_ci=ratio_ci,
        win_rate=win_rate,
        beats_expert=bool(
            ratio > 1.0 + records.REL_TOL and sum(policy_trips) <= sum(expert_trips)
        ),
        expert_trips=expert_trips,
        policy_trips=policy_trips,
        score=score(x_hold, b_hold, ceiling),
        headroom=headroom(b_hold, ceiling),
        room_left=headroom(x_hold, ceiling),
        expert_hold=b_hold,
        policy_hold=x_hold,
        ceiling=ceiling,
        transient_ratio=t_ratio,
        transient_ratio_ci=t_ci,
        transient_win_rate=t_win,
    )
    if band is not None:
        out["expert_settling"] = E.settling_time(expert, band)[0]
        out["policy_settling"] = E.settling_time(policy, band)[0]
        out["settling_ratio"] = _ratio(out["expert_settling"], out["policy_settling"])
    key = next((k for k in ("consumption", "running") if k in expert[0].terms), None)
    if key is not None:
        # Policy over expert. The expert consumes nothing on the tasks whose
        # running weight is zero, so this takes _ratio's convention.
        out["consumption_ratio"] = _ratio(
            E.gain(policy, 0, key)[0], E.gain(expert, 0, key)[0]
        )
    if expert[0].actions is not None:
        for side, episodes in (("expert", expert), ("policy", policy)):
            actions = [ep.actions for ep in episodes]
            out[f"{side}_travel"] = float(np.mean([E.travel(a) for a in actions]))
            out[f"{side}_reversals"] = float(np.mean([E.reversals(a) for a in actions]))
    return out


def seed_scores(
    *,
    expert_cost_per_step,
    policy_cost_per_step,
    ceiling_per_step: float,
    trips=None,
) -> dict:
    """One score per training seed, on the whole episode.

    With ``b_e`` and ``x_e`` the expert's and the policy's cost on episode
    ``e`` and ``c`` the ceiling, a seed scores the mean over its episodes of
    ``clip(log10(max(b_e, c) / max(x_e, c)), -3, +3)``. Above 0 beats the
    expert. 1 is ten times cheaper.

    All three costs are **per step**, in the task's reward units:

    - ``expert_cost_per_step``, ``policy_cost_per_step``: shape ``(seeds,
      episodes)``, the mean of ``-reward`` over every step of the episode, no
      burn-in, which is ``target_gym.eval.gain([episode], 0)[0]``. Never an
      episode's summed cost: the ceiling is per step, so the floor would sit
      an episode length too low. Both sides must come from the same episodes
      in the same order, since the clip and the floor make the pairing matter.
      An infinite policy cost scores -3, so a driver can pass a diverged
      policy as ``inf``.
    - ``ceiling_per_step``: ``ceiling(name, params)``.

    The floor is on both sides, so two costs within the resolution tie at 0
    and swapping the sides negates the score. The clip is there because one
    trip can cost more decades than a good policy gains on a whole episode,
    so it hides how many trips there were. ``trips``, the number of trips in
    each episode (``(seeds, episodes)``, or flags for "at least one"), adds
    ``trip_rate`` to the result: the mean per episode, read from the counts
    and untouched by the clip.

    Returns ``{"score": ..., "trip_rate": ...}``, each of shape ``(seeds,)``,
    the second only with ``trips``. A negative or NaN cost raises
    ``ValueError`` rather than being clamped: reward terms are non-negative,
    so one means a broken measurement, and a NaN would pass silently into
    whatever test reads the score.
    """
    b = np.asarray(expert_cost_per_step, dtype=float)
    x = np.asarray(policy_cost_per_step, dtype=float)
    if b.ndim != 2 or b.shape != x.shape or b.size == 0:
        raise ValueError(
            f"need paired (seeds, episodes) costs, got {b.shape} and {x.shape}"
        )
    if not ceiling_per_step > 0:
        raise ValueError(f"the ceiling must be positive, got {ceiling_per_step}")
    if not (np.isfinite(b).all() and (b >= 0).all() and (x >= 0).all()):
        raise ValueError("costs must be non-negative, and the expert's finite")
    floored = np.log10(np.maximum(b, ceiling_per_step))
    decades = floored - np.log10(np.maximum(x, ceiling_per_step))
    out = {"score": np.clip(decades, -_CLIP_DECADES, _CLIP_DECADES).mean(axis=1)}
    if trips is not None:
        counts = np.asarray(trips, dtype=float)
        if counts.shape != b.shape or not (counts >= 0).all():
            raise ValueError(f"need trips per episode in shape {b.shape}")
        out["trip_rate"] = counts.mean(axis=1)
    return out


def recorded(name: str | None = None) -> dict:
    """``name``'s row of the recorded protocol results (``pid`` on
    ``pid_seeds`` seeds, ``mpc`` on ``seeds``, of the test configuration),
    ``{}`` if it has none, or every row
    by task when ``name`` is None."""
    if name is None:
        return records.load("protocol_results")
    return records.load_row("protocol_results", name)


def best_known(row: dict) -> dict:
    """The oracle's recorded move and settling ratios against the expert's,
    from a ``recorded`` row: the best known on the moves, for information.
    NaN until the row records moves."""
    pid, mpc = row.get("pid", {}), row.get("mpc", {})
    out = {}
    for key, name in (("move_cost", "transient"), ("settling_time", "settling")):
        known = key in pid and key in mpc
        ratio = _ratio(pid[key], mpc[key]) if known else float("nan")
        out[f"oracle_{name}_ratio"] = ratio
    return out


def score_policy(name: str, policy, seeds: int = 10, params=None) -> dict:
    """Run ``policy`` and the expert on seeds ``0 .. seeds - 1`` of the test
    configuration, ``compare`` them, and add the ``best_known`` moves.

    ``policy(obs) -> action``, or ``policy(obs, state)`` for a full-state
    controller. One with a ``reset()`` method is reset before each episode.
    ``target_gym.make_expert(name)`` works as it is. A factory taking no
    arguments, the 0.7.0 form, still works and warns.
    """
    from target_gym.api import as_policy

    policy = as_policy(policy)
    if _takes_no_arguments(policy):
        warnings.warn(
            "score_policy(name, make_policy) is deprecated; pass the policy "
            "itself, score_policy(name, policy)",
            DeprecationWarning,
            stacklevel=2,
        )
        return _score(name, policy, seeds, params)
    return _score(name, lambda: _fresh(policy), seeds, params)


def _takes_no_arguments(f) -> bool:
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return False
    return not any(
        p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        for p in sig.parameters.values()
    )


def _fresh(policy):
    if callable(getattr(policy, "reset", None)):
        policy.reset()
    return policy


def _score(name: str, make_policy, seeds: int, params) -> dict:
    """``score_policy`` on ``make_policy()``, called once per episode."""
    from target_gym.api import as_policy
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY[name]
    p = params if params is not None else spec.make_test_params()
    tolerance = spec_band(name, p)
    record = {"consumption": CONSUMPTION[name]} if name in CONSUMPTION else None

    def run(make):
        return [
            E.run_episode(spec, p, make(), seed=s, band=tolerance, record=record)
            for s in range(seeds)
        ]

    expert = run(lambda: baseline_policy(spec, "pid", p))
    policy = run(lambda: as_policy(make_policy()))
    burn_in, settle = E.scored_burn_in(name, p), fixed_settle(name)
    m = compare(
        policy,
        expert,
        ceiling(name, p),
        burn_in,
        settle,
        band=settle_band(name, p),
    )
    for side, episodes in (("expert", expert), ("policy", policy)):
        share = E.off_spec_fraction(episodes, burn_in, settle)
        m[f"{side}_off_spec"] = float("nan") if tolerance is None else share
    row = recorded(name)
    # The best known tracking hold, since the bound the hold is read against
    # must leave out consumption; rows from before 0.8.0 have only ``rho_floor``.
    bound = row.get("rho_floor_tracking", row.get("rho_floor", float("nan")))
    bound = float("nan") if bound is None else bound
    return {
        **m,
        "s_max": s_max(m["expert_hold"], m["ceiling"], bound),
        **best_known(row),
    }
