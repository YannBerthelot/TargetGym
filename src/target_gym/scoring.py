"""Scoring a policy against the expert and the ceiling.

Three numbers answer three questions (docs/rl-protocol.md, section 4). With
``b`` the expert's cost per step, ``x`` the policy's and ``c`` the ceiling's:

- ``ratio``: ``b / x`` over the whole scored episode, per seed, with
  ``win_rate``, the share of seeds where the policy costs less. Above 1 is
  better than the expert.
- ``score``: ``log(b / x) / log(b / c)`` on the tracking cost while holding: 0
  at the expert, 1 at the ceiling (and capped there), negative below the
  expert. Being a ratio of logarithms it has no units and does not depend on
  the tracking exponent, so it compares across environments.
- ``headroom``: ``log10(b / c)``, the decades of room the task offers beyond
  the expert, and ``room_left``, the decades the policy has left.

The ceiling is as good as a plant can verify: every tracked quantity within
its instrument's resolution, at the reward's price (``ceiling``). Consumption
has no such bound, so ``consumption_ratio`` reports it beside the score, and
``transient_ratio`` compares the cost after each target change with the
expert's.

Score a policy of your own::

    from target_gym.scoring import score_policy
    m = score_policy("cstr", make_policy, seeds=10)
    m["ratio"], m["win_rate"], m["score"], m["headroom"]

``make_policy()`` returns a fresh ``policy(obs)`` or ``policy(obs, state)`` for
each episode. The expert is the environment's shipped ``"pid"`` slot, run on
the same seeds.
"""

from __future__ import annotations

import math

import numpy as np

from target_gym import eval as E
from target_gym.eval import fixed_settle

__all__ = ["TRACKED_TERMS", "ceiling", "compare", "fixed_settle", "headroom"]
__all__ += ["score", "score_policy"]

_ALTITUDE = ("precision_floor", "e_floor_altitude", 1)
_HEADING = ("heading_precision_floor", "e_floor_heading", 1)
_PATH = ("position_precision_floor", "e_floor_path", 1)
_SLOT = ("slot_precision_floor", "e_floor_slot", 1)
_ONE = (("precision_floor", "e_floor", 1),)

#: Per environment, the terms its reward tracks: ``(resolution field, e_floor
#: field, count)`` on its params. ``count`` is ``"wingmen"`` for a term charged
#: once per wingman. hvac prices comfort directly and has no ``e_floor``.
TRACKED_TERMS = {
    "plane": _ONE,
    "plane_energy": _ONE,
    "plane_sine": _ONE,
    "plane3d_heading": (_ALTITUDE, _HEADING),
    "plane3d_circle": (_ALTITUDE, _PATH),
    "plane3d_racetrack": (_ALTITUDE, _PATH),
    "plane3d_figure8": (_PATH,),
    "patrol": (_SLOT, _HEADING),
    "patrol_bearing_only": (_SLOT, _HEADING),
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
    "distillation": (
        ("precision_floor", "e_floor_top", 1),
        ("precision_floor", "e_floor_bottom", 1),
    ),
    "ph_neutralization": _ONE,
    "glass_furnace": _ONE,
    "cement_kiln": _ONE,
    "boiler_drum": (
        ("level_precision_floor", "e_floor_level", 1),
        ("pressure_precision_floor", "e_floor_pressure", 1),
    ),
    "compressor_surge": _ONE,
    "reactor": _ONE,
    "battery": _ONE,
    "wind_turbine": _ONE,
    "hvac": (("precision_floor", None, 1),),
}


def _currency_per_floor(name: str, p) -> float:
    """What one floor-width of tracking error costs on the plants that price
    it (``compute_reward_terms`` of each); 1 on the others."""
    if name == "reactor":
        per_unit_s = (
            p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW * 1000.0
        ) / 3600.0
        return per_unit_s * p.e_floor
    if name in ("battery", "wind_turbine"):
        return p.imbalance_price / 1.0e6 * p.delta_t / 3600.0 * p.e_floor
    return 1.0


def ceiling(name: str, params=None, env=None) -> float:
    """The cost per step of tracking every quantity within its instrument's
    resolution: for each tracked term, the reward's charge for an error one
    resolution beyond its tolerance, ``(resolution / e_floor) ** p``, in the
    plant's currency where the reward prices tracking."""
    from target_gym.registry import REGISTRY

    spec = REGISTRY[name]
    p = params if params is not None else spec.make_test_params()
    exponent = float(p.tracking_exponent)
    if name == "hvac":
        hours = p.delta_t / 3600.0
        return float(p.comfort_price * hours * p.precision_floor**exponent)
    total = 0.0
    for res, floor, count in TRACKED_TERMS[name]:
        if count == "wingmen":
            count = (env if env is not None else spec.make_env()).num_wingmen
        term = (float(getattr(p, res)) / float(getattr(p, floor))) ** exponent
        total += count * term
    return float(_currency_per_floor(name, p) * total)


def score(x: float, b: float, c: float) -> float:
    """``log(b / x) / log(b / c)``, capped at 1; NaN when the expert is already
    within the ceiling, since there is no room to score."""
    if not b > c:
        return float("nan")
    return float(math.log(b / max(x, c)) / math.log(b / c))


def headroom(b: float, c: float) -> float:
    """Decades between a cost ``b`` and the ceiling ``c``; 0 within it."""
    return float(math.log10(b / c)) if b > c else 0.0


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


def _transient(episodes, settle: int) -> float:
    """Mean cost summed over the ``settle`` steps after each target change."""
    sums = [
        ep.cost[i : i + settle].sum()
        for ep in episodes
        for i in np.flatnonzero(ep.target_change)
    ]
    return float(np.mean(sums)) if sums else float("nan")


def compare(policy, expert, ceiling: float, burn_in: int, settle: int) -> dict:
    """The three numbers and their companions for ``Episode`` lists run on the
    same seeds, in the same order (module docstring)."""
    b = np.array([E.gain([ep], burn_in)[0] for ep in expert])
    x = np.array([E.gain([ep], burn_in)[0] for ep in policy])
    logs = np.log10(b / np.maximum(x, 1e-300))
    b_hold = E.hold(expert, burn_in, "tracking", settle)[0]
    x_hold = E.hold(policy, burn_in, "tracking", settle)[0]
    out = dict(
        ratio=float(10 ** _iqm(logs)),
        ratio_ci=_ratio_ci(logs),
        win_rate=float(np.mean(x < b)),
        score=score(x_hold, b_hold, ceiling),
        headroom=headroom(b_hold, ceiling),
        room_left=headroom(x_hold, ceiling),
        expert_hold=b_hold,
        policy_hold=x_hold,
        ceiling=ceiling,
        transient_ratio=_transient(expert, settle) / _transient(policy, settle),
    )
    if "running" in expert[0].terms:
        out["consumption_ratio"] = (
            E.gain(policy, burn_in, "running")[0]
            / E.gain(expert, burn_in, "running")[0]
        )
    return out


def score_policy(name: str, make_policy, seeds: int = 10, params=None) -> dict:
    """Run ``make_policy()`` and the expert on seeds ``0 .. seeds - 1`` of the
    test configuration and ``compare`` them."""
    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY[name]
    p = params if params is not None else spec.make_test_params()
    expert = [
        E.run_episode(spec, p, baseline_policy(spec, "pid", p), seed=s)
        for s in range(seeds)
    ]
    policy = [E.run_episode(spec, p, make_policy(), seed=s) for s in range(seeds)]
    return compare(
        policy, expert, ceiling(name, p), E.scored_burn_in(name, p), fixed_settle(name)
    )
