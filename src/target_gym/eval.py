"""Evaluation protocol for reach-and-hold tasks.

A reach-and-hold controller is judged on two numbers, not one return: the
long-run average cost while holding (the *gain*, what the plant pays forever)
and the excess cost of getting to the target after it moves (the *reach cost*,
paid once per target change). An episode return mixes the two in a proportion
set by the episode length, which is why the recorded baselines are not the
protocol. This module computes:

``gain``            mean per-step cost over every step after ``burn_in``,
                    pooled across episodes, with its 95% interval over
                    episodes: the long-run average cost of the natural
                    process, transients included, which is what Theorem 4
                    decomposes as ``rho = rho_hold + p * reach_cost``. Split
                    into ``tracking`` and ``running`` (and ``failure``) when
                    the environment reports its reward terms. No settle enters
                    it, so two controllers are compared on the same steps.
``hold``            the same over the settled steps only (after each cycle's
                    transient, see ``reach_cost``): ``rho_hold``. Split as
                    ``hold_tracking`` / ``hold_running``.
``reach_cost``      per target-change cycle, the summed cost above that cycle's
                    hold level; ``rho = rho_hold + p * reach_cost`` at change
                    rate ``p``. The hold level is the mean cost over the second
                    half of the cycle and ``settle`` is measured on the cycle
                    itself -- the first step from which the cost stays within
                    twice that level, at least one -- so each controller is
                    scored against its own transient. (A first version applied
                    the MPC's settle to the PID, which on the battery scored
                    the PID's transient after every dispatch block as hold and
                    manufactured an MPC advantage; and with ``settle = 0`` the
                    reach cost was zero by construction.)
``reach_fraction``  share of steps spent before first entering the band.
``failure_rate``    share of cycles in which the plant trips (``info["tripped"]``).
``nea``             normalised expert advantage ``(PID - x) / (PID - floor)``:
                    1 at the achievable floor, 0 at PID parity, negative below
                    PID. In cost units, so it needs ``rho_floor`` -- the
                    floor-normalised reward makes that 1 per tracked output
                    when nothing avoidable is consumed.
``time_in_band``    a KPI only, never a training signal: share of steps with
                    every tracked error inside ``band``.

``burn_in`` is per plant, from ``data/hold_measurements.json``
(``scripts/measure_hold.py``): three cost-bearing time constants. Passing
``settle`` overrides the per-cycle measurement with a fixed count.

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

import numpy as np

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "hold_measurements.json"


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
    """Mean per-step cost over the settled steps only (after the burn-in and
    after each cycle's transient): ``rho_hold``."""
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
    ok = cost <= 2.0 * level + 1e-12
    # last step that is still outside, plus one
    outside = np.flatnonzero(~ok[: n // 2])
    return int(min(max(outside[-1] + 1 if len(outside) else 1, 1), n // 2))


def reach_cost(episodes, settle: int | None = None):
    """Excess cost of each cycle's transient over its hold level."""
    vals = []
    for ep in episodes:
        for a, b in _cycles(ep):
            k = _settle_of(ep.cost[a:b], settle)
            level = ep.cost[a + k : b].mean() if b - a > k else ep.cost[a:b].mean()
            vals.append((ep.cost[a : a + k] - level).sum())
    return (float(np.mean(vals)), _ci(vals)) if vals else (float("nan"), float("nan"))


def hold_mask(ep: Episode, burn_in: int, settle: int | None = None) -> np.ndarray:
    """Steps that count as hold: after the burn-in and after each cycle's
    transient."""
    m = np.zeros(len(ep.cost), bool)
    for a, b in _cycles(ep):
        k = _settle_of(ep.cost[a:b], settle)
        m[a + k : b] = True
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
    out = dict(
        gain=g,
        gain_ci=gci,
        hold=h,
        hold_ci=hci,
        reach_cost=rc,
        reach_cost_ci=rcci,
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
    settings = hold_settings(name)
    return evaluate(
        episodes,
        burn_in=min(settings["burn_in"], int(p.max_steps_in_episode) // 2),
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
