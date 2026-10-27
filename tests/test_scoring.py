"""Scoring a policy against the expert and the ceiling
(docs/proposals/scoring-and-experts.md): the three numbers, every
environment's ceiling, and the fixed settle."""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym import reward as R
from target_gym import scoring
from target_gym.registry import REGISTRY


def _episode(cost, running=None, target_change=None):
    n = len(cost)
    terms = {"tracking": np.asarray(cost, float)}
    if running is not None:
        terms["running"] = np.asarray(running, float)
    tc = np.zeros(n, bool) if target_change is None else target_change
    total = terms["tracking"] + terms.get("running", 0.0)
    return E.Episode(
        cost=total,
        in_band=np.ones(n, bool),
        target_change=tc,
        failed=np.zeros(n, bool),
        terms=terms,
    )


def test_the_score_is_0_at_the_expert_1_at_the_ceiling_and_capped():
    """Two decades from an expert at 100 to a ceiling at 1."""
    assert scoring.score(100.0, 100.0, 1.0) == pytest.approx(0.0)
    assert scoring.score(10.0, 100.0, 1.0) == pytest.approx(0.5)
    assert scoring.score(1.0, 100.0, 1.0) == pytest.approx(1.0)
    assert scoring.score(1e3, 100.0, 1.0) == pytest.approx(-0.5)
    assert scoring.score(0.0, 100.0, 1.0) == 1.0
    assert scoring.headroom(100.0, 1.0) == pytest.approx(2.0)


def test_an_expert_within_the_resolution_leaves_no_room():
    assert scoring.headroom(0.5, 1.0) == 0.0
    assert math.isnan(scoring.score(0.1, 0.5, 1.0))


def test_the_score_does_not_depend_on_the_tracking_exponent():
    """Errors of 30, 3 and 0.3 for the expert, the policy and the ceiling score
    the same whether tracking is linear or squared in the error."""
    expert, policy, ceiling = 30.0, 3.0, 0.3
    assert scoring.score(policy, expert, ceiling) == pytest.approx(
        scoring.score(policy**2, expert**2, ceiling**2)
    )


def test_compare_reads_the_verdict_the_score_and_the_room_from_paired_seeds():
    """An expert holding at 100 per step and a policy at 10, with a ceiling
    of 1: ten times cheaper on every seed, half the two decades of room
    used, and the consumption ratio beside it."""
    expert = [_episode(np.full(200, 100.0), running=np.full(200, 2.0))] * 3
    policy = [_episode(np.full(200, 10.0), running=np.full(200, 4.0))] * 3
    m = scoring.compare(policy, expert, ceiling=1.0, burn_in=20, settle=10)
    assert m["ratio"] == pytest.approx(102.0 / 14.0)
    assert m["win_rate"] == 1.0
    assert m["score"] == pytest.approx(0.5)
    assert m["headroom"] == pytest.approx(2.0)
    assert m["room_left"] == pytest.approx(1.0)
    assert m["consumption_ratio"] == pytest.approx(2.0)


def test_compare_scores_the_hold_after_the_fixed_settle():
    """A policy that pays a large transient after each target change but holds
    at the ceiling scores 1 on the hold, and the transient ratio shows the
    transients it pays for."""
    n, tc = 400, np.zeros(400, bool)
    tc[[0, 100, 200, 300]] = True
    slow = np.full(n, 1.0)
    for i in np.flatnonzero(tc):
        slow[i : i + 20] = 1e4
    expert = [_episode(np.full(n, 100.0), target_change=tc)] * 2
    policy = [_episode(slow, target_change=tc)] * 2
    m = scoring.compare(policy, expert, ceiling=1.0, burn_in=0, settle=25)
    assert m["score"] == pytest.approx(1.0)
    assert m["transient_ratio"] < 1.0


# The ceiling of every environment. hvac prices comfort without
# ``reward.tracking_cost`` and has its own test below.
_VIA_TRACKING_COST = sorted(n for n in REGISTRY if n != "hvac")


@pytest.mark.parametrize("name", _VIA_TRACKING_COST)
def test_the_ceiling_is_the_rewards_tracking_cost_at_one_resolution(name, monkeypatch):
    """Every ``tracking_cost`` the reward calls, at an error one resolution
    beyond its tolerance, sums to the ceiling: a missed term, a wrong count
    or a wrong price fails here."""
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    resolution = {
        round(float(getattr(params, floor)), 12): float(getattr(params, res))
        for res, floor, _ in scoring.TRACKED_TERMS[name]
    }
    real = R.tracking_cost

    def at_one_resolution(error, e_floor, e_tol=0.0, p=2.0, xp=jnp):
        res = resolution[round(float(e_floor), 12)]
        return real(xp.zeros_like(error) + e_tol + res, e_floor, e_tol, p, xp)

    monkeypatch.setattr(R, "tracking_cost", at_one_resolution)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    tracking = float(np.sum(env.reward_terms(state, params)["tracking"]))
    assert tracking == pytest.approx(scoring.ceiling(name, params), rel=1e-5)


def test_the_building_ceiling_is_the_comfort_price_of_one_resolution():
    from target_gym.hvac.env import is_occupied

    spec = REGISTRY["hvac"]
    env, params = spec.make_env(), spec.make_test_params()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    occupied = next(
        t for t in np.arange(0.0, 7 * 86400.0, 900.0) if bool(is_occupied(t, params))
    )
    state = state.replace(
        time=occupied,
        T_air=state.target_T + params.comfort_tolerance + params.precision_floor,
    )
    tracking = float(env.reward_terms(state, params)["tracking"])
    assert tracking == pytest.approx(scoring.ceiling("hvac", params), rel=1e-5)


def test_every_environment_has_a_settle_no_shorter_than_its_measured_ones():
    rows = json.loads(E._DATA.read_text())
    for name in REGISTRY:
        measured = [rows[name][k]["settle"] for k in ("pid", "mpc") if k in rows[name]]
        assert scoring.fixed_settle(name) >= max(measured, default=0), name


def test_score_policy_runs_the_expert_on_the_same_seeds():
    """The expert scored against itself: parity on every seed. Its first-order
    PID already tracks within the resolution, so the task has no room."""
    from target_gym.runners.runners import baseline_policy

    spec = REGISTRY["first_order"]
    params = spec.make_test_params()
    m = scoring.score_policy(
        "first_order", lambda: baseline_policy(spec, "pid", params), seeds=2
    )
    assert m["ratio"] == pytest.approx(1.0)
    assert m["headroom"] == 0.0
    assert math.isnan(m["score"])
