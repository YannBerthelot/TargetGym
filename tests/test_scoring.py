"""Scoring a policy against the expert and the ceiling
(docs/proposals/scoring-and-experts.md): the three numbers, every
environment's ceiling, and the fixed settle."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym import records, scoring
from target_gym import reward as R
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


def _move(per_step, n=200, settle=10):
    """No target change: ``per_step`` over the settle after the reset, then a
    hold at 1."""
    return _episode(np.r_[np.full(settle, per_step), np.ones(n - settle)])


def test_the_transient_ratio_counts_the_approach_from_reset():
    """first_order and cstr draw their target at reset and never change it, so
    the reset is their only move: a policy reaching it ten times cheaper
    scores 10, not NaN."""
    m = scoring.compare([_move(1.0)] * 2, [_move(10.0)] * 2, 0.1, 20, settle=10)
    assert m["transient_ratio"] == pytest.approx(10.0)


def test_the_transient_ratio_is_paired_per_seed_with_a_win_rate():
    """The expert's move costs 100 on three seeds; the policy's cost 10, 50
    and 1000. The per-seed ratios 10, 2 and 0.1 multiply to 2, so their
    geometric mean (the interquartile mean of three logs) is 2 ** (1/3), and
    the policy wins two seeds of three."""
    expert = [_move(10.0)] * 3
    policy = [_move(1.0), _move(5.0), _move(100.0)]
    m = scoring.compare(policy, expert, 0.1, 20, settle=10)
    assert m["transient_ratio"] == pytest.approx(2 ** (1 / 3))
    assert m["transient_win_rate"] == pytest.approx(2 / 3)
    lo, hi = m["transient_ratio_ci"]
    assert 0.1 <= lo <= 2 ** (1 / 3) <= hi <= 10.0


def test_the_settling_ratio_compares_the_steps_to_enter_the_band_and_stay():
    """With a band of 1, the expert's tracking cost stays out of it for 8
    steps after the reset and the policy's for 2: 4 times faster."""
    expert = [_episode(np.r_[np.full(8, 10.0), np.full(92, 0.1)])] * 2
    policy = [_episode(np.r_[np.full(2, 10.0), np.full(98, 0.1)])] * 2
    m = scoring.compare(policy, expert, 0.01, 10, settle=10, band=1.0)
    assert (m["expert_settling"], m["policy_settling"]) == (8.0, 2.0)
    assert m["settling_ratio"] == pytest.approx(4.0)


def test_the_best_known_move_is_the_oracles_recorded_ratio():
    row = {
        "pid": {"move_cost": 300.0, "settling_time": 12.0},
        "mpc": {"move_cost": 100.0, "settling_time": 3.0},
    }
    best = scoring.best_known(row)
    assert best["oracle_transient_ratio"] == pytest.approx(3.0)
    assert best["oracle_settling_ratio"] == pytest.approx(4.0)


def test_without_recorded_moves_the_best_known_is_unknown():
    best = scoring.best_known({"pid": {"hold": 1.0}, "mpc": {"hold": 0.5}})
    assert math.isnan(best["oracle_transient_ratio"])
    assert math.isnan(best["oracle_settling_ratio"])


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


@pytest.mark.parametrize("name", _VIA_TRACKING_COST)
def test_the_settle_band_is_the_rewards_tracking_cost_at_the_floor(name, monkeypatch):
    """Every ``tracking_cost`` the reward calls, at an error at its floor
    beyond its tolerance, sums to the band: 1 per tracked term, in the plant's
    currency where the reward prices tracking."""
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    real = R.tracking_cost

    def at_the_floor(error, e_floor, e_tol=0.0, p=2.0, xp=jnp):
        return real(xp.zeros_like(error) + e_tol + e_floor, e_floor, e_tol, p, xp)

    monkeypatch.setattr(R, "tracking_cost", at_the_floor)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    tracking = float(np.sum(env.reward_terms(state, params)["tracking"]))
    assert tracking == pytest.approx(scoring.settle_band(name, params), rel=1e-5)


def test_the_building_settles_within_its_ceiling():
    """hvac has no floor: its band is the comfort tolerance plus one
    resolution, which is its ceiling."""
    params = REGISTRY["hvac"].make_test_params()
    assert scoring.settle_band("hvac", params) == scoring.ceiling("hvac", params)


def test_every_environment_has_a_settle_no_shorter_than_its_measured_ones():
    rows = records.load("hold_measurements")
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
    assert m["transient_ratio"] == pytest.approx(1.0)
    assert m["settling_ratio"] == pytest.approx(1.0)
