"""Scoring a policy against the expert and the ceiling
(docs/proposals/scoring-and-experts.md): the three numbers, every
environment's ceiling, and the fixed settle."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import eval as E
from target_gym import records, scoring
from target_gym import reward as R
from target_gym.registry import REGISTRY


def _episode(
    cost,
    running=None,
    target_change=None,
    trips=0,
    actions=None,
    consumption=None,
    in_band=None,
):
    n = len(cost)
    terms = {"tracking": np.asarray(cost, float)}
    if running is not None:
        terms["running"] = np.asarray(running, float)
    tc = np.zeros(n, bool) if target_change is None else target_change
    total = terms["tracking"] + terms.get("running", 0.0)
    if consumption is not None:
        terms["consumption"] = np.asarray(consumption, float)
    failed = np.zeros(n, bool)
    failed[:trips] = True
    return E.Episode(
        cost=total,
        in_band=np.ones(n, bool) if in_band is None else np.asarray(in_band, bool),
        target_change=tc,
        failed=failed,
        terms=terms,
        actions=None if actions is None else np.asarray(actions, float),
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


def test_compare_reads_consumption_against_an_expert_that_consumes_nothing():
    """Six rewards weight consumption at zero, so the expert's running cost
    is 0 on those tasks. Assumed contract, following ``_ratio``'s convention
    for the settling ratio (the docs leave this case open): the ratio is 1
    when the policy consumes nothing either, and infinite when it does."""
    expert = [_episode(np.full(200, 100.0), running=np.zeros(200))] * 3
    same = scoring.compare(expert, expert, ceiling=1.0, burn_in=20, settle=10)
    assert same["consumption_ratio"] == 1.0
    policy = [_episode(np.full(200, 100.0), running=np.full(200, 1.0))] * 3
    m = scoring.compare(policy, expert, ceiling=1.0, burn_in=20, settle=10)
    assert m["consumption_ratio"] == float("inf")


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


# The ceiling of every environment. hvac prices comfort, and glass_furnace lost glass,
# without ``reward.tracking_cost``; each has its own test below. Tasks scored on
# the operator's job (charter B5) track no setpoint.
_PRICED_DIRECTLY = ("hvac", "glass_furnace")
_VIA_TRACKING_COST = sorted(
    n for n in REGISTRY if n not in _PRICED_DIRECTLY and n not in scoring.JOB_SCORED
)


def _control_step_tracking(env, state, params) -> float:
    """The tracking charge of one control step: a task that sums physics
    sub-steps into its reward charges a sub-step that many times, and a
    sub-step's terms alone hide that (the reactor sums 6)."""
    period = int(getattr(env, "control_period", 1))
    return period * float(np.sum(env.reward_terms(state, params)["tracking"]))


def test_a_reactor_step_sums_its_sub_steps_at_the_ceilings_price():
    """Hand-worked, independent of ``scoring``: 0.05 K off program for one 30 s
    step, at 1.25 x $80/MWh x 0.23 %/K of 1000 MW, is 2.3 $/(K h) x 1000 x 0.05
    K x 30 / 3600 h = $0.0958."""
    params = REGISTRY["reactor"].make_test_params()
    assert scoring.ceiling("reactor", params) == pytest.approx(0.09583, rel=1e-3)


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
    tracking = _control_step_tracking(env, state, params)
    assert tracking == pytest.approx(scoring.ceiling(name, params), rel=1e-5)


def test_a_furnace_step_one_resolution_short_loses_its_glass():
    """Hand-worked, independent of ``scoring``: 5.79 kg/s for one 120 s step is
    0.6948 t, at 600 EUR/t 416.88 EUR. Short of the pass mark by the reading's
    scatter, the reward charges all of it, and that is the ceiling and the band."""
    spec = REGISTRY["glass_furnace"]
    env, params = spec.make_env(), spec.make_test_params()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    short = state.plant.fining * 0.0 + params.fining_pass - params.e_floor
    state = state.replace(plant=state.plant.replace(fining=short), pull=params.pull)
    tracking = float(env.reward_terms(state, params)["tracking"])
    assert tracking == pytest.approx(416.88, rel=1e-4)
    assert scoring.ceiling("glass_furnace", params) == pytest.approx(416.88, rel=1e-4)
    assert scoring.settle_band("glass_furnace", params) == pytest.approx(
        416.88, rel=1e-4
    )


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
    tracking = _control_step_tracking(env, state, params)
    assert tracking == pytest.approx(scoring.settle_band(name, params), rel=1e-5)


@pytest.mark.parametrize(
    "name, floor, reading",
    [
        ("cstr", "e_floor", "T"),
        ("unstable_cstr", "e_floor", "T"),
        ("distillation", "e_floor_top", "x"),
        ("distillation", "e_floor_bottom", "x"),
        ("boiler_drum", "e_floor_pressure", "pressure"),
        ("wind_turbine", "e_floor", "power"),
    ],
)
def test_no_floor_is_finer_than_its_instruments_scatter(name, floor, reading):
    """A hold finer than the best reading's scatter is one the plant cannot
    verify (docs/proposals/industrial-fixes.md, Constants)."""
    spec = REGISTRY[name]
    sigmas = [r.sigma for r in spec.make_env().readings if r.field == reading]
    assert float(getattr(spec.make_test_params(), floor)) >= min(sigmas)


def test_a_settle_outlasts_the_move_it_follows():
    """A ramp is one move lasting its whole length, and a settle shorter than
    that scores the ramp as hold. The reactor ramps 50 % at 5 %/min in 20 steps
    of 30 s. The wind expert settles in about 36 steps and its oracle in about
    48 (#157). The compressor's ramp takes 10 s, 100 steps."""
    assert scoring.fixed_settle("reactor") >= 20
    assert scoring.fixed_settle("wind_turbine") >= 48
    assert scoring.fixed_settle("compressor_surge") >= 100


_RESCORED = pytest.mark.xfail(
    strict=True, reason="PR 10: this S moved in PR 8, and the row re-records there"
)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(n, marks=_RESCORED) if n in ("reactor", "wind_turbine") else n
        for n in sorted(REGISTRY)
    ],
)
def test_every_environment_is_scored_with_the_settle_its_rows_were(name):
    """S is a per-task constant, frozen at the slower of the shipped
    controllers' measured settles, so every recorded protocol row was scored
    with it (CONTRIBUTING.md, "Frozen scoring constants")."""
    row = records.load_row("protocol_results", name)
    assert scoring.fixed_settle(name) == row["settle"]


def test_a_re_measured_hold_does_not_move_the_settle(tmp_path, monkeypatch):
    """The hold rows are measurements, reported only: re-measuring a
    controller's settle leaves S where it was."""
    before = {name: scoring.fixed_settle(name) for name in REGISTRY}
    rows = records.load("hold_measurements")
    monkeypatch.setattr(records, "DATA", tmp_path)
    for name, row in rows.items():
        slower = {**row["pid"], "settle": 2.0 * before[name] + 10.0}
        records.write("hold_measurements", name, {**row, "pid": slower})
    assert {name: scoring.fixed_settle(name) for name in REGISTRY} == before


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


def test_score_policy_takes_the_policy_itself_and_resets_it_each_seed():
    """The scored expert passed as one object is the expert again on every
    seed, which holds only if its memory is cleared between episodes: the
    cstr PID's integral is far from zero at an episode's end."""
    m = scoring.score_policy("cstr", REGISTRY["cstr"].make_pid(), seeds=2)
    assert m["ratio"] == pytest.approx(1.0)


def test_score_policy_takes_the_expert_itself_with_fresh_memory_each_seed():
    """One expert object across seeds scores as a new one per seed does."""
    import target_gym as tg

    reused = scoring.score_policy("cstr", tg.make_expert("cstr"), seeds=2)
    with pytest.warns(DeprecationWarning):
        fresh = scoring.score_policy("cstr", lambda: tg.make_expert("cstr"), seeds=2)
    assert reused["ratio"] == fresh["ratio"]
    assert reused["win_rate"] == fresh["win_rate"]


def test_score_policy_still_takes_a_factory_with_a_warning():
    def policy(obs):
        return np.zeros(1)

    with pytest.warns(DeprecationWarning, match="the policy itself"):
        old = scoring.score_policy("first_order", lambda: policy, seeds=1)
    new = scoring.score_policy("first_order", policy, seeds=1)
    assert new["ratio"] == old["ratio"]


# The score per training seed (docs/rl-protocol.md, section 4). Contract, from
# the study's formula:
#
#     s = mean over episodes e of clip(log10(b_e / max(x_e, c)), -3, +3)
#
# with costs per step over the whole episode, shaped (seeds, episodes), and
# above 0 beating the expert. Assumed where the formula is silent, and reported
# to the owner: the floor applies to b_e too, so two costs within the
# resolution tie at 0. As written, an expert at b_e = 0 would score a policy at
# the same cost -3, and no policy could reach 0 on an episode with b_e < c. That
# episode is real: the shipped glass_furnace expert costs 0.84 c on seed 1
# (2026-11-05), so a copy of it would score -0.08. Nothing changes for b_e >= c.
def _s(b, x, c=1.0, **kwargs):
    return scoring.seed_scores(
        expert_cost_per_step=b, policy_cost_per_step=x, ceiling_per_step=c, **kwargs
    )


@pytest.mark.parametrize(
    "b, x, want",
    [
        (100.0, 10.0, 1.0),  # ten times cheaper: one decade
        (100.0, 100.0, 0.0),  # level with the expert
        (10.0, 100.0, -1.0),  # ten times dearer
        (100.0, 1.0, 2.0),  # at the ceiling: log10(100 / 1)
        (100.0, 0.5, 2.0),  # past the ceiling reads as the ceiling
        (100.0, 0.0, 2.0),  # zero cost reads as the ceiling, not 300 decades
        (1e3, 1.0, 3.0),  # exactly at the clip
        (1e4, 1.0, 3.0),  # four decades clip to three
        (1.0, 1e3, -3.0),  # exactly at the clip
        (1.0, 1e4, -3.0),  # one trip's worth of cost clips to three
        (1.0, float("inf"), -3.0),  # a diverged policy is the worst score
    ],
)
def test_one_episode_scores_its_decades_against_the_expert(b, x, want):
    assert _s([[b]], [[x]])["score"] == pytest.approx([want])


@pytest.mark.parametrize(
    "b, x, want",
    [
        (0.0, 0.0, 0.0),  # both perfect: a tie, not a ratio of 0 or a NaN
        (0.0, 0.5, 0.0),  # both within the resolution
        (0.5, 0.1, 0.0),  # cheaper, but no plant could verify it
        (0.0, 10.0, -1.0),  # a decade past the ceiling, whatever b is below it
        (0.5, 10.0, -1.0),
    ],
)
def test_an_expert_within_the_resolution_scores_from_the_ceiling(b, x, want):
    assert _s([[b]], [[x]])["score"] == pytest.approx([want])


def test_the_floor_is_the_tasks_ceiling():
    """With a ceiling of 10, a free policy against an expert at 100 has one
    decade to gain, not two."""
    assert _s([[100.0]], [[0.0]], c=10.0)["score"] == pytest.approx([1.0])


def test_each_seed_gets_the_mean_of_its_own_episodes():
    """Rows are seeds. Seed 0 logs +1, +1, +1; seed 1 logs 0, -1, +1."""
    b = np.full((2, 3), 100.0)
    x = [[10.0, 10.0, 10.0], [100.0, 1000.0, 10.0]]
    assert _s(b, x)["score"] == pytest.approx([1.0, 0.0])


def test_each_episode_is_clipped_before_the_mean():
    """+5 clips to +3 beside a -1, so the mean is 1. Clipping the mean of
    the raw logs would give 2."""
    assert _s([[1e5, 1.0]], [[1.0, 10.0]])["score"] == pytest.approx([1.0])


def test_costs_are_paired_episode_by_episode():
    """Logs 0 and +2 give 1. The same costs on the other episodes log -3 and
    +5, which clips to +3, and give 0: without the clip and the floor a mean
    of logs cannot see the pairing, so that is where it has to be tested."""
    b = [[1.0, 1e5]]
    assert _s(b, [[1.0, 1e3]])["score"] == pytest.approx([1.0])
    assert _s(b, [[1e3, 1.0]])["score"] == pytest.approx([0.0])


def test_one_large_target_episode_does_not_dominate():
    """Ten times cheaper at a small target and twice as dear at a large one
    logs +1 and -0.30103, a mean of +0.34949. The ratio of mean costs,
    1001 / 2000.1, would call that a loss."""
    m = _s([[1.0, 1000.0]], [[0.1, 2000.0]], c=0.01)
    assert m["score"] == pytest.approx([0.34948500])


def test_the_score_is_antisymmetric_and_bounded_on_any_costs():
    """Swapping the two sides negates the score, whatever the costs."""
    rng = np.random.default_rng(0)
    b, x = 10.0 ** rng.uniform(-3, 6, (2, 4, 8))
    forward, backward = _s(b, x)["score"], _s(x, b)["score"]
    assert forward == pytest.approx(-backward)
    assert np.all(np.abs(forward) <= 3.0)


def test_the_score_does_not_depend_on_the_cost_unit():
    """A plant priced in euros scores the same in cents."""
    rng = np.random.default_rng(1)
    b, x = 10.0 ** rng.uniform(-3, 6, (2, 4, 8))
    assert _s(b, x, c=2.0)["score"] == pytest.approx(
        _s(100 * b, 100 * x, c=200.0)["score"]
    )


@pytest.mark.parametrize(
    "b, x, c",
    [
        ([[1.0]], [[-1.0]], 1.0),  # reward terms are non-negative
        ([[-1.0]], [[1.0]], 1.0),
        ([[1.0]], [[float("nan")]], 1.0),  # NaN must not reach a test
        ([[float("nan")]], [[1.0]], 1.0),
        ([[float("inf")]], [[1.0]], 1.0),  # no reference to score against
        ([[1.0]], [[1.0]], 0.0),  # no floor
        ([1.0, 1.0], [1.0, 1.0], 1.0),  # no seed axis
        ([[1.0, 1.0]], [[1.0], [1.0]], 1.0),  # episodes cannot be paired
        (np.ones((2, 3)), np.ones((1, 3)), 1.0),  # would broadcast
        (np.ones((2, 0)), np.ones((2, 0)), 1.0),  # nothing to score
    ],
)
def test_costs_that_cannot_be_scored_are_refused(b, x, c):
    with pytest.raises(ValueError):
        _s(b, x, c=c)


def test_the_arguments_must_be_named():
    """Positional costs could swap the sides or pass an episode total without
    a word; the names carry the sign and the unit."""
    with pytest.raises(TypeError):
        scoring.seed_scores([[1.0]], [[1.0]], 1.0)


def test_the_trip_rate_counts_what_the_clip_hides():
    """One trip per episode and three per episode both clip to -3 on every
    episode. The rate tells the seeds apart, and goes past 1 per episode."""
    x = np.full((2, 2), 1e6)
    m = _s(np.ones((2, 2)), x, trips=[[1, 1], [3, 3]])
    assert m["score"] == pytest.approx([-3.0, -3.0])
    assert m["trip_rate"] == pytest.approx([1.0, 3.0])


def test_the_trip_rate_reads_flags_as_the_share_of_episodes():
    m = _s(np.ones((1, 4)), np.ones((1, 4)), trips=[[True, False, False, True]])
    assert m["trip_rate"] == pytest.approx([0.5])


def test_there_is_no_trip_rate_without_trip_information():
    assert "trip_rate" not in _s([[1.0]], [[1.0]])


@pytest.mark.parametrize("trips", [[[1]], [[1, 1, 1]], [[-1, 1]], [[1, float("nan")]]])
def test_trips_that_do_not_match_the_episodes_are_refused(trips):
    with pytest.raises(ValueError):
        _s(np.ones((1, 2)), np.ones((1, 2)), trips=trips)


def test_the_verdict_and_the_seed_score_both_see_an_early_trip():
    """Both read the whole episode. Both controllers cost 10 per step, but the
    policy pays 1e5 on step 5, inside a burn-in of 20 of 100 steps. Over the
    whole episode it costs (99 * 10 + 1e5) / 100 = 1009.9 per step, so the
    ratio is 10 / 1009.9, a log of log10(10 / 1009.9) = -2.00428."""
    spike = np.full(100, 10.0)
    spike[5] = 1e5
    expert = [_episode(np.full(100, 10.0))] * 2
    policy = [_episode(spike)] * 2
    m = scoring.compare(policy, expert, 0.1, 20, settle=10)
    assert m["ratio"] == pytest.approx(10.0 / 1009.9)
    b = [[E.gain([ep], 0)[0] for ep in expert]]
    x = [[E.gain([ep], 0)[0] for ep in policy]]
    assert _s(b, x, c=0.1)["score"] == pytest.approx([-2.00428], abs=1e-5)


# The contract of the rest of this file is docs/proposals/industrial-fixes.md,
# PR 1: what an owner needs beside the ratio. A win has to survive a trip,
# wear is reported, a saving shows, and a score says how far it can go.


def test_compare_counts_each_sides_trips_per_episode():
    expert = [_episode(np.ones(50), trips=n) for n in (0, 1, 0)]
    policy = [_episode(np.ones(50), trips=n) for n in (2, 0, 1)]
    m = scoring.compare(policy, expert, ceiling=0.1, burn_in=10, settle=5)
    assert m["expert_trips"] == [0, 1, 0]
    assert m["policy_trips"] == [2, 0, 1]


@pytest.mark.parametrize(
    "cost, trips, expert_trips, wins",
    [
        (0.5, 0, 0, True),  # cheaper, and no trips
        (0.5, 1, 0, False),  # cheaper, but it trips where the expert does not
        (0.5, 1, 1, True),  # cheaper, and no worse on trips
        (1.0, 0, 0, False),  # a tie is not a win
        (0.995, 0, 0, False),  # nor is a gap two recordings of one row can show
        (0.98, 0, 0, True),  # past that 1%
        (2.0, 0, 0, False),  # dearer
    ],
)
def test_a_win_costs_less_and_trips_no_more_than_the_expert(
    cost, trips, expert_trips, wins
):
    expert = [_episode(np.ones(50), trips=expert_trips)] * 4
    policy = [_episode(np.full(50, cost), trips=trips)] * 4
    m = scoring.compare(policy, expert, ceiling=0.1, burn_in=10, settle=5)
    assert m["beats_expert"] is wins


def test_a_win_survives_no_trip_the_interquartile_mean_trims():
    """The mean trims 2 of 10 seeds from each end. A policy twice as cheap
    that pays 1e9 on 2 seeds keeps a ratio of exactly 2, so only the gate can
    refuse it."""
    trip = np.full(100, 0.5)
    trip[50] = 1e9
    expert = [_episode(np.ones(100))] * 10
    policy = [_episode(np.full(100, 0.5))] * 8
    policy += [_episode(trip, trips=1)] * 2
    m = scoring.compare(policy, expert, ceiling=0.1, burn_in=10, settle=5)
    assert m["ratio"] == pytest.approx(2.0)
    assert m["beats_expert"] is False


@pytest.mark.parametrize(
    "actions, strokes",
    [
        ([[0.5], [0.5], [0.5]], 0.0),
        ([[0.0], [0.5], [0.25]], 0.75),  # 0.5 up and 0.25 down
        ([[0.0, 1.0], [0.5, 1.0], [0.5, 0.0]], 1.5),  # 0.5 and 1 on two actuators
    ],
)
def test_travel_is_the_distance_moved_in_full_strokes(actions, strokes):
    assert E.travel(np.array(actions)) == pytest.approx(strokes)


@pytest.mark.parametrize(
    "position, count",
    [
        ([0.5, 0.5, 0.5, 0.5], 0),
        ([0.0, 0.1, 0.2, 0.3], 0),  # one direction
        ([0.0, 0.5, 0.2, 0.6], 2),  # up, down, up
        ([0.0, 0.5, 0.5, 0.2], 1),  # a pause is not a reversal
        ([0.5, 0.501, 0.499, 0.501], 0),  # inside the 0.25% dead band
    ],
)
def test_a_reversal_is_a_change_of_direction_past_the_dead_band(position, count):
    assert E.reversals(np.array(position)[:, None]) == count


def test_compare_reports_each_sides_travel_and_reversals_per_episode():
    still = np.full((100, 1), 0.5)
    chatter = np.tile([[0.0], [1.0]], (50, 1))
    expert = [_episode(np.ones(100), actions=still)] * 2
    policy = [_episode(np.ones(100), actions=chatter)] * 2
    m = scoring.compare(policy, expert, ceiling=0.1, burn_in=10, settle=5)
    assert m["expert_travel"] == 0.0
    assert m["policy_travel"] == pytest.approx(99.0)  # one stroke per step
    assert m["expert_reversals"] == 0.0
    assert m["policy_reversals"] == 98.0  # every step but the first two


class _BangBang:
    """Full throw each way, three times past the box."""

    def __init__(self):
        self.sign = 1.0

    def __call__(self, obs):
        self.sign = -self.sign
        return np.array([3.0 * self.sign])


def test_an_episode_records_actions_as_fractions_of_the_span_after_the_box():
    """Past the box a command does nothing more, so each step of 150 on
    ``first_order`` moves one full stroke: 149 strokes and 148 reversals."""
    spec = REGISTRY["first_order"]
    ep = E.run_episode(spec, spec.make_test_params(), _BangBang(), seed=0)
    assert ep.actions.min() == 0.0 and ep.actions.max() == 1.0
    assert E.travel(ep.actions) == pytest.approx(149.0)
    assert E.reversals(ep.actions) == 148


def test_the_consumption_ratio_reads_the_quantity_not_the_reward_term():
    """The ratio reads the plant's quantity whatever the running term did with
    it: the policy burns 9 per step against the expert's 10."""
    expert = [
        _episode(np.ones(200), running=np.zeros(200), consumption=np.full(200, 10.0))
    ] * 3
    policy = [
        _episode(np.ones(200), running=np.zeros(200), consumption=np.full(200, 9.0))
    ] * 3
    m = scoring.compare(policy, expert, ceiling=0.1, burn_in=20, settle=10)
    assert m["consumption_ratio"] == pytest.approx(0.9)


@pytest.mark.parametrize("name", sorted(scoring.CONSUMPTION))
def test_the_consumption_is_the_quantity_the_reward_charges(name):
    """The running term is ``w * c / c_hold``, so with ``c_hold`` at twice the
    quantity it is ``w / 2``. A table entry that read another quantity would
    give another number, and a clip at ``c_hold`` would charge nothing."""
    spec = REGISTRY[name]
    env, p = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key, p)
    for _ in range(3):
        action = jnp.zeros(env.action_space(p).shape or (1,))
        _, state, *_ = env.step(key, state, action, p)
    c = float(scoring.CONSUMPTION[name](state, p, False))
    assert c > 0.0
    terms = env.reward_terms(state, p.replace(c_hold=2.0 * c))
    assert float(terms["running"]) == pytest.approx(
        0.5 * float(p.running_weight), rel=1e-5
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_task_states_a_band_per_tracked_output_or_none(name):
    spec = REGISTRY[name]
    band = scoring.spec_band(name)
    if name in _PRICED_DIRECTLY:  # one-sided, and the reward prices it
        assert band is None
        return
    assert band.shape == (len(spec.make_env().tracked_names),)
    assert np.isfinite(band).all() and (band > 0).all()


@pytest.mark.parametrize(
    "name, band",
    [
        ("plane", [65.0 * 0.3048]),  # +-65 ft, 14 CFR 91 Appendix G
        ("plane3d_racetrack", [65.0 * 0.3048]),
        # AFF's 10 ft one sigma on the Hornet [NASA/TM-2002-210729], the
        # owner's choice of airframe (2026-11-10), patrol/PHYSICS.md
        ("patrol", [10.0 * 0.3048]),
        ("formation", [20.0, 10.0 * 0.3048, 10.0 * 0.3048]),  # lead, then wingmen
        ("ph_neutralization", [1.0]),  # the permit range pH 7 +-1 [P7]
    ],
)
def test_a_published_tolerance_is_the_band(name, band):
    np.testing.assert_allclose(scoring.spec_band(name), band)
    assert scoring.SPEC_BAND[name] != "ours"


@pytest.mark.parametrize(
    "name, band",
    [
        ("distillation", [4e-3, 4e-3]),  # floors of 4e-4 mole fraction
        ("cement_kiln", [0.5]),  # a 5e-4 fraction is 0.05 pp of the observed %
    ],
)
def test_a_task_with_no_published_tolerance_gets_ten_floors(name, band):
    np.testing.assert_allclose(scoring.spec_band(name), band)
    assert name not in scoring.SPEC_BAND


def test_a_tolerance_says_how_far_it_was_sourced():
    """A secondary source says so, and a figure no second pass could re-open is
    labelled ours: the rods' +-1.5 F, 1.5 / 1.8 K."""
    assert scoring.SPEC_BAND["reactor"][1].startswith("ours")
    np.testing.assert_allclose(scoring.spec_band("reactor"), [1.5 / 1.8])


def test_off_spec_reads_the_steps_outside_the_band_during_the_hold():
    """Steps 0 and 1 are the burn-in, so their misses do not count. Of the
    other 8 steps, 1 is out on the first episode and 3 on the second."""
    first = _episode(np.ones(10), in_band=[0, 0, 1, 1, 0, 1, 1, 1, 1, 1])
    second = _episode(np.ones(10), in_band=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    assert E.off_spec_fraction([first, second], 2, 1) == pytest.approx((1 + 3) / 16)


@pytest.mark.parametrize(
    "b, c, bound, want",
    [
        (100.0, 1.0, 10.0, 0.5),  # one of the two decades is reachable
        (100.0, 1.0, 0.5, 1.0),  # a bound under the ceiling leaves it reachable
        (100.0, 1.0, 0.0, 1.0),  # no bound known
        (100.0, 1.0, float("nan"), 1.0),
        (100.0, 1.0, 200.0, 0.0),  # the expert is already past the best known
    ],
)
def test_s_max_is_the_score_the_best_known_bound_allows(b, c, bound, want):
    assert scoring.s_max(b, c, bound) == pytest.approx(want)


def test_there_is_no_s_max_without_room():
    assert math.isnan(scoring.s_max(0.5, 1.0, 0.1))


def test_score_policy_reports_the_owners_numbers_for_both_sides():
    """The expert against itself: a tie, so no win, and both sides alike. The
    band is ten floors, and its PID ends inside it, so the hold is on spec."""
    m = scoring.score_policy("first_order", tg.make_expert("first_order"), seeds=2)
    assert m["beats_expert"] is False
    assert m["policy_trips"] == m["expert_trips"] == [0, 0]
    assert m["policy_travel"] == pytest.approx(m["expert_travel"])
    assert m["policy_reversals"] == m["expert_reversals"]
    assert m["policy_off_spec"] == m["expert_off_spec"] == 0.0
    assert m["s_max"] != m["s_max"]  # no room, so no s_max


def test_s_max_reads_the_tracking_bound_not_the_total(monkeypatch):
    """The hold it is read against is tracking alone, and a plant's ``rho_floor``
    also holds consumption. Expert 100, ceiling 1, tracking bound 10: 0.5."""
    row = {"rho_floor_tracking": 10.0, "rho_floor": 1e3}
    monkeypatch.setattr(scoring, "recorded", lambda name: row)
    monkeypatch.setattr(
        scoring,
        "compare",
        lambda *a, **k: dict(expert_hold=100.0, ceiling=1.0, policy_hold=100.0),
    )
    m = scoring.score_policy("first_order", tg.make_expert("first_order"), seeds=1)
    assert m["s_max"] == pytest.approx(0.5)


def _protocol_script():
    path = pathlib.Path(__file__).parent.parent / "scripts" / "evaluate_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_baselines", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_protocol_records_the_expert_on_ten_seeds_and_the_oracle_on_three(
    monkeypatch,
):
    """Three seeds miss the pH expert's cycling on seeds 3 to 5, and an oracle
    costs too much for ten."""
    script = _protocol_script()
    asked, written = {}, {}

    def fake(name, kind, seeds):
        asked[kind] = seeds
        return {"gain": 1.0, "hold": 1.0, "reach_cost": 1.0, "failure_rate": 0.0}

    monkeypatch.setattr(script, "evaluate_controller", fake)
    monkeypatch.setattr(script.records, "load", lambda kind: {})
    monkeypatch.setattr(
        script.records, "write", lambda kind, name, row: written.update(row)
    )
    monkeypatch.setattr(script, "table", lambda rows: "")
    monkeypatch.setattr(sys, "argv", ["evaluate_baselines", "--envs", "first_order"])
    assert script.main() == 0
    assert asked == {"pid": 10, "mpc": 3}
    assert written["pid_seeds"] == 10
    assert written["seeds"] == 3


# --- Tasks scored on their job, zero costs and unknown ceilings ---------------
# Contract (scoring.JOB_SCORED's note and the module docstring): a job-scored task
# has no tracked setpoint, so no tracking term, ceiling or settle band (NaN).
# ``compare`` still gives the paired cost ratio; what is read against a tracking
# hold is unknown there (NaN), not 0 and not an error.


def _job_episode(cost):
    cost = np.asarray(cost, float)
    n = len(cost)
    return E.Episode(
        cost=cost,
        in_band=np.ones(n, bool),
        target_change=np.zeros(n, bool),
        failed=np.zeros(n, bool),
        terms={"energy": cost},
    )


def test_compare_gives_a_job_scored_task_its_ratio_and_no_tracking_numbers():
    # Expert 4 per step, policy 2 per step on both seeds: b / x = 2.
    nan = float("nan")
    expert = [_job_episode(np.full(50, 4.0)) for _ in range(2)]
    policy = [_job_episode(np.full(50, 2.0)) for _ in range(2)]
    m = scoring.compare(policy, expert, nan, burn_in=10, settle=5, band=nan)
    assert m["ratio"] == pytest.approx(2.0)
    assert m["win_rate"] == 1.0 and m["beats_expert"]
    for key in ("score", "headroom", "room_left", "expert_hold", "policy_hold"):
        assert np.isnan(m[key]), key
    for key in ("expert_settling", "policy_settling", "settling_ratio"):
        assert np.isnan(m[key]), key


def test_headroom_to_an_unknown_ceiling_is_unknown():
    assert np.isnan(scoring.headroom(3.0, float("nan")))
    assert np.isnan(scoring.headroom(float("nan"), 1.0))


def test_two_sides_that_cost_nothing_are_level():
    """Neither side ever left its band: the ratio is 1 (``_ratio``'s convention
    for 0 over 0), nobody wins a seed, and the policy does not beat the expert."""
    expert = [_episode(np.zeros(40)) for _ in range(4)]
    policy = [_episode(np.zeros(40)) for _ in range(4)]
    m = scoring.compare(policy, expert, 1.0, burn_in=5, settle=5)
    assert m["ratio"] == pytest.approx(1.0)
    assert m["ratio_ci"] == pytest.approx((1.0, 1.0))
    assert m["transient_ratio"] == pytest.approx(1.0)
    assert m["win_rate"] == 0.0 and not m["beats_expert"]


@pytest.mark.medium
def test_score_policy_scores_a_job_scored_task():
    """hvac has no tracked setpoint (``JOB_SCORED``). Its expert against itself
    on the same seed costs the same, so the ratio is 1 and it does not beat
    itself; the tracking score is unknown."""
    import target_gym as tg

    assert "hvac" in scoring.JOB_SCORED
    m = scoring.score_policy("hvac", tg.make_expert("hvac"), seeds=1)
    assert m["ratio"] == pytest.approx(1.0)
    assert not m["beats_expert"]
    assert np.isnan(m["score"]) and np.isnan(m["s_max"])
