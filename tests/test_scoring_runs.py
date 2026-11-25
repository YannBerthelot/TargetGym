"""Scoring runs: a task scored on its stream pinned to each scenario
(long-strata R2 as the owner cut it for the reactor on 2026-11-24).

Contract, from the owner's answers of 2026-11-24 (relayed by the package review),
written before the code:

- the reactor is scored on four scenarios, baseload, edf_daily, fast_return and
  krsko_slow; edf_short_hold's weight goes to edf_daily and krsko_shallow's to
  krsko_slow, the source's standard day of each pair, fixed by name;
- a run is the stream pinned to one scenario, from the reset as it is, scored over
  7 days of 30 s steps from its first step: no burn-in;
- a seed's cost is the mean cost per step over the scored part, as J4 reads it;
- 10 seeds a scenario; 10 more where the interval of the room on that scenario
  contains 1, once (the stop rule, set before any result);
- the day planner plans to the run's end, not to the one-day window's.

By hand, the weights: the suite gives each load-follow day 250 h a year over a
6.333 h day off full load, over 365 days, over 5 scenarios: 0.0216294 each, and
baseload the rest, 0.8918529. Folded: 0.0432588 for edf_daily and krsko_slow,
0.0216294 for fast_return.
"""

import pytest

from target_gym import scoring_runs as S

EACH = 250.0 / (5.5 + 50.0 / 60.0) / 365.0 / 5.0


def test_the_reactor_is_scored_on_four_scenarios_with_its_twins_weight_folded():
    w = S.weights("reactor")
    assert set(w) == {"baseload", "edf_daily", "fast_return", "krsko_slow"}
    assert w["edf_daily"] == pytest.approx(2 * EACH, rel=1e-6)
    assert w["krsko_slow"] == pytest.approx(2 * EACH, rel=1e-6)
    assert w["fast_return"] == pytest.approx(EACH, rel=1e-6)
    assert w["baseload"] == pytest.approx(1.0 - 5 * EACH, rel=1e-6)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)


def test_a_reactor_run_is_seven_days_of_thirty_second_steps_on_ten_seeds():
    p = S.PROTOCOLS["reactor"]
    assert S.scored_steps("reactor") == 7 * 2880
    assert p.seeds == 10 and p.more == 10


def test_the_first_seeds_are_ten_and_an_extension_adds_ten_new_ones():
    first = S.seeds("reactor")
    more = S.seeds("reactor", extended=True)
    assert len(first) == 10 and len(set(first)) == 10
    assert len(more) == 20 and more[:10] == first


@pytest.mark.parametrize(
    "saving, extend",
    [
        # Reference 10 a step against 8: room 1.25 on every seed, no doubt.
        ([2.0] * 10, False),
        # Saving 2 on half the seeds and -2 on the rest: room near 1, the
        # interval holds 1.
        ([2.0, -2.0] * 5, True),
        # Every seed lost: room under 1, the interval is below 1, not across it.
        ([-2.0] * 10, False),
    ],
)
def test_a_scenario_gets_more_seeds_only_where_its_room_interval_holds_one(
    saving, extend
):
    reference = [10.0] * 10
    other = [10.0 - s for s in saving]
    assert S.needs_more(reference, other) is extend


def test_a_run_starts_scoring_at_its_first_step():
    import jax

    stream, _ = S.stream("reactor", "baseload")
    _, state = stream.reset(jax.random.PRNGKey(3), stream.default_params)
    assert int(state.burn_in) == 0
    assert stream.is_truncated(
        state.replace(time=state.time + S.scored_steps("reactor")), None
    )
    assert not stream.is_truncated(
        state.replace(time=state.time + S.scored_steps("reactor") - 1), None
    )


def test_the_planner_plans_to_the_runs_end():
    p = S.run_params("reactor", "edf_daily")
    assert int(p.max_steps_in_episode) == 7 * 2880


@pytest.mark.medium
def test_a_seeds_cost_is_its_mean_cost_per_step_over_the_scored_part():
    """Two short runs of the reference, read by hand from the stream's own rewards:
    a run of k steps costs the mean of its k step costs, and the same seed gives
    the same cost (the stream's world comes from the seed)."""
    import jax
    import numpy as np

    from target_gym.registry import REGISTRY
    from target_gym.runners.runners import baseline_policy

    steps = 40
    out = S.run("reactor", "baseload", "pid", seeds=[5, 6], steps=steps)
    stream, params = S.stream("reactor", "baseload", steps=steps)
    from target_gym.benchmark import episode_keys

    key = episode_keys(jax.random.PRNGKey(5), 1, 1)[0][0]
    obs, state = stream.reset(key, params)
    pid = baseline_policy(REGISTRY["reactor"], "pid", params)
    costs = []
    for _ in range(steps):
        obs, state, r, _, _ = stream.step_env(None, state, pid(np.asarray(obs)), params)
        costs.append(-float(r))
    assert out["seeds"] == [5, 6]
    assert out["costs"][0] == pytest.approx(np.mean(costs), rel=1e-5)
    assert out["trips"] == [0, 0]
    again = S.run("reactor", "baseload", "pid", seeds=[5], steps=steps)
    assert again["costs"][0] == out["costs"][0]


@pytest.mark.medium
def test_hold_reset_holds_the_resets_move_through_a_run():
    """The do-nothing: rods parked, no boron flow, the turbine on its plan, on a
    run of the stream: no rod step and no kilogram of effluent."""
    out = S.run("reactor", "baseload", "null", seeds=[5], steps=40, terms=True)
    assert out["running"][0] == pytest.approx(0.0, abs=1e-4)


def test_a_runs_core_never_reaches_the_end_of_its_range():
    """The owner's cut (2026-11-25): a 7-day run's core starts at most what 7 days
    burn short of the range's end, at the core's own rate; at the range's fastest
    rate that is 0.9 - 7 x 3.3 / 1690 = 0.886331 of the cycle."""
    import jax
    import numpy as np

    stream, params = S.stream("reactor", "baseload")
    keys = jax.random.split(jax.random.PRNGKey(5), 400)
    _, state = jax.vmap(stream.reset, in_axes=(0, None))(keys, params)
    env = state.env
    run = np.asarray(env.burn_rate) * 7 * 86400.0 / 1690.0
    assert np.all(np.asarray(env.cycle0) + run <= 0.9 + 1e-6)
