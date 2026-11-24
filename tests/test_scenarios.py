"""Scenario suites, the operating windows a task is evaluated on.

Contract, from docs/proposals/task-charter.md, Amendment 3, rule 20:

- A task's suite is a tuple of named scenarios. Each is a set of parameter overrides on
  the scored configuration, with the source it comes from.
- A task that declares no suite has one scenario, ``normal``, with no overrides, so
  every task can be evaluated the same way.
- A suite is not empty and its names are unique.
- An override names a parameter of the task: a typo is an error, not a silent no-op.
- ``params_for`` applies the scenario's overrides, then the caller's.
- ``run`` gives each scenario its own episodes, on the same fixed seeds ``0 .. n-1``,
  so a rerun reproduces them.
"""

import dataclasses

import numpy as np
import pytest

from target_gym import scenarios
from target_gym.registry import REGISTRY
from target_gym.scenarios import Scenario


def declaring(*suite):
    return dataclasses.replace(REGISTRY["first_order"], scenarios=lambda: suite)


#: A task that declares no suite: a registry entry with its suite taken away, so
#: the default holds whatever the registry's tasks declare.
NO_SUITE = dataclasses.replace(REGISTRY["first_order"], scenarios=None)


def test_a_task_with_no_declared_suite_has_one_normal_scenario():
    (only,) = scenarios.suite(NO_SUITE)
    assert only.name == "normal"
    assert dict(only.overrides) == {}


def test_a_declared_suite_is_read_from_the_spec():
    spec = declaring(
        Scenario("slow plant", {"tau": 20.0}, source="test"),
        Scenario("strong plant", {"K": 2.0}, source="test"),
    )
    assert [s.name for s in scenarios.suite(spec)] == ["slow plant", "strong plant"]


def test_an_empty_suite_is_an_error():
    with pytest.raises(ValueError):
        scenarios.suite(declaring())


def test_two_scenarios_of_one_name_are_an_error():
    spec = declaring(Scenario("upset", {"K": 2.0}), Scenario("upset", {"K": 3.0}))
    with pytest.raises(ValueError):
        scenarios.suite(spec)


def test_an_override_naming_no_parameter_is_an_error():
    with pytest.raises(ValueError):
        scenarios.suite(declaring(Scenario("typo", {"tua": 20.0})))


def test_weights_are_shares_of_operating_time():
    spec = declaring(
        Scenario("normal day", {}, weight=3.0),
        Scenario("upset", {"K": 2.0}, weight=1.0),
    )
    assert scenarios.weights(spec) == pytest.approx({"normal day": 0.75, "upset": 0.25})


def test_a_task_with_no_declared_suite_runs_its_normal_scenario_all_the_time():
    assert scenarios.weights(NO_SUITE) == {"normal": 1.0}


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan")])
def test_a_scenario_weight_must_be_positive(weight):
    with pytest.raises(ValueError):
        scenarios.suite(declaring(Scenario("upset", {"K": 2.0}, weight=weight)))


def test_params_for_applies_the_scenario_then_the_caller():
    scenario = Scenario("slow plant", {"tau": 20.0, "K": 2.0})
    p = scenarios.params_for(REGISTRY["first_order"], scenario, K=3.0)
    assert (p.tau, p.K) == (20.0, 3.0)


def test_each_scenario_runs_its_own_episodes_on_fixed_seeds():
    # A constant action on a plant with no gain (K 0) leaves the state where it
    # started, while K 2 moves it: the override must reach the plant, so the two
    # scenarios' costs differ. Seeds are fixed, so a rerun gives the same costs.
    spec = declaring(Scenario("dead", {"K": 0.0}), Scenario("live", {"K": 2.0}))

    def constant(params):
        return lambda obs: np.array([0.5])

    def costs():
        runs = scenarios.run(spec, constant, seeds=2, max_steps_in_episode=20)
        return scenarios.costs(runs)

    first, again = costs(), costs()
    assert list(first) == ["dead", "live"]
    assert all(len(c) == 2 for c in first.values())
    np.testing.assert_array_equal(first["dead"], again["dead"])
    np.testing.assert_array_equal(first["live"], again["live"])
    assert not np.allclose(first["dead"], first["live"])


# --- The evaluation window and seed rule --------------------------------------------
#
# Contract (owner, 2026-11-24, the evaluation cut; long-strata note R2 and R6): a task
# may declare an ``Evaluation`` on its registry entry. Its first ``unscored_steps``
# burn the plant in and its last steps are the ceiling's tail: an evaluation reads only
# the ``scored_steps`` between, as a mean cost per step. Its lines run on ``seeds``,
# and on ``extra_seeds`` too in a scenario where the paired saving of the reference
# over the ceiling on ``seeds`` has a 95 % Student t interval that touches 0. A task
# that declares none scores every step on the ladder's seeds 100 to 109, as before.

LADDER_SEEDS = tuple(range(100, 110))


def evaluating(**fields):
    return dataclasses.replace(
        REGISTRY["first_order"], evaluation=lambda: scenarios.Evaluation(**fields)
    )


def test_a_task_with_no_declared_evaluation_scores_every_step_on_the_ladder_seeds():
    spec = dataclasses.replace(REGISTRY["first_order"], evaluation=None)
    assert scenarios.scored_window(spec, 150) == slice(0, 150)
    assert scenarios.line_seeds(spec) == LADDER_SEEDS
    assert scenarios.line_seeds(spec, [1.0, -1.0, 0.5, -0.5, 0.0]) == LADDER_SEEDS


def test_the_scored_cost_is_the_mean_over_the_window_alone():
    spec = evaluating(unscored_steps=3, scored_steps=4)
    # 9 on the three burn-in steps and the two tail steps, 1 to 4 in the window.
    cost = np.array([9, 9, 9, 1, 2, 3, 4, 9, 9], float)
    assert scenarios.scored_window(spec, 9) == slice(3, 7)
    assert scenarios.scored_cost(spec, cost) == pytest.approx(2.5)


def test_a_window_past_the_episodes_end_is_an_error():
    with pytest.raises(ValueError):
        scenarios.scored_window(evaluating(unscored_steps=3, scored_steps=7), 9)


def test_scenario_costs_read_the_tasks_window():
    from target_gym.eval import Episode

    spec = evaluating(unscored_steps=2, scored_steps=2)
    ep = Episode(
        cost=np.array([5.0, 5.0, 1.0, 3.0, 5.0]),
        in_band=np.ones(5, bool),
        target_change=np.zeros(5, bool),
    )
    out = scenarios.costs({"normal": [ep]}, spec)
    np.testing.assert_allclose(out["normal"], [2.0])


#: Student's t at 97.5 % with 4 degrees of freedom (tables: 2.7764).
T4 = 2.7764


def test_the_saving_interval_is_students_t_at_95_percent():
    # mean 1, sample sd 1 (squares 1, 1, 1, 1, 0 over 4), so 1 +- 2.7764 / sqrt 5.
    lo, hi = scenarios.saving_interval([2.0, 0.0, 2.0, 0.0, 1.0])
    half = T4 / np.sqrt(5.0)
    assert (lo, hi) == (
        pytest.approx(1.0 - half, rel=1e-4),
        pytest.approx(1.0 + half, rel=1e-4),
    )


BASE, EXTRA = tuple(range(100, 105)), tuple(range(105, 110))


@pytest.mark.parametrize(
    "savings, seeds",
    [
        # mean 1.5, sd 0.5: 1.5 +- 0.62, clear of 0, so the five seeds stand.
        ([2.0, 1.0, 2.0, 1.0, 1.5], BASE),
        # mean 1, sd 1: 1 +- 1.24 reaches below 0, so five more run.
        ([2.0, 0.0, 2.0, 0.0, 1.0], BASE + EXTRA),
        # a ceiling worse than the reference on average, its interval through 0 too.
        ([-2.0, 0.0, -2.0, 0.0, -1.0], BASE + EXTRA),
        # no spread: the interval is the point 1.
        ([1.0] * 5, BASE),
    ],
)
def test_extra_seeds_run_only_where_the_saving_interval_touches_zero(savings, seeds):
    spec = evaluating(seeds=BASE, extra_seeds=EXTRA)
    assert scenarios.line_seeds(spec) == BASE
    assert scenarios.line_seeds(spec, savings) == seeds
