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


def test_a_task_with_no_declared_suite_has_one_normal_scenario():
    (only,) = scenarios.suite(REGISTRY["first_order"])
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
    assert scenarios.weights(REGISTRY["first_order"]) == {"normal": 1.0}


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
