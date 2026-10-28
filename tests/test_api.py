"""The front door: ``target_gym.make`` and its siblings.

One way in per need, each resolved through the registry, so the top level and
the registry cannot disagree about what a task, its scored configuration or
its expert is.
"""

from __future__ import annotations

import chex
import gymnasium as gym
import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym import registry
from target_gym.experts.pid import Controller

FRONT_DOOR = (
    "make",
    "make_expert",
    "make_oracle",
    "make_gymnasium",
    "list_envs",
    "score_policy",
)


def test_front_door_is_exported():
    assert set(FRONT_DOOR) <= set(tg.__all__)


def test_score_policy_is_the_scoring_one():
    from target_gym import scoring

    assert tg.score_policy is scoring.score_policy


def test_list_envs_is_the_registry():
    assert tg.list_envs() == registry.env_names()
    aircraft = tg.list_envs("aircraft")
    assert "plane" in aircraft and "cstr" not in aircraft
    with pytest.raises(KeyError):
        tg.list_envs("no_such_group")


@pytest.mark.parametrize("name", registry.env_names())
def test_make_returns_the_shared_env_and_the_scored_params(name):
    spec = registry.get(name)
    env, params = tg.make(name)
    assert env is spec.make_env()
    chex.assert_trees_all_equal(params, spec.make_test_params())


def test_make_overrides_single_fields():
    _, params = tg.make("cstr", max_steps_in_episode=7)
    assert params.max_steps_in_episode == 7


def test_make_names_the_registered_tasks_on_a_typo():
    with pytest.raises(KeyError, match="registered"):
        tg.make("plnae")


def test_make_expert_is_the_scored_expert_in_every_form():
    env, params = tg.make("cstr")
    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
    obs = np.asarray(obs)
    expert = tg.make_expert("cstr")
    assert isinstance(expert, Controller)
    scored = registry.get("cstr").make_pid()(obs)
    np.testing.assert_array_equal(expert.host()(obs), scored)
    policy = expert.batched()
    actions, _ = policy(policy.init_state(1), obs[None])
    np.testing.assert_allclose(np.asarray(actions)[0], scored, rtol=1e-5)


def test_make_oracle_plans_on_the_scored_params():
    oracle = tg.make_oracle("cstr")
    assert hasattr(oracle, "step") and hasattr(oracle, "reset")


@pytest.mark.parametrize("name", ["plane", "cstr", "battery"])
def test_make_gymnasium_runs_any_task_on_the_scored_params(name):
    genv = tg.make_gymnasium(name)
    assert isinstance(genv, gym.Env)
    chex.assert_trees_all_equal(genv.env_params, tg.make(name)[1])
    obs, _ = genv.reset(seed=0)
    assert genv.observation_space.contains(obs)
    obs, reward, terminated, truncated, _ = genv.step(genv.action_space.sample())
    assert obs.shape == genv.observation_space.shape
    assert isinstance(reward, float)


def test_duplicate_expert_paths_warn_and_still_work():
    env, _ = tg.make("cstr")
    with pytest.warns(DeprecationWarning, match="make_expert"):
        assert callable(env.make_pid())
    with pytest.warns(DeprecationWarning, match="make_expert"):
        assert hasattr(env.expert_policy, "init_state")


def test_old_gymnasium_names_warn_and_still_work():
    with pytest.warns(DeprecationWarning, match="make_gymnasium"):
        cls = tg.GymnasiumPlane
    assert isinstance(cls(), gym.Env)
    with pytest.warns(DeprecationWarning, match="make_gymnasium"):
        assert isinstance(tg.gym_wrapper_factory(tg.CSTR)(), gym.Env)
