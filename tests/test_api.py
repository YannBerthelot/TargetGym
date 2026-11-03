"""The front door: ``target_gym.make`` and its siblings.

One way in per need, each resolved through the registry, so the top level and
the registry cannot disagree about what a task, its scored configuration or
its expert is.
"""

from __future__ import annotations

import chex
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import registry

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
    scored = registry.get("cstr").make_pid()(obs)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(expert.host()(obs), scored)
    with pytest.warns(DeprecationWarning):
        policy = expert.batched()
    actions, _ = policy(policy.init_state(1), obs[None])
    np.testing.assert_allclose(np.asarray(actions)[0], scored, rtol=1e-5)


# The expert is used like the environment (owner's decision, 2026-11-03):
#   memory = expert.reset()                     fresh memory for one episode
#   action, memory = expert.step(memory, obs)   the scored expert's action
# The same step runs in a Python loop, under jit and lax.scan, and batched with
# jax.vmap(expert.step) from expert.reset(n). One task per controller shape:
# a single loop, two loops, the aircraft guidance, the compressor's guarded
# pair, a battery's derating, and the multi-agent formation.
SHAPES = [
    "cstr",
    "four_tank",
    "plane3d_heading",
    "compressor_surge",
    "battery",
    "formation",
]


def _scored_episode(name, steps):
    """Observations along ``steps`` steps flown by the scored host expert, and
    its actions: the reference the new form must reproduce."""
    spec = registry.get(name)
    env, params = spec.make_env(), spec.make_test_params()
    pid = spec.make_pid()
    pid.reset()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    span = env.action_space(params)
    span = np.maximum(np.asarray(span.high) - np.asarray(span.low), 1e-12)
    seen, actions = [], []
    for _ in range(steps):
        seen.append(np.asarray(obs))
        actions.append(np.atleast_1d(pid(np.asarray(obs))))
        obs, state, _, _, _ = step(key, state, jnp.asarray(actions[-1]), params)
    return np.stack(seen), np.stack(actions), span


@pytest.mark.parametrize("name", SHAPES)
def test_expert_steps_as_the_scored_expert_from_reset(name):
    seen, scored, span = _scored_episode(name, 60)
    expert = tg.make_expert(name)
    memory = expert.reset()
    worst = 0.0
    for obs, want in zip(seen, scored):
        action, memory = expert.step(memory, obs)
        got = np.atleast_1d(np.asarray(action))
        worst = max(worst, float(np.max(np.abs(got - want) / span)))
    # 1e-3 of the action range: JAX steps in float32 where the scored host
    # PID keeps some float64, as test_make_expert_is_the_scored_pid allows.
    assert worst < 1e-3, f"{name}: {worst:.3g} of the action range"


@pytest.mark.parametrize("name", SHAPES)
def test_vmapped_expert_from_a_batched_reset_steps_each_env_alike(name):
    seen, _, _ = _scored_episode(name, 3)
    expert = tg.make_expert(name)
    batched, _ = jax.vmap(expert.step)(expert.reset(3), jnp.asarray(seen))
    for i, obs in enumerate(seen):
        single, _ = expert.step(expert.reset(), obs)
        # Same float32 operations batched or not.
        np.testing.assert_allclose(batched[i], single, rtol=1e-6, atol=1e-7)


def test_expert_runs_inside_a_jitted_scan_as_in_a_python_loop():
    seen, _, _ = _scored_episode("cstr", 30)
    expert = tg.make_expert("cstr")

    def body(memory, obs):
        action, memory = expert.step(memory, obs)
        return memory, action

    scanned = jax.jit(lambda obs: jax.lax.scan(body, expert.reset(), obs)[1])(
        jnp.asarray(seen)
    )
    memory, looped = expert.reset(), []
    for obs in seen:
        action, memory = expert.step(memory, obs)
        looped.append(action)
    # XLA may fuse the scan's float32 arithmetic differently from one step.
    np.testing.assert_allclose(scanned, np.stack(looped), rtol=1e-5, atol=1e-6)


def test_old_expert_forms_warn_and_still_work():
    seen, scored, _ = _scored_episode("cstr", 1)
    expert = tg.make_expert("cstr")
    with pytest.warns(DeprecationWarning, match=r"reset\(\).*step"):
        host = expert.host()
    np.testing.assert_array_equal(np.atleast_1d(host(seen[0])), scored[0])
    with pytest.warns(DeprecationWarning, match=r"reset\(\).*step"):
        policy = expert.batched()
    assert hasattr(policy, "init_state")
    with pytest.warns(DeprecationWarning, match=r"reset\(\)"):
        memory = expert.init_state
    with pytest.warns(DeprecationWarning, match=r"step"):
        action, _ = expert.act(memory, seen[0])
    np.testing.assert_allclose(np.atleast_1d(action), scored[0], rtol=1e-5)


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
