"""The policy interface and the benchmark runner (``target_gym.benchmark``):
episodes are paired across policies, records are the protocol's, a policy
that cannot run a task says so, only an oracle sees the state, and a family
of per-episode plants is described and cut per episode."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import benchmark as B
from target_gym import eval as E
from target_gym.registry import REGISTRY


class Hold:
    """Holds a constant action; records the observations it was reset on."""

    name = "hold"

    def __init__(self, value: float = 0.0):
        self.value, self.seen = value, []

    def reset(self, key, obs, task):
        self.seen.append(np.asarray(obs))
        return jnp.full((task.n_episodes, task.n_act), self.value)

    def act(self, carry, obs, t):
        return carry, carry


class SingleInputOnly(Hold):
    name = "siso"

    def reset(self, key, obs, task):
        if task.n_act != 1:
            raise B.Unsupported(f"{task.n_act} actuators")
        return super().reset(key, obs, task)


class Oracle(Hold):
    name = "oracle"
    oracle = True

    def act(self, carry, obs, t, state=None):
        self.state = state
        return carry, carry


def _setup(name, n):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params()
    return env, params, B.task_info(name, env, params, n)


def test_a_core_task_keeps_its_benchmark_key():
    """The key is fold_in(PRNGKey(seed), position in REGISTRY), whatever order
    the tasks are named in. TargetFoundation derives the same keys, so the
    positions are pinned as literals."""
    seen = {}

    class KeyHold(Hold):
        def reset(self, key, obs, task):
            seen[task.name] = key
            return super().reset(key, obs, task)

    B.run_policy_on_benchmark(
        KeyHold(), tasks=["battery", "cstr"], n_episodes=2, n_steps=1, verbose=False
    )
    for name, index in (("cstr", 9), ("battery", 20)):
        task_key = jax.random.fold_in(jax.random.PRNGKey(0), index)
        # run_policy hands the policy fold_in(task_key, 1)
        assert (seen[name] == jax.random.fold_in(task_key, 1)).all(), name


def test_a_policy_satisfies_the_protocol():
    assert isinstance(Hold(), B.Policy)
    assert isinstance(B.shipped_policy("pid"), B.Policy)


def test_task_info_describes_the_task():
    env, params, info = _setup("four_tank", 3)
    assert info.n_act == 2 and info.n_episodes == 3
    assert info.e_floor.shape == (3, len(info.value_index))
    assert (info.max_steps == int(params.max_steps_in_episode)).all()
    assert info.dt.shape == (3,) and (info.dt > 0).all()


def test_episodes_are_paired_across_policies_and_runs():
    """The same seed: two policies are reset on the same observations, and the
    same policy twice gives the same record."""
    env, params, info = _setup("cstr", 3)
    key = jax.random.PRNGKey(3)
    a, b = Hold(0.0), Hold(0.5)
    ra = B.run_policy(a, env, params, info, key)
    B.run_policy(b, env, params, info, key)
    np.testing.assert_array_equal(a.seen[0], b.seen[0])
    ra2 = B.run_policy(Hold(0.0), env, params, info, key)
    for k in ("reward", "value", "target", "tripped"):
        np.testing.assert_array_equal(ra[k], ra2[k])


def test_records_are_the_protocols():
    env, params, info = _setup("cstr", 2)
    r = B.run_policy(Hold(0.0), env, params, info, jax.random.PRNGKey(0))
    eps = B.episodes_of(r)
    assert len(eps) == 2
    np.testing.assert_array_equal(eps[0].cost, -r["reward"][0])
    np.testing.assert_array_equal(eps[1].failed, r["tripped"][1])
    s = B.score(r, burn_in=10)
    assert s["metrics"]["gain"] == pytest.approx(E.evaluate(eps, burn_in=10)["gain"])
    assert s["gain"].shape == (2,)


def test_target_changes_follow_run_episodes_rule():
    """More than 5 % of the first target's magnitude, on any output."""
    first = np.array([10.0, 1.0])
    targets = np.array(
        [[10.0, 1.0], [10.4, 1.0], [11.0, 1.0], [11.0, 1.2], [11.0, 1.2]]
    )
    np.testing.assert_array_equal(
        B.target_changes(targets, first), [False, False, True, True, False]
    )


def test_a_task_the_policy_cannot_run_is_reported_not_run():
    res = B.run_policy_on_benchmark(
        SingleInputOnly(), tasks=["cstr", "four_tank"], n_episodes=1, test_params=True,
        verbose=False,
    )  # fmt: skip
    assert "metrics" in res["cstr"]
    assert res["four_tank"] == {"unsupported": "2 actuators"}


def test_only_an_oracle_sees_the_state():
    env, params, info = _setup("cstr", 2)
    p = Oracle()
    B.run_policy(p, env, params, info, jax.random.PRNGKey(0), n_steps=3)
    leaves = jax.tree.leaves(p.state)
    assert leaves and all(x.shape[0] == 2 for x in leaves)


def test_per_episode_plants_are_described_and_cut_per_episode():
    """A family: each episode its own parameters and its own length."""
    env, params, _ = _setup("first_order", 1)
    batch = jax.tree.map(lambda x: jnp.stack([jnp.asarray(x)] * 2), params)
    T = int(params.max_steps_in_episode)
    batch = batch.replace(max_steps_in_episode=jnp.asarray([T, T // 2]))
    info = B.task_info("first_order", env, batch, 2, per_episode_params=True)
    assert list(info.max_steps) == [T, T // 2]
    r = B.run_policy(
        Hold(), env, batch, info, jax.random.PRNGKey(0), per_episode_params=True
    )
    assert list(r["length"]) == [T, T // 2]
    assert [len(e.cost) for e in B.episodes_of(r)] == [T, T // 2]


def test_a_study_declares_its_extras_per_task():
    seen = {}

    class Reads(Hold):
        def reset(self, key, obs, task):
            seen[task.name] = dict(task.extras)
            return super().reset(key, obs, task)

    B.run_policy_on_benchmark(
        Reads(), tasks=["cstr"], n_episodes=1, test_params=True, verbose=False,
        declare=lambda name, env, params: {"direction": name},
    )  # fmt: skip
    assert seen == {"cstr": {"direction": "cstr"}}


class Countdown:
    """A plant that terminates on the step its clock reaches ``params.stop``."""

    obs_value_index, obs_target_index = 0, 1

    def action_space(self, params):
        from gymnax.environments import spaces

        return spaces.Box(-1.0, 1.0, (1,), jnp.float32)

    def reset_env(self, key, params):
        return jnp.zeros(2), jnp.asarray(0)

    def step_env(self, key, state, action, params):
        t = state + 1
        obs = jnp.stack([t.astype(jnp.float32), jnp.asarray(0.0)])
        return obs, t, jnp.asarray(-1.0), t >= params.stop, {}


def test_a_termination_is_recorded_latched_and_ends_the_record():
    from typing import NamedTuple

    class Params(NamedTuple):
        stop: jax.Array
        max_steps_in_episode: jax.Array
        e_floor: jax.Array

    env = Countdown()
    params = Params(jnp.asarray([2, 4]), jnp.asarray([6, 6]), jnp.ones(2))
    info = B.task_info("countdown", env, params, 2, per_episode_params=True)
    r = B.run_policy(
        Hold(), env, params, info, jax.random.PRNGKey(0), per_episode_params=True
    )
    np.testing.assert_array_equal(
        r["terminated"], np.arange(6)[None] >= np.asarray([[1], [3]])
    )
    assert list(r["length"]) == [2, 4]
    np.testing.assert_array_equal(r["reward"] != 0, np.arange(6)[None] < [[2], [4]])


def test_nothing_is_shipped_for_a_task_this_library_does_not_register():
    env, params, info = _setup("cstr", 1)
    info = B.TaskInfo(**{**info.__dict__, "name": "not_a_task"})
    with pytest.raises(B.Unsupported, match="not_a_task"):
        B.shipped_policy("pid").reset(jax.random.PRNGKey(0), None, info)


def test_the_default_is_the_registered_task():
    """plane and plane_energy share one environment; only their registered
    parameters (``make_test_params``) make them two tasks, and the runner
    runs those by default."""
    seen = {}

    class Reads(Hold):
        def reset(self, key, obs, task):
            seen[task.name] = (task.e_floor[0, 0], int(task.max_steps[0]))
            return super().reset(key, obs, task)

    B.run_policy_on_benchmark(
        Reads(), tasks=["plane", "plane_energy"], n_episodes=1, n_steps=1, verbose=False
    )
    assert seen["plane"] != seen["plane_energy"]
    for name in ("plane", "plane_energy"):
        params = REGISTRY[name].make_test_params()
        assert seen[name] == (float(params.e_floor), int(params.max_steps_in_episode))


def test_a_policys_diagnostics_are_recorded_with_its_row():
    class Counts(Hold):
        def diagnostics(self, carry):
            return {"final_action": float(np.asarray(carry)[0, 0])}

    res = B.run_policy_on_benchmark(
        Counts(0.25), tasks=["cstr"], n_episodes=1, n_steps=3, verbose=False
    )
    assert res["cstr"]["diagnostics"] == {"final_action": 0.25}


def test_the_shipped_controller_runs_as_a_policy():
    """The shipped PID through the runner (``CallablePolicy``, the registered
    task's parameters): finite actions within the bounds."""
    env, params, info = _setup("cstr", 2)
    r = B.run_policy(
        B.shipped_policy("pid"), env, params, info, jax.random.PRNGKey(0), n_steps=3
    )
    a = r["action"]
    assert a.shape == (2, 3, 1) and np.isfinite(a).all()
    assert (a >= info.action_low).all() and (a <= info.action_high).all()


def test_a_burn_in_per_episode_is_each_episodes_own():
    """``score`` with a burn-in per episode: each episode scored with its own,
    the metrics the mean over episodes."""
    env, params, info = _setup("cstr", 2)
    r = B.run_policy(Hold(0.0), env, params, info, jax.random.PRNGKey(0))
    s = B.score(r, burn_in=np.array([10, 30]))
    eps = B.episodes_of(r)
    expected = [E.evaluate([e], burn_in=b)["gain"] for e, b in zip(eps, (10, 30))]
    np.testing.assert_allclose(s["gain"], expected)
    assert s["metrics"]["gain"] == pytest.approx(np.mean(expected))


def test_recorded_quantities_are_per_step_and_see_the_trip():
    """A study's per-step quantities: evaluated on the state after each step,
    told whether the step tripped, and averaged over each episode's steps."""
    env, params, info = _setup("cstr", 2)
    record = {
        "time": lambda s, p, tripped: jnp.asarray(s.time, jnp.float32),
        "trip": lambda s, p, tripped: tripped.astype(jnp.float32),
    }
    r = B.run_policy(
        Hold(0.0), env, params, info, jax.random.PRNGKey(0), n_steps=5, record=record
    )
    assert r["records"]["time"].shape == (2, 5)
    np.testing.assert_array_equal(r["records"]["time"][0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(r["records"]["trip"], r["tripped"].astype(float))
    s = B.score({**r, "length": np.array([5, 5])}, burn_in=1)
    np.testing.assert_allclose(s["records"]["time"], [3.0, 3.0])


def test_the_benchmark_passes_a_studys_records_per_task():
    res = B.run_policy_on_benchmark(
        Hold(), tasks=["cstr"], n_episodes=2, n_steps=4, verbose=False,
        record=lambda name, env, params: {"one": lambda s, p, tripped: jnp.float32(1.0)},
    )  # fmt: skip
    np.testing.assert_allclose(res["cstr"]["records"]["one"], [1.0, 1.0])
