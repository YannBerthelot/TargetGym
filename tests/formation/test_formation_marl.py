"""The multi-agent view of the formation: one observation and one action per
aircraft, JaxMARL style, over the same step and the same team reward as the
registered task."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import FormationMARL
from target_gym.formation.marl import LEAD, wingman_name

KEY = jax.random.PRNGKey(0)


def _rand_actions(env, key):
    keys = jax.random.split(key, env.num_agents)
    return {
        a: jax.random.uniform(keys[i], (3,), minval=-1.0, maxval=1.0)
        for i, a in enumerate(env.agents)
    }


class TestAgents:
    def test_default_is_a_lead_and_two_wingmen(self):
        env = FormationMARL()
        assert env.agents == [LEAD, "wingman_0", "wingman_1"]
        assert env.num_agents == 3

    def test_agents_scale_with_the_wingmen(self):
        env = FormationMARL(num_wingmen=4)
        assert env.agents == [LEAD] + [wingman_name(i) for i in range(4)]

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_spaces_per_agent(self, k):
        env = FormationMARL(num_wingmen=k)
        assert env.observation_space(LEAD).shape == (16 + 3 * k,)
        for i in range(k):
            assert env.observation_space(wingman_name(i)).shape == (26,)
        assert set(env.action_spaces) == set(env.agents)
        assert all(s.shape == (3,) for s in env.action_spaces.values())


class TestSameTaskAsTheJointView:
    def test_observations_are_the_joint_observation_split_by_seat(self):
        env = FormationMARL()
        obs, state = env.reset(KEY)
        joint = np.asarray(env.env.get_obs(state))
        split = np.concatenate([np.asarray(obs[a]) for a in env.agents])
        np.testing.assert_array_equal(split, joint)

    def test_every_agent_gets_the_joint_reward(self):
        env = FormationMARL()
        params = env.default_params
        _, state = env.reset(KEY, params)
        actions = _rand_actions(env, jax.random.PRNGKey(1))
        _, s2, rewards, dones, info = env.step_env(KEY, state, actions, params)
        joint = jnp.concatenate([actions[a] for a in env.agents])
        _, j2, r, _, _ = env.env.step_env(KEY, state, joint, params)
        assert set(rewards) == set(env.agents)
        for a in env.agents:
            assert float(rewards[a]) == pytest.approx(float(r))
        assert set(dones) == set(env.agents) | {"__all__"}
        np.testing.assert_array_equal(np.asarray(s2.lead.x), np.asarray(j2.lead.x))

    def test_each_agents_cost_is_reported_and_adds_up(self):
        env = FormationMARL()
        params = env.default_params
        _, state = env.reset(KEY, params)
        _, _, rewards, _, info = env.step_env(
            KEY, state, {a: jnp.zeros(3) for a in env.agents}, params
        )
        assert not bool(info["tripped"])
        total = sum(float(info["costs"][a]) for a in env.agents)
        assert total == pytest.approx(-float(rewards[LEAD]), rel=1e-5)


class TestAutoReset:
    def test_the_episode_ends_at_the_time_limit_and_restarts(self):
        env = FormationMARL()
        params = env.default_params.replace(max_steps_in_episode=3)
        _, state = env.reset(KEY, params)
        zero = {a: jnp.zeros(3) for a in env.agents}
        for t in range(3):
            _, state, _, dones, _ = env.step(jax.random.PRNGKey(t), state, zero, params)
        assert bool(dones["__all__"])
        assert int(state.time) == 0

    @pytest.mark.parametrize("k", [1, 4])
    def test_jit_scan_and_autoreset(self, k):
        env = FormationMARL(num_wingmen=k)
        params = env.default_params.replace(max_steps_in_episode=40)

        def rollout(key):
            obs, state = env.reset(key, params)

            def step(carry, _):
                obs, state, key = carry
                key, ka, ks = jax.random.split(key, 3)
                obs, state, r, d, _ = env.step(
                    ks, state, _rand_actions(env, ka), params
                )
                return (obs, state, key), (r[LEAD], d["__all__"])

            _, out = jax.lax.scan(step, (obs, state, key), None, length=100)
            return out

        rewards, dones = jax.jit(rollout)(KEY)
        assert rewards.shape == (100,)
        assert bool(jnp.all(jnp.isfinite(rewards)))
        assert int(dones.sum()) >= 2

    def test_vmap_reset(self):
        env = FormationMARL(num_wingmen=3)
        keys = jax.random.split(KEY, 6)
        obs, _ = jax.vmap(lambda k: env.reset(k, env.default_params))(keys)
        assert obs["wingman_2"].shape == (6, 26)
        assert obs[LEAD].shape == (6, 25)
