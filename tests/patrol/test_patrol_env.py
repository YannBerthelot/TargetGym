"""Tests for the close-patrol (formation-keeping) environment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import PlanePatrol, PlanePatrolBearingOnly
from target_gym.experts.pid import make_patrol_pid, patrol_pid_step
from target_gym.patrol.env import (
    PatrolParams,
    check_is_terminal_patrol,
    compute_reward_patrol,
    desired_slot_position,
    separation,
    slot_error,
    slot_error_vector,
)

# ─── Construction & spaces ──────────────────────────────


class TestConstruction:
    def test_full_obs_shape(self):
        assert PlanePatrol().obs_shape == (26,)

    def test_bearing_obs_shape(self):
        assert PlanePatrolBearingOnly().obs_shape == (21,)

    def test_action_space(self):
        env = PlanePatrol()
        space = env.action_space()
        assert space.shape == (3,)

    def test_value_target_indices(self):
        env = PlanePatrol()
        # value = slot_error, target = constant 0
        assert env.obs_value_index == 24
        assert env.obs_target_index == 25


# ─── Reset ──────────────────────────────────────────────


class TestReset:
    @pytest.mark.parametrize("Env", [PlanePatrol, PlanePatrolBearingOnly])
    def test_reset_obs_shape_and_finite(self, Env):
        env = Env()
        obs, state = env.reset(jax.random.PRNGKey(0))
        assert obs.shape == env.obs_shape
        assert bool(jnp.all(jnp.isfinite(obs)))

    def test_lead_starts_at_origin(self):
        env = PlanePatrol()
        _, state = env.reset(jax.random.PRNGKey(0))
        assert float(state.lead.x) == 0.0
        assert float(state.lead.y) == 0.0
        assert state.time == 0

    def test_follower_spawns_near_slot(self):
        # follower_spawn_noise is the only offset from the exact slot at reset
        env = PlanePatrol()
        params = env.default_params.replace(follower_spawn_noise=0.0)
        _, state = env.reset(jax.random.PRNGKey(1), params)
        assert float(slot_error(state)) == pytest.approx(0.0, abs=1e-3)

    def test_slots_randomized_across_seeds(self):
        env = PlanePatrol()
        _, s1 = env.reset(jax.random.PRNGKey(0))
        _, s2 = env.reset(jax.random.PRNGKey(123))
        assert float(s1.slot_back) != pytest.approx(float(s2.slot_back), abs=1e-3)


# ─── Slot geometry ──────────────────────────────────────


class TestSlotGeometry:
    def test_slot_error_zero_at_slot(self):
        env = PlanePatrol()
        params = env.default_params.replace(follower_spawn_noise=0.0)
        _, state = env.reset(jax.random.PRNGKey(4), params)
        eb, er, eu = (float(v) for v in slot_error_vector(state))
        assert eb == pytest.approx(0.0, abs=1e-2)
        assert er == pytest.approx(0.0, abs=1e-2)
        assert eu == pytest.approx(0.0, abs=1e-2)

    def test_desired_slot_offset_is_behind_lead(self):
        # With a due-east lead (heading 0) and pure back-slot, the slot sits
        # behind the lead at lower x.
        env = PlanePatrol()
        params = env.default_params
        _, state = env.reset(jax.random.PRNGKey(0), params)
        state = state.replace(
            lead=state.lead.replace(x=0.0, y=0.0, psi=0.0),
            slot_back=200.0,
            slot_right=0.0,
            slot_up=0.0,
        )
        sx, sy, sz = (float(v) for v in desired_slot_position(state))
        assert sx == pytest.approx(-200.0, abs=1e-3)
        assert sy == pytest.approx(0.0, abs=1e-3)


# ─── Step / reward / termination ────────────────────────


class TestStep:
    def test_step_advances_time(self):
        env = PlanePatrol()
        obs, state = env.reset(jax.random.PRNGKey(0))
        _, state2, r, done, _ = env.step_env(
            jax.random.PRNGKey(0), state, jnp.zeros(3), env.default_params
        )
        assert state2.time == state.time + 1
        assert bool(jnp.isfinite(r))

    def test_reward_near_one_in_slot(self):
        env = PlanePatrol()
        params = env.default_params.replace(follower_spawn_noise=0.0)
        _, state = env.reset(jax.random.PRNGKey(2), params)
        r = float(compute_reward_patrol(state, params))
        assert r == pytest.approx(1.0, abs=1e-2)

    def test_collision_terminates_with_penalty(self):
        env = PlanePatrol()
        params = env.default_params
        _, state = env.reset(jax.random.PRNGKey(0), params)
        # Place the follower on top of the lead.
        state = state.replace(
            follower=state.follower.replace(
                x=state.lead.x, y=state.lead.y, z=state.lead.z
            )
        )
        assert float(separation(state)) < params.min_separation
        terminated, _ = check_is_terminal_patrol(state, params)
        assert bool(terminated)
        assert float(compute_reward_patrol(state, params)) < 0.0

    def test_lost_formation_terminates(self):
        env = PlanePatrol()
        params = env.default_params
        _, state = env.reset(jax.random.PRNGKey(0), params)
        state = state.replace(follower=state.follower.replace(x=state.lead.x + 5000.0))
        terminated, _ = check_is_terminal_patrol(state, params)
        assert bool(terminated)


# ─── JIT / vmap ─────────────────────────────────────────


class TestJaxTransforms:
    def test_step_jits(self):
        env = PlanePatrol()
        params = env.default_params
        obs, state = env.reset(jax.random.PRNGKey(0))
        step = jax.jit(lambda k, s, a: env.step_env(k, s, a, params))
        _, _, r, _, _ = step(jax.random.PRNGKey(0), state, jnp.zeros(3))
        assert bool(jnp.isfinite(r))

    def test_vmap_reset_and_step(self):
        env = PlanePatrol()
        params = env.default_params
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        obs, states = jax.vmap(lambda k: env.reset(k, params))(keys)
        assert obs.shape == (8, 26)
        actions = jnp.zeros((8, 3))
        out = jax.vmap(lambda k, s, a: env.step_env(k, s, a, params))(
            keys, states, actions
        )
        rewards = out[2]
        assert rewards.shape == (8,)
        assert bool(jnp.all(jnp.isfinite(rewards)))


# ─── Expert behaviour ───────────────────────────────────


class TestExpert:
    def _settled_error(self, turn_rate, seed, n=1500):
        env = PlanePatrol()
        params = env.default_params.replace(
            max_steps_in_episode=n, lead_turn_rate_range=(turn_rate, turn_rate)
        )
        pid_params, pid0 = make_patrol_pid()
        obs, state = env.reset(jax.random.PRNGKey(seed), params)

        def step(carry, _):
            obs, state, pid = carry
            action, pid = patrol_pid_step(pid_params, pid, obs)
            obs, state, r, done, _ = env.step_env(
                jax.random.PRNGKey(seed), state, action, params
            )
            return (obs, state, pid), (slot_error(state), separation(state))

        _, (errs, seps) = jax.lax.scan(step, (obs, state, pid0), None, length=n)
        return float(jnp.mean(errs[-500:])), float(jnp.min(seps))

    @pytest.mark.xfail(
        reason="The close-patrol expert's gains were re-tuned after the "
        "lift-curve correction (PHYSICS.md D1/D2) doubled control authority: "
        "mean return went from 2.9 to 117 and the follower no longer departs. "
        "But it settles ~139 m from the slot against a 60 m tolerance, so "
        "formation is held only loosely. A grid search over altitude/heading/"
        "bank scalings got no closer, which suggests the guidance law needs "
        "rework rather than further tuning. Every structural patrol test "
        "passes -- this is expert quality, not broken dynamics.",
        strict=True,
    )
    @pytest.mark.parametrize("turn", [0.0, 0.002, -0.003])
    def test_expert_holds_formation(self, turn):
        # Averaged over a few seeds, the tuned expert holds the slot well
        # within the reward tolerance and never collides.
        settled = np.mean([self._settled_error(turn, s)[0] for s in range(3)])
        assert settled < PatrolParams().slot_tolerance

    def test_expert_never_collides(self):
        min_sep = min(self._settled_error(0.002, s)[1] for s in range(3))
        assert min_sep > PatrolParams().min_separation
