"""Tests for the 3D plane environment (all task variants)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane3d.env import (
    PlaneParams3D,
    PlaneState3D,
    check_is_terminal_3d,
    compute_reward_circle,
    compute_reward_figure8,
    compute_reward_heading,
    distance_to_circle,
    wrap_angle,
)
from target_gym.plane3d.env_jax import (
    Plane3DCircle,
    Plane3DFigureEight,
    Plane3DHeading,
)


class TestWrapAngle:
    def test_zero(self):
        assert wrap_angle(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_pi(self):
        assert abs(float(wrap_angle(jnp.pi))) == pytest.approx(jnp.pi, abs=1e-4)

    def test_large_positive(self):
        result = float(wrap_angle(3 * jnp.pi))
        assert abs(result) == pytest.approx(jnp.pi, abs=1e-4)

    def test_large_negative(self):
        result = float(wrap_angle(-5 * jnp.pi))
        assert abs(result) == pytest.approx(jnp.pi, abs=1e-4)


# ─── Heading task ──────────────────────────────────────


class TestHeadingInit:
    def test_init(self):
        env = Plane3DHeading()
        assert env.obs_shape == (15,)

    def test_default_params(self):
        env = Plane3DHeading()
        params = env.default_params
        assert params.I_x == 2_500_000
        assert params.wingspan == 35.8
        assert params.aileron_surface == 6.0


class TestHeadingReset:
    def test_reset(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert obs.shape == (15,)

    def test_reset_state_fields(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        assert float(state.x) == 0.0
        assert float(state.y) == 0.0
        assert float(state.x_dot) == 200.0
        assert float(state.y_dot) == 0.0
        assert float(state.phi) == 0.0
        assert float(state.phi_dot) == 0.0
        assert state.time == 0
        # Unused target fields should be zero
        assert float(state.target_x) == 0.0
        assert float(state.target_y) == 0.0
        assert float(state.target_radius) == 0.0

    def test_target_heading_randomized(self):
        env = Plane3DHeading()
        _, s1 = env.reset(jax.random.PRNGKey(0))
        _, s2 = env.reset(jax.random.PRNGKey(99))
        assert float(s1.target_heading) != pytest.approx(
            float(s2.target_heading), abs=1e-3
        )


class TestHeadingStep:
    def test_step_advances_time(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        action = jnp.array([0.8, 0.0, 0.0])
        obs2, state2, reward, done, info = env.step(key, state, action)
        assert state2.time == state.time + 1

    def test_step_obs_shape(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        action = jnp.array([0.8, 0.0, 0.0])
        obs2, *_ = env.step(key, state, action)
        assert obs2.shape == (15,)

    def test_step_moves_forward(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        action = jnp.array([0.8, 0.0, 0.0])
        _, state2, *_ = env.step(key, state, action)
        assert float(state2.x) > float(state.x)

    def test_no_lateral_motion_without_bank(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        action = jnp.array([0.8, 0.0, 0.0])
        for _ in range(10):
            _, state, *_ = env.step_env(key, state, action)
        assert abs(float(state.y)) < 10.0
        assert abs(float(state.y_dot)) < 1.0

    def test_bank_causes_heading_change(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        params = PlaneParams3D(
            max_steps_in_episode=200,
            initial_altitude_range=(5000, 5000),
            target_altitude_range=(5000, 5000),
        )
        _, state = env.reset_env(key, params)
        initial_psi = float(state.psi)
        action = jnp.array([0.8, 0.0, 0.5])
        for _ in range(50):
            _, state, *_ = env.step_env(key, state, action, params)
        assert float(state.psi) != pytest.approx(initial_psi, abs=0.01)

    def test_bank_causes_lateral_motion(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        params = PlaneParams3D(
            max_steps_in_episode=200,
            initial_altitude_range=(5000, 5000),
            target_altitude_range=(5000, 5000),
        )
        _, state = env.reset_env(key, params)
        action = jnp.array([0.8, 0.0, 0.5])
        for _ in range(50):
            _, state, *_ = env.step_env(key, state, action, params)
        assert abs(float(state.y)) > 1.0


class TestHeadingReward:
    def test_reward_at_target(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        state = state.replace(z=state.target_altitude, psi=state.target_heading)
        reward = compute_reward_heading(state, params)
        assert float(reward) > 0.9

    def test_reward_at_wrong_altitude(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        state = state.replace(
            z=state.target_altitude + 5000.0, psi=state.target_heading
        )
        reward = compute_reward_heading(state, params)
        assert float(reward) < 0.5

    def test_reward_at_wrong_heading(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        state = state.replace(
            z=state.target_altitude, psi=state.target_heading + jnp.pi
        )
        reward = compute_reward_heading(state, params)
        assert float(reward) < 0.1

    def test_penalty_on_terminal(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        state = state.replace(z=-1.0)
        reward = compute_reward_heading(state, params)
        assert float(reward) < -100


# ─── Circle task ───────────────────────────────────────


class TestCircleInit:
    def test_obs_shape(self):
        env = Plane3DCircle()
        assert env.obs_shape == (17,)


class TestCircleReset:
    def test_reset_obs_shape(self):
        env = Plane3DCircle()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert obs.shape == (17,)

    def test_starts_on_circle(self):
        env = Plane3DCircle()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        d = float(distance_to_circle(state))
        assert abs(d) < 1.0  # should be on the circle

    def test_target_radius_in_range(self):
        env = Plane3DCircle()
        params = PlaneParams3D(target_radius_range=(9000, 11000))
        key = jax.random.PRNGKey(42)
        _, state = env.reset_env(key, params)
        assert 9000 <= float(state.target_radius) <= 11000

    def test_heading_tangent_to_circle(self):
        """Initial heading should be roughly tangent to the circle."""
        env = Plane3DCircle()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        # Radial direction from center to aircraft
        radial_angle = jnp.arctan2(state.y - state.target_y, state.x - state.target_x)
        # Tangent is perpendicular to radial
        expected_heading = radial_angle + jnp.pi / 2
        heading_diff = abs(float(wrap_angle(state.psi - expected_heading)))
        assert heading_diff < 0.1


class TestCircleStep:
    def test_step_obs_shape(self):
        env = Plane3DCircle()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        action = jnp.array([0.8, 0.0, 0.0])
        obs2, *_ = env.step(key, state, action)
        assert obs2.shape == (17,)


class TestCircleReward:
    def test_reward_on_circle(self):
        """Reward should be high when on the circle at target altitude."""
        env = Plane3DCircle()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        # State is already on the circle from reset
        state = state.replace(z=state.target_altitude)
        reward = compute_reward_circle(state, params)
        assert float(reward) > 0.5

    def test_reward_far_from_circle(self):
        """Reward should be low when far from the circle."""
        env = Plane3DCircle()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        # Move far from circle
        state = state.replace(
            x=state.target_x + state.target_radius * 3,
            y=state.target_y,
            z=state.target_altitude,
        )
        reward = compute_reward_circle(state, params)
        assert float(reward) < 0.5


# ─── Figure-8 task ─────────────────────────────────────


class TestFigureEightInit:
    def test_obs_shape(self):
        env = Plane3DFigureEight()
        assert env.obs_shape == (19,)


class TestFigureEightReset:
    def test_reset_obs_shape(self):
        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        assert obs.shape == (19,)

    def test_starts_on_twisted_lemniscate(self):
        """Aircraft starts on the 3D twisted lemniscate."""
        from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        params = env.default_params
        _, state = env.reset(key, params)
        _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
        assert float(dist) < 200.0  # should be very close to the curve

    def test_starts_at_rotated_rightmost_point(self):
        """At τ=0 the start position is at (cx + r·cos(θ), cy + r·sin(θ))."""
        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        orientation = float(state.target_heading)
        expected_x = float(state.target_x) + float(state.target_radius) * jnp.cos(
            orientation
        )
        expected_y = float(state.target_y) + float(state.target_radius) * jnp.sin(
            orientation
        )
        assert float(state.x) == pytest.approx(expected_x, rel=1e-3)
        assert float(state.y) == pytest.approx(expected_y, abs=1.0)

    def test_initial_heading_tangent(self):
        """At τ=0, heading is orientation + π/2 (tangent to the lemniscate)."""
        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        expected_heading = float(state.target_heading) + jnp.pi / 2
        assert float(state.psi) == pytest.approx(float(expected_heading), abs=0.1)


class TestFigureEightStep:
    def test_step_obs_shape(self):
        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        action = jnp.array([0.8, 0.0, 0.0])
        obs2, *_ = env.step(key, state, action)
        assert obs2.shape == (19,)


class TestFigureEightReward:
    def test_reward_on_curve(self):
        """Reward should be high when on the lemniscate at target altitude."""
        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        state = state.replace(z=state.target_altitude)
        reward = compute_reward_figure8(state, params)
        assert float(reward) > 0.5

    def test_reward_far_from_curve(self):
        """Reward should be low when far from the lemniscate."""
        env = Plane3DFigureEight()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        state = state.replace(
            x=state.target_x + state.target_radius * 5,
            y=state.target_y + state.target_radius * 5,
            z=state.target_altitude,
        )
        reward = compute_reward_figure8(state, params)
        assert float(reward) < 0.1


# ─── Shared: terminal checks ──────────────────────────


class TestTerminal3D:
    @pytest.mark.parametrize(
        "env_cls", [Plane3DHeading, Plane3DCircle, Plane3DFigureEight]
    )
    def test_not_terminal_in_range(self, env_cls):
        env = env_cls()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        params = PlaneParams3D()
        terminated, truncated = check_is_terminal_3d(state, params)
        assert not bool(terminated)
        assert not bool(truncated)

    def test_terminal_below_ground(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        state = state.replace(z=-1.0)
        terminated, _ = check_is_terminal_3d(state, PlaneParams3D())
        assert bool(terminated)

    def test_terminal_above_max(self):
        params = PlaneParams3D()
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        state = state.replace(z=params.max_alt + 1.0)
        terminated, _ = check_is_terminal_3d(state, params)
        assert bool(terminated)

    def test_truncated_at_max_steps(self):
        params = PlaneParams3D(max_steps_in_episode=100)
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        _, state = env.reset(key)
        state = state.replace(time=101)
        _, truncated = check_is_terminal_3d(state, params)
        assert bool(truncated)


# ─── Action space ──────────────────────────────────────


class TestActionSpace:
    @pytest.mark.parametrize(
        "env_cls", [Plane3DHeading, Plane3DCircle, Plane3DFigureEight]
    )
    def test_action_space_shape(self, env_cls):
        env = env_cls()
        space = env.action_space()
        assert space.shape == (3,)

    def test_sample_action(self):
        env = Plane3DHeading()
        key = jax.random.PRNGKey(42)
        action = env.action_space().sample(key)
        assert action.shape == (3,)
        for i in range(3):
            assert -1 <= float(action[i]) <= 1
