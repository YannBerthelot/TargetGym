"""Wind & turbulence: air-relative aerodynamics, observability, OU gusts.

Wind is a physics-engine property (read from the params/state by
compute_next_state[_3d]), so these tests exercise it through the 2D and 3D
plane envs and confirm it reaches an inheriting env (patrol) for free.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import (
    Plane,
    Plane3D,
    PlaneParams,
    PlanePatrol,
    PlanePatrolMARL,
)
from target_gym.plane.dynamics import total_wind_2d
from target_gym.plane3d.env import PlaneParams3D

FIXED = dict(target_altitude_range=(5000, 5000), initial_altitude_range=(5000, 5000))


def _final_alt_2d(env, params, n=300):
    obs, state = env.reset_env(jax.random.PRNGKey(0), params)

    def step(carry, _):
        s, k = carry
        k, ks = jax.random.split(k)
        _, ns, _, _, _ = env.step_env(ks, s, jnp.array([0.6, 0.0]), params)
        return (ns, k), ns.z

    (s, _), zs = jax.lax.scan(step, (state, jax.random.PRNGKey(0)), None, length=n)
    return zs


class TestBackwardCompatible:
    def test_default_no_wind_obs_shape_2d(self):
        assert Plane().obs_shape == (9,)

    def test_default_no_wind_obs_shape_3d(self):
        assert Plane3D().obs_shape == (15,)

    def test_zero_wind_gust_stays_zero(self):
        env = Plane()
        params = PlaneParams(max_steps_in_episode=50, turbulence_sigma=0.0, **FIXED)
        obs, state = env.reset_env(jax.random.PRNGKey(0), params)
        for _ in range(20):
            _, state, *_ = env.step_env(
                jax.random.PRNGKey(1), state, jnp.zeros(2), params
            )
        assert float(state.gust_x) == 0.0
        assert float(state.gust_z) == 0.0


class TestSteadyWindPhysics:
    def test_headwind_climbs_tailwind_sinks(self):
        env = Plane()
        base = _final_alt_2d(env, PlaneParams(max_steps_in_episode=300, **FIXED))[-1]
        head = _final_alt_2d(
            env, PlaneParams(max_steps_in_episode=300, wind_x=-40.0, **FIXED)
        )[-1]
        tail = _final_alt_2d(
            env, PlaneParams(max_steps_in_episode=300, wind_x=40.0, **FIXED)
        )[-1]
        assert float(head) > float(base) > float(tail)

    def test_wind_changes_trajectory_but_hidden_from_obs(self):
        env = Plane(observe_wind=False)
        p0 = PlaneParams(max_steps_in_episode=5, **FIXED)
        pw = PlaneParams(max_steps_in_episode=5, wind_x=-30.0, **FIXED)
        o0, s0 = env.reset_env(jax.random.PRNGKey(0), p0)
        ow, sw = env.reset_env(jax.random.PRNGKey(0), pw)
        # Same obs at reset (wind not observed) ...
        assert np.allclose(np.array(o0), np.array(ow))
        # ... but the dynamics diverge.
        _, s0b, *_ = env.step_env(jax.random.PRNGKey(0), s0, jnp.zeros(2), p0)
        _, swb, *_ = env.step_env(jax.random.PRNGKey(0), sw, jnp.zeros(2), pw)
        assert not np.isclose(float(s0b.z_dot), float(swb.z_dot))


class TestObservability:
    def test_observable_appends_wind_2d(self):
        env = Plane(observe_wind=True)
        params = PlaneParams(wind_x=-20.0, wind_z=3.0, **FIXED)
        obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
        assert obs.shape == (11,)
        assert np.allclose(np.array(obs[-2:]), [-20.0, 3.0])

    def test_observable_appends_wind_3d(self):
        env = Plane3D(observe_wind=True)
        params = PlaneParams3D(
            wind_x=-20.0,
            wind_y=10.0,
            wind_z=3.0,
            target_heading_range=(0.0, 0.0),
            **FIXED,
        )
        obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
        assert obs.shape == (18,)
        assert np.allclose(np.array(obs[-3:]), [-20.0, 10.0, 3.0])

    def test_hidden_is_prefix_of_observable(self):
        params = PlaneParams3D(wind_x=5.0, target_heading_range=(0.0, 0.0), **FIXED)
        oh, _ = Plane3D(observe_wind=False).reset_env(jax.random.PRNGKey(0), params)
        oo, _ = Plane3D(observe_wind=True).reset_env(jax.random.PRNGKey(0), params)
        assert np.allclose(np.array(oh), np.array(oo[:15]))


class TestTurbulence:
    def _gust_series(self, sigma, n=400):
        env = Plane()
        params = PlaneParams(
            max_steps_in_episode=n,
            turbulence_sigma=sigma,
            turbulence_theta=0.2,
            **FIXED,
        )
        obs, state = env.reset_env(jax.random.PRNGKey(0), params)

        def step(carry, _):
            s, k = carry
            k, ks = jax.random.split(k)
            _, ns, _, _, _ = env.step_env(ks, s, jnp.array([0.6, 0.0]), params)
            return (ns, k), ns.gust_x

        (s, _), g = jax.lax.scan(step, (state, jax.random.PRNGKey(0)), None, length=n)
        return np.array(g)

    def test_zero_sigma_no_gust(self):
        assert np.std(self._gust_series(0.0)) == 0.0

    def test_positive_sigma_produces_correlated_gust(self):
        g = self._gust_series(3.0)
        assert np.std(g) > 1.0  # gust is active
        # OU is temporally correlated, not white noise.
        assert np.corrcoef(g[:-1], g[1:])[0, 1] > 0.5

    def test_observable_wind_tracks_gust(self):
        env = Plane(observe_wind=True)
        params = PlaneParams(
            max_steps_in_episode=30, wind_x=0.0, turbulence_sigma=4.0, **FIXED
        )
        obs, state = env.reset_env(jax.random.PRNGKey(0), params)
        _, state, *_ = env.step_env(jax.random.PRNGKey(3), state, jnp.zeros(2), params)
        obs = env.get_obs(state, params)
        assert float(obs[-2]) == pytest.approx(float(state.gust_x))


class TestEngineBound:
    def test_patrol_inherits_steady_wind(self):
        env = PlanePatrol()

        def lead_y(wy, n=150):
            params = env.default_params.replace(
                max_steps_in_episode=n, lead_turn_rate_range=(0.0, 0.0), wind_y=wy
            )
            _, state = env.reset_env(jax.random.PRNGKey(1), params)

            def step(carry, _):
                (s,) = carry
                _, ns, _, _, _ = env.step_env(
                    jax.random.PRNGKey(1), s, jnp.zeros(3), params
                )
                return (ns,), ns.lead.y

            (s,), ys = jax.lax.scan(step, (state,), None, length=n)
            return float(ys[-1])

        assert not np.isclose(lead_y(0.0), lead_y(25.0), atol=1.0)


class TestWindShear:
    def test_linear_in_altitude(self):
        p = PlaneParams(wind_x=0.0, wind_shear_x=0.02, shear_ref_alt=5000.0)
        assert total_wind_2d(5000.0, 0.0, 0.0, p)[0] == pytest.approx(0.0)
        assert total_wind_2d(6000.0, 0.0, 0.0, p)[0] == pytest.approx(20.0)
        assert total_wind_2d(4000.0, 0.0, 0.0, p)[0] == pytest.approx(-20.0)

    def test_zero_shear_default(self):
        p = PlaneParams()
        assert total_wind_2d(8000.0, 0.0, 0.0, p)[0] == pytest.approx(0.0)

    def test_observable_wind_reflects_shear(self):
        env = Plane(observe_wind=True)
        params = PlaneParams(
            wind_x=0.0,
            wind_shear_x=0.02,
            shear_ref_alt=5000.0,
            target_altitude_range=(6000, 6000),
            initial_altitude_range=(6000, 6000),
        )
        obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
        assert float(obs[-2]) == pytest.approx(20.0, abs=1e-3)


class TestFormationTurbulence:
    def test_patrol_shared_gust(self):
        env = PlanePatrol()

        def gust_std(sigma):
            params = env.default_params.replace(
                max_steps_in_episode=250,
                lead_turn_rate_range=(0.0, 0.0),
                turbulence_sigma=sigma,
            )
            obs, state = env.reset_env(jax.random.PRNGKey(0), params)

            def step(carry, _):
                s, k = carry
                k, ks = jax.random.split(k)
                _, ns, _, _, _ = env.step_env(ks, s, jnp.zeros(3), params)
                # per-plane gusts stay 0; the shared formation gust carries it
                return (ns, k), (ns.gust_x, ns.follower.gust_x)

            (s, _), (g, fg) = jax.lax.scan(
                step, (state, jax.random.PRNGKey(0)), None, length=250
            )
            return float(np.std(np.array(g))), float(np.std(np.array(fg)))

        shared, per_plane = gust_std(4.0)
        assert shared > 1.0  # turbulence reaches the formation
        assert per_plane == 0.0  # shared, not per-aircraft
        assert gust_std(0.0)[0] == 0.0  # sigma=0 -> no turbulence

    def test_marl_formation_turbulence(self):
        env = PlanePatrolMARL(num_wingmen=4)
        params = env.default_params.replace(
            max_steps_in_episode=150, turbulence_sigma=4.0
        )
        obs, state = env.reset(jax.random.PRNGKey(0), params)
        zero = {a: jnp.zeros(3) for a in env.agents}

        def step(carry, _):
            s, k = carry
            k, ks = jax.random.split(k)
            _, ns, _, _, _ = env.step_env(ks, s, zero, params)
            return (ns, k), ns.gust_y

        (s, _), g = jax.lax.scan(step, (state, jax.random.PRNGKey(0)), None, length=150)
        assert float(np.std(np.array(g))) > 1.0
