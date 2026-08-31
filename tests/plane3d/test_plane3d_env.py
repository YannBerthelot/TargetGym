"""Tests for the 3D plane environment (all task variants)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane3d.env import (
    PlaneParams3D,
    check_is_terminal_3d,
    compute_reward_circle,
    compute_reward_figure8,
    compute_reward_heading,
    distance_to_circle,
    nearest_point_on_twisted_lemniscate,
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
        obs2, state2, reward, terminated, truncated, info = env.step(key, state, action)
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
        _jstep = jax.jit(env.step_env)
        for _ in range(10):
            _, state, *_ = _jstep(key, state, action)
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
        _jstep = jax.jit(env.step_env)
        for _ in range(50):
            _, state, *_ = _jstep(key, state, action, params)
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
        _jstep = jax.jit(env.step_env)
        for _ in range(50):
            _, state, *_ = _jstep(key, state, action, params)
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


def test_roll_authority_matches_a_transport_aircraft():
    """Full aileron must roll at a transport's rate, not a fighter's.

    This was PHYSICS.md D2 -- "aileron authority is not calibrated" -- and it
    was not merely unvalidated but wrong by a factor of three. The moment
    applied the *wing's* lift-curve slope directly to the aileron deflection,
    implying a section lift change of 2.20 at full throw: larger than the whole
    wing's CL_max of 1.5, which no control surface can do. The aircraft rolled
    at 84 deg/s.

    A deflected surface only turns the hinged part of the chord, so the
    effective incidence change is tau*delta with tau ~ 0.4 for a quarter-chord
    surface. With that, full aileron gives ~32 deg/s, against an A320's 15 deg/s
    in normal law and 25-30 deg/s of raw aileron authority.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from target_gym.registry import REGISTRY

    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    params = spec.params_cls(max_steps_in_episode=200)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    rates = []
    for _ in range(200):
        _, state, _, terminated, _ = step(
            key, state, jnp.array([0.0, 0.0, 1.0]), params
        )
        rates.append(abs(float(np.rad2deg(state.phi_dot))))
        if bool(terminated):
            break

    steady = float(np.median(rates[-20:]))
    assert 20.0 < steady < 45.0, (
        f"full aileron rolls at {steady:.0f} deg/s; a transport aircraft manages "
        "25-30, and a value far above that means the surface is being credited "
        "with more lift change than the wing can produce"
    )


def test_aileron_cannot_out_lift_the_wing():
    """The consistency check behind the number above.

    A control surface changes the section's effective incidence by tau*delta,
    so the lift it commands must stay inside what the aerofoil can make.
    """
    from target_gym.plane3d.env import PlaneParams3D

    p = PlaneParams3D()
    full_deflection_deg = 25.0
    delta_cl = p.cl_alpha * p.aileron_effectiveness * full_deflection_deg
    assert delta_cl < p.CL_max, (
        f"full aileron commands a section lift change of {delta_cl:.2f} against "
        f"a CL_max of {p.CL_max}"
    )


# ─── Reward precision (see docs/model-review-checklist.md, check 1) ─────


def _heading_reward_at(alt_err=0.0, hdg_err=0.0):
    """Heading-task reward with one axis perturbed and the other on target."""
    _, state = Plane3DHeading().reset(jax.random.PRNGKey(0))
    state = state.replace(
        z=state.target_altitude + alt_err, psi=state.target_heading + hdg_err
    )
    return float(compute_reward_heading(state, PlaneParams3D()))


def _circle_reward_at(cross_track):
    """Circle-task reward at a known radial offset, at the target altitude."""
    _, state = Plane3DCircle().reset(jax.random.PRNGKey(0))
    state = state.replace(
        x=state.target_x + state.target_radius + cross_track,
        y=state.target_y,
        z=state.target_altitude,
    )
    return float(compute_reward_circle(state, PlaneParams3D()))


def _assert_scale_free(reward_at, pairs, axis):
    """Every halving of the error must be worth about the same."""
    gains = [reward_at(b) - reward_at(a) for a, b in pairs]
    assert min(gains) > 0.75 * max(
        gains
    ), f"{axis} reward per halving is not scale-free: " + ", ".join(
        f"{g:.4f}" for g in gains
    )


class TestRewardKeepsPayingForPrecision:
    """The property the benchmark exists to measure.

    TargetGym asks whether a learned policy can hold a setpoint better than a
    PID, so every tracking reward has to keep discriminating all the way down
    to sensor resolution.  These tasks previously normalised the error by the
    *state-space envelope* (``(1 - e/12 km)**10``) and by a Gaussian a tenth of
    the path radius wide, which spent the whole dynamic range on errors the
    controller had already eliminated: the gap between a 10 m and a 1 m
    altitude error was 0.0008, and between a 10 m and a 1 m cross-track error
    it was 5e-5.  Both are invisible next to a return of order 1.
    """

    def test_altitude_error_is_scale_free(self):
        _assert_scale_free(
            lambda e: _heading_reward_at(alt_err=e),
            [(1600, 800), (400, 200), (100, 50), (25, 12.5), (6.25, 3.125)],
            "altitude",
        )

    def test_heading_error_is_scale_free(self):
        _assert_scale_free(
            lambda e: _heading_reward_at(hdg_err=jnp.deg2rad(e)),
            [(64, 32), (16, 8), (4, 2)],
            "heading",
        )

    def test_cross_track_error_is_scale_free(self):
        _assert_scale_free(
            _circle_reward_at,
            [(1600, 800), (400, 200), (100, 50), (25, 12.5)],
            "cross-track",
        )

    def test_metre_scale_tracking_is_visible(self):
        """A tenfold improvement must move the reward by more than rounding."""
        alt_gain = _heading_reward_at(alt_err=1.0) - _heading_reward_at(alt_err=10.0)
        assert (
            alt_gain > 0.1
        ), f"altitude reward barely sees a 10 m -> 1 m improvement: {alt_gain:.4f}"
        track_gain = _circle_reward_at(1.0) - _circle_reward_at(10.0)
        assert (
            track_gain > 0.1
        ), f"cross-track reward barely sees a 10 m -> 1 m improvement: {track_gain:.4f}"

    def test_reward_still_orders_states_far_from_the_path(self):
        """A drifting agent must still be able to tell it is drifting.

        The Gaussian this replaced underflowed to identically zero out here,
        leaving nothing but the terminal penalty to follow home.
        """
        near, far = _circle_reward_at(1600.0), _circle_reward_at(4000.0)
        assert near > far > 0.0, f"no ordering far out: {near:.3e} vs {far:.3e}"


class TestLemniscateDistanceResolution:
    def test_distance_is_not_quantised_by_the_curve_sampling(self):
        """An aircraft flying the curve exactly must be told it is on it.

        The nearest-point search is an argmin over 400 samples of a ~44 km
        curve, so without sub-sample refinement it reported a perfectly flown
        aircraft as up to 66 m off -- above the figure-8 expert's own settled
        error, meaning the reward was scoring its own discretisation rather
        than the controller.
        """
        from target_gym.plane3d.env import _sample_twisted_lemniscate

        params = PlaneParams3D()
        _, state = Plane3DFigureEight().reset(jax.random.PRNGKey(0))
        cx, cy, cz = _sample_twisted_lemniscate(state, params)
        for i in (0, 50, 100, 150, 199):
            midpoint = state.replace(
                x=(cx[i] + cx[i + 1]) / 2,
                y=(cy[i] + cy[i + 1]) / 2,
                z=(cz[i] + cz[i + 1]) / 2,
            )
            dist = float(nearest_point_on_twisted_lemniscate(midpoint, params)[3])
            assert dist < 1.0, f"on-curve point at sample {i} reported {dist:.1f} m off"

    def test_known_offset_is_measured_accurately(self):
        from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

        params = PlaneParams3D()
        _, state = Plane3DFigureEight().reset(jax.random.PRNGKey(0))
        for offset in (100.0, 10.0, 1.0):
            dist = float(
                nearest_point_on_twisted_lemniscate(
                    state.replace(z=state.z + offset), params
                )[3]
            )
            assert dist == pytest.approx(
                offset, rel=0.01
            ), f"{offset} m vertical offset measured as {dist:.3f} m"


_PATH_FOLLOWING_XFAIL = (
    "The path guidance laws do not hold their path. Over three laps the circle "
    "expert wanders 640-1670 m from an 8.4 km circle without ever settling, and "
    "the figure-8 expert is 6-12 km from a curve whose lobes are 8.4 km across, "
    "i.e. not following it at all. Their altitude loops are fine (0.2-1.5 m), "
    "which is what the tuning runs measured; the cross-track error was never "
    "measured. Nothing else sees this: every other test runs the 200-step "
    "episode from EnvSpec.test_params, which is 200 s against a 264 s lap, and "
    "the aircraft is initialised exactly on the path -- so a controller that "
    "simply flies straight ahead looks correct for the whole episode. This is a "
    "guidance-law fault, not a gains fault (see docs/model-review-checklist.md "
    "check 10), and is tracked as an open item there."
)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason=_PATH_FOLLOWING_XFAIL)
@pytest.mark.parametrize(
    "make_env, error_fn",
    [
        (Plane3DCircle, lambda s, p: abs(float(distance_to_circle(s)))),
        (
            Plane3DFigureEight,
            lambda s, p: float(nearest_point_on_twisted_lemniscate(s, p)[3]),
        ),
    ],
    ids=["circle", "figure8"],
)
def test_path_experts_hold_their_path_for_three_laps(make_env, error_fn):
    """A path-following expert must still be on its path after a few laps.

    The bar is 100 m -- about 1% of the path radius, and far looser than the
    3 m position floor the reward is scaled to -- so it fails only a controller
    that is not following the path at all.
    """
    from target_gym.registry import REGISTRY

    spec = REGISTRY[
        "plane3d_circle" if make_env is Plane3DCircle else "plane3d_figure8"
    ]
    params = PlaneParams3D(max_steps_in_episode=800)
    env = spec.make_env()
    step = jax.jit(env.step_env)
    pid = spec.make_pid()
    pid.reset()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    errors = []
    for _ in range(int(params.max_steps_in_episode)):
        obs, state, _, terminated, _ = step(
            key, state, jnp.atleast_1d(jnp.asarray(pid(obs))), params
        )
        errors.append(error_fn(state, params))
        if bool(terminated):
            break

    settled = float(np.mean(errors[int(0.8 * len(errors)) :]))
    assert settled < 100.0, f"settled {settled:.0f} m from the commanded path"
