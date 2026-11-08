"""The fixed inner loop (docs/proposals/industrial-fixes.md, PR 7).

Contract, from the plan and the accepted attitude procedure
(docs/proposals/expert-protocols-aircraft.md, sections 1 to 3.2):

- An action is a 1 Hz demand: full action is 15 deg of pitch and 45 deg of bank.
  The plant flies the surfaces to it at 4 Hz, and a policy never touches them.
- One axis is ``kp s err + integral - kd s^2 rate``, clipped to the surface's
  travel, with ArduPilot's airspeed scaler ``s`` on the angle and squared on the
  rate. The integral freezes while the surface is saturated.
- A surface lags by 0.06 s and moves at no more than 30 deg/s.
- On the tuning test (230 m/s at 5500 m, still air) a step settles and does not
  oscillate. The recipe's reading of that is section 2's: an overshoot under 22%.
  It does not promise 5% on the shipped loop, because step 4 halves I and does
  not fly the result.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane.dynamics import (
    BANK_DEMAND_MAX,
    PITCH_DEMAND_MAX,
    STICK_TRAVEL,
    InnerLoop,
    airspeed_scaler,
    attitude_axis,
    compute_acceleration,
    compute_next_surface,
)
from target_gym.plane.env import compute_next_state
from target_gym.plane.env_jax import Airplane2D
from target_gym.plane3d.env import compute_next_state_3d
from target_gym.plane3d.env_jax import Plane3DCircle, Plane3DHeading

KEY = jax.random.PRNGKey(0)


def _still_3d():
    # The unamended 3D plant, whose bank demand reaches 45 deg: the amended
    # tasks clip it at the protected 30 deg (tests/plane3d/test_plane3d_reality.py).
    env = Plane3DCircle()
    params = env.default_params.replace(turbulence_sigma=0.0, sensor_noise_scale=0.0)
    _, state = env.reset_env(KEY, params)
    state = state.replace(
        z=jnp.float32(5500.0),
        target_altitude=jnp.float32(5500.0),
        x_dot=jnp.float32(230.0),
        y_dot=jnp.float32(0.0),
        power=jnp.float32(0.55),
        psi=jnp.float32(0.0),
        stick=jnp.float32(np.deg2rad(-5.5)),
        pitch_integ=jnp.float32(-0.36),
    )
    return env, params, state


def _fly(env, params, state, action, seconds):
    def body(s, _):
        s = env.step_env(KEY, s, action, params)[1]
        return s, (s.theta, s.phi)

    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=seconds))(state)


# -- One axis -----------------------------------------------------------------


def test_the_scaler_is_one_at_the_tuning_speed_and_clamped_either_side():
    gains = InnerLoop()
    assert float(airspeed_scaler(173.5, gains)) == pytest.approx(1.0)
    # 173.5 / 104 at the slowest the autopilot flies, within the clamp.
    assert float(airspeed_scaler(104.0, gains)) == pytest.approx(1.668, abs=1e-3)
    assert float(airspeed_scaler(50.0, gains)) == pytest.approx(2.0)
    assert float(airspeed_scaler(1000.0, gains)) == pytest.approx(0.3)


def test_the_scaler_acts_once_on_the_angle_and_squared_on_the_rate():
    # 2 * (2 * 0.1) + 0 - 0.5 * 4 * 0.1 = 0.4 - 0.2 = 0.2.
    surface, _ = attitude_axis(2.0, 0.0, 0.5, 0.1, 0.1, 0.0, 2.0, 0.25)
    assert float(surface) == pytest.approx(0.2)


def test_the_integral_grows_on_the_error_and_freezes_at_the_stop():
    # 1 * 1 * 0.1 + 0.3 = 0.4, unsaturated, so 0.3 + 0.2 * 1 * 0.1 * 0.25.
    _, grown = attitude_axis(1.0, 0.2, 0.0, 0.1, 0.0, 0.3, 1.0, 0.25)
    assert float(grown) == pytest.approx(0.305)
    # 1 * 1 * 2 = 2, past the stop.
    surface, held = attitude_axis(1.0, 0.2, 0.0, 2.0, 0.0, 0.3, 1.0, 0.25)
    assert float(surface) == 1.0
    assert float(held) == pytest.approx(0.3)


def test_a_surface_lags_and_then_hits_its_rate_limit():
    limit = np.deg2rad(30.0)
    # A small move is the lag's: 1 - exp(-0.25 / 0.06) = 0.9845 of it.
    small = compute_next_surface(0.01, 0.0, 0.25, 1.0 / 0.06, limit)
    assert float(small) == pytest.approx(0.009845, rel=1e-3)
    # A large one is the rate's: 30 deg/s for 0.25 s is 7.5 deg.
    large = compute_next_surface(1.0, 0.0, 0.25, 1.0 / 0.06, limit)
    assert float(large) == pytest.approx(np.deg2rad(7.5), rel=1e-5)


# -- The loop in the plant ----------------------------------------------------


def test_a_full_bank_demand_is_45_degrees():
    env, params, state = _still_3d()
    state, _ = _fly(env, params, state, jnp.array([0.2, 0.12, 1.0]), 30)
    assert float(state.phi) == pytest.approx(BANK_DEMAND_MAX, abs=np.deg2rad(2.25))


def test_a_pitch_demand_is_held_in_degrees():
    """A demand of 5 deg settles within the droop Kp allows: the trim the loop
    holds is under 0.4 of the stick, so 0.4 / 6.37 = 3.6 deg at the very most."""
    env, params, state = _still_3d()
    demand = np.deg2rad(5.0) / PITCH_DEMAND_MAX
    state, _ = _fly(env, params, state, jnp.array([0.6, demand, 0.0]), 20)
    assert abs(float(jnp.degrees(state.theta)) - 5.0) < 3.6


def test_a_bank_step_settles_and_does_not_oscillate():
    """The recipe's own test: 45 deg, held 30 s, in still air."""
    env, params, state = _still_3d()
    _, (_, phi) = _fly(env, params, state, jnp.array([0.2, 0.12, 1.0]), 30)
    bank = np.degrees(np.asarray(phi))
    assert bank[:15].max() < 45.0 * 1.22
    assert abs(bank[-1] - 45.0) < 45.0 * 0.05


def test_the_roll_integral_trims_out_a_steady_rolling_gust():
    """A steady p_g = 0.02 rad/s holds a droop of L_p p_g / (L_a Kp) with no
    integral, 1.01 * 0.02 / (0.55 * 0.891) = 0.041 rad, 2.4 deg. Kp / Ki is 28 s,
    so 120 s takes off 98%, leaving 0.05 deg, and 0.3 deg is well clear of it."""
    env, params, state = _still_3d()
    gusty = params.replace(gust_roll_rate=0.02)
    state, _ = _fly(env, gusty, state, jnp.array([0.2, 0.12, 0.0]), 120)
    assert abs(float(jnp.degrees(state.phi))) < 0.3
    assert abs(float(state.roll_integ)) > 0.0


def test_the_integrals_survive_a_step_in_both_models():
    # compute_next_state rebuilds its state field by field.
    env2, env3 = Airplane2D(), Plane3DHeading()
    p2 = env2.default_params.replace(turbulence_sigma=0.0)
    p3 = env3.default_params.replace(turbulence_sigma=0.0)
    _, s2 = env2.reset_env(KEY, p2)
    _, s3 = env3.reset_env(KEY, p3)
    a2, b2 = jnp.array([0.0, 0.5]), s2.replace(pitch_integ=jnp.float32(0.2))
    a3 = jnp.array([0.0, 0.5, 0.5])
    b3 = s3.replace(pitch_integ=jnp.float32(0.2), roll_integ=jnp.float32(-0.1))
    n2 = env2.step_env(KEY, b2, a2, p2)[1]
    n3 = env3.step_env(KEY, b3, a3, p3)[1]
    assert 0.2 < float(n2.pitch_integ) < 0.3
    assert 0.2 < float(n3.pitch_integ) < 0.3
    assert -0.1 < float(n3.roll_integ) < 0.0


def test_a_step_is_its_four_sub_steps():
    """No 1 s hold is left: one step of four sub-steps is four steps of one."""
    env, params, state = _still_3d()
    demands = (0.6, 0.1, 0.3)
    whole, _ = compute_next_state_3d(*demands, state, params)
    quarter = params.replace(delta_t=0.25)
    parts = state
    for _ in range(4):
        parts, _ = compute_next_state_3d(*demands, parts, quarter, inner_steps=1)
    for field in ("z", "theta", "phi", "stick", "aileron", "power"):
        a, b = getattr(whole, field), getattr(parts, field)
        np.testing.assert_allclose(float(a), float(b), rtol=1e-4, atol=1e-4)


def test_the_2d_step_is_also_its_four_sub_steps():
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(KEY, params)
    whole, _ = compute_next_state(0.6, 0.05, state, params)
    quarter = params.replace(delta_t=0.25)
    parts = state
    for _ in range(4):
        parts, _ = compute_next_state(0.6, 0.05, parts, quarter, inner_steps=1)
    for field in ("z", "theta", "stick", "power"):
        a, b = getattr(whole, field), getattr(parts, field)
        np.testing.assert_allclose(float(a), float(b), rtol=1e-4, atol=1e-4)


def test_a_policy_does_not_touch_the_surface():
    """The action is a demand, so the aileron moves on the bank error: at the
    demanded bank and at rest, a full action on the old scale asks for nothing."""
    env, params, state = _still_3d()
    level = state.replace(
        phi=jnp.float32(BANK_DEMAND_MAX),
        aileron=jnp.float32(0.0),
        phi_dot=jnp.float32(0.0),
    )
    new = env.step_env(KEY, level, jnp.array([0.2, 0.12, 1.0]), params)[1]
    # The old aileron would have gone to 25 deg within the step.
    assert abs(float(jnp.degrees(new.aileron))) < 5.0


# -- Engaged in trim ----------------------------------------------------------
#
# An autopilot engages trimmed. So a reset aircraft has no pitching moment to shed:
# its elevator, and the integral that holds it, already balance the pitching moment
# at its reset attitude and speed, wherever in the envelope it starts.


@pytest.mark.parametrize("env_cls", (Airplane2D, Plane3DHeading))
@pytest.mark.parametrize("seed", range(4))
def test_a_reset_aircraft_has_no_pitching_moment(env_cls, seed):
    env = env_cls()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(jax.random.PRNGKey(seed), params)
    velocities = jnp.array([state.x_dot, state.z_dot, 0.0])
    if env_cls is Plane3DHeading:
        velocities = velocities.at[0].set(jnp.hypot(state.x_dot, state.y_dot))
    positions = jnp.array([0.0, state.z, state.theta])
    accel, _ = compute_acceleration(velocities, positions, (0.0, state.stick), params)
    # The elevator moves the moment by about 2 rad/s^2 per rad: 0.01 rad/s^2 is a
    # twentieth of a degree of stick, against the 5 deg it used to start from.
    assert abs(float(accel[2])) < 0.01


def test_the_integral_holds_the_trim_it_starts_from():
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(KEY, params)
    assert float(state.pitch_integ) * STICK_TRAVEL == pytest.approx(float(state.stick))


@pytest.mark.parametrize("seed", range(4))
def test_a_small_demand_after_reset_does_not_jolt_the_aircraft(seed):
    """The review's limit is 2.5 g; a 1.8 deg demand at reset used to pull 1.9 g
    in its first second at 3800 m, which is an untrimmed elevator, not the demand.
    A 1.8 deg change of incidence is worth Delta CL 0.16, about 0.5 g at most."""
    env = Airplane2D()
    params = env.default_params.replace(turbulence_sigma=0.0)
    _, state = env.reset_env(jax.random.PRNGKey(seed), params)
    new = env.step_env(KEY, state, jnp.array([0.0, 0.12]), params)[1]
    assert float(new.load_factor) < 1.5
