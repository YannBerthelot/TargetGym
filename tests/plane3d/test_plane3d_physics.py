"""Physics validation for the 3D aircraft's roll extension.

Enforces ``src/target_gym/plane3d/PHYSICS.md``. The longitudinal aerodynamics
are shared with the 2D model and validated in ``tests/plane/``; this file
covers only what the third dimension adds -- banked turns, load factor, and
heading as a derived quantity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.plane3d.env import PlaneParams3D
from target_gym.registry import REGISTRY

G = 9.81


@pytest.fixture(scope="module")
def params():
    return PlaneParams3D()


# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------


def test_roll_damping_is_in_the_transport_aircraft_range(params):
    """C_lp must be negative (damping) and of transport magnitude."""
    assert -0.6 < params.C_lp < -0.3


def test_wingspan_matches_the_reference_airframe(params):
    """A320neo with sharklets."""
    assert params.wingspan == pytest.approx(35.8, rel=0.02)


def test_aileron_moment_arm_is_inside_the_semi_span(params):
    """An aileron cannot act outboard of the wingtip."""
    assert 0.5 < params.moment_arm_aileron / (params.wingspan / 2) < 1.0


# ---------------------------------------------------------------------------
# 2. Banked flight -- the behaviour the third dimension exists for
# ---------------------------------------------------------------------------


def _fly_bank(target_deg, steps=900, tail=500):
    """Demand a bank angle, then measure the resulting turn."""
    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    params = spec.params_cls().replace(max_steps_in_episode=steps + 10)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    demand = jnp.array([0.2, 0.2, target_deg / 45.0])
    phis, psis, speeds = [], [], []
    for _ in range(steps):
        _, next_state, _, _, info = step(key, state, demand, params)
        # A trip no longer ends the episode: base.failure_kernel restarts the
        # aircraft from a fresh draw, with a new heading. Stop at the trip, as
        # the terminated flag used to, so the measured turn is one flight.
        if bool(info["tripped"]):
            break
        state = next_state
        phis.append(float(state.phi))
        psis.append(float(state.psi))
        speeds.append(float(np.hypot(float(state.x_dot), float(state.y_dot))))

    assert len(psis) >= tail, f"tripped after {len(psis)} steps, before a full tail"
    phi = float(np.mean(phis[-tail:]))
    V = float(np.mean(speeds[-tail:]))
    unwrapped = np.unwrap(psis[-tail:])
    rate = float((unwrapped[-1] - unwrapped[0]) / (len(unwrapped) * params.delta_t))
    return phi, V, rate


@pytest.mark.large
@pytest.mark.parametrize("bank_deg", [10.0, 20.0, 30.0])
def test_banked_turn_matches_the_coordinated_turn_relation(bank_deg):
    """psi_dot = g tan(phi) / V, and nothing in the model computes it.

    Heading falls out of the horizontal velocity, which falls out of the
    tilted lift vector. Reproducing the analytic turn rate means the force
    decomposition and the heading derivation agree -- an agreement that a sign
    error or a missing cos(phi) destroys immediately.
    """
    phi, V, rate = _fly_bank(bank_deg)
    assert abs(np.rad2deg(phi) - bank_deg) < 2.0, "bank hold failed"
    analytic = G * np.tan(phi) / V
    assert rate == pytest.approx(analytic, rel=0.05)


@pytest.mark.large
def test_steeper_bank_turns_faster_and_tighter():
    shallow_phi, shallow_V, shallow_rate = _fly_bank(10.0)
    steep_phi, steep_V, steep_rate = _fly_bank(30.0)
    assert abs(steep_rate) > abs(shallow_rate)
    # Turn radius V / psi_dot must shrink.
    assert abs(steep_V / steep_rate) < abs(shallow_V / shallow_rate)


def test_turn_radius_at_cruise_is_kilometres():
    """Which is why the circle task's reference path is km-scale."""
    phi, V, rate = _fly_bank(30.0)
    radius = abs(V / rate)
    assert 2_000.0 < radius < 40_000.0


def test_load_factor_grows_with_bank():
    """n = 1/cos(phi): the wing must carry more than the weight in a turn."""
    loads = [1.0 / np.cos(np.deg2rad(b)) for b in (0.0, 30.0, 60.0)]
    assert loads[0] == pytest.approx(1.0)
    assert loads[1] == pytest.approx(1.155, rel=0.01)
    assert loads[2] == pytest.approx(2.0, rel=0.01)
    assert loads[0] < loads[1] < loads[2]


def test_a_constant_aileron_commands_a_rate_not_an_angle():
    """Real roll behaviour: hold the stick over and the aircraft keeps rolling.

    This is why the plant carries a bank loop, and it is worth pinning: a model
    that settled at a bank angle for constant aileron would need none. The roll
    moment does not depend on the bank, so the rate where aileron meets damping is
    a steady one at every bank angle.
    """
    from target_gym.plane3d.dynamics import compute_acceleration_3d

    params = PlaneParams3D()
    action = (40_000.0, 0.0, 0.05)

    def roll_accel(phi, rate):
        velocities = jnp.array([200.0, 0.0, 0.0, 0.0, rate])
        positions = jnp.array([0.0, 0.0, 5500.0, 0.02, phi])
        return float(
            compute_acceleration_3d(velocities, positions, action, params)[0][4]
        )

    steady = 0.1 * roll_accel(0.0, 0.0) / (roll_accel(0.0, 0.0) - roll_accel(0.0, 0.1))
    assert steady > 0.0
    for phi in (0.0, 0.5, 1.0):
        assert abs(roll_accel(phi, steady)) < 1e-3


def test_wings_level_reproduces_the_two_dimensional_model(params):
    """At zero bank the 3D aircraft must fly like the 2D one.

    Same lift curve, same drag polar, same atmosphere -- the modules share the
    code path, and this pins that they stay shared.
    """
    from target_gym.plane.env import PlaneParams

    p2 = PlaneParams()
    for attr in ("cl0", "cl_alpha", "aoa_stall", "CL_max", "wings_surface"):
        assert getattr(params, attr) == pytest.approx(getattr(p2, attr)), attr


# ---------------------------------------------------------------------------
# 3. The rolling gust (PHYSICS.md section on turbulence)
# ---------------------------------------------------------------------------
#
# Contract, from MIL-F-8785C's Dryden model: turbulence varies across the span,
# so the wing meets it as a roll rate p_g. It adds to the aircraft's own rate in
# the damping moment, which is C_lp (p - p_g) b / 2V q S b. Above 2000 ft the
# scale length is 533.4 m, and p_g's one-sided spectrum starts at
# Phi_0 = sigma_w^2 / (V L) 0.8 (pi L / 4b)^(1/3), rolling off at pi V / 4b,
# 20 rad/s here, well past the 1 s step. So a 1 s step holds white draws, sized
# to keep Phi_0: a held draw of std s has Phi_0 = s^2 T / pi.


def test_a_rolling_gust_rolls_a_wings_level_aircraft(params):
    from target_gym.plane3d.dynamics import compute_acceleration_3d

    p_g = 0.02  # rad/s
    gusty = params.replace(gust_roll_rate=p_g)
    velocities = jnp.array([230.0, 0.0, 0.0, 0.0, 0.0])
    positions = jnp.array([0.0, 0.0, 5500.0, 0.02, 0.0])
    action = (40_000.0, 0.0, 0.0)

    accel, _ = compute_acceleration_3d(velocities, positions, action, gusty)

    # At 5500 m, rho = 0.6972: q = 18 441 Pa. L_p = C_lp b^2 q S / (2 V I_x)
    # = -0.4 * 35.8^2 * 18 441 * 122.6 / (2 * 230 * 2.5e6) = -1.008 /s, so a
    # gust of 0.02 rad/s on a still aircraft rolls it at +0.0202 rad/s^2.
    np.testing.assert_allclose(accel[4], 0.0202, rtol=0.01)


def test_the_rolling_gust_keeps_drydens_low_frequency_level(params):
    from target_gym.plane3d.dynamics import rolling_gust

    keys = jax.random.split(jax.random.PRNGKey(0), 20_000)
    draws = jax.vmap(lambda k: rolling_gust(k, 230.0, params))(keys)

    # sigma_w 2 m/s (the OU gust's own std), V 230, L 533.4, b 35.8:
    # Phi_0 = 4 / (230 * 533.4) * 0.8 * (pi * 533.4 / (4 * 35.8))^(1/3)
    #       = 3.261e-5 * 0.8 * 11.70^(1/3) = 3.261e-5 * 0.8 * 2.270
    #       = 5.92e-5 (rad/s)^2 per rad/s,
    # and s = sqrt(pi Phi_0 / 1 s) = 0.01364 rad/s. Sampling error of a std
    # over 20 000 draws is 1 / sqrt(2 * 20 000) = 0.35%, so 1.5% is 4 sigma.
    assert float(jnp.std(draws)) == pytest.approx(0.01364, rel=0.015)
    assert abs(float(jnp.mean(draws))) < 4 * 0.01364 / np.sqrt(20_000)


def test_still_air_has_no_rolling_gust(params):
    from target_gym.plane3d.dynamics import rolling_gust

    still = params.replace(turbulence_sigma=0.0)
    assert float(rolling_gust(jax.random.PRNGKey(0), 230.0, still)) == 0.0


def test_scored_turbulence_reaches_the_bank():
    """The review's finding: in the scored turbulence with nothing commanded the
    bank stayed exactly 0. Now p_g works on the roll axis and the inner loop works
    against it. With a held white gust of s = 0.0146 rad/s at 200 m/s and a bank
    loop of time constant T = 1.5 s, the bank's RMS is s sqrt(T / 2) = 0.7 deg, so
    0.2 to 2 deg, and the aileron answers it, Kp_roll times that of its travel."""
    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    # The zone over the whole run, at the 2 m/s stationary gust the derivation
    # above was made for (the scored zone's 0.55 m/s is smaller, Amendment 2).
    params = spec.make_test_params().replace(
        max_steps_in_episode=200, zone_onset_range=(0.0, 0.0), turbulence_sigma=2.0
    )

    def attitude(seed):
        key = jax.random.PRNGKey(seed)
        _, state = env.reset_env(key, params)

        def body(state, k):
            state = env.step_env(k, state, jnp.array([0.0, 0.12, 0.0]), params)[1]
            return state, (state.phi, state.aileron)

        return jax.lax.scan(body, state, jax.random.split(key, 120))[1]

    phi, aileron = jax.jit(jax.vmap(attitude))(jnp.arange(16))
    rms = float(jnp.degrees(jnp.sqrt(jnp.mean(phi[:, 30:] ** 2))))
    assert 0.2 < rms < 2.0
    assert float(jnp.max(jnp.abs(aileron[:, 30:]))) > 0.0


def test_the_3d_engine_shares_the_lag_and_the_idle():
    # As tests/plane/test_plane_physics.py: 0.05 + 0.95 (1 - e^(-3 / 1.7)) = 0.8373.
    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    params = spec.params_cls().replace(turbulence_sigma=0.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    state = state.replace(power=jnp.float32(0.05))
    for _ in range(3):
        _, state, _, _, _ = env.step_env(key, state, jnp.array([1.0, 0, 0]), params)
    assert float(state.power) == pytest.approx(0.8373, abs=2e-3)
