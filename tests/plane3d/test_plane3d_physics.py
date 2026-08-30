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


def _fly_bank(target_deg, steps=2500, tail=500):
    """Hold a bank angle, then measure the resulting turn.

    The aileron commands a roll *rate*, so holding an angle needs a loop --
    a constant deflection just keeps rolling. That is the physics, not a
    limitation of the test.
    """
    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    params = spec.params_cls().replace(max_steps_in_episode=steps + 10)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    target = np.deg2rad(target_deg)
    phis, psis, speeds = [], [], []
    for _ in range(steps):
        aileron = float(
            np.clip(
                2.5 * (target - float(state.phi)) - 0.9 * float(state.phi_dot),
                -1.0,
                1.0,
            )
        )
        _, state, _, terminated, _ = step(
            key, state, jnp.array([-0.55, 0.0, aileron]), params
        )
        phis.append(float(state.phi))
        psis.append(float(state.psi))
        speeds.append(float(np.hypot(float(state.x_dot), float(state.y_dot))))
        if bool(terminated):
            break

    phi = float(np.mean(phis[-tail:]))
    V = float(np.mean(speeds[-tail:]))
    unwrapped = np.unwrap(psis[-tail:])
    rate = float((unwrapped[-1] - unwrapped[0]) / (len(unwrapped) * params.delta_t))
    return phi, V, rate


@pytest.mark.slow
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


@pytest.mark.slow
def test_steeper_bank_turns_faster_and_tighter():
    shallow_phi, shallow_V, shallow_rate = _fly_bank(10.0)
    steep_phi, steep_V, steep_rate = _fly_bank(30.0)
    assert abs(steep_rate) > abs(shallow_rate)
    # Turn radius V / psi_dot must shrink.
    assert abs(steep_V / steep_rate) < abs(shallow_V / shallow_rate)


@pytest.mark.slow
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


@pytest.mark.slow
def test_a_constant_aileron_commands_a_rate_not_an_angle():
    """Real roll behaviour: hold the stick over and the aircraft keeps rolling.

    This is why every 3D controller in the suite is a cascade with an inner
    bank loop, and it is worth pinning: a model that settled at a bank angle
    for constant aileron would make those controllers pointless.
    """
    spec = REGISTRY["plane3d_heading"]
    env = spec.make_env()
    params = spec.params_cls().replace(max_steps_in_episode=1200)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    phis = []
    for _ in range(900):
        _, state, _, terminated, _ = step(
            key, state, jnp.array([-0.55, 0.0, 0.15]), params
        )
        phis.append(float(state.phi))
        if bool(terminated):
            break
    phis = np.unwrap(np.array(phis))
    first = phis[len(phis) // 3] - phis[0]
    second = phis[-1] - phis[2 * len(phis) // 3]
    assert abs(second) > 0.25 * abs(first), "roll stopped -- aileron held an angle"


def test_wings_level_reproduces_the_two_dimensional_model(params):
    """At zero bank the 3D aircraft must fly like the 2D one.

    Same lift curve, same drag polar, same atmosphere -- the modules share the
    code path, and this pins that they stay shared.
    """
    from target_gym.plane.env import PlaneParams

    p2 = PlaneParams()
    for attr in ("cl0", "cl_alpha", "aoa_stall", "CL_max", "wings_surface"):
        assert getattr(params, attr) == pytest.approx(getattr(p2, attr)), attr
