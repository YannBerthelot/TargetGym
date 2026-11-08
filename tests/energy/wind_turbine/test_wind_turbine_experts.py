"""The wind turbine's oracle, on the shared solver.

Its claims, each checked below without running a solve (charter rule 9's room
on the v4 plant is not measured yet). A command past either actuator's reach
moves it as a command at it does (T1r). The rotor speed is the one spare degree
of freedom, so it rests at rated, inside the tie's budget (G12 in
docs/proposals/oracle-solver.md). ROSCO's
table is read cell by cell through K2's clip, which at width 0 is its bilinear
lookup, held at its edges, and within w log 2 of the slope's jump above it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.energy.wind_turbine.cp_table import CP_TABLE, CP_TABLE_FEATHER
from target_gym.energy.wind_turbine.env import omega_rated
from target_gym.energy.wind_turbine.oracle import pitch_reach, table_cp, torque_reach
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import Oracle, enable_x64, float64

SPEC = registry.get("wind_turbine")


def test_the_registry_builds_the_shared_oracle():
    env = SPEC.make_env()
    assert isinstance(SPEC.make_mpc(env, SPEC.make_test_params()), Oracle)


def test_the_resting_target_is_rated_speed_and_one_at_the_bands_edge():
    p = plan_params(SPEC, SPEC.make_test_params())
    rest = SPEC.make_mpc(SPEC.make_env(), p).model.rest
    rated = float(omega_rated(p))
    assert float(rest(jnp.array([rated, 0.0, 0.0, 0.0, 0.0]))) == 0.0
    edge = rated * float(p.overspeed_factor)
    assert abs(float(rest(jnp.array([edge, 0.0, 0.0, 0.0, 0.0])))) == pytest.approx(1.0)


def test_the_resting_target_spends_the_tie_budget_over_an_episode_at_the_edge():
    """The planned stage costs the budget over the 2400 steps at the band's
    edge, and K8's replay, which runs ``model.step``, never sees it."""
    env, p = SPEC.make_env(), plan_params(SPEC, SPEC.make_test_params())
    mpc = SPEC.make_mpc(env, p)
    # The tie's, in the score's dollars: half the 1.7 kW resolution, 850 W, for
    # one 0.25 s step at 60 $/MWh.
    budget = 850.0e-6 * 0.25 / 3600.0 * 60.0
    with enable_x64():
        st = float64(env.reset_env(jax.random.PRNGKey(0), p)[1])
        m = mpc.model
        d = {"task": m.data(st), "widths": jnp.asarray(mpc.widths)}
        x = m.state_x(st).at[0].set(float(omega_rated(p)) * 1.1)
        u = m.hold(st)
        x1 = m.step(d, x, u, 0)[0]
        extra = mpc.shooting.model.step(d, x, u, 0)[1] - m.step(d, x, u, 0)[1]
        expected = budget / 2400 * float(m.rest(x1)) ** 2
    assert float(extra) == pytest.approx(expected)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_command_past_the_pitch_reach_moves_it_as_one_at_the_reach(sign):
    """From the reset, 10 deg from both ends of the range: the actuator
    closes the gap at the rate limit until the gap is rate * tau, so a
    command rate * (step + tau) away keeps it at full rate for the whole
    step, as one at the far end of the range does. Float32 env: equal to
    1e-6 relative, on every state the step moves."""
    env, p = SPEC.make_env(), SPEC.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    state = state.replace(pitch=jnp.float32(20.0))

    def raw(pitch):
        span = p.pitch_max - p.pitch_min
        return jnp.asarray(
            [
                2.0 * (pitch - p.pitch_min) / span - 1.0,
                2.0 * float(state.torque) / p.torque_max - 1.0,
            ]
        )

    at = 20.0 + sign * float(pitch_reach(p, 1))
    far = 20.0 + sign * 15.0
    reached = env.step_env(key, state, raw(at), p)[1]
    past = env.step_env(key, state, raw(far), p)[1]
    assert float(reached.pitch) - 20.0 == pytest.approx(
        sign * p.pitch_rate_max * p.delta_t, rel=1e-5
    )
    for field in ("pitch", "omega", "torque"):
        np.testing.assert_allclose(
            getattr(past, field), getattr(reached, field), rtol=1e-6, err_msg=field
        )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_command_past_the_torque_reach_moves_it_as_one_at_the_reach(sign):
    """From 20 kN m: at 15 kN m/s over a 0.25 s step the torque moves 3750 N m,
    and a command 15000 x (0.25 + 0.1) away keeps it at full rate for the step,
    as one 15 kN m away does."""
    env, p = SPEC.make_env(), SPEC.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    state = state.replace(torque=jnp.float32(20e3), torque_cmd=jnp.float32(20e3))

    def raw(torque):
        span = p.pitch_max - p.pitch_min
        pitch = 2.0 * (float(state.pitch) - p.pitch_min) / span - 1.0
        return jnp.asarray([pitch, 2.0 * torque / p.torque_max - 1.0])

    at = 20e3 + sign * float(torque_reach(p, 1))
    reached = env.step_env(key, state, raw(at), p)[1]
    past = env.step_env(key, state, raw(20e3 + sign * 15e3), p)[1]
    assert float(reached.torque) - 20e3 == pytest.approx(sign * 3750.0, rel=1e-5)
    for field in ("pitch", "omega", "torque"):
        np.testing.assert_allclose(
            getattr(past, field), getattr(reached, field), rtol=1e-6, err_msg=field
        )


@pytest.mark.parametrize(
    "row, column",
    [(0.0, 0.0), (10.0, 5.0), (10.5, 5.0), (10.0, 5.5), (10.25, 5.75), (25.0, 30.0)],
)
def test_at_width_zero_the_table_is_read_bilinearly(row, column):
    """By hand from the table's nodes: the node itself, then the mean of two or
    four, or a bilinear share of the four around a point."""
    i, j = int(row), int(column)
    a, b = row - i, column - j
    i2, j2 = min(i + 1, 25), min(j + 1, 30)
    expected = (
        (1 - a) * (1 - b) * CP_TABLE[i, j]
        + (1 - a) * b * CP_TABLE[i, j2]
        + a * (1 - b) * CP_TABLE[i2, j]
        + a * b * CP_TABLE[i2, j2]
    )
    with enable_x64():
        lam, pitch = jnp.float64(2.0 + 0.5 * row), jnp.float64(column)
        assert float(table_cp(lam, pitch, 0.0, 0.0)) == pytest.approx(
            expected, abs=1e-12
        )


def test_past_its_edges_the_table_holds_its_last_nodes():
    with enable_x64():
        low = table_cp(jnp.float64(1.0), jnp.float64(-3.0), 0.0, 0.0)
        high = table_cp(jnp.float64(20.0), jnp.float64(100.0), 0.0, 0.0)
    assert float(low) == pytest.approx(CP_TABLE[0, 0], abs=1e-12)
    assert float(high) == pytest.approx(CP_TABLE_FEATHER[25, 90], abs=1e-12)


def test_the_smoothed_table_stays_within_its_gap_of_the_lookup():
    """K2: a softplus on a cell's clip misses by at most w log 2, in cells, times
    the jump in the slope at the node. At a node of the tip-speed axis at 9 deg
    the jump is |C[i+1, 9] - 2 C[i, 9] + C[i-1, 9]|. Between nodes both are the
    same bilinear value to well under that. The neighbouring nodes' softplus
    tails add exp(-1 / 0.05) = 2e-9 of their jumps, so the bound carries 1e-7."""
    width, i, j = 0.05, 10, 9
    jump = abs(CP_TABLE[i + 1, j] - 2 * CP_TABLE[i, j] + CP_TABLE[i - 1, j])
    with enable_x64():
        lam, pitch = jnp.float64(2.0 + 0.5 * i), jnp.float64(j)
        gap = abs(float(table_cp(lam, pitch, width, 0.0)) - CP_TABLE[i, j])
    assert 0.0 < gap <= width * np.log(2.0) * jump * (1.0 + 1e-7)
