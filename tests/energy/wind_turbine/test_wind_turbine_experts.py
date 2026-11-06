"""The wind turbine's oracle, on the shared solver.

Its claims, each checked below. With the turbulence off the model is the
plant, so the power after every step lands on the target to float32 rounding:
the torque's reach makes that free. A command past the pitch's reach moves
the actuator as a command at it does (T1r, env.py:309). The rotor speed is the
one spare degree of freedom, so it rests at rated, inside the tie's budget (G12
in docs/proposals/oracle-solver.md), and a lull no longer runs it down.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.energy.wind_turbine.env import electrical_power, omega_rated
from target_gym.energy.wind_turbine.oracle import pitch_reach
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import Oracle, enable_x64, float64
from target_gym.runners.runners import mpc_policy, one_episode

SPEC = registry.get("wind_turbine")


def _episode(params, seed, steps=None):
    """Rotor speed over rated, |target - power| and trips over one episode."""
    env = SPEC.make_env()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    oracle.reset()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    w_rated = float(omega_rated(params))
    ratio, error, trips = [], [], 0
    for _ in range(steps or int(params.max_steps_in_episode)):
        action = oracle.step(obs, state)
        obs, state, _, _, info = step(key, state, action, params)
        trips += int(bool(info["tripped"]))
        ratio.append(float(state.omega) / w_rated)
        power = electrical_power(state.omega, state.torque, params)
        error.append(abs(float(state.target_power) - float(power)))
    return np.array(ratio), np.array(error), trips


def test_the_registry_builds_the_shared_oracle():
    env = SPEC.make_env()
    assert isinstance(SPEC.make_mpc(env, SPEC.make_test_params()), Oracle)


@pytest.mark.medium
def test_without_turbulence_the_power_lands_on_the_target():
    """With the wind on its mean path the model is the plant, so the power
    after every step equals the target to float32 rounding (about 0.5 W at
    4-5 MW), within the rows' tolerance. A plan that missed would leave
    kilowatts."""
    params = SPEC.make_test_params().replace(turbulence_std=0.0)
    _, error, trips = _episode(params, 0, steps=20)
    assert trips == 0
    assert error.max() <= 2.0


@pytest.mark.medium
def test_a_lull_does_not_run_the_rotor_down_to_a_power_dip():
    """On seed 1 the wind falls to 14.7 m/s at step 202. A plan that leaves the
    speed free rides it down to about 0.82 of rated and pays there with
    one-step torque dips of 625 to 825 kW, 0.004 to 0.006 dollars a step.
    Resting at rated, the rotor keeps its energy and no step costs a quarter
    of the smaller dip."""
    params = SPEC.make_test_params()
    policy = mpc_policy(SPEC, SPEC.make_env(), params)
    speeds = []

    def spy(obs, state):
        speeds.append(float(state.omega))
        return policy(obs, state)

    spy.controller = policy.controller
    out = one_episode(SPEC, params, spy, 1)
    assert int(np.asarray(out["tripped"]).sum()) == 0
    assert min(speeds) / float(omega_rated(params)) >= 0.9
    assert np.asarray(out["reward"][0, : int(out["length"][0])]).min() >= -0.001


@pytest.mark.medium
def test_the_report_certifies_the_plan_that_applies_not_the_wide_rungs():
    """K7's first rung is smoothed wide on purpose and only starts the next
    one, so K8's report covers the rung that applies, whose output stays
    within the resolution. The first rung's own gap is about 270 resolutions."""
    env, params = SPEC.make_env(), SPEC.make_test_params()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    oracle.reset()
    obs, state = env.reset_env(jax.random.PRNGKey(1), params)
    oracle.step(obs, state)
    assert len(oracle.certificates) > 1
    assert oracle.solver_report()["solver_max_output_gap"] <= 1.0


def test_the_resting_target_is_rated_speed_and_one_at_the_bands_edge():
    p = plan_params(SPEC, SPEC.make_test_params())
    rest = SPEC.make_mpc(SPEC.make_env(), p).model.rest
    rated = float(omega_rated(p))
    assert float(rest(jnp.array([rated, 0.0, 0.0]))) == 0.0
    edge = rated * float(p.overspeed_factor)
    assert abs(float(rest(jnp.array([edge, 0.0, 0.0])))) == pytest.approx(1.0)


def test_the_resting_target_spends_the_tie_budget_over_an_episode_at_the_edge():
    """The planned stage costs the budget over the 400 steps at the band's edge,
    and K8's replay, which runs ``model.step``, never sees it."""
    env, p = SPEC.make_env(), plan_params(SPEC, SPEC.make_test_params())
    mpc = SPEC.make_mpc(env, p)
    budget = 500.0e-6 * 100.0 * 0.25 / 3600.0  # the tie's, in dollars
    with enable_x64():
        st = float64(env.reset_env(jax.random.PRNGKey(0), p)[1])
        m = mpc.model
        d = {"task": m.data(st), "widths": jnp.asarray(mpc.widths)}
        x = m.state_x(st).at[0].set(float(omega_rated(p)) * 1.1)
        u = m.hold(st)
        x1 = m.step(d, x, u, 0)[0]
        extra = mpc.shooting.model.step(d, x, u, 0)[1] - m.step(d, x, u, 0)[1]
        expected = budget / 400 * float(m.rest(x1)) ** 2
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
    state = state.replace(pitch=jnp.float32(20.0), pitch_cmd=jnp.float32(20.0))

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
