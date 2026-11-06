"""The reactor's oracle, on the shared solver.

Contract, from the smoothing protocol (T1, T1r, T2, T3, K8) and the env: a demand
past the rods' travel over the hold moves them as a demand at it does, so bounding
the plan there loses nothing; with the rate corner exact the planner's model and
the env agree on a plan, on the load's ramp and with its wobble, and in a sub-step's
Tavg error and the rods' travel.

By hand, for the reach: 72 steps/min over 228 steps is 1.2 / 228 of the stroke per
second, 0.0105 raw units/s on a span of 2, so 0.3158 raw units over 30 s.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import Oracle, enable_x64, float64
from target_gym.reactor.env import scheduled_load
from target_gym.reactor.oracle import reach_scale

SPEC = registry.get("reactor")


def test_the_registry_builds_the_shared_oracle():
    assert isinstance(SPEC.make_mpc(SPEC.make_env(), SPEC.make_test_params()), Oracle)


def test_the_rods_reach_over_a_step_is_their_speed_for_thirty_seconds():
    env, p = SPEC.make_env(), SPEC.make_test_params()
    assert reach_scale(env, p, 1)[0] / 2.0 == pytest.approx(0.31579, rel=1e-4)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_demand_past_the_rods_reach_moves_them_as_one_at_it(sign):
    """From the reset: a demand at the reach keeps the rods at full speed for
    every sub-step, as one at the far end of their travel does. Float32 env:
    equal to 1e-5 relative."""
    env, p = SPEC.make_env(), SPEC.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    span = p.rho_ext_max - p.rho_ext_min
    raw = 2.0 * (float(state.rho_ext) - p.rho_ext_min) / span - 1.0
    reach = reach_scale(env, p, 1)[0] / 2.0
    at = env.step_env(key, state, jnp.array([raw + sign * reach]), p)[1]
    past = env.step_env(key, state, jnp.array([sign]), p)[1]
    assert (float(at.rho_ext) - float(state.rho_ext)) * sign > 0.0
    moved = 2.0 * (float(at.rho_ext) - float(state.rho_ext)) / span
    assert moved == pytest.approx(sign * reach, rel=1e-3)
    for field in ("rho_ext", "n", "T_coolant", "T_cold"):
        np.testing.assert_allclose(
            getattr(past, field), getattr(at, field), rtol=1e-5, err_msg=field
        )


@pytest.mark.medium
@pytest.mark.parametrize("start_of_ramp", [-3000.0, 20.0])
def test_a_plan_on_the_loads_ramp_replays_on_the_env(start_of_ramp):
    """Ten steps from the reset with the day's ramp moved to ``start_of_ramp`` s
    and a wobble of 0.01, the rods moved by a sine inside their reach: the plan and
    the env differ only in rounding, so 1e-10 and 1e-6 sit many decades above it,
    unless the load's path, a sub-step's error or the travel is off. The widths are
    0, as K5's corner is not exact."""
    env = SPEC.make_env()
    params = SPEC.make_test_params()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    with enable_x64():
        _, state = env.reset_env(jax.random.PRNGKey(100), oracle.params)
        st = float64(state)
        down = jnp.float64(start_of_ramp)
        load = scheduled_load(0.0, down, st.ramp_up_at, oracle.params) + 0.01
        st = st.replace(ramp_down_at=down, load_wobble=jnp.float64(0.01), target_n=load)
        m = oracle.model
        d = {"task": m.data(st), "end": 10, "widths": jnp.zeros(1)}
        moves = m.hold(st)[None] + 0.1 * jnp.sin(0.7 * jnp.arange(oracle.P))[:, None]
        cert = oracle._certify(d, st, m.state_x(st), moves, 10)
    assert not bool(cert["replay_trip"])
    assert float(cert["replay_gap"]) <= 1e-10
    assert float(cert["output_gap"]) <= 1e-6
