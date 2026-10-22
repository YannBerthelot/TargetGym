"""The patrol oracle: the twin-autopilot law with a residual planner on top.

The law flies the lead's own heading autopilot on the follower's state, so in
the slot with nothing to correct it commands exactly what the lead commands,
and the shared gust drops out of the relative position the reward scores. The
planner keeps the best plan it scored, so it never applies one worse than its
warm start. The law and the bearing-only slot are checked here cheaply; the
planner and the closed-loop hold run in the slow job, where the hold would
catch a return to the GradientMPC it replaced, which held about 19 m where
this holds about 0.1 m.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.experts.pid import plane3d_heading_pid_step
from target_gym.patrol.env import (
    desired_slot_position,
    get_obs_heading,
    slot_error,
    wrap_angle,
)
from target_gym.patrol.experts import (
    TWIN_GAINS,
    PatrolTwinOracle,
    twin_law_init,
    twin_law_step,
)


def _reset(name="patrol", seed=0):
    spec = registry.get(name)
    env = spec.make_env()
    params = spec.make_test_params()
    _, state = env.reset_env(jax.random.PRNGKey(seed), params)
    return spec, env, params, state


def test_in_the_slot_the_law_is_the_lead_autopilot():
    """Follower in the slot with the lead's velocity and attitude, nothing
    remembered: every correction is zero, so the law commands what the lead's
    own autopilot commands this step."""
    _, env, _, state = _reset()
    lead = state.lead
    x, y, z = desired_slot_position(state)
    follower = lead.replace(
        x=x, y=y, z=z, target_altitude=lead.target_altitude + state.slot_up
    )
    state = state.replace(follower=follower)
    assert float(slot_error(state)) < 1e-3

    action, _ = twin_law_step(env._lead_pid_params, TWIN_GAINS, twin_law_init(), state)
    lead_cmd = lead.replace(
        target_heading=wrap_angle(lead.target_heading + state.lead_turn_rate)
    )
    lead_action, _ = plane3d_heading_pid_step(
        env._lead_pid_params, state.lead_pid, get_obs_heading(lead_cmd)
    )
    np.testing.assert_allclose(
        np.asarray(action), np.asarray(lead_action), rtol=0, atol=1e-6
    )


def test_without_a_plan_the_oracle_is_the_law():
    """``horizon=0`` gives the law alone, from the same memory."""
    spec, env, params, state = _reset(seed=1)
    oracle = spec.make_mpc(env, plan_params(spec, params), horizon=0)
    assert isinstance(oracle, PatrolTwinOracle)
    oracle.reset()
    law, _ = twin_law_step(env._lead_pid_params, TWIN_GAINS, twin_law_init(), state)
    np.testing.assert_allclose(
        oracle.step(None, state), np.asarray(law), rtol=0, atol=1e-6
    )


def test_the_jitted_step_compiles_once():
    """The law hands back its memory with the types it was given, so the
    oracle's jitted step compiles once per instance. It compiled twice when the
    heading PID's state came back with ``lead_psi_prev`` weakly typed, and with
    the planner on each compile costs tens of seconds per episode. The law
    alone here, to stay cheap."""
    spec, env, params, state = _reset()
    oracle = spec.make_mpc(env, plan_params(spec, params), horizon=0)
    oracle.reset()
    key = jax.random.PRNGKey(0)
    step = jax.jit(env.step_env)
    obs = env.get_obs(state, params)
    for _ in range(3):
        action = oracle.step(obs, state)
        obs, state, *_ = step(key, state, jnp.asarray(action), params)
    assert oracle._jit_step._cache_size() == 1


@pytest.mark.slow
def test_the_plan_is_never_worse_than_its_warm_start():
    """The best iterate is kept, and the warm start is the first one scored.
    From a reset the follower is up to 40 m off its slot, so there is
    something to plan. Slow only for the compile: reverse mode through the
    aircraft's RK4 step costs about 40 s cold."""
    spec, env, params, state = _reset(seed=2)
    oracle = PatrolTwinOracle(env, plan_params(spec, params), horizon=6, n_iter=3)
    carry = twin_law_init()
    warm = jnp.zeros((6, 3))
    best = oracle._solve(state, carry, warm)
    assert float(jnp.max(jnp.abs(best))) <= oracle.max_residual + 1e-6
    f_best = float(oracle._plan_cost(best, state, carry))
    f_warm = float(oracle._plan_cost(warm, state, carry))
    assert f_best <= f_warm * (1.0 + 1e-6)


def test_the_bearing_only_slot_is_the_patrol_oracle_on_the_true_state():
    """It reads the state, never the observation, so from the same seed it
    flies the same actions as on patrol (the law alone, to stay cheap)."""
    flown = {}
    for name in ("patrol", "patrol_bearing_only"):
        spec, env, params, state = _reset(name)
        oracle = spec.make_mpc(env, plan_params(spec, params), horizon=0)
        assert isinstance(oracle, PatrolTwinOracle)
        oracle.reset()
        key = jax.random.PRNGKey(0)
        step = jax.jit(env.step_env)
        obs = env.get_obs(state, params)
        actions = []
        for _ in range(10):
            action = oracle.step(obs, state)
            actions.append(action)
            obs, state, *_ = step(key, state, jnp.asarray(action), params)
        flown[name] = np.array(actions)
    np.testing.assert_allclose(
        flown["patrol_bearing_only"], flown["patrol"], rtol=0, atol=1e-6
    )


@pytest.mark.slow
def test_the_oracle_holds_the_slot_well_inside_the_gps_resolution():
    """Protocol seed 0 from the protocol's burn-in (step 100) for 30 steps:
    the oracle holds a mean slot error of 0.12 m and a heading error of
    1.3e-3 rad there (measured when this test was written), inside the 3 m
    and 0.0087 rad floors, where the law alone held about 4 m RMS and the
    GradientMPC it replaced about 19 m (oracle audit, 2026-10)."""
    spec, env, params, state = _reset()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    oracle.reset()
    key = jax.random.PRNGKey(0)
    step = jax.jit(env.step_env)
    obs = env.get_obs(state, params)
    slot, heading, trips = [], [], 0
    for t in range(130):
        action = oracle.step(obs, state)
        obs, state, _, _, info = step(key, state, jnp.asarray(action), params)
        trips += int(bool(info["tripped"]))
        if t >= 100:
            slot.append(float(slot_error(state)))
            heading.append(abs(float(wrap_angle(state.follower.psi - state.lead.psi))))
    assert trips == 0
    assert np.mean(slot) < 0.5
    assert np.mean(heading) < 0.0087
