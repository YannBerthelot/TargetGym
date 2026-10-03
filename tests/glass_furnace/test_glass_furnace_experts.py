"""The glass furnace's oracle, as changed by the oracle audit (2026-10).

The CasADi MPC applies the first input its plan can move, plans on the known
loads (the pull at its AR(1) conditional mean and the pulsed batch charge at
that pull), and corrects its reduced model with a crown heat-rate disturbance
estimated from the one-step prediction error, where it used a setpoint bias.
The fast tests check each piece on a short horizon. The slow one is the closed
loop on protocol seed 2, where the protocol cost fell from 0.326 to 0.171 per
step and the mean window error from 0.38 K to 0.17 K.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.glass_furnace.env import (
    FUEL_DEAD_TIME_STEPS,
    M_PULL_AR_RHO,
    charge_rate_now,
)
from target_gym.glass_furnace.experts import (
    _FURNACE_DISTURBANCE_GAIN,
    _FURNACE_DISTURBANCE_LIMIT,
    make_glass_furnace_mpc,
)

pytest.importorskip("casadi")
pytest.importorskip("do_mpc")

SPEC = registry.get("glass_furnace")
#: Short enough to build and solve quickly; every piece tested here is the
#: same at the shipped 60.
HORIZON = 12


def _setup(**changes):
    env = SPEC.make_env()
    params = SPEC.make_test_params().replace(**changes)
    oracle = make_glass_furnace_mpc(env, plan_params(SPEC, params), horizon=HORIZON)
    oracle.reset()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    return env, params, oracle, state


def _tvp(oracle, name):
    """The time-varying parameter ``name`` the next solve would get, by node."""
    tpl = oracle._mpc.tvp_fun(0.0)
    return np.array([float(tpl["_tvp", k, name]) for k in range(oracle.horizon + 1)])


def test_the_load_forecast_is_the_noise_free_plant_path():
    """Interval k runs from t + k to t + k + 1, and the plant draws its pull
    after one more AR(1) step, so with the innovation off the forecast is the
    pull and the batch charge the plant applies, floor included. A disturbance
    of -6 kg/s against the 5.79 kg/s nominal pull keeps the 0.1 kg/s floor
    binding for the first five intervals and releases it after."""
    env, params, oracle, state = _setup(m_pull_noise_std=0.0)
    state = state.replace(m_pull_disturbance=jnp.float32(-6.0), time=jnp.int32(37))
    oracle._update_setpoint(state)
    pulls, charges = _tvp(oracle, "m_pull_k"), _tvp(oracle, "charge_k")

    step = jax.jit(env.step_env)
    want_pull, want_charge = [], []
    for _ in range(HORIZON + 1):
        t = int(state.time)
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.zeros(1), params)
        pull = max(float(params.m_pull) + float(state.m_pull_disturbance), 0.1)
        want_pull.append(pull)
        want_charge.append(float(charge_rate_now(pull, t, params)))
    np.testing.assert_allclose(pulls, want_pull, rtol=1e-5)
    # The charge is zero between pulses; float32 rounding at a pulse edge
    # leaves up to about 2e-6 kg/s there, against pulses of 0.3 to 0.5 kg/s.
    np.testing.assert_allclose(charges, want_charge, rtol=1e-5, atol=1e-5)
    assert np.count_nonzero(np.asarray(want_charge) > 0.1) >= 3
    assert pulls[4] == 0.1 and pulls[5] > 0.1
    # The conditional mean decays at the env's own rate.
    d = -6.0 * M_PULL_AR_RHO ** np.arange(1, HORIZON + 2)
    np.testing.assert_allclose(pulls, np.maximum(params.m_pull + d, 0.1), rtol=1e-6)


def test_the_applied_input_is_the_first_one_the_plan_can_move():
    """The first ``FUEL_DEAD_TIME_STEPS`` intervals burn the pipeline, so the
    plan's inputs there never reach its dynamics. The oracle applies the input
    at that index and makes it do-mpc's ``u_prev``; the base class applied
    u_0, which only the move penalty set, a third of the way from ``u_prev``
    to the input that mattered."""
    env, params, oracle, state = _setup()
    m = oracle._mpc
    step = jax.jit(env.step_env)
    for _ in range(3):
        u = oracle.step(None, state)
        planned = m.opt_x_num["_u", FUEL_DEAD_TIME_STEPS, 0] * m._u_scaling
        assert u == pytest.approx(float(np.clip(float(planned), -1.0, 1.0)))
        assert float(m._u0.master) == pytest.approx(float(planned))
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.atleast_1d(u), params)
    first = float(m.opt_x_num["_u", 0, 0] * m._u_scaling)
    assert abs(first - u) > 1e-3  # the reset state is off target: u_0 lags
    assert oracle.solver_report()["solver_failures"] == 0


def test_the_disturbance_estimate_follows_the_one_step_error():
    """Each step folds ``disturbance_gain`` of the crown's one-step prediction
    error, in K/s, into the estimate, clamped to the limit; ``reset`` clears
    it."""
    env, params, oracle, state = _setup()
    assert oracle._q_gain == _FURNACE_DISTURBANCE_GAIN
    oracle.step(None, state)  # no prediction yet: nothing to learn from
    assert oracle._q_crown == 0.0
    predicted = oracle._pred_crown
    assert predicted is not None

    # The crown comes in 1.5 K hotter than the plan predicted.
    nxt = state.replace(T_crown=jnp.float32(predicted + 1.5), time=state.time + 1)
    oracle.step(None, nxt)
    error = float(nxt.T_crown) - predicted
    expected = _FURNACE_DISTURBANCE_GAIN * error / float(params.delta_t)
    assert oracle._q_crown == pytest.approx(expected, rel=1e-9)
    assert _tvp(oracle, "q_crown") == pytest.approx(expected, rel=1e-9)

    # A huge error cannot run it away.
    oracle._pred_crown = float(nxt.T_crown) - 500.0
    oracle.step(None, nxt)
    assert oracle._q_crown == _FURNACE_DISTURBANCE_LIMIT

    oracle.reset()
    assert oracle._q_crown == 0.0 and oracle._pred_crown is None


def test_a_saturated_input_does_not_wind_the_estimate_up():
    """The setpoint bias integrated the tracking error, so with the fuel at
    its bound it kept growing (to -22 K on protocol seed 1). The estimate
    integrates the prediction error instead, and the prediction includes the
    saturated input. A target 30 K below the crown holds the fuel at its
    minimum, and over ten steps the estimate moves by the model mismatch, a
    few hundredths of a kelvin a step, where the old bias at its 0.05 gain
    would have moved by 15 K."""
    env, params, oracle, state = _setup(m_pull_noise_std=0.0)
    schedule = jnp.asarray(state.target_schedule) - 30.0
    state = state.replace(
        target_schedule=schedule, target_T_crown=jnp.float32(schedule[0])
    )
    step = jax.jit(env.step_env)
    for _ in range(10):
        u = oracle.step(None, state)
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.atleast_1d(u), params)
    assert u == pytest.approx(-1.0, abs=1e-6)
    assert float(state.target_T_crown) < float(state.T_crown) - 20.0
    assert abs(oracle._q_crown) * float(params.delta_t) < 0.1  # K a step


def test_a_failed_solve_holds_the_action_and_skips_the_next_update(monkeypatch):
    """On a solve that fails for a reason other than the iteration cap the
    oracle restores its warm start and holds its last action, as the base
    class does, and the step after it learns nothing: there is no prediction
    to compare with."""
    env, params, oracle, state = _setup()
    u0 = oracle.step(None, state)
    nxt = state.replace(
        T_crown=jnp.float32(oracle._pred_crown + 1.5), time=jnp.int32(1)
    )
    monkeypatch.setattr(oracle, "_record_solve", lambda: False)
    assert oracle.step(None, nxt) == u0
    assert oracle._pred_crown is None
    q = oracle._q_crown
    monkeypatch.undo()
    oracle.step(None, nxt.replace(T_crown=jnp.float32(float(nxt.T_crown) + 5.0)))
    assert oracle._q_crown == q
    assert oracle._pred_crown is not None


@pytest.mark.slow
def test_the_oracle_holds_the_crown_on_protocol_seed_2():
    """The protocol's construction (``runners.baseline_policy``: the factory
    on ``plan_params``, then ``reset``) and its rollout (``PRNGKey(seed)`` on
    every step) on seed 2. The audit measured a mean |crown error| of 0.17 K
    over the scored window (the 800 steps after the protocol's burn-in),
    against 0.38 K for the oracle it replaced, with zero solver failures, and
    the heat-rate estimate settled between +0.14 and +0.19 K a step: the plant
    runs hotter than the reduced model."""
    env = SPEC.make_env()
    params = SPEC.make_test_params()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    oracle.reset()
    key = jax.random.PRNGKey(2)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    errors, estimates = [], []
    for t in range(int(params.max_steps_in_episode)):
        u = oracle.step(obs, state)
        obs, state, _, _, info = step(key, state, jnp.atleast_1d(u), params)
        assert not bool(info.get("tripped", False))
        if t >= 800:
            errors.append(abs(float(state.target_T_crown) - float(state.T_crown)))
            estimates.append(oracle._q_crown * float(params.delta_t))
    assert np.mean(errors) < 0.25
    assert 0.1 < min(estimates) and max(estimates) < 0.25
    assert oracle.solver_report()["solver_failures"] == 0
