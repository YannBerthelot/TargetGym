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
from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import rod_worth, scheduled_load
from target_gym.reactor.oracle import boron_step, reach_scale, reactor_model

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
    state = state.replace(rod_in=jnp.asarray(0.5, state.n.dtype))
    raw = 1.0 - 2.0 * float(state.rod_in)
    reach = reach_scale(env, p, 1)[0] / 2.0
    at = env.step_env(key, state, jnp.array([raw + sign * reach, 0.0]), p)[1]
    past = env.step_env(key, state, jnp.array([sign, 0.0]), p)[1]
    moved = 2.0 * (float(state.rod_in) - float(at.rod_in))
    assert moved == pytest.approx(sign * reach, rel=1e-3)
    for field in ("rod_in", "n", "T_coolant", "T_cold"):
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
        # Rods mid-stroke: the plan drops the clip at the stops, so its demand stays
        # inside them.
        st = st.replace(
            ramp_down_at=down, load_wobble=jnp.float64(0.01), target_n=load, rod_in=0.5
        )
        m = oracle.model
        d = {"task": m.data(st), "end": 10, "widths": jnp.zeros(1)}
        moves = m.hold(st)[None] + 0.1 * jnp.sin(0.7 * jnp.arange(oracle.P))[:, None]
        cert = oracle._certify(d, st, m.state_x(st), moves, 10)
    assert not bool(cert["replay_trip"])
    assert float(cert["replay_gap"]) <= 1e-10
    assert float(cert["output_gap"]) <= 1e-6


@pytest.mark.medium
def test_a_plan_mid_stream_and_reloading_replays_on_the_env():
    """The env loads the turbine on the day it carries (its own rate and depth, on a
    plan clock a stream rebased at ``plan_t0``), at most ``pickup`` while it reloads
    after a restart, and burns boron since the last refuelling (``burn_t0``); the
    plan must load and burn the same. A state 20000 sub-steps (27.8 h) into a
    stream, refuelled now, its plan rebased 20 s ago on a day going down at twice
    the window's rate to 0.7, and the turbine reloading at 0.9: the day falls below
    0.9 at 150 s plan time, inside the ten steps (300 s), so the reload is met and
    released mid-plan. Tolerances as on the window's replay above."""
    env = SPEC.make_env()
    params = SPEC.make_test_params()
    oracle = SPEC.make_mpc(env, plan_params(SPEC, params))
    with enable_x64():
        _, state = env.reset_env(jax.random.PRNGKey(100), oracle.params)
        st = float64(state)
        now = 20000
        st = st.replace(
            physics_time=jnp.asarray(now, st.physics_time.dtype),
            plan_t0=jnp.asarray(now - 4, st.plan_t0.dtype),
            burn_t0=jnp.asarray(now, st.burn_t0.dtype),
            plan_rate=jnp.float64(2.0 * oracle.params.load_ramp_rate),
            plan_depth=jnp.float64(0.3),
            ramp_down_at=jnp.float64(0.0),
            pickup=jnp.float64(0.9),
            load_wobble=jnp.float64(0.0),
            target_n=jnp.float64(0.9),
            rod_in=0.5,
        )
        m = oracle.model
        d = {"task": m.data(st), "end": 10, "widths": jnp.zeros(1)}
        moves = m.hold(st)[None] + 0.1 * jnp.sin(0.7 * jnp.arange(oracle.P))[:, None]
        cert = oracle._certify(d, st, m.state_x(st), moves, 10)
    assert not bool(cert["replay_trip"])
    assert float(cert["replay_gap"]) <= 1e-10
    assert float(cert["output_gap"]) <= 1e-6


@pytest.mark.parametrize(
    "arriving, expected",
    [
        (0.0, 1690.0),
        # Clean water at 10 kg/s through 2.5e5 kg for 5 s.
        (-1.0, 1690.0 * np.exp(-2e-4)),
        # Acid at 7800 ppm and 3 kg/s through the same mass for 5 s.
        (1.0, 7800.0 + (1690.0 - 7800.0) * np.exp(-6e-5)),
    ],
)
def test_the_planners_boron_is_the_plants_mixing_at_no_width(arriving, expected):
    p = ReactorParams()
    out = boron_step(jnp.asarray(arriving), jnp.asarray(1690.0), p, 0.0)
    assert float(out) == pytest.approx(expected, rel=1e-5)


def test_every_move_has_a_scale():
    env, p = Reactor(), ReactorParams()
    model = reactor_model(env, p)
    assert len(model.move_scale(1)) == model.nu


def _violated(model, state):
    """The planner's limits a held move breaks on its first step from ``state``."""
    d = {"task": model.data(state), "end": 10, "widths": jnp.zeros(1)}
    x = model.state_x(state)
    _, _, y, *_ = model.step(d, x, model.hold(state), 0)
    q = np.asarray(y, float)[len(model.hinges) :]
    return [
        i for i, (v, lim) in enumerate(zip(q, model.limits))
        if v < lim.lo or v > lim.hi
    ]  # fmt: skip


def _rods_in(state, rod_in, p, balanced):
    """The rods at ``rod_in``; ``balanced``, with the boron cut by their worth, so
    the power holds where it was."""
    rod_in = jnp.asarray(rod_in, state.n.dtype)
    cut = (rod_worth(rod_in, p) - rod_worth(state.rod_in, p)) / p.boron_worth
    return state.replace(rod_in=rod_in, boron=state.boron - cut * balanced)


@pytest.mark.parametrize(
    "timer, setup",
    [
        # The rods driven to 0.8 of the stroke at full power: the power falls to
        # a third within the step and Tavg under its floor of 556 K.
        ("below_floor", lambda s, p: _rods_in(s, 0.8, p, balanced=False)),
        # The rods at 0.30 of the stroke with the boron cut to hold full power,
        # past the insertion limit of 0.22 there (``ril_full``).
        ("ril_outside", lambda s, p: _rods_in(s, 0.30, p, balanced=True)),
    ],
)
def test_every_trip_the_env_times_is_a_planned_limit(timer, setup):
    """The env times two trips, Tavg under its floor and the rods past their
    insertion limit, and trips once either has lasted its allowance. The plan
    looks five minutes ahead, less than the floor's 15 minutes, so a timer it does
    not plan runs out unseen. Like DNB, each is planned as a hard bound: a step
    on which the env starts a timer breaks one of the planner's limits, and the
    reset, which starts none, breaks none."""
    env, params = SPEC.make_env(), SPEC.make_test_params()
    with enable_x64():
        p = float64(params)
        model = reactor_model(env, p)
        key = jax.random.PRNGKey(100)
        _, reset = env.reset_env(key, p)
        reset = float64(reset)
        assert _violated(model, reset) == []
        state = setup(reset, p)
        stepped = env.step_env(key, state, model.hold(state), p)[1]
        assert float(getattr(stepped, timer)) > 0.0  # the env starts its timer
        assert _violated(model, state)
