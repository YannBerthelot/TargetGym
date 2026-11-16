"""The reactor's day planner (``reactor/oracle_day.py``).

Contract, written before the runs.

- A knot's cost is the env's, by hand: the turbine 10 % of rated load under its
  plan through a knot loses 0.1 x 1 GWe x 80 $/MWh x 300 s = 666.67 $; ten rod
  steps of travel cost 10 $; a knot of full dilution arriving is 10 kg/s x 300 s
  of effluent at 0.0628 $/kg, 188.40 $, of full boration 3 kg/s, 56.52 $. (Tavg's
  band is set wide for these, so its term is out of the way.)
- And Tavg past its free band is charged as the env charges it, 0.0639 $/s a
  kelvin, 19.17 $ a knot: a plant lined out on its program pays nothing; with
  10 ppm more boron its Tavg sits 10 x 8 / 15 = 5.33 K under its program at the
  start of the cycle, 3.73 K past the band, 71.56 $. (A quiet plant, its xenon
  balanced with full load: a drawn reset sits a few tenths of a kelvin off.)
  Under the load's wobble, 0.5 % of rated, Tavg's error wobbles by 0.005 x (15.4
  + 3 x 471.2 / 15) = 0.548 K, so a plan with 3 ppm more boron, its Tavg on the
  band's edge (3 x 8 / 15 = 1.6 K), pays the mean crossing, 0.548 / sqrt(2 pi) =
  0.2187 K, 4.19 $.
- The boron line. The planner sends a flow ``committed`` control steps before the
  knot it plans it for, so that must be the env's delay: a command sent on a
  control step first moves the core's boron ``committed`` steps later, and the
  commands already in the line arrive in the order the planner reads them.
- Foresight. The wobble the planner foresees is the one the env draws.
- The oracle on the scored plant (the plan on the wobble's mean path, the plant
  with its wobble and sensor noise), against the reference on the same seed, as the
  records gate asks of an oracle: it never trips, its actions stay in the box, its
  solves all converge, the offset's penalty time stays inside its hour, and it
  costs no more than the reference. On a baseload hour (medium tier) both cost
  next to nothing, so a dollar of slack; on a day of load follow late in the cycle
  (large tier) the reference waits hours for its boron and the oracle must cost
  under half as much.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import enable_x64, float64
from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor import oracle_model as M
from target_gym.reactor.experts import make_reactor_pid_controller
from target_gym.reactor.oracle_day import (
    DayOracle,
    look_ahead,
    plant_x,
    stage_model,
    wobble_path,
)
from target_gym.registry import REGISTRY

START = ReactorParams(cycle_range=(0.0, 0.0))
#: The same with Tavg's free band out of the way.
NO_BAND = ReactorParams(cycle_range=(0.0, 0.0), e_tol=50.0)
#: A plant that resets lined out on its program at full load, and stays there.
QUIET = ReactorParams(
    cycle_range=(0.0, 0.0),
    demand_sigma=0.0,
    burnup_range=(0.0, 0.0),
    carried_days_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(1.0, 1.0),
    ramp_down_range=(1e6, 1e6),
)


def _reset(seed=3, p=START):
    return Reactor().reset_env(jax.random.PRNGKey(seed), p)[1]


@pytest.mark.parametrize(
    "move, size, dollars",
    [
        (None, 0.0, 666.67),
        (M.ROD_IN, 10.0 / 228.0, 676.67),
        (M.ROD_OUT, 10.0 / 228.0, 676.67),
        (M.DILUTE, 1.0, 855.07),
        (M.BORATE, 1.0, 723.19),
    ],
)
def test_a_knot_costs_what_the_env_charges(move, size, dollars):
    with enable_x64():
        p, s = float64(NO_BAND), float64(_reset(p=NO_BAND))
        model = stage_model(Reactor(), p, 4)
        day = look_ahead(s, p, 5, jnp.zeros(6))
        day = day._replace(committed=jnp.asarray(0))  # every flow is the plan's
        x = jnp.asarray([s.I_hat, s.Xe_hat, 0.0, 0.0, s.boron / M.PPM, 0.1, 0.1, 0.0])
        u = jnp.zeros(M.NU)
        u = u if move is None else u.at[move].set(size)
        d = {"day": day, "room": jnp.asarray(1.5), "sigma": jnp.asarray(0.0)}
        cost = model.step(d, x, u, 0)[1]
    assert float(cost) == pytest.approx(dollars, abs=0.01)


@pytest.mark.parametrize(
    "ppm, wobble, dollars", [(0.0, 0.0, 0.0), (10.0, 0.0, 71.56), (3.0, 0.005, 4.19)]
)
def test_a_knot_charges_tavg_past_its_free_band(ppm, wobble, dollars):
    quiet = _reset(p=QUIET)  # in the scored run's float32, as the env resets
    with enable_x64():
        p, s = float64(QUIET), float64(quiet)
        model = stage_model(Reactor(), p, 4)
        day = look_ahead(s, p, 5, jnp.zeros(6))
        x = plant_x(s, p)
        x = x.at[M.BORON].add(ppm / M.PPM)
        d = {"day": day, "room": jnp.asarray(1.5), "sigma": jnp.asarray(wobble)}
        cost = model.step(d, x, jnp.zeros(M.NU), 0)[1]
    # float32 reset: a few mK of Tavg, 0.1 $.
    assert float(cost) == pytest.approx(dollars, abs=0.2)


def test_a_flow_sent_now_reaches_the_core_when_the_planner_says():
    env, p = Reactor(), START
    s = _reset()
    with enable_x64():
        line = int(look_ahead(float64(s), float64(p), 4, jnp.zeros(5)).committed)
    assert 9 <= line <= 29  # the env draws 10 to 30 steps, the last one the core's
    hold = 1.0 - 2.0 * float(s.rod_in)
    key, boron = jax.random.PRNGKey(0), [float(s.boron)]
    step = jax.jit(env.step_env)
    for t in range(line + 2):
        flow = -1.0 if t == 0 else 0.0
        s = step(key, s, jnp.array([hold, flow, 1.0]), p)[1]
        boron.append(float(s.boron))
    moved = np.abs(np.diff(boron)) > 0.0
    assert not moved[:line].any()
    assert moved[line]
    assert not moved[line + 1]


def test_the_line_is_read_in_the_order_it_arrives():
    env, p = Reactor(), START
    s = _reset()
    key, step = jax.random.PRNGKey(0), jax.jit(env.step_env)
    hold = 1.0 - 2.0 * float(s.rod_in)
    sent = [-0.1, 0.2, -0.3, 0.4, -0.5]
    for flow in sent:
        s = step(key, s, jnp.array([hold, flow, 1.0]), p)[1]
    with enable_x64():
        day = look_ahead(float64(s), float64(p), 4, jnp.zeros(5))
    line = int(day.committed)
    arriving = np.asarray(day.arriving)[:line]
    # The five commands are the last to arrive, in the order they were sent.
    np.testing.assert_allclose(arriving[line - 5 :], sent, atol=1e-6)
    np.testing.assert_allclose(arriving[: line - 5], 0.0, atol=1e-6)


def test_the_foreseen_wobble_is_the_one_the_env_draws():
    env, p = Reactor(), START
    s0 = _reset(seed=11)
    key, step = jax.random.PRNGKey(5), jax.jit(env.step_env)
    hold, s, drawn = 1.0 - 2.0 * float(s0.rod_in), s0, []
    for _ in range(20):
        s = step(key, s, jnp.array([hold, 0.0, 1.0]), p)[1]
        drawn.append(float(s.load_wobble))
    path = wobble_path(
        s0.demand_key, jnp.asarray(s0.physics_time, jnp.int32), s0.load_wobble, 120,
        jnp.float32(p.demand_theta), jnp.float32(p.demand_sigma),
        jnp.float32(p.delta_t),
    )  # fmt: skip
    assert np.std(drawn) > 0.0
    np.testing.assert_allclose(np.asarray(path)[5::6], drawn, rtol=1e-5, atol=1e-8)


# --- the oracle on the scored plant ---


def _run(scenario, cycle, steps, seed=100):
    """``steps`` control steps of ``scenario`` under the oracle and under the
    reference, on the same seed: each one's (cost, trips, penalty time in h), the
    oracle's actions and its solver report."""
    spec, env = REGISTRY["reactor"], Reactor()
    (sc,) = [s for s in scenarios.suite(spec) if s.name == scenario]
    p = scenarios.params_for(spec, sc, cycle_range=(cycle, cycle))
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), steps)
    step = jax.jit(env.step_env)

    def play(act):
        obs, s = env.reset_env(jax.random.PRNGKey(seed), p)
        cost, trips, actions = 0.0, 0, []
        for k in keys:
            a = act(obs, s)
            actions.append(np.asarray(a, float))
            obs, s, r, _, info = step(k, s, jnp.asarray(a, jnp.float32), p)
            cost, trips = cost - float(r), trips + int(info["tripped"])
        return (cost, trips, float(s.ao_outside) / 3600.0), np.array(actions)

    oracle = DayOracle(env, plan_params(spec, p))
    expert, memory = make_reactor_pid_controller(), {}

    def reference(obs, _s):
        m = memory.get("m", expert.init_state)
        a, memory["m"] = expert.step(expert.params, m, obs)
        return a

    planned, actions = play(oracle.step)
    return planned, play(reference)[0], actions, oracle.solver_report()


def _well_run(planned, actions, report):
    cost, trips, penalty = planned
    assert trips == 0
    assert np.all(np.abs(actions) <= 1.0) and actions.shape[1] == 3
    assert report["solver_failures"] == 0 and report["solver_capped"] == 0
    assert penalty < 1.0


@pytest.mark.medium
def test_the_oracle_holds_a_baseload_hour_as_cheaply_as_the_reference():
    planned, reference, actions, report = _run("baseload", 0.8, 120)
    _well_run(planned, actions, report)
    assert np.all(actions[:, 2] == 1.0)  # the turbine stays on its plan
    assert planned[0] <= reference[0] + 1.0


@pytest.mark.large
def test_the_oracle_halves_the_references_cost_late_in_the_cycle():
    planned, reference, actions, report = _run("edf_daily", 0.8, 2880)
    _well_run(planned, actions, report)
    assert reference[1] == 0  # the reference does not trip either
    assert planned[0] < 0.5 * reference[0]
