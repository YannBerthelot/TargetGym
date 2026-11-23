"""The reactor planner's model (``reactor/oracle_model.py``).

Contract, written before the runs. The model carries the plant from knot to knot
(5 min) with its heat in balance, so it must say what the env says once the env has
settled, and stay inside the env's limits where it rounds them:

- Balance. At the env's own line-out (a reset) the balance's Tavg is the plant's.
  By hand, with the rods and the boron still: a ppm of boron moves it by
  ``-worth / (alpha_coolant + alpha_fuel)``, -8 / 15 K at the start of the cycle
  and -10.32 / 53.4 K at 80 % of it; a unit of load moves it by the fuel's 471.2 K
  (16.67 K of half rise, 454.5 K of fuel over hot leg) times the Doppler's 3 pcm/K
  over both feedbacks, -94.2 K at the start (3 x 471.2 / 15).
- Mixing. The env's feed and bleed: 30 s of water at 10 kg/s into 250 t takes
  1000 ppm to 1000 exp(-0.0012) = 998.8007; 30 s of 7800 ppm acid at 3 kg/s takes
  it to 7800 - 6800 exp(-0.00036) = 1002.4476; no flow leaves it.
- Penalty time. An offset on a straight line between two knots spends, past a band
  of 5 % counted from 0.5 % inside it: none of the knot from 2 % to 2 %, all of it
  from 8 % to 8 %, (10.5 - 4.5) / (10.5 - 2) = 0.706 of it from 2 % to 10.5 %,
  whichever end comes first and whatever the sign.
- Where it rounds the env's rules it stays on their safe side: the share of the
  time outside that counts is never under the env's, and the offset's room never
  over the env's limit, at every power from 15 % of rated up.
- Tavg's wobble. A Tavg that wobbles with a spread s about the plan's sits past
  its 1.6 K free band, on average: s / sqrt(2 pi) = 0.3989 s with the plan on the
  band's edge (0.1995 K at s = 0.5 K); the plan's own excess, and barely more, far
  outside (1.5 K at 3.1 K off program); and next to nothing on program, 2 x (0.5
  phi(3.2) - 1.6 Phi(-3.2)) = 0.0002 K. The same on the hot side.
- The offset's hour as data. A plan is given the knots it may spend outside the
  band (``Day.free``): reaching any other knot its room is the band, 5 % of rated
  delta I, and reaching a free one the acceptable-operation line (30 % at half
  power). It then counts no penalty time of its own; a plan given no knots may
  spend any and counts them: a knot spent outside at half power is 5 min, 0.0833 h.
- Derivatives. The solver plans on the model's first and second derivatives, so
  a knot's state and rows have finite ones in its state and its moves, whatever
  the env's rates compute on the inputs the model does not use.
- Fidelity (medium tier). The plant follows a plan as the model says. A plan (the
  planner's own first guess for the day, by the model's inverse) is followed on
  the env with the wobble off, every move on a straight line between its knots
  (``oracle.action``), with no solve in between; at every knot the env is
  within the planner's back-offs of the model: Tavg within ``TAVG_BACKOFF``,
  delta I within ``OFFSET_BACKOFF``; the boron within 0.5 ppm and the xenon
  within 0.1 % of full power's (the same rates, a knot apart). The mixing is the
  env's own, so the boron's gap is the env's float32: a sub-step closes
  ``1 - exp(-z)`` of the way to the feed with z near 1e-5 for a small flow, known
  to 6e-8 / 2 of 1e-5, 0.3 % of the step, and a day moves some 150 ppm. The model
  runs the whole day open loop, so its errors add up. The model does not play
  the steam generators' safety valves: a solved plan keeps its steam clear of
  them, and the knots where the first guess does not (at most two, at the bottom
  of a fast ramp down) are left out of the comparison.
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
from target_gym.reactor.env import offset_limit, penalty_rate
from target_gym.reactor.oracle import (
    OFFSET_BACKOFF,
    TAVG_BACKOFF,
    Followed,
    action,
    first_guess,
    look_ahead,
    plant_x,
    roll,
)
from target_gym.reactor.oracle_model import (
    balance,
    mixed,
    noisy_excess,
    offset_room,
    penalty_share,
    time_outside,
)
from target_gym.registry import REGISTRY

P = ReactorParams()


@pytest.mark.parametrize("cycle", [0.0, 0.45, 0.9])
def test_the_balance_is_the_plants_at_its_line_out(cycle):
    env, p = Reactor(), ReactorParams(cycle_range=(cycle, cycle))
    _, s = env.reset_env(jax.random.PRNGKey(3), p)
    T_avg, T_hot, T_cold = balance(s.n, s.rod_in, s.boron, s.Xe_hat, 0.0, s.cycle0, p)
    # float32 env: a few mK at 580 K.
    assert float(T_avg) == pytest.approx(
        float(0.5 * (s.T_coolant + s.T_cold)), abs=0.02
    )
    assert float(T_hot) == pytest.approx(float(s.T_coolant), abs=0.02)
    assert float(T_cold) == pytest.approx(float(s.T_cold), abs=0.02)


@pytest.mark.parametrize("cycle, k", [(0.0, -8.0 / 15.0), (0.8, -10.32 / 53.4)])
def test_a_ppm_of_boron_moves_the_balance_by_both_feedbacks(cycle, k):
    with enable_x64():
        at = lambda b: balance(1.0, 0.1, b, 1.0, 0.0, cycle, P)[0]  # noqa: E731
        boron = 1690.0 * (1.0 - cycle)
        assert float(at(boron + 1.0) - at(boron)) == pytest.approx(k, rel=1e-6)


def test_a_unit_of_load_moves_the_balance_by_the_fuels_doppler():
    with enable_x64():
        at = lambda n: balance(n, 0.1, 1690.0, 1.0, 0.0, 0.0, P)[0]  # noqa: E731
        assert float(at(1.0) - at(0.9)) / 0.1 == pytest.approx(-94.2, rel=1e-3)


@pytest.mark.parametrize(
    "dilute, borate, after",
    [(1.0, 0.0, 998.8007), (0.0, 1.0, 1002.4476), (0.0, 0.0, 1000.0)],
)
def test_the_boron_mixes_as_the_envs_feed_and_bleed(dilute, borate, after):
    with enable_x64():
        out = mixed(jnp.float64(1000.0), dilute, borate, 30.0, P)
    assert float(out) == pytest.approx(after, abs=1e-3)


@pytest.mark.parametrize(
    "before, after, share",
    [
        (0.02, 0.02, 0.0),
        (0.08, 0.08, 1.0),
        (0.02, 0.105, 0.706),
        (0.105, 0.02, 0.706),
        (-0.02, -0.105, 0.706),
    ],
)
def test_the_penalty_time_is_the_crossings_share_of_the_knot(before, after, share):
    with enable_x64():
        out = time_outside(before, after, 0.05, 0.005, 0.01)
    assert float(out) == pytest.approx(share, abs=0.01)


def test_the_time_outside_that_counts_is_never_under_the_envs():
    n = np.linspace(0.15, 1.05, 181)
    counted = np.asarray(penalty_share(n))
    env = np.asarray(penalty_rate(0.2 + 0.0 * n, n, P, np))  # well outside the band
    assert np.all(counted >= env - 1e-9)
    assert float(penalty_share(0.5)) == pytest.approx(1.0)
    assert float(penalty_share(0.4)) == pytest.approx(0.5)


def test_the_offsets_room_is_never_over_the_envs_limit():
    n = np.linspace(0.5, 1.05, 111)
    with enable_x64():  # float32 rounds 0.295 up by 2e-8
        room = np.asarray(offset_room(jnp.asarray(n), P))
        at = lambda v: float(offset_room(jnp.float64(v), P))  # noqa: E731
        edges = at(0.95), at(0.5), at(0.3)
    assert np.all(room <= np.asarray(offset_limit(n, P, np)) + 1e-9)
    # The line stops 5 % short of its 31 % at half power: 0.11 + 0.2 x 0.95.
    assert edges == pytest.approx((0.05, 0.30, 0.30), abs=1e-3)


@pytest.mark.parametrize(
    "error, excess",
    [(1.6, 0.1995), (-1.6, 0.1995), (3.1, 1.5), (-3.1, 1.5), (0.0, 0.0002)],
)
def test_a_wobbling_tavg_pays_its_mean_excess_past_the_band(error, excess):
    with enable_x64():
        out = noisy_excess(jnp.float64(error), 1.6, 0.5)
    assert float(out) == pytest.approx(excess, abs=2e-4)


def _half_power_knots(free):
    """Two knots of the model at half power with the rods 40 % in, far outside
    the band: the offset's room reaching each, and the clock after the first."""
    _, s = Reactor().reset_env(jax.random.PRNGKey(3), ReactorParams())
    with enable_x64():
        p, st = float64(ReactorParams()), float64(s)
        day = look_ahead(st, p, 3, jnp.zeros(4), free=free)
        x = plant_x(st, p).at[M.RODS].set(0.4).at[M.SHORT].set(0.5)
        step = lambda x, k: M.knot_step(  # noqa: E731
            p, day, x, jnp.zeros(M.NU), k, OFFSET_BACKOFF, 0.1
        )
        x1, first = step(x, 0)
        _, second = step(x1, 1)
    assert abs(float(first[2])) > 0.1  # delta I is far outside its band
    return float(first[3]), float(second[3]), float(x1[M.CLOCK] - x[M.CLOCK])


def test_outside_its_free_knots_a_plans_offset_room_is_the_band():
    with enable_x64():
        free = jnp.zeros(4).at[2].set(1.0)
    first, second, clock = _half_power_knots(free)
    assert first == pytest.approx(0.05)
    assert second == pytest.approx(0.30, abs=2e-3)
    assert clock == 0.0  # its knots are given, and already counted


def test_a_plan_given_no_knots_may_spend_any_and_counts_them():
    first, second, clock = _half_power_knots(None)
    assert first == pytest.approx(0.30, abs=2e-3)
    assert second == pytest.approx(0.30, abs=2e-3)
    assert clock == pytest.approx(300.0 / 3600.0, rel=1e-3)


def test_a_knots_derivatives_are_finite():
    _, s = Reactor().reset_env(jax.random.PRNGKey(3), ReactorParams())
    with enable_x64():
        p, st = float64(ReactorParams()), float64(s)
        day = look_ahead(st, p, 3, jnp.zeros(4))
        x = plant_x(st, p)

        def knot(z):
            x1, rows = M.knot_step(p, day, z[: M.NX], z[M.NX :], 0, OFFSET_BACKOFF, 0.1)
            return jnp.concatenate([x1, rows])

        z = jnp.concatenate([x, jnp.zeros(M.NU)])
        first = jax.jacobian(knot)(z)
        second = jax.hessian(lambda z: jnp.sum(knot(z)))(z)
    assert np.all(np.isfinite(np.asarray(first)))
    assert np.all(np.isfinite(np.asarray(second)))


# --- the model against the env, a whole day open loop ---


@pytest.mark.medium
@pytest.mark.parametrize(
    "scenario, cycle", [("edf_daily", 0.8), ("edf_daily", 0.0), ("fast_return", 0.45)]
)
def test_the_plant_follows_a_plan_as_the_model_says(scenario, cycle):
    spec, env, knots = REGISTRY["reactor"], Reactor(), 288
    (sc,) = [s for s in scenarios.suite(spec) if s.name == scenario]
    p = plan_params(spec, scenarios.params_for(spec, sc, cycle_range=(cycle, cycle)))
    _, s = env.reset_env(jax.random.PRNGKey(100), p)
    with enable_x64():
        pp, st = float64(p), float64(s)
        day = look_ahead(st, pp, knots, jnp.zeros(knots + 1))
        x0 = plant_x(st, pp)
        moves = jax.jit(lambda: first_guess(pp, day, x0, knots))()
        states, rows = jax.jit(lambda: roll(pp, day, x0, moves))()
    states, rows = np.asarray(states), np.asarray(rows)
    load = np.asarray(day.plan)[: knots + 1] - states[:, M.SHORT]
    plan = Followed(0, np.asarray(moves), states, load, int(day.committed))
    assert np.abs(np.diff(load)).max() > 0.05  # the plan does move the turbine

    step, key, gaps = jax.jit(env.step_env), jax.random.PRNGKey(0), []
    for t in range((knots - 3) * M.KNOT):  # the last flows are sent 3 knots early
        s = step(key, s, jnp.asarray(action(p, plan, s), jnp.float32), p)[1]
        if (t + 1) % M.KNOT == 0:
            k = (t + 1) // M.KNOT
            gaps.append([
                rows[k - 1, 1] - float(0.5 * (s.T_coolant + s.T_cold)),
                rows[k - 1, 2] - float(s.delta_I - s.n * p.ao_ref),
                states[k, M.BORON] * M.PPM - float(s.boron),
                states[k, M.XENON] - float(s.Xe_hat),
                states[k, M.IODINE] - float(s.I_hat),
                rows[k - 1, 8],
            ])  # fmt: skip
    gaps = np.array(gaps)
    # The model does not play the safety valves, and a plan of the solver's keeps
    # its steam clear of them by the back-off; the first guess does not always (a
    # fast ramp down outruns its boron for a knot), so those knots are left out.
    inside = gaps[:, 5] >= TAVG_BACKOFF
    assert inside.sum() >= len(gaps) - 2
    T_avg, off, boron, xenon, iodine = np.abs(gaps[inside, :5]).max(axis=0)
    assert T_avg < TAVG_BACKOFF
    assert off < OFFSET_BACKOFF
    assert boron < 0.5
    assert xenon < 1e-3 and iodine < 1e-3
