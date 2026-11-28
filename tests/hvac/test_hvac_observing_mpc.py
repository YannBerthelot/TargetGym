"""hvac's observation-only MPC (v1 item 5, as the owner read it on 2026-11-28).

Contract:

- The planner reads the observation only, through the shared estimator
  (``estimation.Estimator``): the room, outdoor and core sensors and the day-ahead
  forecast, from which it estimates the building's mass, the slab core, the air
  and the weather's deviation.
- It prices comfort at its expected cost under the weather's forecast error: the
  room's error at a lead is the plant's own weather innovations to come, carried
  to the room by the building at a held supply (``room_spread``), and the expected
  kelvin-hours outside the band are those of a normal error of that spread
  (``expected_excess``). Nothing is tuned: the spread is the plant's.

Where the expected values come from, none of them the function under test:

- the expected excess of a normal error at the edge is its spread over sqrt(2 pi),
  and far inside the band 0, far outside the mean excess; elsewhere a Monte Carlo
  of normal draws, within its standard error;
- the room's forecast error: the plant itself, run with its own weather noise from
  one state at a held supply over many keys; the spread of the room's temperature
  across them at each lead, within the sampling error of a standard deviation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.hvac.env import HVACParams, steady_state
from target_gym.hvac.env_jax import BuildingHVAC

P = HVACParams()


@pytest.mark.parametrize("spread", [0.05, 0.3, 1.0])
def test_a_room_on_the_edge_expects_its_spread_over_root_two_pi(spread):
    from target_gym.hvac.oracle import expected_excess

    got = float(expected_excess(jnp.asarray(0.0), jnp.asarray(spread)))
    assert got == pytest.approx(spread / np.sqrt(2.0 * np.pi), rel=1e-6)


def test_far_from_the_edge_the_expected_excess_is_the_plain_one():
    from target_gym.hvac.oracle import expected_excess

    assert float(expected_excess(jnp.asarray(-3.0), jnp.asarray(0.2))) < 1e-12
    assert float(expected_excess(jnp.asarray(2.0), jnp.asarray(0.2))) == pytest.approx(
        2.0, rel=1e-9
    )


@pytest.mark.parametrize("mean", [-0.4, 0.15, 0.6])
def test_the_expected_excess_is_a_normal_errors_mean_excess(mean):
    from target_gym.hvac.oracle import expected_excess

    spread = 0.3
    draws = np.random.default_rng(7).normal(mean, spread, 400_000)
    excess = np.maximum(draws, 0.0)
    # Four standard errors of the sample mean.
    tol = 4.0 * excess.std() / np.sqrt(draws.size)
    got = float(expected_excess(jnp.asarray(mean), jnp.asarray(spread)))
    assert abs(got - excess.mean()) < tol


@pytest.mark.medium
def test_the_rooms_forecast_error_is_the_spread_the_plant_draws():
    from target_gym.hvac.oracle import room_spread

    env = BuildingHVAC()
    T_m, T_c, _, T_sup = (
        float(x) for x in steady_state(P, float(P.T_out_mean), float(P.band_low))
    )
    _, s0 = env.reset_env(jax.random.PRNGKey(0), P)
    s0 = s0.replace(
        T_mass=jnp.float32(T_m), T_core=jnp.float32(T_c), weather_dev=jnp.float32(0.0)
    )
    raw = jnp.full(
        (1,), 2.0 * (T_sup - P.T_supply_min) / (P.T_supply_max - P.T_supply_min) - 1.0
    )
    leads = (4, 24, 96)

    def run(key):
        def step(s, k):
            _, s1, *_ = env.step_env(k, s, raw, P)
            return s1, s1.T_air

        return jax.lax.scan(step, s0, jax.random.split(key, max(leads)))[1]

    n = 4096
    rooms = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(11), n))
    spread = np.asarray(room_spread(P, max(leads)))
    for k in leads:
        drawn = float(np.std(np.asarray(rooms[:, k - 1]), ddof=1))
        # A sample standard deviation's standard error is sigma / sqrt(2 (n - 1)).
        tol = 4.0 * drawn / np.sqrt(2.0 * (n - 1))
        assert abs(spread[k - 1] - drawn) < tol, (k, spread[k - 1], drawn)


@pytest.mark.medium
@pytest.mark.parametrize("step", [100, 288])
def test_the_window_planner_converges_from_the_experts_states(step):
    """A cold solve of the registry's window planner from the shipped expert's
    state, before the scored window and at its start, converges: neither capped
    nor failed. The slab heat's two edges are smoothed at the protocol's width
    (``data/oracle_widths/hvac.json``); at width 0 every such solve hit the cap
    (2026-11-28, KKT error near 0.3)."""
    import target_gym as tg
    from target_gym.experts.mpc import plan_params
    from target_gym.hvac.oracle import make_hvac_mpc
    from target_gym.registry import REGISTRY

    env, params = tg.make("hvac", scenario="normal")
    expert = tg.make_expert("hvac")
    key = jax.random.PRNGKey(203)
    obs, state = env.reset_env(key, params)
    memory = expert.reset()
    for k in jax.random.split(key, step):
        action, memory = expert.step(memory, obs)
        obs, state, *_ = env.step_env(k, state, action, params)
    oracle = make_hvac_mpc(env, plan_params(REGISTRY["hvac"], params))
    oracle.step(obs, state)
    for solve in oracle.certificates:
        assert not solve["capped"] and not solve["failed"], solve


# --- The planner prices comfort as the reward does (owner, 2026-11-28) --------
#
# The reward's lost work beyond a band edge is the published curve's fall from
# it, a cubic in the distance d past the edge: its terms are the planner's rows.
# Below the band the three are hinges of degree 1 to 3; above it the cubic term
# is negative, which a hinge cannot carry, so the rows stop at degree 2.


def _curve(T):
    T = min(max(T, 15.0), 35.0)
    return 0.1647524 * T - 0.0058274 * T**2 + 0.0000623 * T**3 - 0.4685328


@pytest.mark.parametrize("d", [0.05, 0.4, 1.0, 2.5])
def test_the_planners_rows_below_the_band_are_the_curves_fall(d):
    from target_gym.hvac.oracle import comfort_rows

    below, _ = comfort_rows(P)
    rows = sum(w * d**k for k, w in enumerate(below, start=1))
    assert rows == pytest.approx(_curve(21.0) - _curve(21.0 - d), rel=1e-9)


@pytest.mark.parametrize("d", [0.05, 0.4, 1.0, 2.0])
def test_the_planners_rows_above_the_band_overstate_the_curve_by_its_cubic_term(d):
    """Above the band the curve's fall from 26 C is a d + b d^2 - c d^3 with the
    curve's cubic coefficient c = 0.0000623 (its Taylor terms at the edge): the
    rows keep the first two, so they overstate the fall by exactly c d^3, 0.5 %
    of it at 1 K and 1.8 % at 2 K."""
    from target_gym.hvac.oracle import comfort_rows

    _, above = comfort_rows(P)
    rows = sum(w * d**k for k, w in enumerate(above, start=1))
    fall = _curve(26.0) - _curve(26.0 + d)
    assert rows - fall == pytest.approx(0.0000623 * d**3, rel=1e-6, abs=1e-12)
    assert min(above) > 0.0


@pytest.mark.parametrize("power", [1, 2, 3])
@pytest.mark.parametrize("mean", [-0.4, 0.15, 0.6])
def test_the_expected_powers_of_the_excess_are_a_normal_errors(power, mean):
    from target_gym.hvac.oracle import expected_power

    spread = 0.3
    draws = np.random.default_rng(11).normal(mean, spread, 400_000)
    moment = np.maximum(draws, 0.0) ** power
    tol = 4.0 * moment.std() / np.sqrt(draws.size)
    got = float(expected_power(jnp.asarray(mean), jnp.asarray(spread), power))
    assert abs(got - moment.mean()) < tol
