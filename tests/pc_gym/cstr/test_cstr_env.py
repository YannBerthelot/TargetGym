"""The jacketed CSTR's balances and step (version 8: the feed is a move).

Contract, from the energy and mass balances in env.py's docstring [S1]: the
reaction consumes A and releases heat; more cooling water cools the jacket,
and a cooler jacket cools the reactor; with no reactant left the cooling wins;
more feed brings more reactant and, colder than the reactor [S1, Tables 3, 4],
cools it at once; RK4 and Euler agree at a small step; each raw command clips
at its valve's range, 40 to 136 g/s of water and 40 to 110 g/s of feed.
"""

import jax.numpy as jnp

from target_gym.pc_gym.cstr.env import (
    CSTRParams,
    CSTRState,
    compute_next_state,
    compute_velocity,
)

P = CSTRParams()
X = jnp.array([0.31, 331.15, 320.15])  # the design point [S1]


def moves(F_w, q=P.q):
    return jnp.array([F_w, q])


def test_compute_velocity_trends():
    """More water cools the jacket; a cooler jacket cools the reactor."""
    more, _ = compute_velocity(X, action=moves(P.Fw_max), params=P)
    less, _ = compute_velocity(X, action=moves(P.Fw_min), params=P)
    assert more[2] < less[2]
    cold, _ = compute_velocity(X.at[2].set(300.0), action=moves(P.Fw_min), params=P)
    assert cold[1] < less[1]
    none, _ = compute_velocity(X.at[0].set(0.0), action=moves(P.Fw_max), params=P)
    assert none[1] < 0.0


def test_the_reaction_consumes_a_and_releases_heat():
    """At the feed's own concentration and temperature the inflow balances,
    so what is left is the reaction: A falls and T rises."""
    p = P.replace(UA=0.0)
    v, _ = compute_velocity(
        jnp.array([P.Caf, 331.15, 331.15]),
        action=moves(0.0),
        params=p,
        extra=(331.15, P.Caf, P.k0),
    )
    assert v[0] < 0.0 and v[1] > 0.0


def test_more_feed_brings_reactant_and_cools_at_once():
    """The feed is richer and colder than the reactor: at the same state, more
    of it raises dC_A/dt and lowers dT/dt."""
    low, _ = compute_velocity(X, action=moves(7200.0, P.q_min), params=P)
    high, _ = compute_velocity(X, action=moves(7200.0, P.q_max), params=P)
    assert high[0] > low[0] and high[1] < low[1]


def _state():
    return CSTRState(time=0, C_a=0.31, T=331.15, target_T=331.15, F_w=5334.0)


def test_integration_methods_agree_for_small_dt():
    params = CSTRParams(delta_t=1e-4)
    raw = jnp.array([0.3, 0.1])
    euler, _ = compute_next_state(raw, _state(), params, integration_method="euler_1")
    rk4, _ = compute_next_state(raw, _state(), params, integration_method="rk4_1")
    for f in ("C_a", "T", "T_j"):
        assert jnp.allclose(getattr(euler, f), getattr(rk4, f), atol=1e-5)


def test_action_clipping():
    """A raw command past the box runs each valve at its end."""
    params = CSTRParams(delta_t=0.1)
    s, _ = compute_next_state(
        jnp.array([100.0, 100.0]), _state(), params, integration_method="euler_1"
    )
    assert s.F_w == params.Fw_max and s.q == params.q_max
    s, _ = compute_next_state(
        jnp.array([-100.0, -100.0]), _state(), params, integration_method="euler_1"
    )
    assert s.F_w == params.Fw_min and s.q == params.q_min
