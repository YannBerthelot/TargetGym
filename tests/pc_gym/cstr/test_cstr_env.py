"""The jacketed CSTR's balances and step (version 5).

Contract, from the energy and mass balances in env.py's docstring [S1]: the
reaction consumes A and releases heat; more cooling water cools the jacket,
and a cooler jacket cools the reactor; with no reactant left the cooling wins;
RK4 and Euler agree at a small step; the raw command clips at the valve's
range.
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


def test_compute_velocity_trends():
    """More water cools the jacket; a cooler jacket cools the reactor."""
    more, _ = compute_velocity(X, action=P.Fw_max, params=P)
    less, _ = compute_velocity(X, action=P.Fw_min, params=P)
    assert more[2] < less[2]
    cold, _ = compute_velocity(X.at[2].set(300.0), action=P.Fw_min, params=P)
    assert cold[1] < less[1]
    none, _ = compute_velocity(X.at[0].set(0.0), action=P.Fw_max, params=P)
    assert none[1] < 0.0


def test_the_reaction_consumes_a_and_releases_heat():
    """At the feed's own concentration and temperature the inflow balances,
    so what is left is the reaction: A falls and T rises."""
    p = P.replace(UA=0.0)
    v, _ = compute_velocity(
        jnp.array([P.Caf, 331.15, 331.15]),
        action=0.0,
        params=p,
        extra=(331.15, P.Caf, P.k0),
    )
    assert v[0] < 0.0 and v[1] > 0.0


def _state():
    return CSTRState(time=0, C_a=0.31, T=331.15, target_T=331.15, F_w=5334.0)


def test_integration_methods_agree_for_small_dt():
    params = CSTRParams(delta_t=1e-4)
    euler, _ = compute_next_state(0.3, _state(), params, integration_method="euler_1")
    rk4, _ = compute_next_state(0.3, _state(), params, integration_method="rk4_1")
    for f in ("C_a", "T", "T_j"):
        assert jnp.allclose(getattr(euler, f), getattr(rk4, f), atol=1e-5)


def test_action_clipping():
    """A raw command past the box runs the valve at its end, 40 or 110 g/s."""
    params = CSTRParams(delta_t=0.1)
    s, _ = compute_next_state(100.0, _state(), params, integration_method="euler_1")
    assert s.F_w == params.Fw_max
    s, _ = compute_next_state(-100.0, _state(), params, integration_method="euler_1")
    assert s.F_w == params.Fw_min
