"""Column A's tray hydraulics and flow limits.

Contract, from Skogestad's 1997 tutorial and ``colamod.m``, not the code: 39 trays
lag by taul = 0.063 min with lambda = 0, so reflux reaches the reboiler after
2.46 min (section 4.1, figure 4); the LV RGA nears 1 above 0.41 rad/min (4.7.3);
the steady gains are eq. 31. Moving holdup conserves the light component. Ours:
every target holds inside the flow limits at feeds within two sigma, and RK4's
real-axis limit, 2.785, holds across the flow box.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import gamma

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from target_gym.pc_gym.distillation.env import N_STAGES as N
from target_gym.pc_gym.distillation.env import (
    DistillationParams,
    compute_next_state,
    compute_velocity,
    steady_liquid,
    tray_holdup,
)
from target_gym.pc_gym.distillation.env_jax import _NOMINAL_PROFILE, DistillationColumn

P = DistillationParams(feed_step=0.0, zF_step=0.0)
ENV, KEY = DistillationColumn(), jax.random.PRNGKey(0)
L0, V0, TRAYS, TAUL = 2.70629, 3.20629, 39, 0.063  # colamod.m
EQ_31 = [[0.8754, -0.8618], [1.0846, -1.0982]]  # d[yD, xB] / d[L, V]


@jax.jit
def _target_steady_state(yD, xB, zF, F=1.0):
    """[x, L, V] holding both products: Newton, float64 under ``enable_x64``."""
    p = P.replace(F=F)

    def residual(z):
        x, L, V = z[:N], z[N], z[N + 1]
        here = jnp.concatenate([x, steady_liquid(L, p)])
        dx = compute_velocity(here, (L, V), zF, p)[0][:N]
        return jnp.concatenate([dx, jnp.stack([x[-1] - yD, x[0] - xB])])

    def newton(_, z):
        return z - jnp.linalg.solve(jax.jacfwd(residual)(z), residual(z))

    start = jnp.concatenate([jnp.asarray(_NOMINAL_PROFILE, float), jnp.array([L0, V0])])
    return jax.lax.fori_loop(0, 15, newton, start)


@pytest.fixture(scope="module")
def response():
    """G(s) from (L, V) to (yD, xB) at the published operating point."""

    def rates(state, u):
        return compute_velocity(state, (u[0], u[1]), 0.5, P)[0]

    with enable_x64():
        z = _target_steady_state(0.99, 0.01, 0.5)
        here = jnp.concatenate([z[:N], steady_liquid(z[N], P)])
        A, B = (np.asarray(J) for J in jax.jacfwd(rates, (0, 1))(here, z[N:]))
    C = np.eye(len(A))[[N - 1, 0]]
    return lambda s: C @ np.linalg.solve(s * np.eye(len(A)) - A, B)


def _rga(G):
    return (G * np.linalg.inv(G).T)[0, 0]


def test_a_reflux_step_reaches_the_reboiler_after_the_liquid_lag():
    """39 equal lags in series step like the gamma CDF. 0.01: RK4 on one lag at
    h / taul = 1 shifts this by about 0.1%."""
    _, state = ENV.reset_env(KEY, P)
    start, raw = state.liquid[0], ENV.initial_action(state.replace(L=state.L + 0.1), P)
    moved = []
    for _ in range(5):
        state = compute_next_state(raw, state, P, KEY)[0]
        moved.append(float(state.liquid[0] - start) / 0.1)
    expected = gamma(TRAYS, scale=TAUL).cdf(np.arange(1, 6))
    np.testing.assert_allclose(moved, expected, atol=0.01)


def test_the_steady_gains_are_the_published_matrix(response):
    G = response(0.0)
    np.testing.assert_allclose(G, EQ_31, atol=1e-4)
    assert _rga(G) == pytest.approx(35.94, abs=0.01)
    # 145.5 from a separate NumPy model written from colamod.m; 1.96 = (yD - xB) / D.
    singular = np.linalg.svd(G, compute_uv=False)
    assert singular[0] / singular[1] == pytest.approx(145.5, abs=0.1)
    assert G[0, 0] + G[1, 0] == pytest.approx(0.98 / 0.5, abs=1e-3)


@pytest.mark.parametrize(
    "omega, reference",
    [(0.41, 1.302), (1.0, 0.571), (2.0, 0.753), (5.0, 1.127), (10.0, 0.998)],
)
def test_the_rga_nears_one_above_the_liquid_lag_frequency(response, omega, reference):
    """Against that NumPy model. A single lag on L would leave 3.4 to 3.7 here,
    since it only scales a column of G."""
    assert abs(_rga(response(1j * omega))) == pytest.approx(reference, rel=0.005)


def test_moving_holdup_conserves_the_light_component():
    """Mid-transient the inventory changes by F zF - D yD - B xB, with B what
    reaches the reboiler less boilup. Holdup is eq. 47: M - M0 = taul (L - L0)."""
    rise = 0.05 * np.linspace(-1.0, 1.0, TRAYS)
    with enable_x64():
        x = jnp.asarray(_NOMINAL_PROFILE, float)
        flow = steady_liquid(P.L_nominal, P) + rise
        rates = compute_velocity(jnp.concatenate([x, flow]), (L0, V0), 0.5, P)[0]
        dx, dflow = rates[:N], rates[N:]
        M = tray_holdup(flow, P)
        np.testing.assert_allclose(M, P.M_tray + TAUL * rise, atol=1e-12)
        trays = jnp.sum(M * dx[1:-1] + x[1:-1] * TAUL * dflow)
        change = float(trays + P.M_drum * (dx[0] + dx[-1]))
        out = float((V0 - L0) * x[-1] + (flow[0] - V0) * x[0])
    assert change == pytest.approx(P.F * 0.5 - out, abs=1e-9)


@pytest.mark.parametrize(
    "F, zF", list(itertools.product((0.8, 1.0, 1.2), (0.4, 0.5, 0.6)))
)
def test_the_specs_hold_inside_the_flow_limits_after_any_feed_step(F, zF):
    """The feed's sourced steps, 20 % in rate and 0.1 in composition [D1,
    Fig. 18], either way: the products can be held on spec in the flow box."""
    with enable_x64():
        z = _target_steady_state(P.yD_spec, P.xB_spec, zF, F)
    assert P.L_min < float(z[N]) < P.L_max
    assert P.V_min < float(z[N + 1]) < P.V_max


@pytest.mark.parametrize("corner", list(itertools.product((-1.0, 1.0), repeat=2)))
def test_the_default_integrator_keeps_its_margin_across_the_flow_box(corner):
    def step(state, _):
        return compute_next_state(jnp.array(corner), state, P, KEY)[0], None

    state = jax.lax.scan(step, ENV.reset_env(KEY, P)[1], None, length=400)[0]
    # Settled, so no limit cycle. Not "monotone": V < L + D_min clips D, and that
    # corner trips long before it settles on its odd profile.
    assert np.abs(np.asarray(step(state, None)[0].x - state.x)).max() < 1e-3
    flows = (float(state.L), float(state.V))
    with enable_x64():
        here = jnp.concatenate([state.x, state.liquid]).astype(float)
        A = jax.jacfwd(lambda s: compute_velocity(s, flows, 0.5, P)[0])(here)
    assert np.abs(np.linalg.eigvals(np.asarray(A))).max() / 16 < 2.785


def test_a_reset_column_holds_with_no_hydraulic_transient():
    """The trays start at the sampled reflux, and ``initial_action`` holds it."""
    _, state = ENV.reset_env(KEY, P)
    held = compute_next_state(ENV.initial_action(state, P), state, P, KEY)[0]
    np.testing.assert_allclose(held.liquid, state.liquid, atol=1e-5)
    np.testing.assert_allclose([held.L, held.V], [state.L, state.V], atol=1e-5)
