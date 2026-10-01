"""Physics validation for the compressor.

Enforces ``src/target_gym/compressor_surge/PHYSICS.md``: the suction
constants against the ISA table, the Moore-Greitzer cubic and its join, two
anchors read off Gravdahl's thesis (a deep-surge orbit and a stable throttle),
the Helmholtz pair and its damping near the line, where the stability
boundaries lie, the fan laws and the surge parabola, the static facts that
make the recycle necessary, the reset, RK4 stability, and the plenum's mass
balance. Deviations D1, D2, D3, D4 and D5 are strict xfails at the end.

Everything here runs the env's own functions: the closed forms with
``xp=np`` (float64), and the velocity, its Jacobian and the step under
``jax.experimental.enable_x64`` unless a test says float32.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.integrate import simpson
from scipy.optimize import brentq

from target_gym.compressor_surge.env import (
    N_DEMAND_BLOCKS,
    N_SETPOINT_BLOCKS,
    CompressorSurgeParams,
    CompressorSurgeState,
    characteristic,
    check_valve_sqrt,
    compute_next_state,
    compute_velocity,
    consumer_flow,
    equilibrium_phi,
    get_obs,
    greitzer_B,
    helmholtz_omega,
    recycle_flow,
    sound_speed,
    suction_density,
    surge_constant,
    surge_flow_per_speed,
)
from target_gym.compressor_surge.env_jax import CompressorSurge

#: U.S. Standard Atmosphere 1976, sea level (read): density and speed of sound.
ISA_DENSITY = 1.2250  # kg/m^3
ISA_SOUND_SPEED = 340.294  # m/s


@pytest.fixture(scope="module")
def params():
    return CompressorSurgeParams()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flow_scale(params, N=1.0):
    """m_c per unit Phi at speed N, rho01 A_c U_r N (kg/s)."""
    return suction_density(params) * params.A_c * params.U_r * N


def _head_scale(params, N=1.0):
    """dp per unit Psi at speed N, rho01 (U_r N)^2 (Pa)."""
    return suction_density(params) * (params.U_r * N) ** 2


def _raw(value, lo, hi):
    """The raw action that ``convert_raw_action_to_range`` maps to ``value``."""
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def _hold_action(params, N, x):
    return jnp.array(
        [_raw(N, params.N_min, params.N_max), _raw(x, 0.0, 1.0)], dtype=jnp.float64
    )


def _state(params, m_c, dp, N, x, u, dtype=jnp.float64, **kw):
    """A state at (m_c, dp, N, x) with the drive setpoint at N, a constant
    consumer opening ``u`` (all six levels), no deviation and a 24 kPa
    setpoint. Build it inside ``enable_x64`` for float64."""
    f = functools.partial(jnp.asarray, dtype=dtype)
    fields = dict(
        time=jnp.asarray(0, jnp.int32),
        m_c=f(m_c),
        dp=f(dp),
        N=f(N),
        N_ramp=f(N),
        x=f(x),
        demand_dev=f(0.0),
        phi_min=f(m_c / _flow_scale(params, N)),
        setpoint_levels=f(np.full(N_SETPOINT_BLOCKS, 24.0e3)),
        demand_levels=f(np.full(N_DEMAND_BLOCKS, u)),
        block_clock=jnp.asarray(0, jnp.int32),
    )
    fields.update(
        {k: (f(v) if k not in ("time", "block_clock") else v) for k, v in kw.items()}
    )
    return CompressorSurgeState(**fields)


def _right_branch_phi(psi, params):
    """The right-branch Phi (>= 2W) with Psi_c(Phi) = psi, float64."""
    return brentq(
        lambda phi: float(characteristic(phi, params, np)) - psi,
        2.0 * params.W,
        0.8316,
        xtol=1e-15,
    )


def _point_at(dp, m, params):
    """The right-branch (Phi, N) that puts the compressor at (m, dp): Psi_c /
    Phi^2 = rho01 A_c^2 dp / m^2. None below the surge flow."""
    c = suction_density(params) * params.A_c**2 * dp / m**2
    g = lambda phi: float(characteristic(phi, params, np)) / phi**2 - c  # noqa: E731
    if g(2.0 * params.W) < 0.0:
        return None
    phi = brentq(g, 2.0 * params.W, 0.8316, xtol=1e-15)
    return phi, m / (_flow_scale(params) * phi)


def _equilibrium(params, N, x, u):
    """The right-branch equilibrium (Phi, m_c, dp) at fixed speed, recycle and
    opening, by bisection on the env's own flow balance, float64."""
    flow, head = _flow_scale(params, N), _head_scale(params, N)

    def residual(phi):
        dp = head * float(characteristic(phi, params, np))
        return (
            flow * phi
            - float(consumer_flow(dp, u, params, np))
            - float(recycle_flow(dp, x, params, np))
        )

    phi = brentq(residual, 2.0 * params.W, 0.8316, xtol=1e-15)
    return phi, flow * phi, head * float(characteristic(phi, params, np))


def _velocity_jacobian(params, m_c, dp, N, x, u):
    """d(dm_c/dt, d dp/dt)/d(m_c, dp) from ``compute_velocity``, float64,
    broadcast over array inputs. Returns (..., 2, 2)."""
    shape = np.broadcast_shapes(*(np.shape(v) for v in (m_c, dp, N, x, u)))
    m_c, dp, N, x, u = (
        np.broadcast_to(np.asarray(v, float), shape).ravel() for v in (m_c, dp, N, x, u)
    )
    with enable_x64():

        def one(m, p, n, xx, uu):
            def f(y):
                v, _ = compute_velocity(
                    jnp.stack([y[0], y[1], n, jnp.zeros_like(n)]),
                    None,
                    n,
                    xx,
                    0.0,
                    jnp.full(N_DEMAND_BLOCKS, uu),
                    params,
                )
                return v[:2]

            return jax.jacfwd(f)(jnp.stack([m, p]))

        J = jax.jit(jax.vmap(one))(
            *(jnp.asarray(v, jnp.float64) for v in (m_c, dp, N, x, u))
        )
    return np.asarray(J).reshape(shape + (2, 2))


def _step_eigenvalues(params, m_c, dp, N, x, u):
    """lambda = ln(mu) / dt for the eigenvalues mu of the (m_c, dp) block of
    ``compute_next_state``'s one-step Jacobian at (m_c, dp), the commands
    holding N and x, float64. The speed is at its drive setpoint, so it does
    not move and the block is exact."""
    key = jax.random.PRNGKey(0)
    with enable_x64():
        state = _state(params, m_c, dp, N, x, u)
        action = _hold_action(params, N, x)

        def f(y):
            s = compute_next_state(
                action, state.replace(m_c=y[0], dp=y[1]), params, key
            )[0]
            return jnp.stack([s.m_c, s.dp])

        J = np.asarray(jax.jacfwd(f)(jnp.array([m_c, dp], jnp.float64)))
    mu = np.linalg.eigvals(J)
    return np.log(mu.astype(complex)) / params.delta_t


def _upper(lams):
    """The eigenvalue with positive imaginary part."""
    return lams[np.argmax(lams.imag)]


def _peak_opening(params, N=1.0, x=0.0):
    """The consumer opening that puts the equilibrium on the peak."""
    m = float(surge_flow_per_speed(params)) * N
    dp = _head_scale(params, N) * float(characteristic(2.0 * params.W, params, np))
    return (
        (m - float(recycle_flow(dp, x, params, np)))
        / float(consumer_flow(dp, 1.0, params, np)),
        m,
        dp,
    )


def _run(params, state, action, n, method="rk4_10"):
    """``n`` steps of ``compute_next_state`` (no trip, no restart) under a
    constant key. Returns the stacked states."""
    key = jax.random.PRNGKey(0)

    def body(s, _):
        s2 = compute_next_state(action, s, params, key, integration_method=method)[0]
        return s2, s2

    return jax.jit(lambda s: jax.lax.scan(body, s, None, length=n)[1])(state)


# ---------------------------------------------------------------------------
# 1. Read constants and the characteristic
# ---------------------------------------------------------------------------


def test_suction_constants_are_isa(params):
    """rho01 and a01 against the ISA sea-level table, 1.2250 kg/m^3 and
    340.294 m/s, to 5e-5 relative. The code computes them from P01, T01,
    R_gas and gamma; R_gas 287.05 against the standard's 287.05287 puts the
    density 1.0e-5 high."""
    rho = float(suction_density(params))
    a = float(sound_speed(params, np))
    assert rho == pytest.approx(ISA_DENSITY, rel=5e-5)
    assert a == pytest.approx(ISA_SOUND_SPEED, rel=5e-5)
    assert rho / ISA_DENSITY - 1.0 == pytest.approx(1.0e-5, abs=0.2e-5)


def test_characteristic_is_the_moore_greitzer_cubic(params):
    """The peak is at (2W, psi_c0 + 2H) = (0.5, 0.66). Value and slope are
    continuous at the join, 0.78, where the slope is -3.774, and the line
    reaches zero head at Phi 0.8316."""
    psi = lambda phi: float(characteristic(phi, params, np))  # noqa: E731
    assert psi(0.5) == pytest.approx(0.66, abs=1e-12)
    assert psi(0.5 - 1e-4) < psi(0.5) and psi(0.5 + 1e-4) < psi(0.5)
    # Values read off the cubic: Psi_c(0) = psi_c0, Psi_c(W) = psi_c0 + H.
    assert psi(0.0) == pytest.approx(params.psi_c0, abs=1e-12)
    assert psi(params.W) == pytest.approx(params.psi_c0 + params.H, abs=1e-12)

    j, eps = params.phi_join, 1e-7
    assert psi(j + eps) == pytest.approx(psi(j - eps), abs=1e-6)
    with enable_x64():
        slope = jax.grad(lambda p: characteristic(p, params))
        left = float(slope(jnp.float64(j - 1e-9)))
        right = float(slope(jnp.float64(j + 1e-9)))
    assert left == pytest.approx(right, rel=1e-6)
    assert right == pytest.approx(-3.774, abs=1e-3)
    zero_head = brentq(psi, 0.79, 0.9, xtol=1e-14)
    assert zero_head == pytest.approx(0.8316, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Anchors read off Gravdahl (1998), thesis, figures 2.8 and 2.10
# ---------------------------------------------------------------------------


def _throttle_params(params, gain):
    """Gravdahl's throttle Phi_T = gain sqrt(Psi) in the dimensional plant:
    no static head, recycle shut, consumer opening k_d u / (A_c sqrt(rho01))
    = gain. Returns the params and the opening."""
    p = params.replace(dp_out=0.0, demand_sigma=0.0)
    u = gain * params.A_c * math.sqrt(suction_density(params)) / params.k_d
    return p, u


def test_gravdahl_fig_2_8_deep_surge_orbit(params):
    """At rated speed B = 1.80, the thesis' surge value, and the model is the
    thesis' eq. (2.6) with time rescaled. Fig. 2.8 (B 1.8, throttle gain 0.61,
    from Phi = Psi = 0.6) shows deep surge with Phi from the axis floor, -0.2,
    to about 0.75 and Psi from about 0.21 to 0.67 over its first two flow
    collapses. The env, stepped by ``compute_next_state`` at 5 ms steps (no
    trip) for 2.2 s, gives Phi -0.194 to 0.761 and Psi 0.212 to 0.669
    (derived, scripts/compressor_surge_numbers.py --section anchor)."""
    assert float(greitzer_B(1.0, params, np)) == pytest.approx(1.80, abs=0.005)
    p, u = _throttle_params(params, 0.61)
    assert u == pytest.approx(0.5626, abs=1e-4)
    p = p.replace(delta_t=0.005)
    with enable_x64():
        state = _state(p, 0.6 * _flow_scale(p), 0.6 * _head_scale(p), 1.0, 0.0, u)
        states = _run(p, state, _hold_action(p, 1.0, 0.0), 440)
        phi = np.concatenate([[0.6], np.asarray(states.m_c) / _flow_scale(p)])
        psi = np.concatenate([[0.6], np.asarray(states.dp) / _head_scale(p)])
    assert np.all(np.asarray(states.N) == pytest.approx(1.0, abs=1e-12))
    assert phi.min() <= -0.17
    assert phi.max() == pytest.approx(0.75, abs=0.03)
    assert psi.min() == pytest.approx(0.21, abs=0.03)
    assert psi.max() == pytest.approx(0.67, abs=0.03)
    # Two flow collapses within the window.
    down = np.flatnonzero((phi[:-1] >= 0.2) & (phi[1:] < 0.2))
    assert len(down) >= 2


def test_gravdahl_fig_2_10_throttle_is_stable(params):
    """Fig. 2.10 starts flat at about Phi 0.53, Psi 0.66 with throttle gain
    0.65, where the model's equilibrium is (0.5268, 0.6568) and decays; at
    gain 0.61 (0.4955, 0.6599, just left of the peak) it grows."""
    for gain, phi_eq, psi_eq, stable in (
        (0.65, 0.5268, 0.6568, True),
        (0.61, 0.4955, 0.6599, False),
    ):
        p, u = _throttle_params(params, gain)
        flow, head = _flow_scale(p), _head_scale(p)
        phi = brentq(
            lambda f: flow * f
            - float(consumer_flow(head * float(characteristic(f, p, np)), u, p, np)),
            0.3,
            0.8,
            xtol=1e-15,
        )
        psi = float(characteristic(phi, p, np))
        assert phi == pytest.approx(phi_eq, abs=1e-3)
        assert psi == pytest.approx(psi_eq, abs=1e-3)
        lams = _step_eigenvalues(p, flow * phi, head * psi, 1.0, 0.0, u)
        assert (lams.real.max() < 0.0) == stable, gain
        if stable:
            assert phi == pytest.approx(0.53, abs=0.02)
            assert psi == pytest.approx(0.66, abs=0.02)


# ---------------------------------------------------------------------------
# 3. Linear dynamics
# ---------------------------------------------------------------------------


def test_helmholtz_pair_on_the_peak(params):
    """At an equilibrium on the peak (rated speed, recycle shut, opening
    0.683) the compressor slope vanishes, so the (m_c, dp) pair has modulus
    omega_H = a01 sqrt(A_c / (V_p L_c)) = 13.892 rad/s and real part
    -(a01^2 / V_p) dm_out/d dp / 2, -1.058 /s (derived). Measured from the
    one-step Jacobian: |lambda| to 1 % and Re to 5 %."""
    u, m, dp = _peak_opening(params)
    assert u == pytest.approx(0.683, abs=1e-3)
    omega_H = float(helmholtz_omega(params, np))
    assert omega_H == pytest.approx(13.892, abs=1e-3)
    with enable_x64():
        g = float(jax.grad(lambda p: consumer_flow(p, u, params))(jnp.float64(dp)))
    re = -(sound_speed(params, np) ** 2 / params.V_p) * g / 2.0
    assert re == pytest.approx(-1.058, abs=1e-3)

    lam = _upper(_step_eigenvalues(params, m, dp, 1.0, 0.0, u))
    assert abs(lam) == pytest.approx(omega_H, rel=0.01)
    assert lam.real == pytest.approx(re, rel=0.05)


def _pair_at(params, dp, u, x=0.0):
    """(zeta, damped Hz) of the (m_c, dp) pair at the right-branch
    equilibrium delivering opening ``u`` at header pressure ``dp``, with the
    speed that puts it there."""
    m = float(consumer_flow(dp, u, params, np)) + float(recycle_flow(dp, x, params, np))
    phi, N = _point_at(dp, m, params)
    lam = _upper(np.linalg.eigvals(_velocity_jacobian(params, m, dp, N, x, u)))
    return -lam.real / abs(lam), lam.imag / (2.0 * math.pi), phi, N


def test_off_peak_pair(params):
    """At 24 kPa, demand 0.8, recycle shut (Phi 0.5358, speed 86.53 %):
    zeta 0.567, damped 2.023 Hz, to 1 %."""
    zeta, f, phi, N = _pair_at(params, 24.0e3, 0.8)
    assert phi == pytest.approx(0.5358, abs=1e-4)
    assert N == pytest.approx(0.8653, abs=1e-4)
    assert zeta == pytest.approx(0.567, rel=0.01)
    assert f == pytest.approx(2.023, rel=0.01)


def test_damping_collapses_toward_the_line(params):
    """At 24 kPa with the recycle shut, the pair's damping ratio falls
    monotonically as the equilibrium approaches the line: 0.619 at an 8 %
    margin, 0.105 on it (41 margins from 10 % to 0)."""
    dp = 24.0e3
    rho = suction_density(params)
    margins = np.linspace(0.10, 0.0, 41)
    phi = 2.0 * params.W * (1.0 + margins)
    psi = np.asarray(characteristic(phi, params, np))
    m = np.sqrt(dp * rho * params.A_c**2 * phi**2 / psi)
    N = m / (_flow_scale(params) * phi)
    u = m / np.asarray(consumer_flow(dp, 1.0, params, np))
    lams = np.linalg.eigvals(_velocity_jacobian(params, m, dp, N, 0.0, u))
    upper = lams[np.arange(len(lams)), np.argmax(lams.imag, axis=1)]
    zeta = -upper.real / np.abs(upper)
    assert np.all(np.diff(zeta) < 0.0)
    assert zeta[margins.round(4) == 0.08][0] == pytest.approx(0.619, abs=2e-3)
    assert zeta[-1] == pytest.approx(0.105, abs=2e-3)


def test_boundaries_lie_left_of_the_peak(params):
    """Over the action box (15 speeds from 70 to 105 %, 21 recycle openings
    from 0 to 1), 107 pairs have an equilibrium family, parameterised by the
    consumer opening in [0.2, 1], that reaches the peak. On the peak the
    trace is negative in every one (at most -1.58 /s), so no equilibrium at
    or right of the line is unstable. Walking left from the peak, 105 of the
    families meet a Hopf point before their opening leaves [0.2, 1], and it
    closes a stable sliver of 0.70 % (105 %, recycle 0.30) to 3.77 % (70 %,
    recycle shut) of the peak's Phi (derived, scripts/compressor_surge_numbers.py
    --section linear)."""
    rho = suction_density(params)
    Ns = np.linspace(params.N_min, params.N_max, 15)
    xs = np.linspace(0.0, 1.0, 21)
    phi = np.linspace(0.45, 0.5, 2001)
    NN, XX, PP = np.meshgrid(Ns, xs, phi, indexing="ij")
    dp = rho * (params.U_r * NN) ** 2 * np.asarray(characteristic(PP, params, np))
    m = rho * params.A_c * params.U_r * NN * PP
    u = (m - np.asarray(recycle_flow(dp, XX, params, np))) / np.asarray(
        consumer_flow(dp, 1.0, params, np)
    )
    valid = (u >= 0.2) & (u <= 1.0)
    reaches = valid[..., -1]
    assert reaches.sum() == 107

    J = _velocity_jacobian(
        params, m[reaches], dp[reaches], NN[reaches], XX[reaches], u[reaches]
    )
    trace = J[..., 0, 0] + J[..., 1, 1]
    assert trace[:, -1].max() < 0.0
    assert trace[:, -1].max() == pytest.approx(-1.58, abs=0.01)

    slivers = []
    for tr, ok, (i, k) in zip(trace, valid[reaches], np.argwhere(reaches)):
        # The family's valid segment ending at the peak, and its Hopf point.
        start = np.flatnonzero(~ok).max() + 1 if (~ok).any() else 0
        unstable = np.flatnonzero(tr[start:] >= 0.0)
        if unstable.size:
            slivers.append((1.0 - phi[start + unstable.max()] / 0.5, Ns[i], xs[k]))
    slivers = np.array(slivers)
    assert len(slivers) == 105
    lo, hi = slivers[:, 0].argmin(), slivers[:, 0].argmax()
    assert slivers[lo, 0] == pytest.approx(0.0070, abs=3e-4)
    assert slivers[hi, 0] == pytest.approx(0.0377, abs=3e-4)
    assert (slivers[lo, 1], slivers[lo, 2]) == pytest.approx((1.05, 0.30))
    assert (slivers[hi, 1], slivers[hi, 2]) == pytest.approx((0.70, 0.0))


# ---------------------------------------------------------------------------
# 4. Scaling, the surge line and the steady-state facts of the task
# ---------------------------------------------------------------------------


def test_fan_laws(params):
    """With no static head and the recycle shut, a fixed consumer opening
    holds the same Phi at every speed: flow scales as N and pressure as N^2,
    to 1e-5, at 70, 87.5 and 105 %. Each point is a fixed point of the env's
    velocity."""
    p = params.replace(dp_out=0.0)
    u = 0.6
    with enable_x64():
        rows = []
        for N in (0.70, 0.875, 1.0, 1.05):
            phi = float(equilibrium_phi(N, 0.0, u, p, 30, xp=np))
            m = _flow_scale(p, N) * phi
            dp = _head_scale(p, N) * float(characteristic(phi, p, np))
            v, _ = compute_velocity(
                jnp.array([m, dp, N, 0.0]), None, N, 0.0, 0.0, jnp.full(6, u), p
            )
            assert np.abs(np.asarray(v[:3])).max() < 1e-6
            rows.append((N, m, dp))
    _, m1, dp1 = rows[2]
    for N, m, dp in rows:
        assert m / m1 == pytest.approx(N, rel=1e-5)
        assert dp / dp1 == pytest.approx(N**2, rel=1e-5)


def test_surge_line_is_a_parabola(params):
    """At 80, 90 and 100 % with the recycle shut, the smallest consumer
    opening whose equilibrium does not trip (found by bisection on the env's
    Newton equilibrium and its trip rule) puts the compressor on dp = K m^2,
    K = 215.51 Pa/(kg/s)^2 (derived), to 1 %. Held there with 2 % more
    opening the env does not trip; with 2 % less it does."""
    K = float(surge_constant(params))
    assert K == pytest.approx(215.51, abs=0.01)
    assert float(surge_flow_per_speed(params)) == pytest.approx(12.25, abs=1e-3)
    env = CompressorSurge()
    key = jax.random.PRNGKey(0)
    quiet = params.replace(demand_sigma=0.0)

    @jax.jit
    def hold(state, action):
        def body(s, _):
            _, s2, _, _, info = env.step_env(key, s, action, quiet)
            return s2, info["tripped"]

        return jax.lax.scan(body, state, None, length=100)[1]

    for N in (0.8, 0.9, 1.0):
        with enable_x64():
            phi_of = jax.jit(lambda u, N=N: equilibrium_phi(N, 0.0, u, params, 30))

            def trips(u):
                return not (float(phi_of(u)) >= 2.0 * params.W)

            lo, hi = 0.2, 1.0
            assert trips(lo) and not trips(hi)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if trips(mid) else (lo, mid)
            u_star = hi
            phi = float(phi_of(u_star))
            phi_hi = float(phi_of(1.02 * u_star))
            m = _flow_scale(params, N) * phi
            dp = _head_scale(params, N) * float(characteristic(phi, params, np))
        assert dp == pytest.approx(K * m**2, rel=0.01), N

        # Held from the equilibrium at 2 % more opening, and with the opening
        # then cut to 2 % less (float32, as shipped).
        action = jnp.array([_raw(N, params.N_min, params.N_max), -1.0])
        for scale, should_trip in ((1.02, False), (0.98, True)):
            s = _state(
                quiet,
                _flow_scale(params, N) * phi_hi,
                _head_scale(params, N) * float(characteristic(phi_hi, params, np)),
                N,
                0.0,
                scale * u_star,
                dtype=jnp.float32,
            )
            tripped = np.asarray(hold(s, action))
            assert tripped.any() == should_trip, (N, scale)


def test_every_block_is_feasible(params):
    """At 105 % the compressor delivers 18.70 kg/s at 20 kPa and 17.08 at
    28 kPa, above the 12.00 and 16.10 kg/s the consumers take at full
    opening. Every schedule corner (20 or 28 kPa, opening 0.35, 0.95 or the
    clip's 1.0) needs a speed between 78.6 and 101.8 % of rated: on the
    surge line where the consumers take less than the surge flow, and at
    their flow otherwise."""
    rho = suction_density(params)
    deliverable = {}
    for dp in (20.0e3, 28.0e3):
        phi = _right_branch_phi(dp / _head_scale(params, params.N_max), params)
        deliverable[dp] = _flow_scale(params, params.N_max) * phi
    assert deliverable[20.0e3] == pytest.approx(18.70, abs=0.01)
    assert deliverable[28.0e3] == pytest.approx(17.08, abs=0.01)
    assert float(consumer_flow(20.0e3, 1.0, params, np)) == pytest.approx(
        12.00, abs=0.01
    )
    assert float(consumer_flow(28.0e3, 1.0, params, np)) == pytest.approx(
        16.10, abs=0.01
    )

    speeds = []
    for dp in (20.0e3, 28.0e3):
        for u in (0.35, 0.95, 1.0):
            m = float(consumer_flow(dp, u, params, np))
            point = _point_at(dp, m, params)
            if point is None:  # below the surge flow: recycle is forced
                N = math.sqrt(dp / (rho * params.U_r**2 * 0.66))
            else:
                N = point[1]
            speeds.append(N)
    assert min(speeds) == pytest.approx(0.786, abs=1e-3)
    assert max(speeds) == pytest.approx(1.018, abs=1e-3)
    assert params.N_min <= min(speeds) and max(speeds) <= params.N_max


def test_most_blocks_force_recycle(params):
    """Over 1e5 uniform draws of a setpoint level and a demand level, the
    consumers take less than the surge flow at the setpoint in 66 % of them
    (+-1 %): the recycle must open there to hold the pressure without
    surge."""
    rng = np.random.default_rng(0)
    dp = rng.uniform(*params.p_ref_range, 100_000)
    u = rng.uniform(*params.demand_range, 100_000)
    surge_flow = np.sqrt(dp / float(surge_constant(params)))
    forced = np.asarray(consumer_flow(dp, u, params, np)) < surge_flow
    assert forced.mean() == pytest.approx(0.66, abs=0.01)


def test_no_fixed_valve_is_both_safe_and_able(params):
    """A fixed recycle opening must be at least 0.256 to keep every block's
    equilibrium off the surge line (20 kPa, demand 0.35) and at most 0.071
    for 105 % to hold every block's pressure (28 kPa, demand 0.95). No
    opening does both, so the recycle has to move."""
    dp = np.linspace(*params.p_ref_range, 41)[:, None]
    u = np.linspace(*params.demand_range, 41)[None, :]
    m_d = np.asarray(consumer_flow(dp, u, params, np))
    per_x = np.asarray(recycle_flow(dp, 1.0, params, np))
    x_min = np.maximum((np.sqrt(dp / float(surge_constant(params))) - m_d) / per_x, 0.0)
    phi_max = np.vectorize(
        lambda p: _right_branch_phi(p / _head_scale(params, params.N_max), params)
    )(dp)
    x_max = (_flow_scale(params, params.N_max) * phi_max - m_d) / per_x
    assert x_min.max() == pytest.approx(0.256, abs=1e-3)
    assert x_max.min() == pytest.approx(0.071, abs=1e-3)
    assert np.unravel_index(x_min.argmax(), x_min.shape) == (0, 0)
    assert np.unravel_index(x_max.argmin(), x_max.shape) == (40, 40)
    assert x_min.max() > x_max.min()


def test_monotonicity(params):
    """At equilibrium: pressure rises with speed; opening the recycle lowers
    the pressure and raises the compressor flow and the surge margin;
    lowering the demand lowers the margin."""
    with enable_x64():
        eq = lambda N, x, u: float(
            equilibrium_phi(N, x, u, params, 30, xp=np)
        )  # noqa: E731

        def point(N, x, u):
            phi = eq(N, x, u)
            return (
                phi,
                _flow_scale(params, N) * phi,
                _head_scale(params, N) * float(characteristic(phi, params, np)),
            )

        by_speed = np.array([point(N, 0.2, 0.7) for N in np.linspace(0.8, 1.05, 11)])
        by_valve = np.array([point(0.95, x, 0.8) for x in np.linspace(0.0, 1.0, 11)])
        by_demand = np.array([point(0.95, 0.2, u) for u in np.linspace(1.0, 0.5, 11)])
    assert np.all(np.diff(by_speed[:, 2]) > 0.0)
    assert np.all(np.diff(by_valve[:, 2]) < 0.0)
    assert np.all(np.diff(by_valve[:, 1]) > 0.0)
    assert np.all(np.diff(by_valve[:, 0]) > 0.0)
    assert np.all(np.diff(by_demand[:, 0]) < 0.0)


def test_dead_head_speed(params):
    """The shutoff head, rho01 (U_r N)^2 Psi_c(0), equals the consumers'
    static head at N = 0.825: below that speed a compressor pushing no flow
    cannot lift the header to where the consumers take any."""
    shutoff = lambda N: _head_scale(params, N) * float(
        characteristic(0.0, params, np)
    )  # noqa: E731
    N = brentq(lambda n: shutoff(n) - params.dp_out, 0.5, 1.0, xtol=1e-14)
    assert N == pytest.approx(0.825, abs=5e-4)


def test_reset_is_surge_safe_and_converged(params):
    """The reset's six Newton steps from Phi 0.70 (float32, as shipped)
    equal a bisection root of the same flow balance to 1e-6 for openings 0.2
    to 1.0, and every reset equilibrium lies right of the line, Phi 0.530
    to 0.728."""
    openings = np.linspace(0.2, 1.0, 161)
    newton = np.asarray(
        equilibrium_phi(
            params.N_reset,
            params.x_reset,
            jnp.asarray(openings, jnp.float32),
            params,
            params.reset_newton_iters,
        )
    )
    assert newton.dtype == np.float32
    exact = np.array(
        [_equilibrium(params, params.N_reset, params.x_reset, u)[0] for u in openings]
    )
    assert np.abs(newton - exact).max() <= 1e-6
    assert exact.min() == pytest.approx(0.5299, abs=5e-4)
    assert exact.max() == pytest.approx(0.7275, abs=5e-4)
    assert params.reset_newton_iters == 6


# ---------------------------------------------------------------------------
# 5. Integration and conservation
# ---------------------------------------------------------------------------


def test_rk4_is_stable_on_the_reachable_set(params):
    """Along full travel high (105 %, recycle open) over 20 drawn schedules,
    the stiffest eigenvalue of the three ODEs, the duct on the steep right
    branch, puts h |lambda| at 1.94 at the 10 ms substep, under RK4's
    real-axis limit of 2.785; every eigenvalue's RK4 amplification is below
    1 (float32 rollout, float64 Jacobians)."""
    env = CompressorSurge()
    action = jnp.array([1.0, 1.0])

    def episode(key):
        _, state = env.reset_env(key, params)

        def body(s, _):
            _, s2, _, _, info = env.step_env(key, s, action, params)
            return s2, (s2, info["tripped"])

        return jax.lax.scan(body, state, None, length=params.max_steps_in_episode)[1]

    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(20))
    states, tripped = jax.jit(jax.vmap(episode))(keys)
    assert not np.asarray(tripped).any()

    flat = jax.tree_util.tree_map(
        lambda a: np.asarray(a, np.float64).reshape((-1,) + a.shape[2:]), states
    )
    with enable_x64():

        def jac(s):
            def f(y):
                v, _ = compute_velocity(
                    jnp.concatenate([y, (s.block_clock * params.delta_t)[None]]),
                    None,
                    s.N_ramp,
                    s.x,
                    s.demand_dev,
                    s.demand_levels,
                    params,
                )
                return v[:3]

            return jax.jacfwd(f)(jnp.stack([s.m_c, s.dp, s.N]))

        J = jax.jit(jax.vmap(jac))(
            jax.tree_util.tree_map(lambda a: jnp.asarray(a), flat)
        )
    h = params.delta_t / 10
    z = h * np.linalg.eigvals(np.asarray(J))
    assert np.abs(z).max() == pytest.approx(1.94, abs=0.01)
    amplification = np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
    assert amplification.max() < 1.0


def test_plenum_mass_balance_closes(params):
    """With both valves sealed (k_d = k_r = 0) the plenum's pressure is its
    stored mass: V_p / a01^2 (dp_end - dp_0) = integral of m_c dt. From an
    empty plenum with the compressor at its zero-head flow (Phi 0.8316, rated
    speed), stepped by ``compute_next_state`` at 5 ms steps through the fill
    and into deep surge for 2 s, the two agree to 0.1 % of the integral of
    |m_c|, with a01 from the ISA table rather than from the code
    (docs/PHYSICS_METHODOLOGY.md, conservation). A wrong plenum coefficient,
    or any other term feeding dp, breaks it."""
    p = params.replace(k_d=0.0, k_r=0.0, delta_t=0.005, demand_sigma=0.0)
    with enable_x64():
        state = _state(p, 0.8316 * _flow_scale(p), 0.0, 1.0, 0.0, 0.5)
        states = _run(p, state, _hold_action(p, 1.0, 0.0), 400)
        m = np.concatenate([[float(state.m_c)], np.asarray(states.m_c)])
        dp = np.concatenate([[0.0], np.asarray(states.dp)])
    t = np.arange(len(m)) * p.delta_t
    assert m.min() < -1.0  # deep surge: the flow reverses
    stored = p.V_p / ISA_SOUND_SPEED**2 * (dp[-1] - dp[0])
    delivered = simpson(m, x=t)
    assert abs(stored - delivered) <= 1e-3 * simpson(np.abs(m), x=t)


# ---------------------------------------------------------------------------
# 6. Known deviations (PHYSICS.md section 9), each a strict xfail: the test
#    states what a real plant does, and fails the suite if the model starts
#    doing it without PHYSICS.md and this marker being updated.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D1: no rotating stall; left of the peak the machine sits on the unstalled cubic.",
)
def test_equilibrium_left_of_the_peak_sits_below_the_cubic(params):
    """D1. Left of the peak a real compressor runs in rotating stall, whose
    annulus-averaged pressure rise lies below the unstalled characteristic.
    Here the stable sliver at 70 % with the recycle shut holds an equilibrium
    2 % left of the peak, and stepping the env there for 10 s leaves dp on
    the cubic, where the test asks for 5 % below it."""
    N = 0.70
    phi = 0.98 * 2.0 * params.W
    m = _flow_scale(params, N) * phi
    dp = _head_scale(params, N) * float(characteristic(phi, params, np))
    u = m / float(consumer_flow(dp, 1.0, params, np))
    p = params.replace(demand_sigma=0.0)
    with enable_x64():
        states = _run(p, _state(p, m, dp, N, 0.0, u), _hold_action(p, N, 0.0), 100)
        m_end, dp_end = float(states.m_c[-1]), float(states.dp[-1])
    phi_end = m_end / _flow_scale(p, N)
    if abs(phi_end - phi) > 1e-3:
        pytest.fail(
            f"the sliver equilibrium did not hold: Phi {phi:.4f} -> {phi_end:.4f}"
        )
    cubic = _head_scale(p, N) * float(characteristic(phi_end, p, np))
    assert dp_end <= 0.95 * cubic


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D2: incompressible duct; the flow is rho01 A_c times the throughflow velocity.",
)
def test_duct_flow_is_compressible(params):
    """D2. At the largest flow coefficient the plant reaches, 0.777 at 105 %
    with both valves open, the duct's throughflow velocity is Mach 0.48, where
    the isentropic density is 10.6 % below the suction density. Here the env's
    flow per unit velocity and area is the suction density, where the test
    asks for at least 5 % less."""
    with enable_x64():
        phi = float(equilibrium_phi(params.N_max, 1.0, 1.0, params, 30, xp=np))
    assert phi == pytest.approx(0.777, abs=1e-3)
    velocity = params.N_max * params.U_r * phi
    mach = velocity / float(sound_speed(params, np))
    assert mach == pytest.approx(0.48, abs=0.01)
    m_c = _flow_scale(params, params.N_max) * phi
    assert m_c / (params.A_c * velocity) <= 0.95 * float(suction_density(params))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D3: the plenum's speed of sound is taken at suction temperature.",
)
def test_helmholtz_frequency_sees_the_discharge_temperature(params):
    """D3. The plenum holds gas compressed to about 1.32 times suction
    pressure, at least 23.7 K warmer (isentropic, derived,
    scripts/compressor_surge_numbers.py --section deviations), so its speed
    of sound, and with it the Helmholtz frequency, is about 4 % above the
    suction value. Here f_H, measured from the env's one-step Jacobian at an
    equilibrium on the peak, equals the value computed from the ISA suction
    temperature, where the test asks for at least 3 % more."""
    u, m, dp = _peak_opening(params)
    lam = _upper(_step_eigenvalues(params, m, dp, 1.0, 0.0, u))
    suction = ISA_SOUND_SPEED * math.sqrt(params.A_c / (params.V_p * params.L_c))
    assert abs(lam) >= 1.03 * suction


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D4: the trip is instantaneous; any substep below the line trips.",
)
def test_a_short_excursion_does_not_trip(params):
    """D4. Surge takes time to develop, so a real machine carried across its
    surge line for less than a Helmholtz period (0.45 s) recovers without
    surging. Here, from an equilibrium 5 % right of the line at rated speed
    with the flow knocked to 1 % left of it, the plant is back right of the
    line within one period (stepping ``compute_next_state``), and the env
    trips on the first step, where the test asks for no trip."""
    phi_eq = 1.05 * 2.0 * params.W
    m, dp = _flow_scale(params) * phi_eq, _head_scale(params) * float(
        characteristic(phi_eq, params, np)
    )
    u = m / float(consumer_flow(dp, 1.0, params, np))
    p = params.replace(demand_sigma=0.0)
    kicked = _state(
        p,
        0.99 * float(surge_flow_per_speed(params)),
        dp,
        1.0,
        0.0,
        u,
        dtype=jnp.float32,
    )
    action = jnp.array([_raw(1.0, p.N_min, p.N_max), -1.0], jnp.float32)
    period_steps = int(np.ceil(2 * math.pi / float(helmholtz_omega(p, np)) / p.delta_t))
    states = _run(p, kicked, action, period_steps)
    if not (np.asarray(states.m_c)[-1] / _flow_scale(p) > 2.0 * p.W):
        pytest.fail("the excursion lasted longer than a Helmholtz period")
    _, _, _, _, info = CompressorSurge().step_env(
        jax.random.PRNGKey(0), kicked, action, p
    )
    assert not bool(info["tripped"])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D5: the smooth check valve leaks at zero pressure drop.",
)
def test_check_valve_is_tight(params):
    """D5. A consumer valve passes nothing at zero pressure drop. Here the
    smooth one-sided square root leaks: at the static head and full opening
    the observed delivered flow is k_d sqrt(s ln 2) = 0.706 kg/s, where the
    test asks for under 1e-3 kg/s."""
    state = _state(params, 10.0, params.dp_out, 1.0, 0.0, 1.0, dtype=jnp.float32)
    leak = float(get_obs(state, params)[2])
    assert leak == pytest.approx(
        params.k_d * float(check_valve_sqrt(0.0, params, np)), rel=1e-5
    )
    assert leak < 1e-3
