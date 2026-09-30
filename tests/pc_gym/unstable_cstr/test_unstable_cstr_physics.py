"""Physics validation for the unstable CSTR.

Enforces ``src/target_gym/pc_gym/unstable_cstr/PHYSICS.md``. The balances
are cstr's, imported; these tests check that the import is exact, that the
model reproduces steady states published by two separate implementations,
that every target of the task is a saddle, where the point of no return lies
and that the line the controllers read stays below it, that energy is
conserved, and that RK4 is stable wherever the plant can go. Deviations D1
and D4 are strict xfails at the end. Everything here runs the env's own
functions: the closed forms with ``xp=np`` (float64), and the velocity, its
Jacobian and the step under ``jax.experimental.enable_x64``, since the
figures are quoted to 0.01 K.
"""

import ast
import functools
import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.optimize import brentq

import target_gym.pc_gym.cstr.env as cstr_env
from target_gym.pc_gym.unstable_cstr.env import (
    N_BLOCKS,
    UnstableCSTRParams,
    UnstableCSTRState,
    check_is_terminal,
    compute_next_state,
    compute_velocity,
    steady_coolant,
    steady_temperature,
)
from target_gym.pc_gym.unstable_cstr.experts import (
    MPC_PNR_MARGIN_K,
    pnr_line,
    point_of_no_return_batch,
    raw_from_coolant,
    trips_under_full_cooling,
)

#: The grid of targets PHYSICS.md tabulates, spanning the band.
TARGETS = (0.45, 0.50, 0.55, 0.60, 0.65)
#: Open-loop growth rate at each target (/min), PHYSICS.md section 2.
LAMBDA_PLUS = (3.158, 2.834, 2.422, 1.940, 1.391)
PHYSICAL = ("q", "V", "rho", "C", "deltaHr", "EA_over_R", "k0", "UA", "Ti", "Caf")


@pytest.fixture(scope="module")
def params():
    return UnstableCSTRParams()


def _steady_states(T_c, params, Ti_dev=0.0):
    """Every equilibrium ``(C_a, T)`` at jacket temperature ``T_c``, float64.

    Each steady state has its own concentration L, with T = T*(L), so the
    equilibria are the roots in L of Tc*(L, dTi) = T_c.
    """
    levels = np.linspace(1e-4, 1.0 - 1e-4, 20001)
    f = steady_coolant(levels, Ti_dev, params, np) - T_c
    roots = [
        brentq(
            lambda L: steady_coolant(L, Ti_dev, params, np) - T_c,
            levels[i],
            levels[i + 1],
            xtol=1e-16,
            rtol=1e-15,
        )
        for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
    ]
    return [(L, float(steady_temperature(L, params, np))) for L in roots]


def _jacobian(C_a, T, T_j, params, Ti_dev=0.0):
    """d(velocity)/d(C_a, T, T_j), float64, from the env's own velocity."""
    with enable_x64():
        J = jax.jacfwd(lambda x: compute_velocity(x, T_j, Ti_dev, params)[0])(
            jnp.array([C_a, T, T_j], dtype=jnp.float64)
        )
    return np.asarray(J)


def _at_target(L, params, Ti_dev=0.0):
    """The equilibrium the env holds at level L: (L, T*(L), Tc*(L, dTi))."""
    return (
        L,
        float(steady_temperature(L, params, np)),
        float(steady_coolant(L, Ti_dev, params, np)),
    )


def _spectral_radii(points, params):
    """max |eig| of d(velocity)/d(C_a, T, T_j) at each row of ``points``
    (n, 3), float64. The command is a constant (the jacket temperature), so
    the lag row keeps its -1/tau_j; the drift enters additively and does not
    move the Jacobian."""
    with enable_x64():
        x = jnp.asarray(np.asarray(points, float), jnp.float64)
        J = jax.vmap(jax.jacfwd(lambda y, u: compute_velocity(y, u, 0.0, params)[0]))(
            x, x[:, 2]
        )
    return np.abs(np.linalg.eigvals(np.asarray(J))).max(axis=1)


def _states64(C_a, T, T_j, Ti_dev):
    """A float64 batch of states from broadcastable arrays, each holding its
    own C_a as the level. Call inside ``enable_x64``."""
    C_a, T, T_j, Ti_dev = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(v, float)) for v in (C_a, T, T_j, Ti_dev))
    )
    f64 = functools.partial(jnp.asarray, dtype=jnp.float64)
    return UnstableCSTRState(
        time=jnp.zeros(C_a.size, jnp.int32),
        C_a=f64(C_a),
        T=f64(T),
        T_j=f64(T_j),
        Ti_dev=f64(Ti_dev),
        target_levels=f64(np.repeat(C_a[:, None], N_BLOCKS, axis=1)),
        block_clock=jnp.zeros(C_a.size, jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=("n",))
def _hold(states, raw, params, n):
    """``n`` steps of ``compute_next_state`` from each state at the constant
    raw action ``raw``, the drift held at its value, no restart. Returns the
    C_a, T and T_j paths and the trip check of each proposal, each (batch, n)."""
    key = jax.random.PRNGKey(0)

    def one(state):
        def body(s, _):
            s2 = compute_next_state(raw, s, params, key)[0].replace(Ti_dev=s.Ti_dev)
            return s2, (s2.C_a, s2.T, s2.T_j, check_is_terminal(s2, params)[0])

        return jax.lax.scan(body, state, None, length=n)[1]

    return jax.vmap(one)(states)


def _unstable_direction(L, params):
    """dC_a/dT along the unstable eigenvector of the two balances at target L."""
    w, v = np.linalg.eig(_jacobian(*_at_target(L, params), params)[:2, :2])
    vector = v[:, np.argmax(w.real)].real
    return vector[0] / vector[1]


def _adiabatic_batch(params, C_a0=0.1, T0=340.0, n=400):
    """The reactor sealed and insulated (q 0, UA 0) from (C_a0, T0), stepped
    ``n`` times by ``compute_next_state`` in float64. Returns the C_a and T
    paths, the start included."""
    batch = params.replace(q=0.0, UA=0.0, Ti_sigma=0.0)
    with enable_x64():
        C_a, T, _, _ = _hold(_states64(C_a0, T0, 300.0, 0.0), 0.0, batch, n)
    C_a = np.concatenate([[C_a0], np.asarray(C_a)[0]])
    T = np.concatenate([[T0], np.asarray(T)[0]])
    return C_a, T


# ---------------------------------------------------------------------------
# 1. The model is cstr's (check 12: imported, never copied)
# ---------------------------------------------------------------------------


def test_the_ten_parameters_are_cstrs(params):
    """The ten physical values and the analyser resolution are cstr's defaults,
    and the two balances are cstr's ``compute_velocity``, bit for bit, with the
    jacket temperature in the coolant slot and the drifted feed temperature."""
    shipped = cstr_env.CSTRParams()
    for name in (*PHYSICAL, "precision_floor"):
        assert getattr(params, name) == getattr(shipped, name), name

    rng = np.random.default_rng(0)
    n = 200
    x = jnp.asarray(
        np.stack(
            [
                rng.uniform(0.05, 0.99, n),
                rng.uniform(300.0, 380.0, n),
                rng.uniform(params.T_c_min, params.T_c_max, n),
            ],
            axis=1,
        ),
        jnp.float32,
    )
    command = jnp.asarray(rng.uniform(params.T_c_min, params.T_c_max, n), jnp.float32)
    Ti_dev = jnp.asarray(rng.normal(0.0, 3.0, n), jnp.float32)

    ours = jax.vmap(lambda p, u, d: compute_velocity(p, u, d, params)[0])(
        x, command, Ti_dev
    )
    theirs = jax.vmap(
        lambda p, d: cstr_env.compute_velocity(
            p[:2], p[2], shipped.replace(Ti=shipped.Ti + d)
        )[0]
    )(x, Ti_dev)
    np.testing.assert_array_equal(np.asarray(ours[:, :2]), np.asarray(theirs))
    # The third row is the jacket lag and nothing else.
    np.testing.assert_allclose(
        np.asarray(ours[:, 2]),
        np.asarray((command - x[:, 2]) / params.tau_j),
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# 2. Published steady states (anchors outside this model)
# ---------------------------------------------------------------------------


def test_published_low_branch_state(params):
    """APMonitor's stirred-reactor page gives the low state at a 300 K jacket
    to 15 digits: (0.87725294608097 mol/L, 324.475443431599 K).
    https://apmonitor.com/pdc/index.php/Main/StirredReactor"""
    C_a, T = min(_steady_states(300.0, params), key=lambda s: s[1])
    assert C_a == pytest.approx(0.87725294608097, abs=1e-8)
    assert T == pytest.approx(324.475443431599, abs=1e-6)


@pytest.mark.parametrize(
    "T_c, C_a_published, T_published",
    [(299.709, 0.483, 350.970), (299.413, 0.465, 352.000)],
)
def test_published_middle_branch_states(params, T_c, C_a_published, T_published):
    """Decardi-Nelson and Liu (2022), Section 5.3, hold the reactor on two
    middle-branch states: (0.483, 350.970) at 299.709 K and (0.465, 352.000) at
    299.413 K. Matched within their rounding, and both are saddles.
    https://arxiv.org/html/2109.09810v1"""
    C_a, T = sorted(_steady_states(T_c, params), key=lambda s: s[1])[1]
    assert C_a == pytest.approx(C_a_published, abs=6e-4)
    assert T == pytest.approx(T_published, abs=5e-3)
    J = _jacobian(C_a, T, T_c, params)[:2, :2]
    assert np.linalg.det(J) < 0.0


def test_nominal_middle_state(params):
    """At a 300 K jacket the middle state is (0.499918, 350.0055 K), with
    eigenvalues -0.4542 and +2.8344 /min. The textbook's (0.5, 350) is close
    but not an exact steady state."""
    C_a, T = sorted(_steady_states(300.0, params), key=lambda s: s[1])[1]
    assert C_a == pytest.approx(0.499918, abs=1e-6)
    assert T == pytest.approx(350.0055, abs=1e-3)
    eig = np.sort(np.linalg.eigvals(_jacobian(C_a, T, 300.0, params)[:2, :2]).real)
    np.testing.assert_allclose(eig, [-0.4542, 2.8344], atol=1e-3)

    with enable_x64():
        v = np.asarray(
            compute_velocity(jnp.array([0.5, 350.0, 300.0]), 300.0, 0.0, params)[0]
        )
    assert v[0] == pytest.approx(3.4e-5, abs=1e-6)
    assert v[1] == pytest.approx(-7.1e-3, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Multiplicity and folds
# ---------------------------------------------------------------------------


def test_steady_state_counts_and_roots(params):
    """1, 3, 3 and 1 equilibria at 295, 300, 302 and 305 K, at the temperatures
    cstr's PHYSICS.md tabulates (to its 0.1 K rounding)."""
    shipped_table = {
        295.0: [317.7],
        300.0: [324.5, 350.0, 369.7],
        302.0: [328.7, 343.5, 373.6],
        305.0: [378.1],
    }
    for T_c, expected in shipped_table.items():
        found = sorted(T for _, T in _steady_states(T_c, params))
        assert len(found) == len(expected), (T_c, found)
        np.testing.assert_allclose(found, expected, atol=0.05, err_msg=str(T_c))


def test_multiplicity_window_and_folds(params):
    """Three equilibria exactly for a jacket in (298.080, 303.229) K. The folds
    are where Tc*(L) turns: the extinction fold at T 360.511 K and the ignition
    fold at T 335.654 K."""
    with enable_x64():
        slope = jax.vmap(jax.grad(lambda L: steady_coolant(L, 0.0, params)))
        grid = np.linspace(0.05, 0.95, 2001)
        g = np.asarray(slope(jnp.asarray(grid)))
        folds = [
            brentq(
                lambda L: float(slope(jnp.array([L]))[0]),
                grid[i],
                grid[i + 1],
                xtol=1e-14,
            )
            for i in np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
        ]
    assert len(folds) == 2
    extinction, ignition = sorted(folds)  # the hot fold has the lower C_a
    for L, T_fold, T_c_fold in (
        (extinction, 360.511, 298.080),
        (ignition, 335.654, 303.229),
    ):
        assert float(steady_temperature(L, params, np)) == pytest.approx(
            T_fold, abs=0.01
        )
        assert float(steady_coolant(L, 0.0, params, np)) == pytest.approx(
            T_c_fold, abs=0.01
        )

    for T_c, n in ((298.06, 1), (298.10, 3), (303.21, 3), (303.25, 1)):
        assert len(_steady_states(T_c, params)) == n, T_c


# ---------------------------------------------------------------------------
# 4. Stability of the targets and the branches
# ---------------------------------------------------------------------------


def _check_7_entry():
    """``KNOWN_OPEN_LOOP_UNSTABLE["unstable_cstr"]``, read from the conformance
    suite's source (importing it would build every spec)."""
    path = pathlib.Path(__file__).resolve().parents[2] / "test_env_conformance.py"
    for node in ast.parse(path.read_text()).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "KNOWN_OPEN_LOOP_UNSTABLE"
        ):
            return ast.literal_eval(node.value).get("unstable_cstr")
    return None


def test_every_target_is_a_saddle(params):
    """At every target the two balances have one negative and one positive
    eigenvalue, lambda+ from 3.158 /min at 0.45 to 1.391 at 0.65. The jacket
    row of the three-state Jacobian is (0, 0, -1/tau_j), so the lag adds
    exactly that eigenvalue and moves neither of the others."""
    for L, lam in zip(TARGETS, LAMBDA_PLUS):
        J = _jacobian(*_at_target(L, params), params)
        J2 = J[:2, :2]
        assert np.linalg.det(J2) < 0.0, L
        eig2 = np.sort(np.linalg.eigvals(J2).real)
        assert eig2[0] < 0.0 < eig2[1]
        assert eig2[1] == pytest.approx(lam, abs=0.01), L

        np.testing.assert_array_equal(J[2], [0.0, 0.0, -1.0 / params.tau_j])
        eig3 = np.sort(np.linalg.eigvals(J).real)
        np.testing.assert_allclose(
            eig3, np.sort([*eig2, -1.0 / params.tau_j]), atol=1e-9
        )


def test_the_check_7_allowlist_quotes_these_rates():
    """The conformance suite skips check 7 for this task on the strength of
    the saddle test above, so its entry must quote the same rates."""
    entry = _check_7_entry()
    assert entry is not None, "no KNOWN_OPEN_LOOP_UNSTABLE entry for unstable_cstr"
    rates = [float(x) for x in re.findall(r"lambda\+ ([\d.]+) to ([\d.]+)", entry)[0]]
    assert rates == pytest.approx([min(LAMBDA_PLUS), max(LAMBDA_PLUS)], abs=0.01)
    assert "test_every_target_is_a_saddle" in entry


def test_upper_branch_is_an_unstable_focus(params):
    """The ignited states at 300 and 305 K are unstable foci (trace +2.715 and
    +0.587 /min), not stable states: the branch turns stable at a Hopf point,
    Tc 306.22 K and T 379.61 K. At a 310 K jacket, full heating, the only
    equilibrium is a stable focus at 383.89 K, above the trip."""
    for T_c, trace in ((300.0, 2.715), (305.0, 0.587)):
        C_a, T = min(_steady_states(T_c, params), key=lambda s: s[0])
        J2 = _jacobian(C_a, T, T_c, params)[:2, :2]
        assert np.trace(J2) == pytest.approx(trace, abs=0.005), T_c
        assert np.trace(J2) ** 2 < 4.0 * np.linalg.det(J2)  # complex pair

    def trace_on_branch(L):
        return np.trace(_jacobian(*_at_target(L, params), params)[:2, :2])

    L_hopf = brentq(trace_on_branch, 0.10, 0.14, xtol=1e-12)
    assert float(steady_coolant(L_hopf, 0.0, params, np)) == pytest.approx(
        306.22, abs=0.05
    )
    assert float(steady_temperature(L_hopf, params, np)) == pytest.approx(
        379.61, abs=0.05
    )

    (only,) = _steady_states(params.T_c_max, params)
    assert only[1] == pytest.approx(383.89, abs=0.01)
    assert only[1] > params.T_trip
    J2 = _jacobian(*only, params.T_c_max, params)[:2, :2]
    assert np.all(np.linalg.eigvals(J2).real < 0.0)


# ---------------------------------------------------------------------------
# 5. The closed forms reset, PID and MPC use
# ---------------------------------------------------------------------------


def test_steady_coolant_is_an_equilibrium(params):
    """At (L, T*(L), Tc*(L, dTi)) the velocity is zero for every drift, which
    is also what shows T* holds whatever the drift. Tc* moves by
    -dTi / beta, and the actuator keeps headroom on both sides."""
    levels = np.linspace(*params.target_CA_range, 21)
    beta = params.UA / (params.rho * params.C * params.V)
    assert beta == pytest.approx(2.0921, abs=1e-4)
    with enable_x64():
        for Ti_dev in (-6.0, 0.0, 6.0):
            for L in levels:
                x = jnp.array(_at_target(L, params, Ti_dev))
                v = np.asarray(compute_velocity(x, x[2], Ti_dev, params)[0])
                assert np.max(np.abs(v)) < 1e-9, (L, Ti_dev, v)

    for Ti_dev in (-6.0, 6.0):
        shift = steady_coolant(levels, Ti_dev, params, np) - steady_coolant(
            levels, 0.0, params, np
        )
        np.testing.assert_allclose(shift, -Ti_dev / beta, atol=1e-9)

    lo, hi = params.target_CA_range
    assert float(steady_coolant(lo, 0.0, params, np)) == pytest.approx(
        299.187, abs=0.01
    )
    assert float(steady_coolant(hi, 0.0, params, np)) == pytest.approx(
        302.502, abs=0.01
    )

    def headroom(Ti_dev):
        T_c = steady_coolant(levels, Ti_dev, params, np)
        return min(np.min(T_c - params.T_c_min), np.min(params.T_c_max - T_c))

    assert headroom(0.0) > 7.49  # 7.50 K, at 0.65
    # 4.63 K at dTi -6 K (0.65, upper bound), 6.32 K at +6 K (0.45, lower)
    assert min(headroom(-6.0), headroom(6.0)) > 4.6


def test_no_p_or_pi_loop_on_c_a_stabilises(params):
    """On the three-state linearisation the plant polynomial is
    (s + 1/tau_j)(s^2 - tr s + det), and the path from the coolant to C_a has
    a constant numerator. A loop on C_a alone adds a multiple of that constant
    to the lowest coefficient, so under P the s^1 coefficient, and under PI
    the s^2 coefficient, stays det - tr/tau_j: negative at every target
    whatever the gains. A controller must use T or derivative action."""
    C = np.array([1.0, 0.0, 0.0])
    Ti_int = 0.5  # min, any positive integral time
    expected = {0.45: -28.9, 0.65: -9.4}
    for L in TARGETS:
        x = _at_target(L, params)
        A = _jacobian(*x, params)
        with enable_x64():
            B = np.asarray(
                jax.jacfwd(lambda u: compute_velocity(jnp.array(x), u, 0.0, params)[0])(
                    jnp.float64(x[2])
                )
            )
        tr, det = np.trace(A[:2, :2]), np.linalg.det(A[:2, :2])
        plant = np.polymul([1.0, 1.0 / params.tau_j], [1.0, -tr, det])
        np.testing.assert_allclose(np.poly(A), plant, rtol=1e-9, atol=1e-9)
        # Relative degree three: C B = C A B = 0, so the numerator is C A^2 B.
        assert C @ B == 0.0 and C @ A @ B == 0.0
        assert C @ A @ A @ B != 0.0

        stuck = det - tr / params.tau_j
        assert stuck < 0.0
        if L in expected:
            assert stuck == pytest.approx(expected[L], abs=0.1), L
        for gain in (100.0, 1000.0):  # K of coolant per mol/L, raised with C_a
            p_loop = np.poly(A + gain * np.outer(B, C))
            assert p_loop[2] == pytest.approx(stuck, rel=1e-9), (L, gain)
            pi_loop = np.poly(
                np.block(
                    [
                        [A + gain * np.outer(B, C), (gain / Ti_int) * B[:, None]],
                        [C[None, :], np.zeros((1, 1))],
                    ]
                )
            )
            assert pi_loop[2] == pytest.approx(stuck, rel=1e-9), (L, gain)


def test_short_term_and_static_signs_are_opposite(params):
    """Raising the coolant 1 K from an equilibrium lowers C_a within two steps
    (the reactor heats and burns faster), while the steady coolant rises with
    the target across the band: holding a higher C_a needs a warmer jacket."""
    p = params.replace(Ti_sigma=0.0)
    key = jax.random.PRNGKey(0)
    with enable_x64():
        for L in TARGETS:
            C_a, T, T_c = _at_target(L, p)
            state = UnstableCSTRState(
                time=jnp.int64(0),
                C_a=jnp.float64(C_a),
                T=jnp.float64(T),
                T_j=jnp.float64(T_c),
                Ti_dev=jnp.float64(0.0),
                target_levels=jnp.full((N_BLOCKS,), L, jnp.float64),
                block_clock=jnp.int64(0),
            )
            held, raised = state, state
            for _ in range(2):
                held, _ = compute_next_state(
                    jnp.float64(raw_from_coolant(T_c, p)), held, p, key
                )
                raised, _ = compute_next_state(
                    jnp.float64(raw_from_coolant(T_c + 1.0, p)), raised, p, key
                )
            assert float(raised.C_a) < float(held.C_a), L

    levels = np.linspace(*params.target_CA_range, 41)
    assert np.all(np.diff(steady_coolant(levels, 0.0, params, np)) > 0.0)


# ---------------------------------------------------------------------------
# 6. The point of no return and the line the controllers read
# ---------------------------------------------------------------------------


def test_point_of_no_return_at_the_hottest_target(params):
    """At 0.45 mol/L, the hottest target and the one closest to the edge,
    full cooling from the equilibrium's jacket still brings T back from
    3.30 K above T* with no drift and from 2.48 K above it at a +6 K drift,
    raising T alone. Along the unstable eigenvector, where C_a falls as T
    rises, the edge is 5.10 K out (dTi 0). All three are derived
    (scripts/unstable_cstr_numbers.py --section pnr). The margin in T grows
    with the target, so 0.45 is the tightest one at every drift."""
    L = TARGETS[0]
    T_star = float(steady_temperature(L, params, np))
    for Ti_dev, margin in ((0.0, 3.304), (6.0, 2.476)):
        T_j = float(steady_coolant(L, Ti_dev, params, np))
        assert point_of_no_return_batch(L, T_j, Ti_dev, params)[
            0
        ] - T_star == pytest.approx(margin, abs=0.01), Ti_dev

    levels = np.array(TARGETS)
    for Ti_dev in (-6.0, 0.0, 6.0):
        T_j = steady_coolant(levels, Ti_dev, params, np)
        margins = point_of_no_return_batch(
            levels, T_j, Ti_dev, params
        ) - steady_temperature(levels, params, np)
        assert np.all(np.diff(margins) > 0.0), (Ti_dev, margins)

    # Bisection along the eigenvector: the smallest offset s (K of T) from
    # which full cooling from (L + s dC_a/dT, T* + s) still trips.
    slope = _unstable_direction(L, params)
    assert slope < 0.0
    T_j = float(steady_coolant(L, 0.0, params, np))
    lo, hi = 0.0, 40.0
    with enable_x64():
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            trips = trips_under_full_cooling(
                *(jnp.float64(v) for v in (L + mid * slope, T_star + mid, T_j, 0.0)),
                params,
            )
            lo, hi = (lo, mid) if bool(trips) else (mid, hi)
    assert hi == pytest.approx(5.098, abs=0.01)


def test_pnr_line_is_conservative(params):
    """``pnr_line``, which caps the PID's setpoint and bounds the MPC's T, lies
    at or below the exact point of no return (jacket at the steady coolant)
    over C_a 0.40 to 0.70 at a drift of -6, 0 and +6 K. The grid is 0.0025
    mol/L, twice as fine as the one the line was fitted on, and holds every
    target. The line was fitted at +6 K, where it touches the curve. The
    smallest gap is 0.017 K, at C_a 0.5475 on this grid (derived). The MPC's
    bound, the line less ``MPC_PNR_MARGIN_K``, still lies above T*(L) at
    every target in the band (by 1.067 K at 0.45, derived), so it never cuts
    a target off."""
    grid = np.round(np.arange(0.40, 0.70 + 1e-9, 0.0025), 4)
    assert set(TARGETS) <= set(grid.tolist())
    gaps = {}
    for Ti_dev in (-6.0, 0.0, 6.0):
        exact = point_of_no_return_batch(
            grid, steady_coolant(grid, Ti_dev, params, np), Ti_dev, params
        )
        gaps[Ti_dev] = exact - pnr_line(grid)
        worst = int(np.nanargmin(gaps[Ti_dev]))
        assert np.all(gaps[Ti_dev] >= 0.0), (Ti_dev, grid[worst], gaps[Ti_dev][worst])
    assert gaps[6.0].min() < 0.05
    assert gaps[0.0].min() > gaps[6.0].min() and gaps[-6.0].min() > gaps[0.0].min()

    band = np.linspace(*params.target_CA_range, 2001)
    headroom = pnr_line(band) - MPC_PNR_MARGIN_K - steady_temperature(band, params, np)
    assert headroom.min() > 1.0  # 1.067 K, at 0.45


# ---------------------------------------------------------------------------
# 7. Conservation and the integrator
# ---------------------------------------------------------------------------

#: RK4's stability limit on the negative real axis, |lambda| dt (read).
RK4_REAL_LIMIT = 2.785


def test_energy_balance_closes_in_an_adiabatic_batch(params):
    """Sealed and insulated (q 0, UA 0), every mole converted heats the
    reactor by A = -deltaHr / (rho C), so T + A C_a is constant. RK4 keeps a
    linear invariant exactly, so the env's own step holds it to rounding over
    a run to 99 % conversion (64 steps from 0.1 mol/L at 340 K)."""
    A = -params.deltaHr / (params.rho * params.C)
    C_a, T = _adiabatic_batch(params)
    converted = np.flatnonzero(C_a <= 0.01 * C_a[0])
    assert converted.size, "the batch did not reach 99 % conversion"
    end = converted[0] + 1
    np.testing.assert_allclose(
        T[:end] + A * C_a[:end], T[0] + A * C_a[0], rtol=0.0, atol=1e-6
    )
    assert T[end - 1] - T[0] == pytest.approx(A * (C_a[0] - C_a[end - 1]), abs=1e-6)


def test_rk4_is_stable_where_the_plant_can_go(params):
    """RK4 at ``delta_t`` is stable for |lambda| dt below 2.785 on the real
    axis (read). The fastest dynamics are the runaway just below the trip,
    so the paths checked are full heating until the trip, from the 135
    corners of the reset box (five targets, 3 x 3 corners, drift -6, 0 and
    +6 K) and from 33 extinguished states (jacket 290 to 300 K, the same
    drifts).

    Each set of paths gets its own bound, just above what it reaches
    (derived, scripts/unstable_cstr_numbers.py --section integrator). The
    reset-box paths reach 0.711. A catch from the extinguished branch reaches
    the trip with more reactant left (C_a 0.69 at 364.6 K, against 0.54 from
    the box), and the runaway's rate grows with C_a, so those paths reach
    0.989. A bound set from the box alone (0.8) would not cover them. Both
    stay well inside the limit, at 26 and 36 % of it. Anywhere with
    0 <= C_a <= Caf below the trip, reachable or not, the largest is 1.616,
    at C_a = Caf and T = 365 K."""
    dt = params.delta_t
    box = np.array(
        [
            (L + a * params.initial_CA_offset, T_star + b * params.initial_T_offset)
            + (float(steady_coolant(L, d, params, np)), d)
            for L, T_star in (
                (x, float(steady_temperature(x, params, np))) for x in TARGETS
            )
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for d in (-6.0, 0.0, 6.0)
        ]
    ).T
    extinguished = np.array(
        [
            (*min(_steady_states(T_j, params, d), key=lambda s: s[1]), T_j, d)
            for d in (-6.0, 0.0, 6.0)
            for T_j in np.arange(290.0, 300.5, 1.0)
        ]
    ).T
    assert box.shape[1] == 135 and extinguished.shape[1] == 33

    def worst_on_the_way_to_the_trip(starts, n=80):
        with enable_x64():
            paths = _hold(_states64(*starts), 1.0, params, n)
        C_a, T, T_j, tripped = (np.asarray(x) for x in paths)
        assert tripped.any(axis=1).all(), "full heating did not trip in 80 steps"
        before = np.arange(n)[None, :] < np.argmax(tripped, axis=1)[:, None]
        points = np.concatenate(
            [starts[:3].T, np.stack([C_a[before], T[before], T_j[before]], axis=1)]
        )
        return _spectral_radii(points, params).max() * dt

    assert worst_on_the_way_to_the_trip(box) <= 0.75  # 0.711
    assert worst_on_the_way_to_the_trip(extinguished) <= 1.0  # 0.989

    C_a, T = np.meshgrid(
        np.linspace(0.0, params.Caf, 201),
        np.linspace(280.0, params.T_trip, 171),
        indexing="ij",
    )
    grid = np.stack([C_a.ravel(), T.ravel(), np.full(C_a.size, 300.0)], axis=1)
    anywhere = _spectral_radii(grid, params).max() * dt
    assert anywhere <= 1.62  # 1.616
    assert anywhere < RK4_REAL_LIMIT


# ---------------------------------------------------------------------------
# 8. Known deviations (PHYSICS.md section 9), each a strict xfail: the test
#    states what a real plant does, and fails the suite if the model starts
#    doing it without PHYSICS.md and this marker being updated.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Deviation D1: C = 0.239 J/(g K), the value every source uses, gives an "
        "adiabatic rise of 209 K; an aqueous feed would rise about 12 K."
    ),
)
def test_adiabatic_rise_matches_an_aqueous_feed(params):
    """D1. Converted in a sealed, insulated batch, an aqueous feed with this
    heat of reaction (C 4.184 J/(g K), read) would rise about 12 K (derived).
    The rise is measured by stepping the env: the heating per mole converted
    in the adiabatic batch, times Caf."""
    C_a, T = _adiabatic_batch(params)
    rise = params.Caf * (T[-1] - T[0]) / (C_a[0] - C_a[-1])
    assert rise <= 20.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Deviation D4: Caf is constant; only the feed temperature drifts, so with "
        "that drift off nothing moves the plant off an equilibrium."
    ),
)
def test_feed_composition_moves_the_equilibrium(params):
    """D4. A real feed's concentration varies, and on a saddle any variation
    moves the plant off its equilibrium. With the temperature drift off
    (``Ti_sigma`` 0) and the coolant held at Tc*, a minute of steps under
    fresh keys should move C_a by more than 1 % of the analyser's
    resolution at some target."""
    p = params.replace(Ti_sigma=0.0)
    keys = jax.random.split(jax.random.PRNGKey(0), 20)
    moved = 0.0
    with enable_x64():
        for L in TARGETS:
            state = _states64(*_at_target(L, p), 0.0)
            state = jax.tree_util.tree_map(lambda x: x[0], state)
            raw = jnp.float64(raw_from_coolant(float(state.T_j), p))
            for key in keys:
                state, _ = compute_next_state(raw, state, p, key)
                moved = max(moved, abs(float(state.C_a) - L))
    assert moved > 1e-6
