"""The four-tank task against its plant record (REALITY.md, COLD_REVIEW.md
tests 1 to 6).

Contract, from Johansson and Nunes 1998 [S1] and the copies of the rig in
REALITY.md, not from the code:
- with the Lund rig's constants the model settles at the printed P+ levels,
  12.6, 13.0, 4.8, 4.9 cm at 3.15 V, each within 5 % or 0.1 cm [S1, p. 2046];
- its time constants are (63, 91, 39, 56) s within 10 % [S1, p. 2047];
- the relative gain is gamma1 gamma2 / (gamma1 + gamma2 - 1) within 5 %
  [S3, eq. (2)];
- the shipped task's cross gains are 1.3 to 2.6 times its direct gains
  [S1, eq. (3); S5, Table 3];
- its right half-plane zero lies in 0.010 to 0.040 rad/s [S1, p. 2047; S5,
  p. 4];
- its upper tanks sit below and run faster than its lower tanks: level ratios
  at most 0.6, time-constant ratios below 1 [S1; S5].
Tolerances are the audit's, fixed before any run.

Units: the task is in SI; the paper's cm figures are converted here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from target_gym.pc_gym.four_tank.env import FourTankParams, compute_velocity
from target_gym.pc_gym.four_tank.env_jax import FourTank

CM = 1e-2
#: The Lund rig [S1, p. 2046], in SI: cm2 -> m2 is 1e-4, cm3 -> m3 is 1e-6.
LUND = dict(
    A1=28e-4, A2=32e-4, A3=28e-4, A4=32e-4,
    a1=0.071e-4, a2=0.057e-4, a3=0.071e-4, a4=0.057e-4, g=9.81,
)  # fmt: skip
LUND_PLUS = dict(LUND, k1=3.14e-6, k2=3.29e-6, gamma1=0.43, gamma2=0.34)


def quiet(**kw):
    """No setpoint moves, no extra outlet, no cup, no instrument noise."""
    return FourTankParams(
        sensor_noise_scale=0.0,
        setpoint_step=0.0,
        leak_fraction=0.0,
        cup_volume=0.0,
        **kw,
    )


@pytest.fixture(scope="module")
def env():
    return FourTank()


def raw(v, p):
    return 2.0 * (jnp.asarray(v) - p.v_min) / (p.v_max - p.v_min) - 1.0


def run(env, p, v, h0, steps):
    """Levels over ``steps`` seconds at constant pump voltages ``v``."""
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(h1=h0[0], h2=h0[1], h3=h0[2], h4=h0[3])
    action = raw(v, p)

    def body(s, _):
        _, s, *_ = env.step_env(jax.random.PRNGKey(0), s, action, p)
        return s, jnp.stack([s.h1, s.h2, s.h3, s.h4])

    return jax.lax.scan(body, s, None, length=steps)[1]


def settle(env, p, v, h0=(0.12, 0.12, 0.05, 0.05)):
    return np.asarray(run(env, p, v, jnp.asarray(h0), 6000)[-1])


def tau63(trace, dt=1.0):
    """Time for a step response to cover 63 % of its change."""
    y = np.asarray(trace)
    frac = (y - y[0]) / (y[-1] - y[0])
    i = int(np.argmax(frac >= 1.0 - np.exp(-1.0)))
    # Linear interpolation between the samples either side of the crossing.
    lo = frac[i - 1]
    return dt * (i - 1 + ((1.0 - np.exp(-1.0)) - lo) / (frac[i] - lo) + 1)


def test_the_lund_rig_settles_at_its_printed_levels(env):
    """[S1, p. 2046], P+ at 3.15 V on both pumps."""
    p = quiet(**LUND_PLUS)
    h = settle(env, p, (3.15, 3.15)) / CM
    for got, want in zip(h, (12.6, 13.0, 4.8, 4.9)):
        assert abs(got - want) <= max(0.05 * want, 0.1), (h, got, want)


def test_the_lund_rig_has_its_printed_time_constants(env):
    """[S1, p. 2047]: P+ (T1, T2, T3, T4) = (63, 91, 39, 56) s. A small step
    on pump 1 moves tank 1 directly and tank 4 alone, each first order; pump
    2 moves tank 2 and tank 3."""
    p = quiet(**LUND_PLUS)
    h0 = settle(env, p, (3.15, 3.15))
    one = np.asarray(run(env, p, (3.15 * 1.02, 3.15), h0, 1500))
    two = np.asarray(run(env, p, (3.15, 3.15 * 1.02), h0, 1500))
    trace = lambda tr, i: np.concatenate([[h0[i]], tr[:, i]])  # noqa: E731
    got = (
        tau63(trace(one, 0)),
        tau63(trace(two, 1)),
        tau63(trace(two, 2)),
        tau63(trace(one, 3)),
    )
    for g, want in zip(got, (63, 91, 39, 56)):
        assert abs(g - want) <= 0.10 * want, got


def _gains(env, p):
    """Steady-state gain matrix of (h1, h2) on (v1, v2), by 2 % steps from
    the task's nominal pumps."""
    v0 = np.array([float(p.v_nominal)] * 2)
    base = settle(env, p, tuple(v0))
    G = np.zeros((2, 2))
    for j in range(2):
        v = v0.copy()
        v[j] *= 1.02
        G[:, j] = (settle(env, p, tuple(v), base)[:2] - base[:2]) / (0.02 * v0[j])
    return G, base, v0


def test_the_relative_gain_follows_the_valve_split(env):
    """[S3, eq. (2)]: lambda = gamma1 gamma2 / (gamma1 + gamma2 - 1)."""
    p = quiet()
    G, _, _ = _gains(env, p)
    lam = G[0, 0] * G[1, 1] / (G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0])
    want = p.gamma1 * p.gamma2 / (p.gamma1 + p.gamma2 - 1.0)
    assert lam == pytest.approx(want, rel=0.05)


def test_cross_gains_are_within_the_rigs_range_of_the_direct_ones(env):
    """[S1, eq. (3)] G+(0) = 1.5, 2.5, 2.5, 1.6; [S5, Table 3]. 1.3 to 2.6."""
    p = quiet()
    G, _, _ = _gains(env, p)
    for ratio in (G[0, 1] / G[0, 0], G[1, 0] / G[1, 1]):
        assert 1.3 <= ratio <= 2.6, G


def test_the_right_half_plane_zero_is_within_the_rigs_range(env):
    """[S1, p. 2047] P+ zero at +0.013; [S5, p. 4] +0.0324. 0.010 to 0.040."""
    p = quiet()
    _, h, v = _gains(env, p)
    f = lambda x, u: compute_velocity(x, u, p)[0]  # noqa: E731
    A = np.asarray(jax.jacfwd(f, 0)(jnp.asarray(h), jnp.asarray(v)))
    B = np.asarray(jax.jacfwd(f, 1)(jnp.asarray(h), jnp.asarray(v)))
    C = np.eye(4)[:2]
    M = np.block([[A, B], [-C, np.zeros((2, 2))]])
    N = np.block([[np.eye(4), np.zeros((4, 2))], [np.zeros((2, 6))]])
    zeros = scipy.linalg.eigvals(M, N)
    zeros = zeros[np.isfinite(zeros)].real
    rhp = zeros[zeros > 0]
    assert len(rhp) == 1 and 0.010 <= rhp[0] <= 0.040, zeros


def _corners(p):
    """Steady pump voltages for the lower levels an episode can ask for: the
    nominal point and 1.25 cm (5 percentage points of a 25 cm tank [S4, p. 9])
    either side on each tank. Solved from the rig's eq. (1) [S1], linear in
    the voltages at steady state."""
    v0 = float(p.v_nominal)
    g = p.g
    M = np.array(
        [
            [p.gamma1 * p.k1, (1 - p.gamma2) * p.k2],
            [(1 - p.gamma1) * p.k1, p.gamma2 * p.k2],
        ]
    )
    q = M @ np.array([v0, v0])
    h1 = (q[0] / p.a1) ** 2 / (2 * g)
    h2 = (q[1] / p.a2) ** 2 / (2 * g)
    out = []
    for d1 in (-1.25 * CM, 1.25 * CM):
        for d2 in (-1.25 * CM, 1.25 * CM):
            rhs = np.array(
                [p.a1 * np.sqrt(2 * g * (h1 + d1)), p.a2 * np.sqrt(2 * g * (h2 + d2))]
            )
            out.append(tuple(np.linalg.solve(M, rhs)))
    return out


def test_upper_tanks_sit_below_and_run_faster_than_lower_tanks(env):
    """[S1, p. 2046] h3/h1 about 0.38 (P+); [S5, Table 3] 0.51, 0.42; every
    published upper time constant is shorter than its lower one."""
    p = quiet()
    for v in _corners(p):
        h = settle(env, p, v)
        assert h[2] / h[0] <= 0.6 and h[3] / h[1] <= 0.6, h
        T = [
            getattr(p, f"A{i + 1}") / getattr(p, f"a{i + 1}") * np.sqrt(2 * h[i] / p.g)
            for i in range(4)
        ]
        assert T[2] / T[0] < 1.0 and T[3] / T[1] < 1.0, T
