"""Calcination's stable step in the cement kiln.

Contract. The model is the ODE of PHYSICS.md section 1, for the residual
r = 1 - alpha of each cell, at a rate k_i = A exp(-E / (R T_i)) and with the
solid crossing a cell in tau / n, so at v = n / tau:

    dr_i/dt = -k_i r_i + v (r_{i-1} - r_i),    r_{-1} = 1 (the raw inlet).

Above 1429 K, k h is past what explicit RK4 can follow at the env's 30 s step
(its real limit is 2.785). ``calcination_step(alpha, T_solid, rpm, raw_meal, h,
params)`` steps this equation with the temperatures held, by exponential Euler
(Hochbruck and Ostermann 2010, eq. 1.6), and returns the end state and the
calcination rate each cell ran at on average. It promises:

- a cell with a constant inflow is exact (the equation is then linear);
- a steady state of the ODE does not move (the steady residual is the
  recurrence r_i = v r_{i-1} / (k_i + v));
- residual is conserved down the column: over a step the kiln gains what
  came in and loses what left and what reacted;
- fractions stay fractions, and a deviation never grows, for any h.

The plant's step takes its heat from that mean rate, so ``compute_next_state``
promises the ODE's trajectory to within a tenth of the reward's floor, against
a 0.25 s explicit RK4 of ``compute_velocity``, where h k_calc(1900 K) = 0.8.

Oracles: scipy's Radau on the ODE, the ODE's fixed point, the balance above,
that fine reference, and hand arithmetic for the heat.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.tree_util import Partial as partial
from scipy.integrate import solve_ivp

from target_gym.cement_kiln.env import (
    CementKilnParams,
    calcination_step,
    compute_next_state,
    compute_velocity,
    residence_time,
    steady_profile,
)
from target_gym.cement_kiln.env_jax import CementKiln
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import enable_x64, float64

R_GAS = 8.314
H = 30.0  # the env's step


def rate(T, p):
    """The Arrhenius law of PHYSICS.md, section 3."""
    A, E = float(p.A_calcination), float(p.E_calcination)
    return A * np.exp(-E / (R_GAS * np.asarray(T)))


def crossing_rate(rpm, p):
    """The solid crosses one of the n cells in tau / n."""
    with enable_x64():
        return p.n_zones / float(residence_time(rpm, p))


@pytest.fixture(scope="module")
def p():
    with enable_x64():
        return float64(CementKilnParams())


@pytest.fixture(scope="module")
def nominal(p):
    return float(p.rpm_nominal), float(p.raw_meal_nominal)


def _step(alpha, T, nominal, h, p):
    rpm, meal = nominal
    with enable_x64():
        end, mean_rate = calcination_step(
            jnp.asarray(alpha), jnp.asarray(T), rpm, meal, h, p
        )
        return np.asarray(end), np.asarray(mean_rate)


def _profile(seed, p):
    rng = np.random.default_rng(seed)
    n = p.n_zones
    return rng.uniform(0.0, 1.0, n), rng.uniform(300.0, 2400.0, n)


def test_the_first_cell_follows_the_ode_exactly(p, nominal):
    """Cell 0 sees only the inlet, which is constant, so its equation is linear
    with a constant inflow: Radau's solution over the step, with the reacted
    amount as a second state, is the answer to its tolerance."""
    alpha, T = np.full(p.n_zones, 0.4), np.full(p.n_zones, 1800.0)
    end, mean_rate = _step(alpha, T, nominal, H, p)
    k, v = rate(1800.0, p), crossing_rate(nominal[0], p)
    ode = solve_ivp(
        lambda _, y: [-(k + v) * y[0] + v, k * y[0]],
        (0.0, H),
        [0.6, 0.0],
        method="Radau",
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(1.0 - end[0], ode.y[0, -1], rtol=1e-8)
    np.testing.assert_allclose(mean_rate[0] * H, ode.y[1, -1], rtol=1e-8)


@pytest.mark.parametrize("h", [1.0, 30.0, 600.0])
def test_a_steady_residual_does_not_move(p, nominal, h):
    """The ODE's steady residual is a fixed point of the step at any h, and its
    rate is k r: a stable step must not move a steady plant."""
    T = np.linspace(1090.0, 1750.0, p.n_zones)
    k, v = rate(T, p), crossing_rate(nominal[0], p)
    steady = np.empty(p.n_zones)
    inflow = 1.0
    for i in range(p.n_zones):  # the recurrence, from the ODE at d/dt = 0
        inflow = steady[i] = v * inflow / (k[i] + v)
    end, mean_rate = _step(1.0 - steady, T, nominal, h, p)
    np.testing.assert_allclose(1.0 - end, steady, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(mean_rate, k * steady, rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("seed", range(4))
def test_the_step_conserves_residual_down_the_column(p, nominal, seed):
    """Sum of r changes by v h (the raw inlet, r = 1), less what the last cell
    passed on, v h times its mean r, less everything that reacted. Passing a
    cell's own start value downstream instead of its mean breaks this."""
    alpha, T = _profile(seed, p)
    end, mean_rate = _step(alpha, T, nominal, H, p)
    v = crossing_rate(nominal[0], p)
    left = v * H * mean_rate[-1] / rate(T[-1], p)
    change = np.sum(alpha - end)
    np.testing.assert_allclose(
        change, v * H - left - H * np.sum(mean_rate), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("h", [1.0, 30.0, 600.0])
def test_fractions_stay_fractions_at_any_step(p, nominal, seed, h):
    alpha, T = _profile(seed, p)
    end, mean_rate = _step(alpha, T, nominal, h, p)
    eps = 1e-12
    assert np.all(end >= -eps) and np.all(end <= 1.0 + eps)
    assert np.all(mean_rate >= 0.0)


def _reset(p):
    with enable_x64():
        _, state = CementKiln().reset_env(jax.random.PRNGKey(0), p)
    return state


def _next(state, p):
    env = CementKiln()
    with enable_x64():
        s, _ = compute_next_state(
            jnp.zeros(2),
            state,
            p,
            jax.random.PRNGKey(1),
            integration_method=env.integration_method,
        )
    return s


def test_a_deviation_in_the_fractions_does_not_grow_over_a_step(p):
    """Each cell's next value is a weighted mean of its own and its inflow's,
    with weights summing to at most 1 (eq. 1.6 on a decay), so the worst
    deviation cannot rise. Every cell is moved: where an explicit step only
    overshoots 1 its clip hides the gain, so the test needs the cells past RK4's
    limit where the residual stays positive (cells 8 and 9 at 30 s)."""
    state = _reset(p)
    size = 1e-6
    off = state.replace(alpha=state.alpha - size)
    moved = np.abs(np.asarray(_next(off, p).alpha - _next(state, p).alpha))
    assert np.max(moved) <= size * (1.0 + 1e-6)


def test_a_step_takes_the_heat_of_the_carbonate_that_reacted(p):
    """By hand: what the kiln still calcines is (1 - 0.92) of 0.79 carbonate,
    at 1780 kJ/kg and 1 kJ/(kg K), so 112.5 K per unit of alpha. Cell 13 runs
    at about 1710 K, k h of about 29, so 10 % left there reacts within the
    step: 0.99 of 11.25 K, at q = 0.371 K/s while the rate is held. The charge
    relaxes against that at r = 0.0233 /s: advection 0.0107, gas and wall
    0.0028 (9520 W/K over m c), and clinkering 0.0098, which slows as the
    charge cools (420 K per unit of lime times k_sint L E / (R T^2), at
    L = 0.15). So d = (q / r) (1 - exp(-r h)) = 8.0 K, and with
    h_clinkerisation = 0 the same sum gives 9.2 K, which the plant meets at
    9.0 K. The 5 % is that estimate's own miss. Heat from the instantaneous
    rate at RK4's stages took 150 K here."""
    state = _reset(p)
    left = state.replace(alpha=state.alpha.at[13].set(0.9))
    cooled = float(_next(state, p).T_solid[13] - _next(left, p).T_solid[13])
    assert cooled == pytest.approx(8.0, rel=0.05)


def _rollout(step, y0, actions):
    return np.asarray(jax.lax.scan(lambda y, a: (step(y, a),) * 2, y0, actions)[1])


@pytest.mark.parametrize(
    "fuel, rpm",
    [(0.0, 0.0), (0.08, 0.0), (-0.08, 0.0), (0.0, 0.2)],
    ids=["hold", "fuel up", "fuel down", "speed up"],
)
def test_the_plant_follows_the_odes_discharge_lime(p, fuel, rpm):
    """Against RK4 of the same ODE at 0.25 s, from one lined-out start, with a
    relative move on fuel and speed after 20 steps: free lime stays within a
    tenth of e_floor, the reward's resolution, so no score can read the
    integrator's error. T_bz within 0.1 K: a tenth of a limit's back-off."""
    with enable_x64():
        q = p.replace(feed_noise_std=0.0)
        n, env = q.n_zones, CementKiln()
        state = _reset(q).replace(raw_meal=jnp.asarray(q.raw_meal_nominal))
        steps = 160
        moved = jnp.arange(steps) >= 20
        f0, r0 = q.fuel_nominal, q.rpm_nominal
        fuels = f0 * (1.0 + fuel * moved)
        rpms = r0 * (1.0 + rpm * moved)
        y0 = jnp.concatenate([state.T_solid, state.T_wall, state.alpha, state.lime])

        def fine(y, a):
            velocity = partial(
                compute_velocity, action=(a[0], a[1]), raw_meal=q.raw_meal_nominal,
                params=q,
            )  # fmt: skip
            return integrate_dynamics(
                positions=y, delta_t=H, method="rk4_120", compute_velocity=velocity
            )[0]

        reference = _rollout(fine, y0, jnp.stack([fuels, rpms], axis=1))

        def raw(value, lo, hi):
            return 2.0 * (value - lo) / (hi - lo) - 1.0

        def plant(s, a):
            s1, _ = compute_next_state(
                jnp.stack(
                    [raw(a[0], q.fuel_min, q.fuel_max), raw(a[1], q.rpm_min, q.rpm_max)]
                ),
                s,
                q,
                jax.random.PRNGKey(1),
                integration_method=env.integration_method,
            )
            return s1, (s1.lime[-1], jnp.max(s1.T_solid))

        lime, T_bz = jax.lax.scan(plant, state, jnp.stack([fuels, rpms], axis=1))[1]
    np.testing.assert_allclose(lime, reference[:, -1], atol=0.1 * float(q.e_floor))
    np.testing.assert_allclose(T_bz, np.max(reference[:, :n], axis=1), atol=0.1)


def test_the_lined_out_residual_falls_down_the_kiln(p):
    """At a steady state r_i = v r_{i-1} / (k_i + v) < r_{i-1}, so no cell can
    hold more than the one before it. A march that leans on a clip at alpha = 1
    leaves a jitter of zeros and leftovers along the hot cells."""
    with enable_x64():
        _, _, alpha, _ = steady_profile(p)
    assert np.all(np.diff(1.0 - np.asarray(alpha)) <= 4 * np.finfo(float).eps)
