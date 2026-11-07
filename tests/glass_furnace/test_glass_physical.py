"""Attempt 2's physics, from the equations, written before the plant.

Contract, from the sources and from conservation, not from any code:
- Planck's band fractions and the Rosseland form of Liu 2016 (eqs 5 to 7);
- the parallel-squares view factor 0.1998, closure and reciprocity of exchange areas,
  and an isothermal enclosure exchanging nothing (Feischl eqs 5 to 11);
- the long-tank flow: for uniform viscosity and gradient a, free top and no-slip floor,
  net flux zero, the surface moves at g rho beta a H^3 / (48 mu), by hand below;
- Pilon's foam thickness, eq 19 of JCIS2001;
- blanket mass advected at the charging speed and melted by the heat it takes;
- a step's books balance: stored energy changes by what enters less what leaves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace import physical as P


@pytest.mark.parametrize(
    "lam_T, fraction", [(3000.0, 0.2732), (5000.0, 0.6337), (10000.0, 0.9142)]
)
def test_band_fraction_is_plancks(lam_T, fraction):
    assert P.band_fraction(lam_T) == pytest.approx(fraction, abs=2e-3)


def test_rosseland_limit_of_one_gray_band():
    # Both bands at kappa and opaque past 5 um: kr = 16 n^2 sigma T^3 w / (3 kappa),
    # with w the share of dB/dT under 5 um. Hand: w at 1500 K, lam T = 7500, is under 1.
    T = 1500.0
    kr = P.radiative_conductivity(T, 100.0, 100.0, 1.5)
    limit = 16 * 1.5**2 * P.SIGMA * T**3 / (3 * 100.0)
    assert 0.5 * limit < kr < limit


def test_conductivity_is_liu_at_1400_c():
    # kc = 1.31 + 5.90e-4 * 1400 = 2.136 W/(m K), and kr is of order tens at 1673 K
    # with kappa about 50 /m, 16 * 2.25 * 5.67e-8 * 1673^3 / 150 = 64 at most.
    k = float(P.melt_conductivity(jnp.asarray(1673.15)))
    assert 2.136 + 15 < k < 2.136 + 64


def test_viscosity_lookup_stays_on_the_vft():
    T = jnp.linspace(1400.0, 1950.0, 37)
    exact = P.vft_viscosity(T)
    np.testing.assert_allclose(P.viscosity(T), exact, rtol=2e-3)


def test_parallel_squares_view_factor():
    # Unit squares a unit apart, no gas: F = 0.1998 (Incropera, aligned plates).
    ex = P.direct_areas(n_bays=1, length=1.0, width=1.0, height=1.0, kappa=0.0, walls=False)
    assert ex["ss"][0, 1] == pytest.approx(0.1998, abs=2e-3)


@pytest.fixture(scope="module")
def zones():
    return P.exchange(n_bays=3, length=12.0, width=6.0, height=2.0, kappa=0.08, eps=(0.8, 0.8))


def test_exchange_areas_close_and_are_reciprocal():
    ex = P.direct_areas(n_bays=3, length=12.0, width=6.0, height=2.0, kappa=0.08, walls=True)
    np.testing.assert_allclose(ex["ss"], ex["ss"].T, rtol=1e-9)
    leaving = ex["ss"].sum(1) + ex["sg"].sum(1)
    np.testing.assert_allclose(leaving, ex["area"], rtol=1e-6)
    np.testing.assert_allclose(
        ex["sg"].sum(0) + ex["gg"].sum(1), ex["emission"], rtol=1e-6
    )


def test_isothermal_enclosure_exchanges_nothing(zones):
    E = np.full(6, 3.0)
    H = zones["Hs"] @ E + zones["Hg"] @ np.full(3, 3.0)
    np.testing.assert_allclose(H, 3.0, rtol=1e-9)
    net = zones["Ugg"] @ np.full(3, 3.0) - zones["Ugs"] @ E
    np.testing.assert_allclose(net, 0.0, atol=1e-8)


def test_gas_of_one_path_length_has_gray_emissivity():
    # Hoogendoorn: k = 0.08 /m gives gas emissivity about 0.2 on his 2.4 m chamber's
    # beam, 1 - exp(-0.08 L) with L = 3.5 V / A; a 2.4 m high, 11 m wide duct.
    L = 3.5 * (2.4 * 11.0) / (2 * (2.4 + 11.0))
    assert 1 - np.exp(-0.08 * L) == pytest.approx(0.2, abs=0.05)


def test_long_tank_flow_is_birikh_for_uniform_viscosity():
    # d/dz (mu du/dz) = G + c z, with c = -g rho beta a, z down from the free top:
    # mu u_z = G z + c z^2 / 2, u(z) = -(G (H^2 - z^2) / 2 + c (H^3 - z^3) / 6) / mu.
    # No net flux: G = -3 c H / 8. So u(0) = c H^3 / (48 mu).
    H, mu, a = 1.2, 10.0, 5.0
    dz = jnp.full((400,), H / 400)
    grad = jnp.full((400, 1), a)
    u = P.column_flow(
        grad, jnp.full((400, 1), mu), dz, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1),
        g_rho_beta=9.81 * 2319 * 6e-5,
    )  # fmt: skip
    expected = -9.81 * 2319 * 6e-5 * a * H**3 / (48 * mu)
    assert float(u[0, 0]) == pytest.approx(expected, rel=1e-2)
    # float32: the sum of 400 terms carries about 1e-6 of the scale u H
    assert float(jnp.sum(u[:, 0] * dz)) == pytest.approx(0.0, abs=1e-5 * abs(expected) * H)
    assert float(u[-1, 0]) * expected < 0  # the return runs the other way


def test_long_tank_flow_carries_its_flux_and_follows_the_blanket():
    dz = jnp.full((200,), 0.006)
    grad = jnp.zeros((200, 2))
    mu = jnp.full((200, 2), 10.0)
    u = P.column_flow(
        grad, mu, dz, jnp.array([1e-3, 1e-3]), jnp.array([0.0, 1.0]),
        jnp.array([0.0, 2e-3]), g_rho_beta=1.0,
    )  # fmt: skip
    np.testing.assert_allclose(np.asarray(jnp.sum(u * dz[:, None], 0)), 1e-3, rtol=1e-6)
    assert float(u[0, 1]) == pytest.approx(2e-3, rel=2e-2)


def test_foam_thickness_is_pilons_equation():
    # H = 2905 sigma (mu j)^0.8 / (r0^2.6 (rho g)^1.8), hand: 0.0823 m at these values.
    H = P.foam_thickness(j=1e-3, mu=10.0, sigma=0.3, r0=8.3e-3, rho=2300.0, g=9.81)
    assert float(H) == pytest.approx(0.0823, rel=2e-2)


def test_foam_passes_less_heat_and_runs_hotter_on_top():
    # Same irradiation; with a foam its top runs hotter and less enters the glass.
    H = 800e3  # over sigma T^4 = 595 kW/m^2 at 1800 K
    bare = P.bare_flux(H, 1800.0, 0.8)
    foam, top = P.foam_flux(H, 1800.0, thickness=0.05, eps=0.8)
    assert 0 < foam < bare and top > 1800.0


def test_blanket_moves_at_its_speed_and_loses_what_melted():
    dx = jnp.full((20,), 1.0)
    m, c = jnp.zeros(20), jnp.zeros(20)
    melt = jnp.zeros(20)
    for _ in range(400):
        m, c = P.advance_blanket(m, c, melt, 280.0, 0.25, 2e-3, dx, 30.0)
    # 400 * 30 s * 2 mm/s = 24 m... past the tank's end for a front; the mass in
    # the bed is fed 280 kg/m^2 at v, so the charged mass less what left is kept.
    assert float(m[0]) == pytest.approx(280.0, rel=1e-3)
    front = int(jnp.sum(m > 140.0))
    assert 19 <= front <= 20
    m2, _ = P.advance_blanket(m, c, jnp.full((20,), 0.1), 280.0, 0.25, 0.0, dx, 30.0)
    np.testing.assert_allclose(m - m2, 3.0, rtol=1e-6)


def test_wedge_cover_reads_the_tip_inside_a_cell():
    m = jnp.array([280.0, 280.0, 70.0, 0.0])
    # A linear wedge from the full cell's value to zero: the cell of mean 70 kg/m^2
    # next to one of 280 is half covered.
    np.testing.assert_allclose(P.cover(m), [1.0, 1.0, 0.5, 0.0])


def test_a_step_keeps_its_books():
    g = P.Grid()
    p = P.PhysicalParams()

    def run(s, t):
        return P.plant_step(s, 0.46, 5.79, 0.25, 0.2, t, g, p)[0], None

    # Past the first steps: a foam's top is carried one step, so it only lags once
    # the start's guess has gone.
    s = jax.jit(lambda: jax.lax.scan(run, P.initial_fields(g, p), jnp.arange(400))[0])()
    new, books = jax.jit(lambda: P.plant_step(s, 0.46, 5.79, 0.25, 0.2, 400, g, p))()
    change = float(P.stored_energy(new, g, p) - P.stored_energy(s, g, p))
    flows = 30.0 * float(books["in"] - books["out"])
    assert change == pytest.approx(flows, abs=5e-3 * abs(30.0 * float(books["in"])))
