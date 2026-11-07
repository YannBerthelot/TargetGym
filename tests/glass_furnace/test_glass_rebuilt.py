"""The rebuilt furnace against its sources (PHYSICS.md, "Rebuilt plant").

Contract, from the proposal and the cited figures, not from the code:
- the melt's viscosity is DGG X's measured table; a cell's flow is Poles's solution;
- the settled orbit is settled, and it sits on the published operating figures:
  4 to 6 GJ/t, a crown cold face near 1200 C and under about 2 kW/m^2, and 6 to 8
  passes of the hot spot;
- a hidden move ramps and stays in its band, is invisible to the observation, and
  more cullet burns less fuel at the same crown (2.5 % per 10 points);
- the score is the hot face against the target, plus fuel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace import rebuilt as B


@pytest.fixture(scope="module")
def env():
    return B.RebuiltFurnace()


def quiet(**kw):
    return B.RebuiltParams(
        sensor_noise_scale=0.0, m_pull_noise_std=0.0, lhv_noise_std=0.0,
        charge_duty=1.0, charge_mass_jitter=0.0, cullet_move=0.0, thickness_move=0.0,
        **kw,
    )  # fmt: skip


def hold(env, p, raw, steps, state):
    key = jax.random.PRNGKey(0)

    def f(s, _):
        s = B.next_state(raw, s, p, key, "rk4_2")
        x = B.positions_of(s)
        r1 = B.cell_flow(x[B.HS] - x[B.BS], 0.5 * (x[B.BD] + x[B.HD]), p)
        return s, (s.T_crown, s.fuel_flow, r1, s.T_bath, s.T_work)

    return jax.lax.scan(f, state, None, length=steps)


@pytest.fixture(scope="module")
def orbit(env):
    p = quiet()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    raw, x = env.settled(jnp.asarray(1590.0))
    s = s.replace(
        T_crown=x[0], T_roof=x[: B.N_ROOF], T_bath=x[B.BS : B.WORK], T_work=x[B.WORK],
        T_rA=x[B.RA : B.RB], T_rB=x[B.RB :], m_batch=x[B.BATCH],
        fuel_pipeline=jnp.full((2,), 0.513 + 0.5 * (raw + 1) * (0.698 - 0.513)),
        lhv_deviation=jnp.zeros(()),
    )  # fmt: skip
    return p, raw, s, jax.jit(lambda: hold(env, p, raw, 2880, s))()


@pytest.mark.parametrize(
    "T, eta_dPas", [(1100.0, 3.13e3), (1200.0, 9.17e2), (1400.0, 1.46e2)]
)
def test_viscosity_reads_the_standard_glass_table(T, eta_dPas):
    # The fit's stated error is 0.96 K, about 1 % in viscosity here.
    assert float(B.glass_viscosity(jnp.asarray(T))) == pytest.approx(
        eta_dPas / 10.0, rel=0.03
    )


def test_cell_flow_is_poles_solution_and_turns_with_the_gradient():
    # dT 60.6 K over a 6.06 m zone is 10 K/m, h^4 = 3.3215, nu = 14.6 / 2319:
    # 0.00542 * 9.81 * 5.95e-5 * 10 * 3.3215 / 6.296e-3 = 0.01669 m^2/s per metre
    # of width, by 11 m at 2319 kg/m^3 is 425.8 kg/s.
    p = B.RebuiltParams(current_scale=1.0)
    dT = 10.0 * p.A_melt / p.tank_width / 3.0
    assert float(B.cell_flow(dT, 1400.0, p)) == pytest.approx(425.8, rel=0.03)
    assert float(B.cell_flow(-dT, 1400.0, p)) < 0.0


def test_brick_reads_between_its_cells():
    centres = np.cumsum(B.ROOF_CELLS) - 0.5 * np.array(B.ROOF_CELLS)
    T = jnp.asarray(1600.0 - 800.0 * centres)  # a straight profile, 800 K/m
    assert float(B.brick_temperature(T, 0.05)) == pytest.approx(1560.0, abs=0.01)


def test_a_move_ramps_and_stays_in_its_band():
    times = jnp.array([0.0, 1e9])
    sizes = jnp.array([0.04, 0.09])
    assert float(B.moved(1800.0, times, sizes, 3600.0, 0.05)) == pytest.approx(0.02)
    assert float(B.moved(3600.0, times, sizes, 3600.0, 0.05)) == pytest.approx(0.04)
    assert float(B.moved(2e9, times, sizes, 3600.0, 0.05)) == pytest.approx(0.05)


def test_reset_sits_on_its_orbit(orbit):
    _, _, _, (_, (crown, *_)) = orbit
    crown = np.asarray(crown)
    first, last = crown[:300].mean(), crown[-300:].mean()
    assert first == pytest.approx(1590.0, abs=1.0)
    assert last == pytest.approx(first, abs=0.5)


def test_orbit_sits_on_the_published_figures(orbit):
    p, _, _, (s, (_, fuel, r1, bath, work)) = orbit
    energy = float(np.mean(fuel)) * p.LHV / p.m_pull / 1e6  # GJ/t
    assert 4.0 <= energy <= 6.0
    passes = 1.0 + float(np.mean(r1)) / p.m_pull
    assert 6.0 <= passes <= 8.0  # Beerkens: the hot spot is passed 6 to 8 times
    bath = np.asarray(bath).mean(0)
    assert bath[0] < bath[2] and bath[1] < bath[0] + 1.0  # batch side colder
    assert float(np.mean(work)) < bath[2]  # the working end is the cold end
    cells = np.array(B.ROOF_CELLS)
    back = float(s.T_roof[-1])
    cold = p.T_ambient + (back - p.T_ambient) / (
        1 + p.U_roof_back * 0.5 * cells[-1] / p.k_brick
    )
    assert 1150.0 <= cold <= 1250.0  # W2000a: 1195 to 1210 C on a 1600 C face
    assert p.U_roof_back * (cold - p.T_ambient) <= 2300.0  # about 2 kW/m^2


def test_more_cullet_burns_less_fuel_at_the_same_crown(env):
    # 10 points of cullet: 2.5 % less fuel. Compare the crown at fuels 2.5 % apart.
    p = quiet()
    down = 0.025 * 0.59 / (0.5 * (p.fuel_max - p.fuel_min))
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    raw, _ = env.settled(jnp.asarray(1590.0))
    fat = s.replace(
        move_times=s.move_times.at[0, 0].set(-1e6),
        move_sizes=s.move_sizes.at[0, 0].set(0.05),
    )
    far = 2880 * 12
    _, a = jax.jit(lambda: hold(env, p.replace(cullet_band=0.10), raw, far, s))()
    fat = fat.replace(move_sizes=fat.move_sizes.at[0, 0].set(0.10))
    _, b = jax.jit(
        lambda: hold(env, p.replace(cullet_band=0.10), raw - down, far, fat)
    )()
    assert float(jnp.mean(b[0][-300:])) == pytest.approx(
        float(jnp.mean(a[0][-300:])), abs=2.0
    )


def test_moves_never_show_in_the_observation(env):
    p = B.RebuiltParams()
    obs, s = env.reset_env(jax.random.PRNGKey(3), p)
    assert obs.shape == (6,)
    moved_state = s.replace(move_sizes=s.move_sizes * 3.0)
    assert jnp.array_equal(
        env.get_obs(s, p.replace(sensor_noise_scale=0.0)),
        env.get_obs(moved_state, p.replace(sensor_noise_scale=0.0)),
    )


def test_score_is_the_hot_face_plus_fuel(env):
    p = B.RebuiltParams(sensor_noise_scale=0.0)
    _, s = env.reset_env(jax.random.PRNGKey(3), p)
    s = s.replace(T_crown=s.target_T_crown + 3.0, fuel_flow=jnp.asarray(p.c_hold))
    assert float(env.compute_reward(s, p)) == pytest.approx(-(9.0 + 1.0), abs=1e-3)


def test_a_crown_past_its_limit_trips(env):
    p = B.RebuiltParams()
    _, s = env.reset_env(jax.random.PRNGKey(3), p)
    assert not bool(B.is_tripped(s, p))
    assert bool(B.is_tripped(s.replace(T_crown=p.T_crown_max + 1.0), p))
