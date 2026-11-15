"""What a temperature move costs the glass, written before the code.

Contract, from REALITY.md section 10 and the owner's go of 2026-11-15, not from any code:
- the tank blocks give off defects when their temperature rises, and only then [TR4; S7].
  A block follows the glass it touches over ``block_lag``. A block at or above
  ``block_release_T`` that rises by dT adds ``block_release * dT * contact`` defects per
  plate to its cell's glass, ``contact`` being the wall and floor area the cell touches
  per kg of its glass. A block that cools, holds, or sits under that temperature adds
  nothing;
- reboil [S45; TR3]: past the last port, the fining that still runs makes new bubbles.
  Glass doing a share ds of its fining there gains ``reboil_plate * ds`` defects per
  plate. In the fired zone the same reaction is fining and adds nothing;
- defects are carried with the glass as the fining share is, batch glass arriving with
  none, and nothing clears them: what the blocks made is in the tank or has left;
- the cold end rejects plates by their defects: with d expected defects per plate in the
  glass leaving, a share 1 - exp(-d) of the pull is lost at the lost-glass price. Glass
  under a mark is lost whole, and never more than the whole pull;
- the packaged design state is settled: its blocks are at their glass's temperature and
  its glass carries no defects, so a furnace held there loses next to nothing to them.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import physical as P

ENV = E.GlassFurnace()


@pytest.fixture(scope="module")
def plant():
    g, p = P.Grid(), P.PhysicalParams()
    return g, p, P.initial_fields(g, p)


def _contact(g, p):
    """Block area a cell touches per kg of its glass, m^2/kg, by hand: a cell dz thick,
    dx long and w wide holds rho dz dx w kg and touches 2 dz dx of side wall, so
    2 / (rho w); the bottom cell also touches dx w of floor, so 1 / (rho dz) more."""
    a = P.layout(g)
    dz, w = np.asarray(a.dz, float), np.asarray(a.wc, float)
    contact = np.tile(2.0 / (p.rho_glass * w), (g.n_depth, 1))
    contact[-1] += 1.0 / (p.rho_glass * dz[-1])
    return contact


def _cell_mass(g, p):
    a = P.layout(g)
    dz, dx, w = (np.asarray(v, float) for v in (a.dz, a.dx, a.wc))
    return p.rho_glass * dz[:, None] * dx * w


def test_a_hot_block_that_rises_sheds_by_its_rise_and_a_cold_one_does_not(plant):
    g, p, s = plant
    p = p.replace(block_release=3.0, reboil_plate=0.0)  # the blocks' defects alone
    start = s.replace(block=s.T - 60.0)  # every block far colder than its glass
    new, _ = P.plant_step(start, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    follow = 1.0 - np.exp(-p.delta_t / p.block_lag)
    before = np.asarray(start.block, float)
    block = before + (np.asarray(new.T, float) - before) * follow
    # float32 near 1800 K resolves 1e-4 K, against rises of about 1 K
    np.testing.assert_allclose(new.block, block, rtol=1e-6)
    rise = np.maximum(block - before, 0.0)
    hot = block >= p.block_release_T
    assert (rise[hot] > 0.0).any() and (rise[~hot] > 0.0).any()
    expected = np.where(hot, 3.0 * _contact(g, p) * rise, 0.0)
    np.testing.assert_allclose(new.defects, expected, rtol=5e-3, atol=1e-9)


def test_a_block_that_cools_or_holds_sheds_nothing(plant):
    g, p, s = plant
    p = p.replace(block_release=3.0, reboil_plate=0.0)
    # far hotter than any change the unsettled start makes in one step
    cooling = s.replace(block=s.T + 500.0)
    new, _ = P.plant_step(cooling, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    assert float(jnp.max(new.block - cooling.block)) < 0.0
    assert float(jnp.max(jnp.abs(new.defects))) == 0.0
    still = p.replace(block_lag=1.0e30)  # blocks that never move
    new, _ = P.plant_step(
        s.replace(block=s.T - 60.0), 0.46, 5.79, 0.25, 0.2, 0.0, g, still
    )
    assert float(jnp.max(jnp.abs(new.defects))) == 0.0


def test_fining_past_the_last_port_is_reboil_and_in_the_fired_zone_is_not(plant):
    g, p, s = plant
    a = P.layout(g)
    p = p.replace(reboil_plate=7.0, block_release=0.0)  # reboil alone
    # A uniform share stays uniform where no batch glass arrives, so past the ports
    # the share fined in the step is the new share less 0.3.
    start = s.replace(fining=jnp.full_like(s.fining, 0.3))
    new, _ = P.plant_step(start, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    done = np.asarray(new.fining, float)[:, a.nm :] - 0.3
    assert done.max() > 1e-3  # some glass past the ports is inside the band
    np.testing.assert_allclose(new.defects[:, a.nm :], 7.0 * done, rtol=1e-3, atol=1e-5)
    assert float(jnp.max(jnp.abs(new.defects[:, : a.nm]))) == 0.0


def test_defects_read_the_leaving_glass_by_its_flow(plant):
    g, p, s = plant
    a = P.layout(g)
    profile = jnp.linspace(0.4, 0.0, g.n_depth)
    s = s.replace(defects=s.defects.at[:, -1].set(profile))
    weigh = lambda x: float(jnp.sum(x * a.dz) / jnp.sum(a.dz))  # noqa: E731
    assert float(P.sensors(s, g, p)["defects"]) == pytest.approx(
        weigh(profile), rel=1e-6
    )


def test_what_the_blocks_made_is_in_the_tank_or_has_left(plant):
    g, p, s = plant
    a = P.layout(g)
    p = p.replace(block_release=3.0, reboil_plate=0.0)
    mass, contact = jnp.asarray(_cell_mass(g, p)), jnp.asarray(_contact(g, p))
    start = s.replace(block=s.T - 60.0)

    def f(s, _):
        new, b = P.plant_step(s, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
        rise = jnp.maximum(new.block - s.block, 0.0) * (new.block >= p.block_release_T)
        made = jnp.sum(3.0 * contact * rise * mass)
        # the glass leaves evenly over depth, as much as melted
        leaving = jnp.sum(b["melt"] * a.dx[: a.nm] * a.width) * p.delta_t
        left = leaving * jnp.sum(s.defects[:, -1] * a.dz) / jnp.sum(a.dz)
        return new, (made, left)

    end, (made, left) = jax.jit(lambda: jax.lax.scan(f, start, None, length=720))()
    held = float(jnp.sum(end.defects * mass))
    made, left = float(jnp.sum(made)), float(jnp.sum(left))
    assert made > 0.0 and left > 0.0
    # float32 sums over a day of steps
    assert held + left == pytest.approx(made, rel=5e-3)


def test_the_cold_end_rejects_plates_by_their_defects():
    p = ENV.default_params
    dt = p.delta_t
    pull = 1.1 * p.pull
    worth = pull * dt / 1000.0 * p.lost_glass_price
    good = dict(
        canal=p.canal_target + 0.5, fining=p.fining_pass + 0.01, pull=pull, params=p
    )
    assert float(E.costs(0.45, **good, defects=0.0)["tracking"]) == 0.0
    # one plate in two carries a defect at ln 2 defects per plate
    half = E.costs(0.45, **good, defects=np.log(2.0))
    assert float(half["tracking"]) == pytest.approx(0.5 * worth, rel=1e-6)
    few = E.costs(0.45, **good, defects=0.01)
    assert float(few["tracking"]) == pytest.approx(
        (1.0 - np.exp(-0.01)) * worth, rel=1e-5
    )
    # never more than the whole pull, and a lost mark loses it whole
    assert float(E.costs(0.45, **good, defects=50.0)["tracking"]) == pytest.approx(
        worth, rel=1e-6
    )
    under = {**good, "fining": p.fining_pass - 0.01}
    assert float(E.costs(0.45, **under, defects=0.01)["tracking"]) == pytest.approx(
        worth, rel=1e-6
    )
    # defects charge nothing on the fuel
    assert float(half["running"]) == float(
        E.costs(0.45, **good, defects=0.0)["running"]
    )


def _quiet():
    return ENV.default_params.replace(
        walk_sd=(0.0,) * len(E.WALKS), sensor_noise_scale=0.0
    )


def _day_at(valve, p, steps=720):
    _, state = ENV.reset_env(jax.random.PRNGKey(0), p)

    def f(st, k):
        _, st, r, _, _ = ENV.step_env(k, st, E.to_action(jnp.asarray(valve), p), p)
        terms = E.compute_reward_terms(st, p)
        return st, (P.sensors(st.plant, p.grid, p.plant)["defects"], terms["tracking"])

    keys = jax.random.split(jax.random.PRNGKey(1), steps)
    return jax.jit(lambda: jax.lax.scan(f, state, keys)[1])()


def test_the_design_state_is_settled_for_its_blocks_and_carries_no_defects():
    _, _, fields = E.design_state()
    np.testing.assert_array_equal(fields["block"], fields["T"])
    assert float(np.abs(fields["defects"]).max()) == 0.0
    p = _quiet()
    fuel, _, _ = E.design_state()
    _, lost = _day_at(fuel, p)
    worth = p.pull * p.delta_t / 1000.0 * p.lost_glass_price
    assert float(np.asarray(lost).mean()) < 1e-3 * worth  # under 0.1 % of the glass


def test_firing_harder_sheds_defects_and_firing_less_does_not():
    p = _quiet()
    fuel, _, _ = E.design_state()
    up, _ = _day_at(1.03 * fuel, p)
    down, _ = _day_at(0.97 * fuel, p)
    assert float(np.asarray(up)[-1]) > 0.0
    assert float(np.asarray(down).max()) < 0.1 * float(np.asarray(up).max())
