"""The storey the zone is, and the heat it stores (owner, 2026-11-25, the hvac
review's findings 1 and 5).

Contract:

- The zone is one storey of a stack of like storeys, so both faces of its TABS
  slab heat it and it has no roof: its envelope is the walls, the glazing and
  the ventilation, ``0.28 x 120 + 1.30 x 25 + 0.34 x 0.5 x 150 x 2.6`` = 33.6 +
  32.5 + 66.3 = 132.4 W/K.
- The slab is modelled on its own, so the 5R1C mass is what is left beside it:
  ISO 13790's "light" class, 110 kJ/(m2 K) and A_m = 2.5 A_floor, not the
  "medium" class (165), whose figure counts the floors.
- Stored heat ``E = C_m T_mass + C_slab T_core``, with C_m = 110 000 x 150 and
  C_slab = 2400 x 1000 x 0.25 x 150 J/K, is priced at the cheapest heat the
  plant can store: the valle tariff over the COP at the design supply (-10 C
  outdoors, the room at 21 C). Every run records the store's value at that price
  after each step, as ``stored_value``, and hvac's evaluation names it as its
  edge charge: the store lost across the scored window is charged once, out of
  the reward (owner, 2026-11-25, P1 as amended), so the reward stays minus a sum
  of non-negative terms.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.hvac.env import (
    HVACParams,
    steady_state,
    total_heat_loss_coefficient,
    zone_conductances,
)
from target_gym.hvac.env_jax import BuildingHVAC

P = HVACParams()


def test_a_middle_storey_loses_heat_through_its_walls_glazing_and_air_alone():
    assert float(total_heat_loss_coefficient(P)) == pytest.approx(132.4, abs=0.05)


def test_the_mass_beside_the_slab_is_the_light_class():
    c = zone_conductances(P)
    assert float(c["C_m"]) == pytest.approx(110_000.0 * 150.0)
    assert float(c["A_m"]) == pytest.approx(2.5 * 150.0)


def _store(s):
    return 110_000.0 * 150.0 * float(s.T_mass) + 2400.0 * 1000.0 * 0.25 * 150.0 * float(
        s.T_core
    )


def _price():
    # The design supply is the steady state's; the COP by hand from its law,
    # 0.52 of Carnot's between the supply and the brine.
    T_d = float(steady_state(P, -10.0, 21.0)[3])
    cop = P.cop_carnot_fraction * (T_d + 273.15) / (T_d - P.T_brine)
    return P.price_valle / cop / 3.6e6  # EUR per J


def test_a_run_records_the_store_at_the_price_of_the_cheapest_stored_heat():
    from target_gym.benchmark import run_policy, task_info

    class Moves:
        name = "moves"

        def reset(self, key, obs, task):
            return 0

        def act(self, carry, obs, t):
            u = jnp.asarray([[0.8 if t % 6 < 3 else -0.6]])
            return u, carry + 1

    env = BuildingHVAC()
    task = task_info("hvac", env, P, 1)
    nodes = {"T_mass": lambda s, p, t: s.T_mass, "T_core": lambda s, p, t: s.T_core}
    out = run_policy(Moves(), env, P, task, seeds=[0], n_steps=24, record=nodes)
    rec = out["records"]
    store = (
        110_000.0 * 150.0 * rec["T_mass"]
        + 2400.0 * 1000.0 * 0.25 * 150.0 * rec["T_core"]
    )
    assert np.ptp(store) > 1e5  # the moves did move the store
    # float32 states near 2.3e9 J at about 6e-9 EUR/J: 1e-5 EUR covers rounding.
    np.testing.assert_allclose(rec["stored_value"], _price() * store, rtol=0, atol=1e-5)


def test_the_reward_carries_no_stored_heat_and_the_evaluation_charges_its_edges():
    from target_gym import evaluation
    from target_gym.registry import REGISTRY

    assert evaluation.of(REGISTRY["hvac"]).edge_charge == "stored_value"
    env = BuildingHVAC()
    _, s = env.reset_env(jax.random.PRNGKey(0), P)
    s = s.replace(T_core=jnp.asarray(18.0), T_mass=jnp.asarray(19.0))
    _, s1, r, _, _ = env.step_env(jax.random.PRNGKey(5), s, jnp.ones(1), P)
    terms = env.reward_terms(s1, P)
    # Full supply into a cold slab stores heat; no term credits it.
    assert all(float(v) >= 0.0 for v in terms.values())
    assert float(r) == pytest.approx(-sum(float(v) for v in terms.values()), rel=1e-6)
