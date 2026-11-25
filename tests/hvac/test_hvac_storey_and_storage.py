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
  outdoors, the room at 21 C). Each step charges that price on the heat it draws
  from the store, so over any run of steps the charge is the price times the
  store at its start less the store at its end.
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


def test_the_stored_heat_charge_over_a_run_is_the_price_on_the_change_in_store():
    env = BuildingHVAC()
    _, s = env.reset_env(jax.random.PRNGKey(0), P)
    moves = jax.random.uniform(jax.random.PRNGKey(1), (24,), minval=-1.0, maxval=1.0)
    states, charged = [s], []
    for k, u in enumerate(moves):
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(10 + k), s, u[None], P)
        states.append(s)
        charged.append(float(env.reward_terms(s, P)["stored_heat"]))
    # Steps 6 to 18: the store at the start of step 6 less the store after 18.
    expected = _price() * (_store(states[6]) - _store(states[18]))
    assert abs(expected) > 1e-3  # the moves did move the store
    assert sum(charged[6:18]) == pytest.approx(expected, rel=2e-3)


def test_heat_put_into_the_store_is_credited():
    """Full supply into a cold slab stores heat, so its step's charge is
    negative."""
    env = BuildingHVAC()
    _, s = env.reset_env(jax.random.PRNGKey(0), P)
    s = s.replace(T_core=jnp.asarray(18.0), T_mass=jnp.asarray(19.0))
    _, s1, _, _, _ = env.step_env(jax.random.PRNGKey(5), s, jnp.ones(1), P)
    assert float(env.reward_terms(s1, P)["stored_heat"]) < 0.0
    assert np.isfinite(float(env.reward_terms(s1, P)["stored_heat"]))
