"""What hvac's evaluation charges for the store its scored window leaves (owner,
2026-11-27, the RL research review's H2, decision "No credit, charge a shortfall").

Contract:

- A store left fuller at the end of the scored window (days 4 to 10) than it began
  earns nothing. One left emptier is charged its shortfall, once, at the dearest
  price heat can cost the plant. Ending at the start level is neutral, so neither
  filling past the start level nor draining below it at the edge can pay.
- The dearest price heat can cost is the highest tariff over the lowest COP the
  heat pump reaches in its supply range. The COP, 0.52 (T + 273.15) / (T - 0) in
  kelvin over the lift from the brine at 0 C, falls as the supply T rises, so its
  lowest is at the valve's top, 49 C: 0.52 x 322.15 / 49 = 3.41874. The punta
  tariff, EUR 0.263/kWh, over it: EUR 0.076929 per kWh of heat, 2.13690e-8 per J
  (PHYSICS.md, "What is scored"). No heat the plant buys costs more.
- The review's experiment: the expert, overridden to full supply in one night of
  the window, gains no more on the window's last scored day than on a middle day;
  the expert with the supply shut on the last scored day pays no less than the
  expert. Both run the shipped expert on hvac's own evaluation (medium tests).
- The reward carries no edge charge (``test_hvac_storey_and_storage``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import evaluation, scenarios
from target_gym.eval import Episode
from target_gym.hvac.env import HVACParams, electricity_price, heat_pump_cop
from target_gym.hvac.env_jax import BuildingHVAC
from target_gym.registry import REGISTRY

SPEC = REGISTRY["hvac"]
P = HVACParams()
ENV = BuildingHVAC()
DAY = 96  # steps of 15 min
#: The scored window, days 4 to 10 of the twelve (``scenarios.hvac_evaluation``).
FIRST, END, STEPS = 3 * DAY, 10 * DAY, 12 * DAY

#: The derivation above, by hand.
COP_LOWEST = 0.52 * (49.0 + 273.15) / 49.0
DEAREST = 0.263 / COP_LOWEST / 3.6e6  # EUR per J of heat
C_SLAB = 2400.0 * 1000.0 * 0.25 * 150.0  # J/K, the slab's capacity
#: float32 store values near EUR 65 carry about 8e-6 EUR each, so a difference of
#: two is good to 2e-5 EUR: 3e-6 of the 4 K shortfall's EUR 7.7 below.
STORE_RTOL = 1e-5


def _value(T_core):
    """The env's value of the store with the slab core at ``T_core``, EUR."""
    _, s = ENV.reset_env(jax.random.PRNGKey(0), P)
    return float(ENV.stored_value(s.replace(T_core=jnp.float32(T_core)), P))


def _episode(before, after, cost=0.01):
    """Twelve days at ``cost`` a step, the store worth ``before`` after the step
    before the window and ``after`` after its last step, and more in between."""
    store = np.full(STEPS, 2.0 * max(before, after))
    store[FIRST - 1], store[END - 1] = before, after
    return Episode(
        cost=np.full(STEPS, cost),
        in_band=np.ones(STEPS, bool),
        target_change=np.zeros(STEPS, bool),
        terms={"stored_value": store},
    )


def test_ending_the_window_at_its_start_level_is_neutral():
    v = _value(26.0)
    assert evaluation.episode_cost(SPEC, _episode(v, v)) == pytest.approx(0.01)


def test_a_store_left_fuller_than_it_began_earns_nothing():
    """So a window that cost nothing scores 0, not a credit: a credit made one
    shoulder-season seed's cost negative, which the headline refuses (H2)."""
    ep = _episode(_value(26.0), _value(30.0), cost=0.0)
    assert evaluation.episode_cost(SPEC, ep) == 0.0


def test_a_store_left_emptier_pays_its_shortfall_at_the_dearest_heat_price():
    ep = _episode(_value(30.0), _value(26.0))
    shortfall = 4.0 * C_SLAB * DEAREST  # EUR 7.69, once over the window's steps
    assert evaluation.episode_cost(SPEC, ep) == pytest.approx(
        0.01 + shortfall / (END - FIRST), rel=STORE_RTOL
    )


def test_no_heat_the_plant_buys_costs_more_than_the_shortfall_price():
    # The store's price, read off the env's value of 20 K of the core.
    charged = (_value(40.0) - _value(20.0)) / (20.0 * C_SLAB)
    assert charged == pytest.approx(DEAREST, rel=STORE_RTOL)
    tariffs = np.unique(np.asarray(electricity_price(jnp.arange(DAY), P)))
    supply = jnp.linspace(P.T_supply_min, P.T_supply_max, 69)
    cop = np.asarray(heat_pump_cop(supply, P), float)
    per_J = tariffs[:, None] / cop[None, :] / 3.6e6
    assert per_J.max() <= charged * (1.0 + STORE_RTOL)
    assert per_J.max() == pytest.approx(DEAREST, rel=1e-6)


# --- The review's experiment, on hvac's own evaluation (medium) ---------------
#
# The shipped expert and the same heating curve overridden on one day of the
# episode (1-based), as the review ran them: full supply in the night tariff, and
# the supply shut. Each line is J, the scenarios' costs at their weights, on the
# evaluation's first five seeds with its edge charge. On this protocol today's
# credit pays the last night's charge (+0.96 % against -0.74 % on day 7) and the
# supply shut after 08:00 on day 10 (+4.8 %); under the shortfall rule they lose
# 7.6 % and 11.8 %. Inside the cold snap alone a 04:00 to 08:00 charge on day 10
# still saves 0.6 % to day 7's 0.4 %: there the expert ends the window short on
# half its seeds, and heat bought at night costs less than the shortfall's price,
# so topping the store back up at the edge pays against being charged for it.

#: The seeds the evaluation scores first (``scenarios.hvac_evaluation``).
SEEDS = list(range(100, 105))
#: (name, day, hours, raw supply) of each line; the expert has no day.
LINES = (
    ("expert", None, (0.0, 0.0), 0.0),
    ("charge day 7 00-08", 7, (0.0, 8.0), 1.0),
    ("charge day 10 00-08", 10, (0.0, 8.0), 1.0),
    ("charge day 7 04-08", 7, (4.0, 8.0), 1.0),
    ("charge day 10 04-08", 10, (4.0, 8.0), 1.0),
    ("shut day 10 00-24", 10, (0.0, 24.0), -1.0),
    ("shut day 10 08-24", 10, (8.0, 24.0), -1.0),
    ("shut day 10 16-24", 10, (16.0, 24.0), -1.0),
)


class _Lines:
    """The heating curve on every episode, overridden in episode ``i`` as line
    ``i // len(SEEDS)`` of ``LINES`` says."""

    name = "curve"

    def __init__(self):
        from target_gym.hvac.experts import make_hvac_curve

        self.curve = make_hvac_curve()
        rows = np.repeat(np.arange(len(LINES)), len(SEEDS))
        self.day = np.array([-1 if LINES[i][1] is None else LINES[i][1] for i in rows])
        self.lo = np.array([LINES[i][2][0] for i in rows])
        self.hi = np.array([LINES[i][2][1] for i in rows])
        self.u = np.array([LINES[i][3] for i in rows])

    def reset(self, key, obs, task):
        return None

    def act(self, carry, obs, t):
        from target_gym.hvac.experts import CurveState, curve_step

        a, _ = curve_step(self.curve, CurveState(), np.asarray(obs, np.float64), np)
        hour = (t % DAY) / 4.0
        on = (t // DAY + 1 == self.day) & (self.lo <= hour) & (hour < self.hi)
        return np.where(on[:, None], self.u[:, None], a).astype(np.float32), carry


@pytest.fixture(scope="module")
def lines() -> dict:
    """Each line's J over hvac's scenarios, as the evaluation reads it."""
    from target_gym.benchmark import episodes_of, run_policy, task_info

    weights = scenarios.weights(SPEC)
    J = np.zeros(len(LINES))
    for s in scenarios.suite(SPEC):
        p = scenarios.params_for(SPEC, s)
        task = task_info("hvac", ENV, p, len(LINES) * len(SEEDS))
        run = run_policy(_Lines(), ENV, p, task, seeds=SEEDS * len(LINES))
        costs = [evaluation.episode_cost(SPEC, ep) for ep in episodes_of(run)]
        J += weights[s.name] * np.reshape(costs, (len(LINES), len(SEEDS))).mean(1)
    return {name: float(j) for (name, *_), j in zip(LINES, J)}


@pytest.mark.medium
@pytest.mark.parametrize("hours", ["00-08", "04-08"])
def test_a_night_charge_gains_no_more_on_the_last_scored_day_than_on_day_7(
    lines, hours
):
    assert lines[f"charge day 10 {hours}"] >= lines[f"charge day 7 {hours}"]


@pytest.mark.medium
@pytest.mark.parametrize("hours", ["00-24", "08-24", "16-24"])
def test_shutting_the_supply_on_the_last_scored_day_never_pays(lines, hours):
    assert lines[f"shut day 10 {hours}"] >= lines["expert"]
