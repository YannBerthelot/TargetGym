"""The reference through the fuel cycle, and its rule for the turbine (owner,
2026-11-15).

Contract, written before the runs. The reference is the Westinghouse procedure of
PHYSICS.md section 6, run from the operator's curve book at the day's cycle point:

- The curve book is the plant's own (``env.core_at``), read at observation 13.
- Feedforward from the balances: the program asks ``ppm_per_load`` of boron per unit
  of load, the power defect on the program over the boron's worth. By hand, the fuel
  runs 243.3 K hotter at full load than at half (7.7 K of program, 8.33 K of core
  rise, 227.27 K of fuel to coolant) and Tavg 7.7 K hotter: 3 x 243.3 + 12 x 7.7 =
  822.3 pcm per 50 % at the start of the cycle, 205.6 ppm per unit of load at
  8 pcm/ppm; at 80 % of the cycle 3 x 243.3 + 50.4 x 7.7 = 1118.0 pcm per 50 %,
  216.7 ppm per unit of load at 10.32 pcm/ppm.
- A ppm of boron moves Tavg by ``-worth / (alpha_coolant + alpha_fuel)``: -8 / 15 K
  at the start, -10.32 / 53.4 K at 80 %.
- The zone's back-off (charter J1, k = 2): the operator's tool and the turbine act
  two standard deviations of Tavg inside the edge of its 1.6 K free band. Under the
  load's wobble, 0.5 % of rated (4.08e-4 over sqrt(2 / 300 s)), with rods and boron
  still, a unit of load moves Tavg's error by ``3 x 471.2 / (3 + MTC) + 15.4`` K
  (the fuel's 471.2 K per unit of load at a fixed Tavg, given back through both
  feedbacks, and the program's 15.4 K span): 109.6 K at the start, 41.9 K at 80 %,
  so 0.548 K and 0.209 K of spread, and bands of 0.504 K and 1.182 K.
- A family member's operator has that member's curve book (charter B7's sweep
  runs the expert on drawn members): the expert is built for a plant, the shipped
  member by default. By hand on the first step of a day lined out at full load at
  the start of the cycle, with the plan's ramp down (2 %/min) under way a boron
  lead ahead: the plan asks 205.6 ppm per unit of load, 2.056 ppm for the step's
  1 % of load, against the 2.200 ppm a step of full boration brings at 1690 ppm
  (3 kg/s of 7800 ppm acid into 250 t), so the flow asked is 0.935 of full
  boration. On a member whose boron is worth 10 pcm/ppm, not 8, the same power
  defect asks 164.5 ppm per unit of load, and the flow is 0.748.
- The turbine loads at the rate boron carries: full dilution at the boron read,
  plus xenon's rate, over ``ppm_per_load``. At 80 % of the cycle with 450 ppm read
  and xenon in balance: 0.018 ppm/s over 216.7 ppm = 8.31e-5 of rated load a
  second, 0.50 %/min, a tenth of the turbine's 5 %/min. At the start with 1690 ppm:
  0.0676 over 205.6 = 3.29e-4, 1.97 %/min (OAPS's 1 to 2).
- Tavg's free band is reactivity too. The turbine loads at the carried rate with
  Tavg at the cold side of the backed-off band, and faster the warmer Tavg is: a
  kelvin is ``1 / (tavg_per_ppm x ppm_per_load)`` of rated load, taken over the
  boron line's lead. It tapers to a stop as Tavg cools from that cold side to the
  free band's cold edge, and to the turbine's full rate as it warms from the hot
  side to the hot edge.
- Through the cycle, on every kind of day (medium tier): no trip, the offset's
  penalty time inside its hour, the rods above their insertion limit.
- Where boron carries the plan (a start-of-cycle Krsko day at 8 %/h, any baseload
  day), the turbine stays on it: no output is lost.
- Late in the cycle on an EDF day the turbine waits for the boron: by the balance
  108 ppm at about 1 ppm/min, so it is back on its plan between 60 and 150 min
  after the plan's return starts, with Tavg inside its free band meanwhile but for
  a twentieth of the day at most (the back-off's two sigma).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import core_at, rod_insertion_limit
from target_gym.reactor.experts import (
    ReferenceParams,
    carried_rate,
    curve_book,
    make_reactor_pid_controller,
    ppm_per_load,
    tavg_dead_band,
    tavg_per_ppm,
    tavg_spread,
    turbine_demand,
)
from target_gym.registry import REGISTRY

BOOK = ReactorParams()


# --- the curve book and what is read off it ---


@pytest.mark.parametrize("cycle", [0.0, 0.37, 0.8, 1.0])
def test_the_curve_book_is_the_plants(cycle):
    read, plant = curve_book(cycle, BOOK, np), core_at(cycle, BOOK)
    for name in ("boron_ref", "alpha_coolant", "boron_worth"):
        assert float(getattr(read, name)) == pytest.approx(
            float(getattr(plant, name)), rel=1e-6, abs=1e-9
        )


@pytest.mark.parametrize("cycle, ppm", [(0.0, -205.6), (0.8, -216.7)])
def test_the_program_asks_boron_by_the_power_defect(cycle, ppm):
    assert float(ppm_per_load(curve_book(cycle, BOOK, np))) == pytest.approx(
        ppm, rel=2e-3
    )


@pytest.mark.parametrize("cycle, k", [(0.0, -8.0 / 15.0), (0.8, -10.32 / 53.4)])
def test_a_ppm_moves_tavg_by_both_feedbacks(cycle, k):
    assert float(tavg_per_ppm(curve_book(cycle, BOOK, np))) == pytest.approx(
        k, rel=1e-5
    )


@pytest.mark.parametrize(
    "cycle, spread, band", [(0.0, 0.548, 0.504), (0.8, 0.209, 1.182)]
)
def test_the_back_off_is_two_spreads_of_tavg_under_the_wobble(cycle, spread, band):
    plant = curve_book(cycle, BOOK, np)
    assert float(tavg_spread(plant, np)) == pytest.approx(spread, abs=0.002)
    assert float(tavg_dead_band(ReferenceParams(), plant, np)) == pytest.approx(
        band, abs=0.004
    )


# --- the turbine's rule ---


@pytest.mark.parametrize(
    "cycle, boron, xenon, per_min",
    [
        (0.0, 1690.0, 0.0, 1.97),  # OAPS: 1 to 2 %/min at the start of a cycle
        (0.8, 450.0, 0.0, 0.498),
        (0.8, 450.0, -0.002, 0.443),  # xenon building takes a ninth of the line
        (0.8, 450.0, 0.004, 0.609),  # its burnout adds to it
        (0.8, 450.0, -0.03, 0.0),  # never backwards
    ],
)
def test_the_turbine_loads_at_the_rate_boron_carries(cycle, boron, xenon, per_min):
    rate = carried_rate(boron, xenon, curve_book(cycle, BOOK, np), np)
    assert float(rate) * 6000.0 == pytest.approx(per_min, rel=5e-3, abs=1e-6)


@pytest.mark.parametrize(
    "error, action",
    [
        (1.18, 0.1),  # Tavg at the cold side of the backed-off band: the carried rate
        (0.0, 0.156),  # 1.18 K warmer: 4.70e-5 of rated load a second more
        (1.39, 0.05),  # halfway to the cold edge
        (1.6, 0.0),  # at the cold edge: stop
        (2.5, 0.0),
        (-1.18, 0.213),  # 2.36 K warmer than the cold side
        (-1.39, 0.611),  # halfway to the hot edge: 0.223 + 0.777 / 2
        (-1.6, 1.0),  # at the hot edge: the turbine's own rate
    ],
)
def test_the_turbine_spends_tavgs_band_and_stops_at_its_cold_edge(error, action):
    """At 80 % of the cycle the backed-off band is 1.18 K, and a kelvin of Tavg is
    1 / (0.19326 K/ppm x 216.66 ppm) = 2.388 % of rated load, taken over the
    line's 600 s lead: 3.981e-5 of rated load a second per kelvin warmer than the
    band's cold side. The carried rate here is a tenth of the turbine's 8.333e-4,
    so a kelvin adds 0.0478 to the action."""
    plant = curve_book(0.8, BOOK, np)
    out = turbine_demand(error, 8.333e-5, 1.18, 600.0, plant, np)
    assert float(out) == pytest.approx(action, abs=0.005)


@pytest.mark.parametrize("worth, flow", [(None, 0.935), (-10.0e-5, 0.748)])
def test_the_expert_sends_the_boron_its_own_members_book_asks(worth, flow):
    member = None if worth is None else ReactorParams(boron_worth=worth)
    expert = make_reactor_pid_controller(plant=member)
    obs = np.array(
        [1.0, 596.87, 1.0 - 2.0 * 3.0 / 228.0, 1.0, 580.2, 580.2, 0.0, 1690.0,
         0.1, 5.0, 1.0, 2.0, 0.5, 0.0, 1.0, 0.0],
        np.float32,
    )  # fmt: skip
    action, _ = expert.step(expert.params, expert.init_state, obs, np)
    assert float(action[1]) == pytest.approx(flow, abs=0.005)


# --- days through the cycle ---

ENV, EXPERT, SPEC = Reactor(), make_reactor_pid_controller(), REGISTRY["reactor"]
STEPS = 2880


@jax.jit
def _day(p, seed):
    obs, s = ENV.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), STEPS)

    def step(c, k):
        s, obs, m = c
        a, m = EXPERT.step(EXPERT.params, m, obs)
        obs, s, _, _, info = ENV.step_env(k, s, a, p)
        out = dict(
            tripped=info["tripped"],
            error=s.T_ref - 0.5 * (s.T_coolant + s.T_cold),
            load=obs[3],
            plan=obs[14],
            penalty=s.ao_outside,
            past=s.rod_in - rod_insertion_limit(s.n, p),
            flow=a[1],
            # The plan the turbine does not take, of rated load: the plant's own
            # count (``pickup`` is the turbine's load while it is under its plan).
            short=jnp.maximum(obs[14] - s.pickup, 0.0),
        )
        return (s, obs, m), out

    return jax.lax.scan(step, (s, obs, EXPERT.init_state), keys)[1]


def day(scenario, seed, cycle):
    """One scored day (sensor noise on) of ``scenario`` at ``cycle``, under the
    reference: its params and what each step showed."""
    (sc,) = [s for s in scenarios.suite(SPEC) if s.name == scenario]
    p = scenarios.params_for(SPEC, sc, cycle_range=(cycle, cycle))
    return p, {k: np.asarray(v) for k, v in _day(p, jnp.int32(seed)).items()}


DAYS = ["edf_daily", "edf_short_hold", "krsko_slow", "krsko_shallow", "fast_return"]


@pytest.mark.medium
@pytest.mark.parametrize("cycle", [0.0, 0.45, 0.9])
@pytest.mark.parametrize("scenario", [*DAYS, "baseload"])
def test_the_reference_runs_every_day_through_the_cycle(scenario, cycle):
    for seed in (100, 104):
        p, out = day(scenario, seed, cycle)
        assert not out["tripped"].any(), (seed, int(np.argmax(out["tripped"])))
        assert float(out["penalty"][-1]) < p.ao_allowed
        assert float(out["past"].max()) <= 1e-6


@pytest.mark.medium
@pytest.mark.parametrize(
    "scenario, cycle", [("krsko_slow", 0.0), ("baseload", 0.0), ("baseload", 0.9)]
)
def test_where_boron_carries_the_plan_the_turbine_stays_on_it(scenario, cycle):
    """Krsko's 8 %/h is a fifteenth of what boron carries at the start of the
    cycle, and a baseload plan does not move. Lost output is $666.67 a step per
    unit of rated load: under 0.05 of a load-step over the day, $33, is none to
    speak of."""
    for seed in (100, 104):
        _, out = day(scenario, seed, cycle)
        assert float(out["short"].sum()) < 0.05


@pytest.mark.medium
@pytest.mark.parametrize("seed", [100, 104])
def test_late_in_the_cycle_the_turbine_waits_for_the_boron(seed):
    p, out = day("edf_daily", seed, 0.8)
    plan = out["plan"]
    returning = int(np.argmax(np.diff(plan) > 0.0))  # the plan's return starts
    behind = out["short"]
    lagging = np.flatnonzero(behind[returning:] > 0.02)
    assert lagging.size > 0  # it does fall behind its plan
    back = (lagging[-1] + 1) * 30.0 / 60.0  # min after the return starts
    assert 60.0 <= back <= 150.0
    # While it lags by more than a tenth of rated load the line dilutes at full flow.
    far = returning + np.flatnonzero(behind[returning:] > 0.1)
    assert np.all(out["flow"][far] == -1.0)
    # Tavg stays inside its free band but for a twentieth of the day.
    assert float(np.mean(np.abs(out["error"]) > p.e_tol)) < 0.05


@pytest.mark.medium
@pytest.mark.parametrize("cycle", [0.0, 0.9])
def test_the_reference_holds_tavg_in_its_band_on_a_baseload_day(cycle):
    for seed in (100, 104):
        p, out = day("baseload", seed, cycle)
        assert float(np.mean(np.abs(out["error"]) > p.e_tol)) < 0.05
