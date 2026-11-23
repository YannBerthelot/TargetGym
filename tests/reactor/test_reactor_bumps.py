"""The reference recipe's boron bump on Tavg reads the plant, not the day's drift.

Contract (owner, 2026-11-10, after the family sweep): a bump test reads the response
to its step, so the day's own drift (xenon, burnup, a starting tilt) is taken out by
a paired run without the step, as a practitioner steps from a lined-out plant or
subtracts the drift.

Hand value, from the plant's balance (PHYSICS.md): with the rods held and the turbine
on its load, boron changes are met by Tavg, the fuel following the coolant at a fixed
power, so Tavg moves ``-boron_worth / (alpha_coolant + alpha_fuel)`` K per ppm.
Dilution at a fraction ``a`` of full flow takes ppm out at ``a * dilution_flow *
boron / primary_mass`` per second. The slope per unit of action is the product:

    slope / a = -boron_worth / (alpha_coolant + alpha_fuel)
                * dilution_flow * boron / primary_mass

The step itself is sized from the curve book so the plant stays inside its limits
for the test (owner, 2026-11-10): a bump test that trips the plant reads the restart,
not the response.

Tolerance 20 %: the hour's last 40 samples read a ramp still settling from the line's
delay and the steam generator's lag.

The expert's two bump-test numbers are the recipe's on the shipped member (owner's
rule: an expert's numbers come from its recipe, run as written). The recipe lines
its bump points out by hand at the start of the cycle, so that is where it is run:
bank D on the offset gives ``wd_Kc`` and ``wd_Ti``, and boron on Tavg the dead time
``theta`` (its SIMC integral time over 8). Large tier: the recipe is some hundred
bump tests.
"""

import importlib
import pathlib
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import tavg_dnb_limit
from target_gym.reactor.experts import ReferenceParams

#: The recipe lines its bump points out by hand for a core at the start of its cycle
#: (``reactor_ladder.point``), so its bump tests run there.
START_OF_CYCLE = {"cycle_range": (0.0, 0.0)}
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
ladder = importlib.import_module("reactor_ladder")

# The sweep's draw 1 (seed 20261110): a strong MTC, where the drift won.
STRONG = dict(alpha_coolant=-19.306e-5, boron_worth=-9.646e-5)


def hand_slope(q, s):
    k = -q.boron_worth / (q.alpha_coolant + q.alpha_fuel)
    return k * q.dilution_flow * float(s.boron) / q.primary_mass


@pytest.mark.medium
@pytest.mark.parametrize("plant", [{}, STRONG], ids=["nominal", "strong_mtc"])
@pytest.mark.parametrize("seed", [100, 103, 107])
def test_the_boron_bump_on_tavg_reads_the_plant(plant, seed):
    env = Reactor()
    p = ReactorParams(
        demand_sigma=0.0, sensor_noise_scale=0.0, **START_OF_CYCLE, **plant
    )
    for q, s in ladder.points(env, p, seed):
        slope, _ = ladder.boron_on_tavg(env, q, s)
        assert slope == pytest.approx(hand_slope(q, s), rel=0.2)


@pytest.mark.medium
def test_the_reading_ignores_the_day_drift():
    """The same plant with a day's burnup and debt reads the same slope."""
    env = Reactor()
    quiet = ReactorParams(
        demand_sigma=0.0, sensor_noise_scale=0.0, burnup_range=(0.0, 0.0),
        carried_days_range=(0.0, 0.0), initial_tilt_share=0.0, **START_OF_CYCLE,
        **STRONG,
    )  # fmt: skip
    busy = quiet.replace(burnup_range=(3.3, 3.3), initial_tilt_share=1.0)
    q0, s0 = ladder.points(env, quiet, 100)[0]
    q1, s1 = ladder.points(env, busy, 100)[0]
    a, _ = ladder.boron_on_tavg(env, q0, s0)
    b, _ = ladder.boron_on_tavg(env, q1, s1)
    assert b == pytest.approx(a, rel=0.05)


@pytest.mark.medium
@pytest.mark.parametrize("plant", [{}, STRONG], ids=["nominal", "strong_mtc"])
def test_the_boron_bump_keeps_the_plant_inside_its_limits(plant):
    env = Reactor()
    p = ReactorParams(
        demand_sigma=0.0, sensor_noise_scale=0.0, **START_OF_CYCLE, **plant
    )
    for q, s in ladder.points(env, p, 100):
        a = jnp.array([1.0 - 2.0 * s.rod_in, ladder.tavg_bump(q, s), 1.0])
        tr = ladder.closed(env, q, s, lambda m, o: (a, m), jnp.asarray(0.0), 120)
        # 0.05 K: the hour's reading ends at the limit, within its last step's rise.
        assert np.max(ladder.tavg(tr)) <= float(tavg_dnb_limit(q)) + 0.05
        assert np.max(np.asarray(tr.T_coolant)) < q.T_coolant_max
        assert float(ladder.tavg(tr)[-1]) > float(ladder.tavg(s)) + 0.5  # it moved


@pytest.mark.medium
def test_a_bump_that_trips_the_plant_is_refused():
    """A run that trips restarts the plant, and a fit across the restart reads it as
    the response (the v9 sweep's sign flip). The recipe stops instead of reading it.
    The old step, 20 % dilution for an hour at full power, reaches the 610 K hot leg."""
    env = Reactor()
    p = ReactorParams(demand_sigma=0.0, sensor_noise_scale=0.0, **START_OF_CYCLE)
    q, s = ladder.points(env, p, 100)[0]
    a = jnp.array([1.0 - 2.0 * s.rod_in, -0.2, 1.0])
    with pytest.raises(ladder.TrippedDuringTest):
        ladder.closed(env, q, s, lambda m, o: (a, m), jnp.asarray(0.0), 120)


@pytest.mark.large
def test_the_experts_bump_numbers_are_the_recipes_on_the_shipped_member():
    tune, _ = ladder.bumps(Reactor(), ReactorParams(**START_OF_CYCLE))
    shipped = ReferenceParams()
    assert shipped.wd_Kc == pytest.approx(tune.wd_Kc, rel=1e-3)
    assert shipped.wd_Ti == pytest.approx(tune.wd_Ti, rel=1e-6)
    assert shipped.theta == pytest.approx(tune.t_Ti / 8.0, rel=1e-6)
