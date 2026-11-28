"""What a day leaves owed at its end (owner, 2026-11-23).

Contract, written before the code. A day is a window cut in a plant that runs
on, and its score charges nothing for the state it ends in: a policy that dilutes
nothing leaves the day's burnup for the next one. The family sweep prices that
end (``reactor_ladder.owed``); the task's score does not.

What is owed is what it takes to put the plant back on its feet: the rods back at
their park, at their price a step ($0 since the owner dropped it on 2026-11-28; it
was a dollar), and the boron moved by what then closes Tavg's error,
at the effluent's price. By hand at the start of the cycle (15 pcm/K of both
feedbacks, 8 pcm a ppm, 1690 ppm, a primary of 2.5e5 kg, $0.0628 a kg, acid at
7800 ppm):

- on its program with the rods parked, nothing;
- Tavg 1 K under its program: 15 / 8 = 1.875 ppm to dilute, 2.5e5 x ln(1690 /
  1688.125) = 277.5 kg, $17.43;
- Tavg 1 K over it: 1.875 ppm to borate, 2.5e5 x ln(6110 / 6108.125) = 76.7 kg,
  $4.82;
- the rods 10 steps in from their park at 3, Tavg on its program: their travel
  back, free at the task's price, and the worth they give back, 1500 pcm x
  (f(13 / 228) - f(3 / 228)) with f(y) = y - sin(2 pi y) / (2 pi), 1500 x 0.0011967
  = 1.795 pcm, is 0.2244 ppm to borate, 9.18 kg, $0.58.

A day's change in it: a quiet plant on full load whose fuel burns 3 ppm a day,
left alone (rods parked, no boron), loses 24 pcm, so Tavg ends the day 1.6 K
under its program, at the edge of its free band and never past it: the day
charges nothing. It owes the 3 ppm: 2.5e5 x ln(1690 / 1687) = 444.2 kg, $27.9.
"""

import importlib
import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest

from target_gym.reactor import Reactor, ReactorParams

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
ladder = importlib.import_module("reactor_ladder")

QUIET = dict(
    demand_sigma=0.0,
    sensor_noise_scale=0.0,
    carried_days_range=(0.0, 0.0),
    cycle_range=(0.0, 0.0),
    initial_tilt_share=0.0,
    initial_xenon_power_range=(1.0, 1.0),
    ramp_down_range=(1e6, 1e6),
)
STILL = ReactorParams(burnup_range=(0.0, 0.0), **QUIET)
BURNING = ReactorParams(burnup_range=(3.0, 3.0), **QUIET)


def _lined_out(under=0.0, steps_in=0.0):
    """The plant on full load with Tavg ``under`` K under its program and the rods
    ``steps_in`` steps in from their park."""
    _, s = Reactor().reset_env(jax.random.PRNGKey(0), STILL)
    assert float(s.boron) == pytest.approx(1690.0, abs=5.0)
    shift = s.T_ref - under - 0.5 * (s.T_coolant + s.T_cold)
    return s.replace(
        T_coolant=s.T_coolant + shift,
        T_cold=s.T_cold + shift,
        rod_in=s.rod_in * 0 + (3.0 + steps_in) / 228.0,
    )


@pytest.mark.parametrize(
    "under, steps_in, dollars",
    [(0.0, 0.0, 0.0), (1.0, 0.0, 17.43), (-1.0, 0.0, 4.82), (0.0, 10.0, 0.58)],
)
def test_a_plant_owes_what_puts_it_back_on_its_program(under, steps_in, dollars):
    owed = float(ladder.owed(_lined_out(under, steps_in), STILL))
    assert owed == pytest.approx(dollars, rel=0.01, abs=0.02)


@pytest.mark.medium
def test_a_days_burnup_left_undiluted_is_owed_at_its_end():
    env = Reactor()
    hold = lambda m, o: (jnp.array([ladder.PARK, 0.0, 1.0]), m)  # noqa: E731
    out = ladder.episode(env, BURNING, 0, hold, jnp.asarray(0.0))
    assert int(out["trips"]) == 0
    assert float(out["cost"]) < 1.0
    assert float(out["owed"]) == pytest.approx(27.9, rel=0.03)
