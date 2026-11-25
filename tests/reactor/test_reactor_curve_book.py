"""The expert's curve book, measured (owner, 2026-11-25).

Contract, from the owner's answer and the practice of reload startup physics tests
(ANSI/ANS-19.6.1), written before the code:

- The operator does not read the core's parameters; the recipe measures them at
  power, at the two ends of the load-following range: a dilution bump at fixed rods
  reads K of Tavg per ppm, a rod bump at fixed boron, read against the rods'
  calibrated worth curve, reads the feedback ``alpha_coolant + alpha_fuel``. The
  worth is ``-K x feedback`` and the moderator coefficient the feedback less the
  design's Doppler coefficient.
- A measurement on a noise-free plant reads what it measures: within 3 % of the
  core's own values (core_at), at the start of the cycle (-12 pcm/K, -8 pcm/ppm)
  and at 90 % of it (-12 - 0.9 x 48 = -55.2 pcm/K; -8 - 0.9 x 2.9 = -10.61
  pcm/ppm). 3 % covers the bumps' linear readings over an hour; the rods move the
  offset too.
- The expert reads straight lines through the two measured ends, carried on to
  the cycle's end: a value ``v0`` at the start and ``v1`` at 0.9 is
  ``v0 + (v1 - v0) / 0.9`` at the end.
"""

import sys
from pathlib import Path

import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.experts import ReferenceParams, measured

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_the_expert_reads_straight_lines_through_the_measured_ends():
    ref = ReferenceParams(
        book_alpha_coolant=-10e-5, book_alpha_coolant_end=-55e-5,
        book_boron_worth=-8e-5, book_boron_worth_end=-11e-5, book_cycle_end=0.9,
    )  # fmt: skip
    book = measured(ReactorParams(), ref)
    assert book.alpha_coolant == pytest.approx(-10e-5)
    assert book.alpha_coolant_end == pytest.approx(-10e-5 - 45e-5 / 0.9)
    assert book.boron_worth == pytest.approx(-8e-5)
    assert book.boron_worth_end == pytest.approx(-8e-5 - 3e-5 / 0.9)


@pytest.mark.medium
@pytest.mark.parametrize(
    "cycle, alpha_coolant, worth",
    [(0.0, -12.0e-5, -8.0e-5), (0.9, -55.2e-5, -10.61e-5)],
)
def test_the_physics_tests_read_the_cores_own_coefficients(cycle, alpha_coolant, worth):
    import reactor_ladder as L

    env = Reactor()
    p = ReactorParams(
        demand_sigma=0.0, sensor_noise_scale=0.0, cycle_range=(cycle, cycle)
    )
    read_alpha, read_worth = L.book_at(env, *L.point(env, p, 100, 1.0))
    assert read_alpha == pytest.approx(alpha_coolant, rel=0.03)
    assert read_worth == pytest.approx(worth, rel=0.03)
