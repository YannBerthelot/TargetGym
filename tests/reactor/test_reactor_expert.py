"""The published rod program: Westinghouse PWR rod control on average temperature.

The task's expert is now the ladder's best rung (``experts.reference_step``). This
program stays as a row in the ladder's table, so its contract stays here.

Contract, from the NRC's Westinghouse Technology training manual (HRTD 8.1,
rod control; HRTD 5.0, the Tavg program and the 40 s rate/lag), not from the
code:

* Tavg is programmed linearly in load from no-load to full power, and the plant
  reports both.
* The total error, in deg F, is ``(Tref - Tavg)`` plus the power-mismatch
  term: a rate/lag ``40 s / (1 + 40 s)`` on (load - power) in % power,
  through a nonlinear gain of 0.3 F per % up to 1 % and 1.5 F per % beyond,
  times a variable gain of 2 up to 50 % load and 1 / load above.
* Rod speed: none inside +-1.5 F from rest; once moving, the rods stop only
  at 1.0 F (lockup); 8 steps/min to 3 F, then 32 steps/min per F, 72 steps/min
  from 5 F. A positive error withdraws.
* The 228-step stroke spans the rods' range, so a speed of ``s`` steps/min moves
  the command by ``2 s * 30 / (60 * 228)`` of the [-1, 1] action per 30 s step
  (72 steps/min: 0.31579).
"""

import numpy as np
import pytest

from target_gym.experts.pid import Controller
from target_gym.reactor.experts import (
    RodControlParams,
    RodControlState,
    rod_control_step,
)
from target_gym.registry import REGISTRY

P = REGISTRY["reactor"].make_test_params()
PER_SPM = 2.0 * 30.0 / (60.0 * 228.0)  # action per step per step/min


def _obs(error_F=0.0, n=0.8, target=0.8, rod=0.0):
    """An observation whose temperature error (Tref - Tavg) is ``error_F``."""
    T_ref = 570.0
    T_avg = T_ref - error_F / 1.8
    return np.array([n, 590.0, rod, target, T_avg, T_ref], np.float32)


def _speed(expert, obs):
    """The rod speed commanded, in steps/min, signed (withdrawal positive)."""
    return (float(np.asarray(expert(obs))[0]) - float(obs[2])) / PER_SPM


def _expert():
    expert = Controller(RodControlParams(), RodControlState(), rod_control_step).host()
    expert.reset()
    return expert


@pytest.mark.parametrize(
    "error_F, spm",
    [(0.0, 0.0), (1.2, 0.0), (2.0, 8.0), (4.0, 40.0), (6.0, 72.0), (-6.0, -72.0)],
)
def test_the_rod_speed_program(error_F, spm):
    """From rest, on the temperature error alone. 4 F: 8 + 32 * (4 - 3) = 40."""
    assert _speed(_expert(), _obs(error_F)) == pytest.approx(spm, abs=1e-2)


def test_moving_rods_stop_only_inside_the_lockup():
    expert = _expert()
    assert _speed(expert, _obs(2.0)) == pytest.approx(8.0, abs=1e-2)
    assert _speed(expert, _obs(1.2)) == pytest.approx(8.0, abs=1e-2)
    assert _speed(expert, _obs(0.9)) == pytest.approx(0.0, abs=1e-2)
    assert _speed(expert, _obs(1.2)) == pytest.approx(0.0, abs=1e-2)


def test_a_demand_step_moves_the_rods_through_the_rate_lag():
    """Load (80 %) 5 % above power, Tavg on its program. The rate/lag leaves
    5 exp(-30 k / 40) % after k steps of 30 s: 2.362, 1.116, 0.527. Through the gains
    (0.3 + 1.5 (m - 1)) * 1.25 F: 2.93, 0.59, 0.20. So 8 steps/min (under 3 F), then
    below the 1.0 F lockup, stopped."""
    expert = _expert()
    _speed(expert, _obs(n=0.8, target=0.8))
    speeds = [_speed(expert, _obs(n=0.75, target=0.80)) for _ in range(3)]
    np.testing.assert_allclose(speeds, [8.0, 0.0, 0.0], atol=0.05)


def test_the_mismatch_gain_doubles_below_half_load():
    """3 % of mismatch leaves 1.417 % after one 30 s step, 0.926 F through the
    nonlinear gain: times 2 at 40 % load it moves the rods (1.85 F), times 1 / 0.8
    at 80 % it does not (1.16 F)."""
    low, high = _expert(), _expert()
    _speed(low, _obs(n=0.40, target=0.40))
    _speed(high, _obs(n=0.80, target=0.80))
    assert _speed(low, _obs(n=0.37, target=0.40)) == pytest.approx(8.0, abs=1e-2)
    assert _speed(high, _obs(n=0.77, target=0.80)) == pytest.approx(0.0, abs=1e-2)


def test_power_below_demand_withdraws_and_above_inserts():
    up, down = _expert(), _expert()
    _speed(up, _obs())
    _speed(down, _obs())
    assert _speed(up, _obs(n=0.75, target=0.80)) > 0.0
    assert _speed(down, _obs(n=0.85, target=0.80)) < 0.0


# Spec, stated before measuring: Westinghouse rod control holds Tavg inside its
# deadband. On held-out seeds, over the scored hold, Tavg is within 1.5 F of
# Tref on at least 90 % of steps, and nothing trips.
HELD_OUT = range(200, 210)
