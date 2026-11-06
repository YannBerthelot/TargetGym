"""The inner loop's tuning (scripts/tune_inner_loop.py).

Contract: the readings of section 2 of docs/proposals/expert-protocols-aircraft.md,
on hand-built logs, and that the gains in ``plane.dynamics.InnerLoop`` are what the
script returns, since a value typed by hand has no procedure behind it. A step's
overshoot is read on the first half of its hold.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from target_gym.plane.dynamics import InnerLoop

ROOT = pathlib.Path(__file__).resolve().parents[1]
HOLD = 20.0  # s, so 80 samples at 4 Hz
LEVELS = (0.0, 1.0)


def _script():
    spec = importlib.util.spec_from_file_location(
        "_tune_inner_loop", ROOT / "scripts" / "tune_inner_loop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(segment):
    return _script().reads(np.concatenate([np.zeros(80), segment]), LEVELS, HOLD)


def test_a_clean_step_reads_neither():
    assert _read(1.0 - np.exp(-np.arange(80) / 8.0)) == (False, False)


def test_a_peak_past_5_percent_is_an_overshoot_but_not_an_oscillation():
    response = 1.0 - np.exp(-np.arange(80) / 8.0)
    response[10:14] = 1.08
    assert _read(response) == (True, False)


def test_a_peak_inside_5_percent_is_no_overshoot():
    response = 1.0 - np.exp(-np.arange(80) / 8.0)
    response[10:14] = 1.04
    assert _read(response) == (False, False)


def test_a_late_drift_past_the_target_is_no_overshoot():
    """The speed mode, peaking 15 to 20 s after the step, in the hold's second half."""
    response = 1.0 - np.exp(-np.arange(80) / 8.0)
    response[60:] = 1.1
    assert _read(response)[0] is False


def test_two_sign_changes_past_the_band_after_the_peak_are_an_oscillation():
    # Approach, past (+0.2), under (-0.2), past: four excursions.
    t = np.arange(80)
    response = 1.0 + 0.2 * np.where(t < 4, -5.0, np.sign(np.sin(t / 6.0 + 0.1)))
    assert _read(response)[1] is True


@pytest.mark.medium
def test_the_shipped_gains_are_what_the_procedure_gives():
    """Against the numbers typed into ``InnerLoop``. About 3 s."""
    gains = _script().tuned_gains()
    shipped = InnerLoop()
    for name in ("Kp_theta", "Ki_theta", "Kd_theta", "Kp_roll", "Ki_roll", "Kd_roll"):
        assert getattr(gains, name) == pytest.approx(getattr(shipped, name), rel=1e-3)
