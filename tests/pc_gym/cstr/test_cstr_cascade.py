"""The CSTR expert is a temperature cascade: concentration trims a T loop.

Contract (PHYSICS.md section 6, from reactor practice): the outer PI on C_a
sets a temperature setpoint around the target's steady temperature; the
inner PI on T moves the coolant. On this plant a hotter reactor converts
more, so a C_a below its target calls for a cooler reactor and the coolant
goes down; above it, up. Neither integral winds up while the coolant is
saturated.
"""

import numpy as np

from target_gym.pc_gym.cstr.experts import cstr_cascade, make_cstr_pid_controller

TARGET, T_NOW = 0.875, 325.0


def _first_action(C_a, T=T_NOW, target=TARGET):
    pid = make_cstr_pid_controller().host()
    return float(pid(np.array([C_a, T, target], np.float32))[0])


def test_too_much_conversion_cools_the_reactor():
    assert _first_action(TARGET - 0.002) < _first_action(TARGET)


def test_too_little_conversion_warms_the_reactor():
    assert _first_action(TARGET + 0.002) > _first_action(TARGET)


def test_a_hotter_reactor_than_the_setpoint_gets_colder_coolant():
    pid = make_cstr_pid_controller().host()
    held = float(pid(np.array([TARGET, T_NOW, TARGET], np.float32))[0])
    hotter = float(pid(np.array([TARGET, T_NOW + 1.0, TARGET], np.float32))[0])
    assert hotter < held


def test_a_saturated_coolant_winds_up_neither_integral():
    """After many steps pinned at full coolant, the same small reversed error
    gets the same answer as a fresh controller's: nothing was stored."""
    loop = cstr_cascade(Kp=-130.0, Ti=0.9, Kp_T=0.27, Ti_T=0.63).host()
    pinned = np.array([TARGET + 0.05, T_NOW, TARGET], np.float32)
    for _ in range(50):
        assert float(loop(pinned)[0]) == 1.0
    reversed_error = np.array([TARGET - 0.001, T_NOW, TARGET], np.float32)
    fresh = cstr_cascade(Kp=-130.0, Ti=0.9, Kp_T=0.27, Ti_T=0.63).host()
    np.testing.assert_allclose(loop(reversed_error), fresh(reversed_error), rtol=1e-6)
