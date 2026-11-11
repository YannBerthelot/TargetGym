"""The glass furnace expert: a PI on the fining share read at the cold end, held two
sigma above its pass mark.

Contract, from the tuning rule stored with the gains (``pid_gains.json``) and the
charter's two-sigma back-off, not from the code:

* SIMC with the closed-loop time equal to the dead time, on the stored valve step:
  ``Kc = tau / (k (tc + theta))`` and ``Ti = min(tau, 4 (tc + theta))`` (Skogestad
  2003), ``k`` in share per kg/s, the action's 2 raw units spanning the valve's range.
* The setpoint is the pass mark plus twice the stored spread.
* At its setpoint with no integral, the valve rests where the design state holds it.
* The integral stops while the valve is at a limit.
"""

import json

import numpy as np
import pytest

from target_gym.glass_furnace import env_jax as E
from target_gym.provenance import GAINS_PATH
from target_gym.registry import REGISTRY

STORED = json.loads(GAINS_PATH.read_text())["glass_furnace"]
P0 = REGISTRY["glass_furnace"].make_test_params()


def _pi(**changes):
    expert = REGISTRY["glass_furnace"].make_pid_controller()
    expert = expert.replace(params=expert.params.replace(**changes)).host()
    expert.reset()
    return expert


def _obs(fining):
    obs = np.zeros(len(E.OBS_NAMES), np.float32)
    obs[4], obs[5] = fining, P0.fining_pass
    return obs


def test_the_gains_follow_simc_for_the_stored_step_test():
    """The store keeps k, tau and theta to four or five digits and Ti to 0.1 h of
    30.7 h, so the gains agree within 0.2 %."""
    m = STORED["model"]
    k, tau, theta = m["k"], m["tau_h"], m["theta_h"]
    raw_per_kg_s = 2.0 / (P0.fuel_max - P0.fuel_min)
    kc = raw_per_kg_s * tau / (k * (theta + theta))
    ti = min(tau, 4.0 * (theta + theta)) * 3600.0
    assert STORED["Kp"] == pytest.approx(kc, rel=1e-3)
    assert STORED["Ki"] == pytest.approx(kc / ti, rel=2e-3)
    assert STORED["Kd"] == 0.0


def test_the_setpoint_sits_two_sigma_above_the_pass_mark():
    assert STORED["backoff"] == pytest.approx(2.0 * STORED["model"]["sigma"])


def test_at_its_setpoint_the_valve_holds_the_design_fuel():
    pi = _pi()
    u = float(pi(_obs(P0.fining_pass + STORED["backoff"]))[0])
    fuel, _, _ = E.design_state()
    assert float(E.to_fuel(u, P0)) == pytest.approx(fuel, abs=1e-4)


def test_a_shortfall_opens_the_valve_by_the_proportional_gain():
    """No integral: 0.01 short at Kp 2 opens the valve 0.02 above its rest."""
    rest = float(_pi(Kp=2.0, Ki=0.0)(_obs(P0.fining_pass + STORED["backoff"]))[0])
    short = float(
        _pi(Kp=2.0, Ki=0.0)(_obs(P0.fining_pass + STORED["backoff"] - 0.01))[0]
    )
    assert short - rest == pytest.approx(0.02, rel=1e-4)


def test_the_integral_stops_while_the_valve_is_at_its_limit():
    """Ten steps pinned open by a large shortfall, then on the setpoint: with the
    integral stopped the valve returns to its rest. Wound up, it would sit 0.06 higher
    (10 steps x 0.5 x 120 s x 1e-4)."""
    on = P0.fining_pass + STORED["backoff"]
    rest = float(_pi(Kp=200.0, Ki=1e-4)(_obs(on))[0])
    pi = _pi(Kp=200.0, Ki=1e-4)
    for _ in range(10):
        assert float(pi(_obs(on - 0.5))[0]) == 1.0
    assert float(pi(_obs(on))[0]) == pytest.approx(rest, abs=1e-6)
