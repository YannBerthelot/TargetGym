"""The pH expert linearises the titration curve and biases the base valve.

Contract (PHYSICS.md section 7, from pH practice): the PI acts on reagent
demand, the steady base flow the titration curve says a pH needs at the
nominal buffer flow, so ``reagent_demand`` must invert the plant's own steady
state. Its bias is the target's demand, so at rest on target with the
nominal buffer it asks for the flow that holds the target. A pH below target
opens the base valve further; above it, closes it.

The oracle for the curve is the environment's own steady state: mixing
(``steady_state_invariants``) and the bisection on the charge balance
(``solve_pH``), not the closed form under test.
"""

import numpy as np
import pytest

from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    solve_pH,
    steady_state_invariants,
)
from target_gym.pc_gym.ph_neutralization.experts import (
    TitrationCurve,
    make_ph_pid_controller,
    reagent_demand,
)

P = PHParams()
CURVE = TitrationCurve.of(P)
#: The bisection resolves pH to 14 / 2**20 = 1.3e-5, and over the valve's
#: range (pH 4.0 to 10.3 at the nominal buffer) the steady base flow moves at
#: most 9.6 mL/s per pH, at the alkaline end (measured on ``_steady_pH``), so
#: the inverse is good to 1.3e-4 mL/s.
DEMAND_TOL = 2e-4


def _steady_pH(q3, q2=P.q2_nominal):
    return float(solve_pH(*steady_state_invariants(q3, q2, P), P))


@pytest.mark.parametrize("q3", [10.0, 12.5, 15.0, 16.0, 17.5, 20.0, 22.0])
def test_demand_inverts_the_plants_steady_state(q3):
    assert abs(float(reagent_demand(_steady_pH(q3), CURVE, np)) - q3) < DEMAND_TOL


def _raw(q3):
    return 2.0 * (q3 - P.q3_min) / (P.q3_max - P.q3_min) - 1.0


@pytest.mark.parametrize("target", [6.5, 7.0, 8.0])
def test_on_target_it_asks_for_the_flow_that_holds_the_target(target):
    pid = make_ph_pid_controller().host()
    raw = float(pid(np.array([target, 50.0, target], np.float32))[0])
    q3 = P.q3_min + 0.5 * (raw + 1.0) * (P.q3_max - P.q3_min)
    assert _steady_pH(q3) == pytest.approx(target, abs=1e-3)


def test_acid_effluent_opens_the_base_valve_and_alkaline_closes_it():
    on, acid, alkaline = (
        float(make_ph_pid_controller().host()(np.array([pH, 50.0, 7.0]))[0])
        for pH in (7.0, 6.8, 7.2)
    )
    assert acid > on > alkaline
