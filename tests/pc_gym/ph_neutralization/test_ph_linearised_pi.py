"""The pH expert linearises the titration curve and biases the base valve.

Contract (PHYSICS.md section 7, from pH practice): the PI acts on reagent
demand, the steady base flow the titration curve says a pH needs at the
nominal buffer flow, so ``reagent_demand`` must invert the plant's own steady
state. Its bias is the target's demand, so at rest on target with the
nominal buffer it asks for the flow that holds the target, with the stage-2
trim shut. A pH below target opens the base valve further; above it, closes
it. The plant is staged: the stage-1 valve reads the stage-1 probe.

Owner, 2026-11-25: the set point is the operator's, held in the expert's own
parameters (``setpoint_pH``), not read from the observation, whose target channel
shows the permit's centre; and the demand is fed forward from the metered influent
(obs 6), as a pH loop on a metered stream is, so on target the bias holds the
target at the influent the meter reads.

The oracle for the curve is the environment's own steady state: mixing
(``steady_state_invariants``) and the bisection on the charge balance
(``solve_pH``), not the closed form under test.
"""

import numpy as np
import pytest

from target_gym.experts.pid import gain
from target_gym.pc_gym.ph_neutralization.env import (
    PHParams,
    solve_pH,
    steady_state_invariants,
)
from target_gym.pc_gym.ph_neutralization.experts import (
    TitrationCurve,
    make_ph_pid_controller,
    ph_linearised_pi,
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


#: The observation's target channel: the permit's centre, never the set point.
CENTRE = 0.5 * (P.permit_low + P.permit_high)


def _obs(pH_1, pH_2=None, q1=P.q1, shown=CENTRE):
    """The 8-channel observation with both probes at ``pH_1`` unless the
    discharge's ``pH_2`` is given, the meter reading ``q1``."""
    pH_2 = pH_1 if pH_2 is None else pH_2
    return np.array([pH_2, 50.0, shown, 0.0, pH_1, 0.0, q1, 0.0], np.float32)


def _expert(target):
    """The scored expert's gains, holding ``target``."""
    return ph_linearised_pi(
        Kp=gain("ph_neutralization", "Kp"),
        Ti=gain("ph_neutralization", "Ti"),
        params=P.replace(setpoint_pH=target),
        Tf=gain("ph_neutralization", "Tf"),
    ).host()


def _raw(q3):
    return 2.0 * (q3 - P.q3_min) / (P.q3_max - P.q3_min) - 1.0


@pytest.mark.parametrize("target", [6.5, 7.0, 8.0])
def test_on_target_it_asks_for_the_flow_that_holds_the_target(target):
    action = np.asarray(_expert(target)(_obs(target)))
    q3 = P.q3_min + 0.5 * (float(action[0]) + 1.0) * (P.q3_max - P.q3_min)
    assert _steady_pH(q3) == pytest.approx(target, abs=1e-3)
    assert float(action[1]) == pytest.approx(-1.0)  # the trim rests shut


def test_acid_effluent_opens_the_base_valve_and_alkaline_closes_it():
    on, acid, alkaline = (float(_expert(7.0)(_obs(pH))[0]) for pH in (7.0, 6.8, 7.2))
    assert acid > on > alkaline


def test_an_acid_discharge_opens_the_trim():
    trim = float(_expert(7.0)(_obs(7.0, pH_2=6.8))[1])
    assert trim > -1.0


def test_the_scored_expert_holds_the_operators_set_point_not_the_observations():
    """The scored expert holds ``PHParams().setpoint_pH`` whatever the target
    channel shows."""
    pid = make_ph_pid_controller().host()
    sp = float(P.setpoint_pH)
    a = np.asarray(pid(_obs(sp, shown=CENTRE)))
    b = np.asarray(make_ph_pid_controller().host()(_obs(sp, shown=6.0)))
    np.testing.assert_array_equal(a, b)
    q3 = P.q3_min + 0.5 * (float(a[0]) + 1.0) * (P.q3_max - P.q3_min)
    assert _steady_pH(q3) == pytest.approx(sp, abs=1e-3)


@pytest.mark.parametrize("scale", [0.8, 1.2])
def test_on_target_it_feeds_forward_the_metered_influent(scale):
    """With the meter reading ``scale`` times the nominal influent, the base
    flow it asks for on target holds the target at that influent: the plant's
    own steady state at ``q1 = scale * nominal`` (``steady_state_invariants``,
    ``solve_pH``)."""
    q1 = scale * float(P.q1)
    action = np.asarray(_expert(6.5)(_obs(6.5, q1=q1)))
    q3 = P.q3_min + 0.5 * (float(action[0]) + 1.0) * (P.q3_max - P.q3_min)
    plant = P.replace(q1=q1)
    pH = float(solve_pH(*steady_state_invariants(q3, P.q2_nominal, plant), plant))
    assert pH == pytest.approx(6.5, abs=1e-3)
