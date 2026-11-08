"""What the process experts do with the filter and gate their recipes size
(docs/proposals/expert-protocols-noise.md, N5 and N8).

Contract, from the proposal: ``Tf`` is a first-order filter on the reading, so a
step of ``d`` reaches the controller as ``a d``, ``a = 1 - exp(-dt / Tf)``, and
``Tf = 0`` leaves the reading alone. An analyser loop counts updates, acts only
on a move past its band, and after a move sits out the updates that cannot show
it. Expected values are worked by hand from those statements.
"""

import math

import numpy as np
import pytest

from target_gym.pc_gym.cstr.experts import cstr_temperature_pi
from target_gym.pc_gym.distillation.experts import distillation_base_layer
from target_gym.pc_gym.first_order.experts import first_order_pi
from target_gym.pc_gym.ph_neutralization.experts import (
    TitrationCurve,
    ph_linearised_pi,
    reagent_demand,
)
from target_gym.registry import REGISTRY


def _run(controller, observations):
    pid = controller.host()
    return [np.asarray(pid(np.asarray(o, np.float32))) for o in observations]


def _coefficient(dt, Tf):
    return -math.expm1(-dt / Tf)


@pytest.mark.parametrize("Tf", [0.0, 0.1])
def test_first_order_reads_its_filter(Tf):
    # P only, gain 2, on x stepping 0 to 0.1 with the target at 0: the first
    # call starts the filter at the reading, the second moves it a of the way.
    a = _coefficient(0.05, Tf) if Tf else 1.0
    out = _run(first_order_pi(2.0, 0.0, Tf), [[0.0, 0.0], [0.1, 0.0]])
    assert out[0][0] == pytest.approx(0.0)
    assert out[1][0] == pytest.approx(-2.0 * a * 0.1, rel=1e-5)


@pytest.mark.parametrize("Tf", [0.0, 0.5])
def test_the_cstr_temperature_loop_reads_its_filter(Tf):
    # P only, gain 0.1 per K, on T stepping 1 K above its setpoint: the first
    # update starts the filter, the second moves it by a of the step (v5).
    p = REGISTRY["cstr"].make_test_params()
    a = _coefficient(float(p.delta_t), Tf) if Tf else 1.0
    c = cstr_temperature_pi(0.1, 1e12, p, Tf=Tf)
    sp = float(p.T_setpoint)
    out = _run(c, [[sp, 320.0, 50.0, sp], [sp + 1.0, 320.0, 50.0, sp]])
    assert out[1][0] - out[0][0] == pytest.approx(0.1 * a, rel=1e-3)


@pytest.mark.parametrize("Tf", [0.0, 20.0])
def test_ph_reads_its_filter_in_demand_units(Tf):
    # The filter runs on the linearised reading, what the PI acts on.
    p = REGISTRY["ph_neutralization"].make_test_params()
    a = _coefficient(float(p.delta_t), Tf) if Tf else 1.0
    curve = TitrationCurve.of(p)
    d0 = float(reagent_demand(7.0, curve, np))
    d1 = float(reagent_demand(7.2, curve, np))
    span = float(p.q3_max - p.q3_min)
    c = ph_linearised_pi(1.0, 1e12, p, Tf=Tf)
    # Staged (v5): the stage-1 loop reads the stage-1 probe, channel 4.
    out = _run(
        c,
        [
            [7.0, 50.0, 7.0, 0.0, 7.0, 0.0, 16.6, 0.0],
            [7.0, 50.0, 7.0, 0.0, 7.2, 0.0, 16.6, 0.0],
        ],
    )
    expected = -2.0 / span * a * (d1 - d0)
    assert out[1][0] - out[0][0] == pytest.approx(expected, rel=1e-3)


# --- The analyser gate ------------------------------------------------------


def _column(**over):
    p = REGISTRY["distillation"].make_test_params()
    gains = dict(
        Kp_T=0.1,
        Ti_T=1e12,
        Kp_D=10.0,
        Ti_D=1e12,
        Kp_B=-100.0,
        Ti_B=1e12,
        Tf_T=0.0,
        band_D=0.002,
        band_B=0.002,
        every=5,
        wait=2,
        sp_rate=1e6,
    )
    return distillation_base_layer(**{**gains, **over}, params=p), p


def _obs(yD=0.99, xB=0.01, T=351.0, tD=0.99, tB=0.01):
    return [yD, xB, 0.5, 0.5, tD, tB, T]


def _reflux(raw, p):
    return p.L_min + (raw + 1.0) / 2.0 * (p.L_max - p.L_min)


def test_an_analyser_loop_ignores_a_move_inside_its_band():
    c, p = _column()
    steady = [_obs()] * 5
    out = _run(c, steady + [_obs(yD=0.9915)])  # 0.0015 is inside 0.002
    assert out[-1][0] == pytest.approx(out[0][0])


def test_an_analyser_loop_acts_on_the_update_after_a_move_past_its_band():
    c, p = _column()
    # The cycle is 5 steps: updates at 0, 5, 10. A move shows at step 5.
    out = _run(c, [_obs()] * 5 + [_obs(yD=0.994)] * 5)
    # e = -0.004, Kp_D 10, so L falls 0.04 at step 5 and holds through step 9.
    moved = _reflux(out[5][0], p) - _reflux(out[4][0], p)
    assert moved == pytest.approx(-0.04, rel=1e-3)
    assert out[9][0] == pytest.approx(out[5][0])


def test_an_analyser_loop_sits_out_the_update_that_cannot_show_its_move():
    # wait 2: acted at step 5, so step 10's update is ignored and step 15's reads.
    c, p = _column()
    obs = [_obs()] * 5 + [_obs(yD=0.994)] * 5 + [_obs(yD=0.998)] * 5 + [_obs(yD=0.998)]
    out = _run(c, obs)
    assert out[10][0] == pytest.approx(out[5][0])
    assert _reflux(out[15][0], p) - _reflux(out[5][0], p) == pytest.approx(
        -0.04, rel=1e-3
    )


def test_a_target_past_the_band_counts_as_a_move():
    c, p = _column()
    out = _run(c, [_obs()] * 5 + [_obs(tD=0.995)])
    assert _reflux(out[5][0], p) - _reflux(out[0][0], p) == pytest.approx(
        0.05, rel=1e-3
    )


def test_the_temperature_setpoint_ramps_to_what_the_bottoms_loop_asks():
    # xB 0.004 over its target asks -100 * -0.004 = +0.4 K at step 5, at 0.1 K per
    # minute: 0.1 K of it at once, and the temperature loop's P is 0.1 on that.
    c, p = _column(sp_rate=0.1)
    out = _run(c, [_obs()] * 5 + [_obs(xB=0.014)] * 3)
    V_per_raw = (p.V_max - p.V_min) / 2.0
    assert (out[5][1] - out[4][1]) * V_per_raw == pytest.approx(0.01, rel=1e-3)
    assert (out[7][1] - out[4][1]) * V_per_raw == pytest.approx(0.03, rel=1e-3)
