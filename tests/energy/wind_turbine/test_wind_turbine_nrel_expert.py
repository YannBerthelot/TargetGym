"""The wind turbine expert: the NREL 5 MW baseline controller.

Contract, from Jonkman, Butterfield, Musial and Scott (2009), NREL/TP-500-38060,
chapter 7 (not from the code):

* Generator speed is filtered by a single pole at 0.25 Hz,
  ``alpha = exp(-2 pi Ts fc)``; both loops read the filtered speed.
* Generator torque (generator side) by region: none below 670 rpm; a line
  from 670 to 871 rpm up to the region-2 curve; ``0.0255764 N m/rpm^2 w^2``;
  a 10 % slip line ending at 1161.963 rpm (99 % of rated) on the region-3
  torque; and region 3, ``P / w``, whenever the previous pitch command is at
  least 1 deg or the filtered speed is at least 1161.963 rpm. Torque is
  limited to 47,402.91 N m and to 15,000 N m/s.
* Pitch: a PI on the filtered generator-speed error against 1173.7 rpm, with
  ``Kp = 2 I W0 zeta wn / (N (-dP/dtheta))`` and ``Ki = I W0 wn^2 / (N
  (-dP/dtheta))``, ``wn`` 0.6 rad/s and ``zeta`` 0.7; the integral is
  saturated at the pitch limits; the command moves at most 8 deg/s; it starts
  bumpless from the measured pitch.
* Ours, for a curtailed task: ``P`` is the dispatched target over the
  generator efficiency (Jonkman's 5.296610 MW is 5 MW / 0.944), and the
  below-rated torque never exceeds region 3's. Gains are fixed: on our Cp
  surface the sensitivity's best-fit line is flat, so the schedule ``GK``
  is 1.
"""

import json
import math

import jax
import numpy as np
import pytest

import target_gym as tg
from target_gym.energy.wind_turbine.env import WindTurbineParams
from target_gym.provenance import GAINS_PATH
from target_gym.registry import REGISTRY

WP = WindTurbineParams()
N = float(WP.N_gear)
ETA = float(WP.eta_gen)
TS = float(WP.delta_t)
RATED_GEN_RPM = 1173.7
RPM = 2.0 * math.pi / 60.0
K2 = 0.0255764  # N m per rpm^2, generator side


def _obs(gen_rpm, pitch=10.0, torque_gen=None, target_MW=4.0):
    torque = 0.0 if torque_gen is None else torque_gen
    return np.array(
        [gen_rpm / N, pitch, 100.0 * torque / float(WP.torque_max), 0.0, target_MW],
        np.float32,
    )


def _torque(action):
    return 0.5 * (float(action[1]) + 1.0) * float(WP.torque_max)


def _pitch(action):
    return 0.5 * (float(action[0]) + 1.0) * float(WP.pitch_max)


def _expert():
    expert = tg.make_expert("wind_turbine").host()
    expert.reset()
    return expert


def test_region_3_holds_the_dispatched_power():
    """4 MW at rated speed: (4e6 / 0.944) / (1173.7 rpm) = 34,474 N m."""
    tq = 4.0e6 / ETA / (RATED_GEN_RPM * RPM)
    a = _expert()(_obs(RATED_GEN_RPM, torque_gen=tq))
    assert _torque(a) == pytest.approx(34474.0, rel=1e-3)


@pytest.mark.parametrize(
    "gen_rpm, expected",
    [
        (997.6, K2 * 997.6**2),  # region 2: 25,454 N m
        (800.0, K2 * 871.0**2 * (800.0 - 670.0) / (871.0 - 670.0)),  # 1.5: 12,550
    ],
)
def test_below_rated_at_fine_pitch_the_torque_follows_the_baseline_table(
    gen_rpm, expected
):
    a = _expert()(_obs(gen_rpm, pitch=0.0, torque_gen=expected))
    assert _torque(a) == pytest.approx(expected, rel=1e-3)


def test_the_torque_moves_at_most_15_kNm_per_second():
    """From 20,000 N m toward region 3's 34,474: 15,000 * 0.25 s = 3,750."""
    a = _expert()(_obs(RATED_GEN_RPM, torque_gen=20000.0))
    assert _torque(a) == pytest.approx(23750.0, rel=1e-3)


def test_the_pitch_gains_are_jonkmans_pole_placement():
    m = json.loads(GAINS_PATH.read_text())["wind_turbine"]["model"]
    I = float(WP.J_rotor) + N**2 * float(WP.J_gen)
    W0 = float(WP.omega_rated_rpm) * RPM
    den = N * -m["dP_dtheta"]
    params = tg.make_expert("wind_turbine").params
    assert params.Kp == pytest.approx(2 * I * W0 * 0.7 * 0.6 / den, rel=1e-9)
    assert params.Ki == pytest.approx(I * W0 * 0.6**2 / den, rel=1e-9)


def test_a_speed_step_reaches_the_pitch_through_the_quarter_hertz_filter():
    """Bumpless at rated speed and 10 deg, then 1 % overspeed: the filter
    passes (1 - alpha) of it, and the PI moves the pitch by
    (Kp + Ki Ts) (1 - alpha) de."""
    params = tg.make_expert("wind_turbine").params
    expert = _expert()
    tq = 4.0e6 / ETA / (RATED_GEN_RPM * RPM)
    assert _pitch(expert(_obs(RATED_GEN_RPM, 10.0, tq))) == pytest.approx(
        10.0, abs=1e-4
    )
    alpha = math.exp(-2.0 * math.pi * TS * 0.25)
    de = 0.01 * RATED_GEN_RPM * RPM * (1.0 - alpha)
    moved = math.degrees((params.Kp + params.Ki * TS) * de)
    a = expert(_obs(1.01 * RATED_GEN_RPM, 10.0, tq))
    assert _pitch(a) - 10.0 == pytest.approx(moved, rel=1e-3)


def test_the_pitch_command_moves_at_most_8_deg_per_second():
    expert = _expert()
    expert(_obs(RATED_GEN_RPM, 10.0))
    a = expert(_obs(1.2 * RATED_GEN_RPM, 10.0))
    assert _pitch(a) - 10.0 == pytest.approx(8.0 * TS, abs=1e-4)


def test_the_integral_waits_at_fine_pitch():
    """Long underspeed at 0 deg does not wind the integral below the stop, so
    the first overspeed pitches at once."""
    expert = _expert()
    for _ in range(100):
        expert(_obs(0.9 * RATED_GEN_RPM, 0.0))
    for _ in range(20):  # let the filtered speed rise past rated
        a = expert(_obs(1.02 * RATED_GEN_RPM, 0.0))
    assert _pitch(a) > 0.0


# Spec, stated before measuring: the baseline regulates rated speed and holds
# the dispatched power. On held-out seeds, over the scored hold, rotor speed is
# within 10 % of rated on at least 99 % of steps, the mean power error is under
# 1 % of the target, and nothing trips.
HELD_OUT = range(200, 210)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="Finding (2026-10-28): speed holds within 10 % on every hold step, "
    "but the mean power error is 0.25-1.10 % of the target, 1.10 % on seed 200 "
    "(the PID it replaces: 0.08-0.14 %). Region 3's torque divides by the "
    "0.25 Hz filtered speed, so the power lags the rotor's speed swings.",
)
def test_speed_and_power_hold_on_held_out_seeds():
    from target_gym.eval import scored_burn_in

    spec = REGISTRY["wind_turbine"]
    env, p = spec.make_env(), spec.make_test_params()
    step = jax.jit(env.step_env)
    burn = scored_burn_in("wind_turbine", p)
    for seed in HELD_OUT:
        expert = tg.make_expert("wind_turbine").host()
        expert.reset()
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, p)
        speed_ok, power_err, targets = [], [], []
        for t in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, _, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"])
            if t >= burn:
                o = np.asarray(obs)
                speed_ok.append(abs(o[0] / float(p.omega_rated_rpm) - 1.0) <= 0.10)
                power_err.append(abs(o[3] - o[4]))
                targets.append(o[4])
        assert np.mean(speed_ok) >= 0.99, f"seed {seed}: {np.mean(speed_ok):.3f}"
        assert np.mean(power_err) < 0.01 * np.mean(targets), f"seed {seed}"
