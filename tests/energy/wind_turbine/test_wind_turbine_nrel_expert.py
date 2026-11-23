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
  ``Kp = 2 I W0 zeta wn / (N (-dP/dtheta(0)))`` and ``Ki = I W0 wn^2 / (N
  (-dP/dtheta(0)))``, ``wn`` 0.6 rad/s and ``zeta`` 0.7, each times the schedule
  ``GK = 1 / (1 + theta / theta_K)``; the integral is saturated at the pitch
  limits; the command moves at most 8 deg/s; it starts bumpless from the
  measured pitch.
* ``dP/dtheta`` is the sensitivity with the wake frozen, as S1 linearises it
  (pp. 22-25) and as the 0.6 rad/s loop meets it on this plant, whose induced
  velocity lags a pitch move by Oye's 3 to 9 s (PHYSICS.md sec. 2). With that
  lag switched off the plant has only its steady table, and the recipe reads it.
* Ours, for a curtailed task: ``P`` is the dispatched target over the
  generator efficiency (Jonkman's 5.296610 MW is 5 MW / 0.944), and the
  below-rated torque never exceeds region 3's.
* Ours (docs/proposals/expert-protocols-process.md, R10): ``GK`` reads the last
  pitch command, held inside the region-3 curve's pitch range. Derived (R11):
  the integral is in pitch units and adds ``dt GK Ki e``, as ROSCO's does.
"""

import importlib
import json
import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.energy.wind_turbine.env import (
    WindTurbineParams,
    aerodynamic_torque,
    compute_velocity,
    drivetrain_inertia,
    induction,
)
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
    assert params.theta_K_deg == pytest.approx(m["theta_K_deg"], rel=1e-9)


def _gk(theta_deg, params):
    return 1.0 / (1.0 + theta_deg / params.theta_K_deg)


def _rated_power_pitch(v, p=WP):
    """The pitch (deg) that leaves rated mechanical power at rated speed in wind v."""
    from scipy.optimize import brentq

    w0 = float(p.omega_rated_rpm) * RPM
    P0 = float(p.P_rated) / ETA
    return brentq(lambda t: float(aerodynamic_torque(w0, v, t, p)[1]) - P0, 0.0, 30.0)


def _power_the_rotor_takes(v, theta, wake_of, p=WP):
    """Aerodynamic power (W) at rated speed and pitch ``theta`` while the wake is
    still the one in equilibrium with pitch ``wake_of``: read off the plant's own
    right-hand side with the generator unloaded, power = J dw/dt w."""
    w0 = float(p.omega_rated_rpm) * RPM
    w_induced, _ = induction(w0, v, wake_of, p)
    state = jnp.stack([w0, theta, 0.0, 0.0, 0.0, w_induced]).astype(jnp.float32)
    domega = compute_velocity(state, (0.0, 0.0), v, p)[0][0]
    return float(domega) * float(drivetrain_inertia(p)) * w0


def test_the_schedule_reads_the_sensitivity_the_speed_loop_sees():
    """Pole placement needs dP/dtheta at the pitch it flies, as the loop meets
    it: with the wake frozen. From the plant's dynamics by a central difference
    at rated speed, not through the fitted line: at 15, 16 and 18 m/s the
    rated-power pitch's wake-frozen sensitivity is dP/dtheta(0) / GK within
    15 %, the table's own scatter about the line."""
    params = tg.make_expert("wind_turbine").params
    m = json.loads(GAINS_PATH.read_text())["wind_turbine"]["model"]
    for v in (15.0, 16.0, 18.0):
        th = _rated_power_pitch(v)
        d = 1.0  # deg, a table cell either side
        seen = (
            _power_the_rotor_takes(v, th + d, th)
            - _power_the_rotor_takes(v, th - d, th)
        ) / math.radians(2 * d)
        assert m["dP_dtheta"] / _gk(th, params) == pytest.approx(seen, rel=0.15)


def test_without_the_wake_lag_the_recipe_reads_the_steady_table():
    """The recipe follows the plant. With ``dynamic_inflow`` off nothing holds
    the wake through a pitch move, so the curve it fits is the steady table's
    own sensitivity: at 13 and 15 m/s, where the frozen wake would read 30 to
    50 % more, within the table's 15 % scatter."""
    scripts = pathlib.Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    x64 = jax.config.jax_enable_x64
    try:
        recipe = importlib.import_module("wind_turbine_gains")
    finally:  # the script turns x64 on for its own run
        jax.config.update("jax_enable_x64", x64)
        sys.path.remove(str(scripts))
    p = WindTurbineParams(dynamic_inflow=0.0)
    curve = recipe.sensitivity_curve(p)
    w0 = float(p.omega_rated_rpm) * RPM
    for v in (13.0, 15.0):
        th = _rated_power_pitch(v, p)
        d = 1.0
        steady = (
            float(aerodynamic_torque(w0, v, th + d, p)[1])
            - float(aerodynamic_torque(w0, v, th - d, p)[1])
        ) / math.radians(2 * d)
        fitted_on = np.interp(th, curve[:, 0], curve[:, 1])
        assert fitted_on == pytest.approx(steady, rel=0.15)


def test_a_speed_step_reaches_the_pitch_through_the_quarter_hertz_filter():
    """Bumpless at rated speed and 10 deg, then 1 % overspeed: the filter
    passes (1 - alpha) of it, and the PI moves the pitch by
    GK(10 deg) (Kp + Ki Ts) (1 - alpha) de."""
    params = tg.make_expert("wind_turbine").params
    expert = _expert()
    tq = 4.0e6 / ETA / (RATED_GEN_RPM * RPM)
    assert _pitch(expert(_obs(RATED_GEN_RPM, 10.0, tq))) == pytest.approx(
        10.0, abs=1e-4
    )
    alpha = math.exp(-2.0 * math.pi * TS * 0.25)
    de = 0.01 * RATED_GEN_RPM * RPM * (1.0 - alpha)
    moved = math.degrees(_gk(10.0, params) * (params.Kp + params.Ki * TS) * de)
    a = expert(_obs(1.01 * RATED_GEN_RPM, 10.0, tq))
    assert _pitch(a) - 10.0 == pytest.approx(moved, rel=1e-3)


def test_the_gain_falls_as_the_blade_feathers():
    """The same speed error at 20 deg moves the pitch GK(20) / GK(10) as far,
    about half here, where the sensitivity has doubled."""
    params = tg.make_expert("wind_turbine").params
    tq = 4.0e6 / ETA / (RATED_GEN_RPM * RPM)

    def first_move(pitch):
        expert = _expert()
        expert(_obs(RATED_GEN_RPM, pitch, tq))
        return _pitch(expert(_obs(1.01 * RATED_GEN_RPM, pitch, tq))) - pitch

    ratio = _gk(20.0, params) / _gk(10.0, params)
    assert first_move(20.0) / first_move(10.0) == pytest.approx(ratio, rel=1e-3)


@pytest.mark.parametrize(
    "outside, end", [(0.0, "schedule_lo_deg"), (35.0, "schedule_hi_deg")]
)
def test_the_schedule_holds_at_the_ends_of_the_curve(outside, end):
    """Past either end of the region-3 curve the line says nothing, so the
    gain stays what it was at that end (R10)."""
    params = tg.make_expert("wind_turbine").params
    tq = 4.0e6 / ETA / (RATED_GEN_RPM * RPM)
    expert = _expert()
    expert(_obs(RATED_GEN_RPM, outside, tq))
    alpha = math.exp(-2.0 * math.pi * TS * 0.25)
    de = 0.001 * RATED_GEN_RPM * RPM * (1.0 - alpha)
    gk = _gk(getattr(params, end), params)
    moved = math.degrees(gk * (params.Kp + params.Ki * TS) * de)
    a = expert(_obs(1.001 * RATED_GEN_RPM, outside, tq))
    assert _pitch(a) - outside == pytest.approx(moved, rel=1e-3)


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
# within 10 % of rated on at least 99 % of steps and nothing trips. The power
# clause (mean error under 1 % of the target) measured the v4 score's error off
# the rating; v5 scores the shortfall against the wind's power at peak Cp, a
# bound no fixed-speed region 3 meets in a lull, so it has no threshold here.
HELD_OUT = range(200, 210)


@pytest.mark.large
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
        speed_ok = []
        for t in range(int(p.max_steps_in_episode)):
            key, k = jax.random.split(key)
            obs, state, _, _, info = step(k, state, expert(np.asarray(obs)), p)
            assert not bool(info["tripped"])
            if t >= burn:
                o = np.asarray(obs)
                speed_ok.append(abs(o[0] / float(p.omega_rated_rpm) - 1.0) <= 0.10)
        assert np.mean(speed_ok) >= 0.99, f"seed {seed}: {np.mean(speed_ok):.3f}"


# ---------------------------------------------------------------------------
# The second published candidate: S1's controller with S2's peak shaving
# ---------------------------------------------------------------------------
# Contract, from Abbas et al. (2022) [S2 Sect. 5.3] and ROSCO's toolbox (its
# `ps_percent`, default 0.8, and wind filter, 0.20944 rad/s):
#
# * Peak shaving is a minimum pitch scheduled on wind speed, so that the
#   steady rotor thrust stays under `ps_percent` of its peak over the
#   operating curve. The schedule comes from the Ct table alone: at each wind,
#   the smallest pitch at the operating tip-speed ratio whose thrust is under
#   the cap.
# * The minimum pitch is the lower saturation of the pitch loop, integral
#   included, read at the low-passed wind.
# * Ours: the wind is the anemometer channel (ROSCO has its own estimator), and
#   S1's switch to its constant-power torque law, 1 deg of pitch, is measured
#   from the shaved minimum; on the pitch alone the rotor would be held at
#   rated power below rated wind, and stall.
# * With no schedule it is S1's controller.

SHAVE = 0.8


def _obs7(gen_rpm, pitch, wind, torque_gen=0.0, target_MW=5.0):
    return np.array(
        [
            gen_rpm / N,
            pitch,
            100.0 * torque_gen / float(WP.torque_max),
            0.0,
            target_MW,
            0.0,
            wind,
        ],
        np.float32,
    )


def _shaved():
    from target_gym.energy.wind_turbine.experts import (
        make_wind_turbine_shaved_controller,
    )

    return make_wind_turbine_shaved_controller()


def _operating_thrust(v):
    """Steady thrust (N) on the operating curve at wind ``v``, and the speed."""
    from target_gym.energy.wind_turbine.env import power_coefficient, rotor_thrust

    lam = np.arange(4.0, 12.0, 0.05)
    best = float(lam[int(np.argmax(np.asarray(power_coefficient(lam, 0.0, WP))))])
    w0 = float(WP.omega_rated_rpm) * RPM
    speed = min(w0, best * v / float(WP.R))
    P0 = float(WP.P_rated) / ETA
    pitch = 0.0
    if float(aerodynamic_torque(speed, v, 0.0, WP)[1]) > P0:
        pitch = _rated_power_pitch(v)
    return float(rotor_thrust(speed, v, pitch, WP)), speed


def test_the_shaving_schedule_holds_the_steady_thrust_under_its_cap():
    """At every wind of the schedule the thrust at the scheduled minimum pitch is
    under 0.8 of the operating curve's peak, and where the minimum is above 0 it
    is the smallest such pitch: half a degree less is over the cap. 1 %: the
    schedule's pitch is found to 0.05 deg, about 4 kN of thrust."""
    from target_gym.energy.wind_turbine.env import rotor_thrust

    params = _shaved().params
    winds = np.asarray(params.shave_wind, float)
    floors = np.asarray(params.shave_pitch_deg, float)
    assert winds[0] <= 4.0 and winds[-1] >= 25.0 and len(winds) == len(floors)
    peak = max(_operating_thrust(v)[0] for v in np.arange(8.0, 14.01, 0.1))
    cap = SHAVE * peak
    assert floors.max() > 2.0  # it does shave: several degrees near rated wind
    for v, floor in zip(winds, floors):
        speed = _operating_thrust(v)[1]
        assert float(rotor_thrust(speed, v, floor, WP)) <= cap * 1.01
        if floor > 0.5:
            assert float(rotor_thrust(speed, v, floor - 0.5, WP)) > cap


def _floor_at(wind):
    params = _shaved().params
    return float(np.interp(wind, params.shave_wind, params.shave_pitch_deg))


def test_the_shaved_pitch_never_goes_under_the_scheduled_minimum():
    """At rated wind and 10 % underspeed S1's loop asks for fine pitch; the
    shaved one holds the minimum scheduled at that wind."""
    floor = _floor_at(11.4)
    expert = _shaved().host()
    expert.reset()
    for _ in range(40):  # 10 s: the command settles on the minimum
        a = expert(_obs7(0.9 * RATED_GEN_RPM, floor, 11.4))
    assert _pitch(a) == pytest.approx(floor, abs=1e-3)


def test_the_shaved_integral_waits_at_the_minimum():
    """Long underspeed does not wind the integral under the minimum, so the pitch
    leaves it as soon as the filtered speed passes rated. From 0.90 of rated
    speed the 0.25 Hz filter passes rated on the fifth reading of 1.02; by the
    eighth, 2 s in, the error is 1.5 % of rated speed and the proportional term
    alone is 0.9 deg over the minimum. An integral wound down to 0 would still
    be climbing the 2.9 deg back to it."""
    floor = _floor_at(11.4)
    expert = _shaved().host()
    expert.reset()
    for _ in range(100):
        expert(_obs7(0.9 * RATED_GEN_RPM, floor, 11.4))
    for _ in range(8):
        a = expert(_obs7(1.02 * RATED_GEN_RPM, floor, 11.4))
    assert _pitch(a) > floor + 0.3


def test_the_minimum_follows_the_wind_through_rosco_s_filter():
    """A first-order low-pass at 0.20944 rad/s: after a step of the anemometer
    from 8 m/s (no shaving) to rated wind, held 4.775 s (one time constant, 19
    steps of 0.25 s is 4.75 s), the scheduled wind has moved 1 - exp(-4.75 x
    0.20944) = 63.0 % of the way."""
    params = _shaved().params
    expert = _shaved().host()
    expert.reset()
    expert(_obs7(0.9 * RATED_GEN_RPM, 0.0, 8.0))
    for _ in range(19):
        a = expert(_obs7(0.9 * RATED_GEN_RPM, 0.0, 11.4))
    seen = 8.0 + (11.4 - 8.0) * (1.0 - math.exp(-19 * TS * 0.20944))
    expected = float(np.interp(seen, params.shave_wind, params.shave_pitch_deg))
    # The command climbs to the minimum at 8 deg/s from 0 deg: well within 4.75 s.
    assert _pitch(a) == pytest.approx(expected, abs=0.02)


def test_below_rated_speed_the_shaved_rotor_keeps_s1s_below_rated_torque():
    """At rated wind the minimum pitch is several degrees, over S1's 1 deg
    switch. Read from the minimum, the switch stays off at 90 % of rated speed,
    and the torque is S1's 10 % slip line or its optimal curve, whichever is
    higher, far under the constant-power torque that would stall the rotor."""
    floor = _floor_at(11.4)
    assert floor > 1.0
    gen = 0.9 * RATED_GEN_RPM
    expert = _shaved().host()
    expert.reset()
    tq0 = 30_000.0
    for _ in range(40):
        a = expert(_obs7(gen, floor, 11.4, torque_gen=tq0))
        tq0 = _torque(a)
    w = gen * RPM
    rated3 = 1161.963 * RPM
    sync = rated3 / 1.10
    power = 5.0e6 / ETA
    line = (power / rated3) / (rated3 - sync) * (w - sync)
    curve = K2 * gen**2
    assert _torque(a) == pytest.approx(max(line, curve), rel=1e-3)
    assert _torque(a) < 0.8 * power / w


def test_without_a_schedule_the_shaved_controller_is_s1s():
    """The same step function with an empty schedule gives S1's actions, on
    whatever it reads."""
    from target_gym.energy.wind_turbine.experts import (
        NRELBaselineState,
        nrel_baseline_step,
    )

    plain = tg.make_expert("wind_turbine").params
    emptied = _shaved().params.replace(shave_wind=(), shave_pitch_deg=())
    rng = np.random.default_rng(0)
    s_plain, s_empty = NRELBaselineState(), NRELBaselineState()
    for _ in range(50):
        obs = _obs7(
            RATED_GEN_RPM * rng.uniform(0.8, 1.15),
            rng.uniform(0.0, 20.0),
            rng.uniform(8.0, 25.0),
            torque_gen=rng.uniform(1e4, 4.5e4),
        )
        a, s_plain = nrel_baseline_step(plain, s_plain, obs, np)
        b, s_empty = nrel_baseline_step(emptied, s_empty, obs, np)
        np.testing.assert_array_equal(a, b)
