"""Behaviour tests for the wind turbine against the real plant (charter step 3).

Each test is one of the twelve proposed tests in ``COLD_REVIEW.md``. Its
source, the quantity and the tolerance are the audit's, written before any
run. The sources are in ``REALITY.md``: S1 is Jonkman et al.'s NREL 5 MW
definition, S4 the GE overspeed patent, S5 Duckwitz and Shan's tower damping
field test, S9 ROSCO's examples, S13 IEC 61400-1 as quoted by pyiecwind.

The closed-loop tests run S1's baseline controller with its published gains
(KP(0) 0.01882681 s, KI(0) 0.008068634, theta_K 6.302336 deg) as a fixed
recipe, never refit, through the plant's own step (``compute_next_state``) and
its true observation (``get_obs``), so they read the plant and the published
controller, not the task's expert gains.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.wind_turbine import env as W
from target_gym.energy.wind_turbine.experts import (
    NRELBaselineParams,
    NRELBaselineState,
    nrel_baseline_step,
)

DT = 0.25
RPM = 2.0 * math.pi / 60.0
RATED_GEN_RPM = 1173.7  # S1 Table 7-2
RATED_ROTOR = RATED_GEN_RPM / 97.0 * RPM  # rad/s


def _params(**kw):
    return W.WindTurbineParams(sensor_noise_scale=0.0, **kw)


def _quiet(v, **kw):
    """Steady wind ``v``: no turbulence (Iref 0), the mean fixed at ``v``."""
    return _params(turbulence_iref=0.0, v_mean_range=(v, v), **kw)


def _published(p):
    """S1's baseline controller with its published gains (Table 7-2)."""
    return NRELBaselineParams(
        Kp=0.01882681,
        Ki=0.008068634,
        theta_K_deg=6.302336,
        schedule_lo_deg=0.0,
        schedule_hi_deg=90.0,
        N_gear=float(p.N_gear),
        eta_gen=float(p.eta_gen),
        torque_max_plant=float(p.torque_max),
        pitch_max_deg=float(p.pitch_max),
        dt=float(p.delta_t),
    )


def _reset(p, seed=0):
    from target_gym.energy.wind_turbine.env_jax import WindTurbine

    return WindTurbine().reset_env(jax.random.PRNGKey(seed), p)[1]


def _closed_loop(p, state, steps, wind_step=None, key=0):
    """Rotor speed (rad/s), electrical power (W), generator torque (N m),
    pitch (deg) and the observation, one row per step, under S1's controller.
    ``wind_step`` = (step, dv) raises the mean wind by dv at that step."""
    cp = _published(p)

    def body(carry, k):
        s, c = carry
        if wind_step is not None:
            s = s.replace(
                v_mean=jnp.where(k == wind_step[0], s.v_mean + wind_step[1], s.v_mean)
            )
        obs = W.get_obs(s, p)
        a, c = nrel_baseline_step(cp, c, obs)
        s, _ = W.compute_next_state(a, s, p, jax.random.PRNGKey(key))
        out = (s.omega, W.electrical_power(s.omega, s.torque, p), s.torque, s.pitch)
        return (s, c), out

    _, out = jax.lax.scan(body, (state, NRELBaselineState()), jnp.arange(steps))
    return [np.asarray(o, float) for o in out]


# 1. Rated torque. S1 p. 19 and Table 7-2: 43,093.55 N m at 1173.7 rpm, 5 MW
# electrical, 94.4 %. Tolerance 0.01 %: four exact published figures.
def test_rated_generator_torque_is_the_published_figure():
    assert float(W.rated_generator_torque(_params())) == pytest.approx(
        43_093.55, rel=1e-4
    )


# 2. Peak Cp, restated for the family (Amendment 3, rule 19). Pooled over five
# pitch-regulated three-bladed rotors: S1's 0.482 at 7.55; ROSCO's tables for
# the NREL 5 MW (0.466 at 7.5), IEA 15 MW (0.470 at 8.5), BAR 10 (0.454 at 10)
# and NREL 2.8 MW (0.477 at 8.2) [S18 and its siblings at the same commit].
# Range, written before the rerun: peak Cp 0.45 to 0.49, at a tip speed ratio
# of 7 to 10.5.
def test_peak_power_coefficient_at_fine_pitch_is_in_the_familys_range():
    p = _params()
    lam = jnp.linspace(2.0, 14.5, 1251)
    cp = np.asarray(W.power_coefficient(lam, jnp.zeros_like(lam), p))
    assert 0.45 <= cp.max() <= 0.49
    assert 7.0 <= float(lam[cp.argmax()]) <= 10.5


# 3. Region 3 hold with the published controller. S1 Sec. 7: pitch holds rated
# speed and torque holds 5 MW. Zero turbulence, 600 s, mean of the last 60 s.
# Tolerance 1 % on each.
@pytest.mark.medium
@pytest.mark.parametrize("v", [13.0, 16.0, 20.0, 25.0])
def test_region_3_holds_rated_speed_and_power_with_the_published_controller(v):
    p = _quiet(v)
    omega, power, _, _ = _closed_loop(p, _reset(p), 2400)
    last = slice(-240, None)
    assert omega[last].mean() == pytest.approx(RATED_ROTOR, rel=0.01)
    assert power[last].mean() == pytest.approx(5.0e6, rel=0.01)


# 4. Closed-loop speed response, restated for the family. Published pitch loops
# are placed at 0.2 rad/s (ROSCO's floating example [S2 Sect. 6.2]) to 0.6 rad/s
# (S1 pp. 22-25), damping 0.7: peak times pi / (w sqrt(1 - 0.49)) of 7.33 s to
# 22.0 s. Range, written before the rerun: S1's gains on the plant give a peak
# 5.5 s (7.33 less the audit's 25 %) to 22 s after a 1 m/s wind step at 15 m/s,
# and an overshoot under 15 %, read as the opposite excursion after the peak
# over the peak excursion.
@pytest.mark.medium
def test_a_wind_step_is_regulated_at_the_published_bandwidth():
    p = _quiet(15.0)
    settle = 1200  # 300 s at 15 m/s before the step
    omega, _, _, _ = _closed_loop(p, _reset(p), settle + 240, wind_step=(settle, 1.0))
    base = omega[settle - 40 : settle].mean()
    after = omega[settle:] - base
    k_peak = int(after.argmax())
    peak_time = (k_peak + 1) * DT
    overshoot = -after[k_peak:].min() / after[k_peak]
    assert 5.5 <= peak_time <= 22.0, peak_time
    assert overshoot < 0.15


# 5. Pitch sensitivity, restated as the family property every gain schedule
# rests on: the power's sensitivity to pitch grows as the blades feather along
# the Region 3 curve (S1 pp. 22-25, Fig. 7-4, where it doubles by 6.3 deg; ROSCO
# schedules every rotor's gains on it [S2 Sect. 3]). Written before the rerun:
# at rated speed, 12 to 25 m/s in 1 m/s steps, |dP/dtheta| from a 0.1 deg step
# never falls from one wind to the next, and at least doubles over the curve.
def test_power_sensitivity_grows_as_the_blades_feather():
    p = _params()
    w = jnp.asarray(RATED_ROTOR)
    thetas, slopes = [], []
    for v in np.arange(12.0, 25.5, 1.0):
        grid = np.arange(0.0, 30.0, 0.01)
        _, power = W.aerodynamic_torque(w, v, jnp.asarray(grid), p)
        theta = grid[int(np.argmax(np.asarray(power) < 5.296610e6))]
        _, p0 = W.aerodynamic_torque(w, v, theta, p)
        _, p1 = W.aerodynamic_torque(w, v, theta + 0.1, p)
        thetas.append(theta)
        slopes.append((float(p1) - float(p0)) / 0.1)
    size = np.abs(np.array(slopes))
    assert np.all(np.diff(size) >= 0.0), size
    assert size[-1] >= 2.0 * size[0], size


# 6. Actuator limits are hard saturations. S1 Table 7-2: pitch rate 8 deg/s,
# torque 47,402.91 N m, torque rate 15,000 N m/s; pitch 0 to 90 deg (S1 p. 26).
# Full-range step commands. No excess at all on the rates and the torque;
# pitch reaches 90 deg.
def test_full_range_commands_meet_the_published_saturations():
    p = _quiet(15.0)
    s = _reset(p)
    s = s.replace(pitch=jnp.asarray(0.0), torque=jnp.asarray(0.0))
    pitch, torque = [float(s.pitch)], [float(s.torque)]
    for _ in range(60):  # 15 s
        s, _ = W.compute_next_state(jnp.array([1.0, 1.0]), s, p, jax.random.PRNGKey(0))
        pitch.append(float(s.pitch))
        torque.append(float(s.torque))
    pitch, torque = np.array(pitch), np.array(torque)
    assert np.diff(pitch).max() <= 8.0 * DT + 1e-4
    assert np.diff(torque).max() <= 15_000.0 * DT + 1e-2
    assert torque.max() <= 47_402.91 + 1e-2
    assert pitch.max() == pytest.approx(90.0, abs=1e-3)


def _spectrum(signal, window=True):
    """Power per frequency bin. A Hann window for a stationary record; none for
    a transient that starts the record, which a Hann window would erase."""
    x = np.asarray(signal, float) - np.mean(signal)
    taper = np.hanning(len(x)) if window else np.ones(len(x))
    power = np.abs(np.fft.rfft(x * taper)) ** 2
    return np.fft.rfftfreq(len(x), DT), power


# 7. Tower fore-aft mode. S1 Table 9-1: 0.3240 Hz fore-aft. Zero turbulence,
# 15 m/s, constant pitch and torque at balance, a 1 deg pitch step; the peak of
# the tower-top acceleration spectrum (observation channel 5, the nacelle
# accelerometer) over 600 s. Tolerance 0.02 Hz.
@pytest.mark.medium
def test_a_pitch_step_rings_the_tower_at_its_first_fore_aft_frequency():
    p = _quiet(15.0)
    s = _reset(p)
    s = s.replace(pitch=s.pitch + 1.0)
    held = jnp.stack(
        [2.0 * s.pitch / p.pitch_max - 1.0, 2.0 * s.torque / p.torque_max - 1.0]
    )

    def body(s, _):
        s, _ = W.compute_next_state(held, s, p, jax.random.PRNGKey(0))
        return s, W.get_obs(s, p)[5]

    _, accel = jax.lax.scan(body, s, None, length=2400)
    f, power = _spectrum(accel, window=False)
    assert float(f[1:][power[1:].argmax()]) == pytest.approx(0.3240, abs=0.02)


# 8. Tower damping, restated as the family's property (S5 Sect. 3.3 and 4; CART
# damps fore-aft with pitch too [S3]): a tower-velocity loop on collective pitch
# takes power out of the tower band with little pitch, and more as its gain
# rises. S5's own figure ("fully suppressed") is for its hand-tuned gain, which
# it does not print. Test 3's recipe in turbulence at 15 m/s, same seed, with no
# loop and with loops adding 5 % and 10 % of critical damping through the
# plant's thrust sensitivity to pitch (ours). No 3p notch: the plant has no 3p.
# Written before the rerun: power in 0.30 to 0.35 Hz falls with each step of
# gain, by at least half at 10 %, with RMS pitch modulation in that band under
# 0.5 deg [S5].
@pytest.mark.medium
def test_a_tower_velocity_loop_takes_power_out_of_the_tower_band():
    p = _params(v_mean_range=(15.0, 15.0))
    w_t = 2.0 * math.pi * float(p.tower_frequency_hz)
    s0 = _reset(p, seed=3)
    v = float(s0.v_mean)
    dF = float(W.rotor_thrust(RATED_ROTOR, v, 10.1, p)) - float(
        W.rotor_thrust(RATED_ROTOR, v, 10.0, p)
    )
    dF_dtheta = dF / math.radians(0.1)  # N per rad
    critical = 2.0 * float(p.tower_modal_mass) * w_t
    per_unit = critical / -dF_dtheta  # rad of pitch per m/s, per unit damping
    cp = _published(p)
    w1 = w_t / 3.0  # the leaky integral's corner, below the tower frequency

    def run(k_td):
        def body(carry, _):
            s, c, vel = carry
            obs = W.get_obs(s, p)
            a, c = nrel_baseline_step(cp, c, obs)
            vel = vel + DT * (obs[5] - w1 * vel)
            extra = k_td * vel
            pitch = 0.5 * (a[0] + 1.0) * p.pitch_max + jnp.degrees(extra)
            a = a.at[0].set(2.0 * pitch / p.pitch_max - 1.0)
            s, _ = W.compute_next_state(a, s, p, jax.random.PRNGKey(7))
            return (s, c, vel), (obs[5], jnp.degrees(extra))

        _, out = jax.lax.scan(
            body, (s0, NRELBaselineState(), jnp.asarray(0.0)), None, length=2400
        )
        return [np.asarray(o, float)[400:] for o in out]

    powers = []
    for zeta in (0.0, 0.05, 0.10):
        accel, extra = run(zeta * per_unit)
        f, pw = _spectrum(accel)
        powers.append(pw[(f >= 0.30) & (f <= 0.35)].sum())
    fe, pe = _spectrum(extra, window=False)  # Parseval needs no taper
    in_band = np.sqrt(2.0 * pe[(fe >= 0.30) & (fe <= 0.35)].sum()) / len(extra)
    assert powers[0] > powers[1] > powers[2], powers
    assert powers[2] <= 0.5 * powers[0] and in_band < 0.5, (powers, in_band)


def _wind(p, seconds, key, mean, hold=True):
    s = _reset(p).replace(v_mean=jnp.asarray(mean), v_wind=jnp.asarray(mean))

    def body(s, _):
        a = jnp.stack(
            [2.0 * s.pitch / p.pitch_max - 1.0, 2.0 * s.torque / p.torque_max - 1.0]
        )
        s, _ = W.compute_next_state(a, s, p, key)
        return s, s.v_wind

    return jax.lax.scan(body, s, None, length=int(seconds / DT))[1]


# 9. Turbulence grows with wind. S13: sigma1 = Iref (0.75 Vhub + 5.6). Twenty
# 10 min records at means 13.5 and 20 m/s; the ratio of the standard
# deviations, expected 20.6 / 15.725 = 1.31. Tolerance 10 %.
@pytest.mark.medium
def test_turbulence_grows_with_the_mean_wind_as_iec_scales_it():
    stds = {}
    for mean in (13.5, 20.0):
        p = _params(v_mean_range=(mean, mean))
        records = [
            np.asarray(_wind(p, 600.0, jax.random.PRNGKey(seed), mean))
            for seed in range(20)
        ]
        stds[mean] = np.sqrt(np.mean([r.var() for r in records]))
    assert stds[20.0] / stds[13.5] == pytest.approx(1.31, rel=0.10)


# 10. Extreme operating gust. S13: magnitude min(1.35 (Ve1 - Vhub), 3.3 sigma1
# / (1 + 0.1 D / 42 m)), Ve1 = 0.8 x 1.4 Vref, over T = 10.5 s, shape Vhub -
# 0.37 Vgust sin(3 pi t / T)(1 - cos(2 pi t / T)). At 15 m/s with D = 126 m and
# Iref 0.14: sigma1 = 0.14 (0.75 x 15 + 5.6) = 2.359 m/s, so Vgust = 3.3 x
# 2.359 / 1.3 = 5.988 m/s (the 1.35 (56 - 15) = 55.4 m/s branch is far larger).
# The shape peaks at t = T / 2 at 0.37 x 2 x Vgust = 4.431 m/s. Tolerance 1 %
# on amplitude, duration exact to one step. The gust is the trace with the gust
# less the same seed's trace without it.
def test_the_extreme_operating_gust_has_the_iec_amplitude_and_duration():
    base = dict(v_mean_range=(15.0, 15.0), turbulence_iref=0.14)
    key = jax.random.PRNGKey(5)
    calm = np.asarray(_wind(_params(**base), 60.0, key, 15.0))
    gusty = _params(gust=1.0, gust_window=(20.0, 20.0), **base)
    gust = np.asarray(_wind(gusty, 60.0, key, 15.0))
    shape = gust - calm
    assert shape.max() == pytest.approx(4.431, rel=0.01)
    moving = np.flatnonzero(np.abs(shape) > 1e-6)
    duration = (moving[-1] - moving[0] + 2) * DT
    assert duration == pytest.approx(10.5, abs=DT)


# 11. Overspeed safety system feathers. S4: above a stored permissible speed
# the pitch system feathers independently of the main controller. 20 m/s, a
# grid loss (torque command 0), pitch command held at balance. Tolerance: once
# the speed passes the level, pitch rises at 8 deg/s (within one step) to
# 90 deg, and speed falls monotonically after feather starts.
def test_overspeed_feathers_the_blades_whatever_the_controller_sends():
    p = _quiet(20.0)
    s = _reset(p)
    hold = 2.0 * s.pitch / p.pitch_max - 1.0
    action = jnp.stack([hold, jnp.asarray(-1.0)])
    omega, pitch = [], []
    for _ in range(240):  # 60 s
        s, _ = W.compute_next_state(action, s, p, jax.random.PRNGKey(0))
        omega.append(float(s.omega))
        pitch.append(float(s.pitch))
    omega, pitch = np.array(omega), np.array(pitch)
    level = float(p.overspeed_factor) * float(W.omega_rated(p))
    k = int(np.argmax(omega >= level))
    assert omega[k] >= level
    rises = np.diff(pitch[k:])
    start = int(np.argmax(rises > 0.0))
    assert start <= 1
    climbing = rises[start:][pitch[k + start + 1 :] < 90.0 - 1e-6]
    np.testing.assert_allclose(climbing[:-1], 8.0 * DT, rtol=1e-3)
    assert pitch[-1] == pytest.approx(90.0, abs=1e-3)
    assert np.all(np.diff(omega[k + start :]) <= 0.0)


# 12. A de-rate is a reference change. S9 Ex. 29: a power rating R sets the
# references; S11 Fig. ES-8: a de-rate step. Rating 1.0 to 0.9 at 15 m/s, zero
# turbulence. Power within 1 % of 4.5 MW; speed and torque at the rating's
# references within 1 %. Assumed (ours, S9 gives no formula in REALITY): the
# speed reference stays rated and the torque reference is 4.5 MW over the
# efficiency at rated speed.
@pytest.mark.medium
def test_a_derating_lands_on_the_ratings_references():
    p = _quiet(15.0)
    s = _reset(p)
    p9 = p.replace(rating=0.9)
    s = s.replace(target_power=jnp.asarray(4.5e6))
    omega, power, torque, _ = _closed_loop(p9, s, 2400)
    last = slice(-240, None)
    assert power[last].mean() == pytest.approx(4.5e6, rel=0.01)
    assert omega[last].mean() == pytest.approx(RATED_ROTOR, rel=0.01)
    torque_ref = 4.5e6 / (0.944 * RATED_GEN_RPM * RPM)
    assert torque[last].mean() == pytest.approx(torque_ref, rel=0.01)
