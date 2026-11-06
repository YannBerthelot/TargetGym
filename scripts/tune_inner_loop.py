"""Tune the aircraft's fixed inner loop once, by the accepted manual procedure.

Runs section 3.2 of ``docs/proposals/expert-protocols-aircraft.md`` (ArduPlane
4.0.9, G-MAN Method 2) with its amendment of 2026-11-06: still air, 5500 m, 230 m/s,
with the throttle holding the speed. Roll first, bank demand 0, +45, -45, 0 deg,
30 s each. Then pitch, trim +5, -5, trim deg, 20 s each, wings level. Steps 1 to 4
of the recipe, and the readings of its section 2, as they stand. Steps 5 and 6 set
guidance values and belong to PR 9.

The result ships whatever it is. No scored or held-out seed is flown: the tests
are in still air, so nothing here reads a result. It prints the values typed into
``plane.dynamics.InnerLoop``, one tune for every task, and their tests.

    uv run python scripts/tune_inner_loop.py
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import fsolve

from target_gym.plane.dynamics import (
    STICK_TRAVEL,
    InnerLoop,
    compute_thrust_output,
)
from target_gym.plane3d.dynamics import compute_acceleration_3d
from target_gym.plane3d.env import PlaneParams3D, PlaneState3D, compute_next_state_3d

ALTITUDE, SPEED = 5500.0, 230.0
#: The loop's sub-step, so every test logs what a flight log would, at 4 Hz.
SUBSTEP = 0.25
#: ArduPilot's units to ours: (180 / pi) / 45 (the proposal's section 1).
UNIT = 180.0 / np.pi / 45.0
ROLL_TEST = (0.0, 45.0, -45.0, 0.0)
PITCH_TEST = (0.0, 5.0, -5.0, 0.0)  # about trim
ROLL_HOLD, PITCH_HOLD = 30.0, 20.0
#: The proposal's steps, in ArduPilot units.
P_START, P_STEP, P_MAX = 1.0, 0.1, 5.0
D_START = {"roll": 0.08, "pitch": 0.04}
D_STEP, D_MAX, I_STEP = 0.01, 0.1, 0.05
#: The speed hold's sample, the plant's action rate (the amendment).
SPEED_HOLD_SAMPLE = 1.0
#: Band of the readings: 5% of the step (section 2).
BAND = 0.05


def speed_hold_gains(slope, lag):
    """The speed hold's PI, ``(gain, integral time)``, by the amendment's SIMC rule:
    the lever moves the speed as ``slope / (s (lag s + 1))``, so with half a sample
    of delay and the half rule, theta_e = sample / 2 + lag / 2, and the tight setting
    tau_c = theta_e gives 1 / (2 slope theta_e) and 8 theta_e."""
    delay = 0.5 * SPEED_HOLD_SAMPLE + 0.5 * lag
    return 1.0 / (2.0 * slope * delay), 8.0 * delay


def trim(params):
    """``(pitch, stick, lever, slope)`` of level flight at the test's point, with
    ``slope`` the acceleration a unit of lever gives."""

    def residual(x):
        theta, stick, thrust = float(x[0]), float(x[1]), float(x[2]) * 1e5
        v = jnp.array([SPEED, 0.0, 0.0, 0.0, 0.0])
        p = jnp.array([0.0, 0.0, ALTITUDE, theta, 0.0])
        a, _ = compute_acceleration_3d(v, p, (thrust, stick, 0.0), params)
        return [float(a[0]), float(a[2]), float(a[3])]

    # The plant is float32, so the Jacobian's differences must be well above 1e-7.
    theta, stick, thrust = fsolve(residual, [0.03, -0.09, 0.4], epsfcn=1e-5)
    state = _level(params, theta, stick, 0.0, 0.0)
    unit = compute_thrust_output(
        power=1.0,
        thrust_output_at_sea_level=params.thrust_output_at_sea_level,
        M=SPEED / float(state.speed_of_sound),
        rho=state.rho,
    )
    power = float(thrust * 1e5 / unit)
    idle = params.idle_power
    slope = float(unit) * (1.0 - idle) / params.initial_mass
    return float(theta), float(stick), (power - idle) / (1.0 - idle), slope


def _level(params, theta, stick, power, integ):
    return PlaneState3D(
        x=0.0,
        x_dot=SPEED,
        y=0.0,
        y_dot=0.0,
        z=ALTITUDE,
        z_dot=0.0,
        theta=theta,
        theta_dot=0.0,
        phi=0.0,
        phi_dot=0.0,
        psi=0.0,
        alpha=theta,
        gamma=0.0,
        m=params.initial_mass,
        power=power,
        stick=stick,
        aileron=0.0,
        fuel=params.initial_fuel_quantity,
        target_altitude=ALTITUDE,
        target_heading=0.0,
        target_x=0.0,
        target_y=0.0,
        target_radius=0.0,
        time=0,
        pitch_integ=integ,
    )


def make_flight(params, trimmed):
    """A jitted flight of ``(pitches, banks)`` held at the 4 Hz sub-step, logging
    pitch and bank at each, with the throttle holding the speed at 1 Hz. The
    plant's own step, one sub-step at a time."""
    theta, stick, lever, slope = trimmed
    idle = params.idle_power
    gain, integral_time = speed_hold_gains(slope, 1.0 / params.power_response_rate)
    per_sample = int(SPEED_HOLD_SAMPLE / SUBSTEP)
    power = idle + (1.0 - idle) * lever
    start = _level(params, theta, stick, power, stick / STICK_TRAVEL)
    start = jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float32), start)
    start = start.replace(time=jnp.asarray(0))

    def fly(gains, pitches, banks):
        p = params.replace(inner_loop=gains, delta_t=SUBSTEP)

        def body(carry, demand):
            state, held, integral = carry
            pitch, bank, step = demand
            sample = step % per_sample == 0
            speed = jnp.sqrt(state.x_dot**2 + state.y_dot**2 + state.z_dot**2)
            error = SPEED - speed
            wanted = lever + gain * (error + integral / integral_time)
            clipped = jnp.clip(wanted, 0.0, 1.0)
            grown = jnp.where(wanted == clipped, integral + error, integral)
            held = jnp.where(sample, clipped, held)
            integral = jnp.where(sample, grown, integral)
            state, _ = compute_next_state_3d(held, pitch, bank, state, p, inner_steps=1)
            return (state, held, integral), (state.theta, state.phi, speed)

        steps = jnp.arange(len(pitches))
        carry = (start, jnp.float32(lever), jnp.float32(0.0))
        return jax.lax.scan(body, carry, (pitches, banks, steps))[1]

    return jax.jit(fly)


def schedule(levels, hold):
    return np.repeat(np.asarray(levels, float), int(hold / SUBSTEP))


def excursions(log, target, step):
    """The signs of the error's excursions past the band, in order, with the
    approach first (positive). ``step`` is the move into this segment."""
    err = (target - log) * np.sign(step)
    band = BAND * abs(step)
    signs = [1]
    for e in err:
        if abs(e) > band and np.sign(e) != signs[-1]:
            signs.append(int(np.sign(e)))
    return signs


def reads(log, levels, hold):
    """``(overshoots, oscillates)`` over a test's segments (section 2).

    Overshoots: the peak passes the target by more than 5% of the step.
    Oscillates: after the first peak the error changes sign at least twice, each
    excursion leaving the 5% band. The approach and the first peak are the first
    two excursions, so that is four or more.
    """
    per = int(hold / SUBSTEP)
    over = osc = False
    for i in range(1, len(levels)):
        step = levels[i] - levels[i - 1]
        segment = log[i * per : (i + 1) * per]
        err = (levels[i] - segment) * np.sign(step)
        over |= bool(err.min() < -BAND * abs(step))
        osc |= len(excursions(segment, levels[i], step)) >= 4
    return over, osc


def tune(axis, run, gains):
    """Steps 1 to 4 for one axis, in ArduPilot units. ``run(gains)`` flies the
    test and gives ``(overshoots, oscillates)``. Returns the gains and the log."""
    names = dict(roll=("Kp_roll", "Ki_roll", "Kd_roll"))
    names["pitch"] = ("Kp_theta", "Ki_theta", "Kd_theta")
    kp, ki, kd = names[axis]

    def with_(p, i, d):
        return gains.replace(**{kp: p * UNIT, ki: i * UNIT, kd: d * UNIT})

    log = []

    def test(p, i, d):
        over, osc = run(with_(p, i, d))
        log.append(
            dict(P=round(p, 4), I=round(i, 4), D=round(d, 4), over=over, osc=osc)
        )
        return over, osc

    # 1. Defaults, halving P while the first test oscillates.
    p, d = P_START, D_START[axis]
    while test(p, 0.0, d)[1]:
        p /= 2.0

    # 2. P up until a test overshoots or oscillates, then back one step. Where
    # the defaults already do, one step back does not clear it, so P walks down
    # to the last step that is clean, which is what the step is after (ours).
    if any(test(p, 0.0, d)):
        p -= P_STEP
        while any(test(p, 0.0, d)):
            p -= P_STEP
    else:
        while p + P_STEP <= P_MAX + 1e-9 and not any(test(p + P_STEP, 0.0, d)):
            p += P_STEP

    # 3. D up until a test oscillates, then halve; at the guide's limit it ships
    # there (the amendment), since past it this loop only slows down.
    while True:
        if test(p, 0.0, d)[1]:
            d /= 2.0
            break
        if d + D_STEP > D_MAX + 1e-9:
            break
        d += D_STEP

    # 4. I up from 0 until a test overshoots or oscillates, then halve.
    i = 0.0
    while True:
        i += I_STEP
        if any(test(p, i, d)):
            i /= 2.0
            break
        if i > 20.0:  # guard only: the rule above is what stops it
            raise RuntimeError("no overshoot from I")
    return with_(p, i, d), log


GAINS = ("Kp_theta", "Ki_theta", "Kd_theta", "Kp_roll", "Ki_roll", "Kd_roll")


def tuned_gains(report=None):
    """The recipe, roll first on the pitch loop's defaults: an ``InnerLoop``.
    ``report``, a dict, collects each axis's tests."""
    params = PlaneParams3D(turbulence_sigma=0.0)
    trimmed = trim(params)
    theta0 = trimmed[0]
    fly = make_flight(params, trimmed)

    def demands(pitches, banks):
        return jnp.asarray(pitches, jnp.float32), jnp.asarray(banks, jnp.float32)

    bank_levels = np.deg2rad(ROLL_TEST)
    banks = schedule(bank_levels, ROLL_HOLD)

    def run_roll(gains):
        _, phi, _ = fly(gains, *demands(np.full_like(banks, theta0), banks))
        return reads(np.asarray(phi), bank_levels, ROLL_HOLD)

    pitch_levels = theta0 + np.deg2rad(PITCH_TEST)
    pitches = schedule(pitch_levels, PITCH_HOLD)

    def run_pitch(gains):
        theta, _, _ = fly(gains, *demands(pitches, np.zeros_like(pitches)))
        return reads(np.asarray(theta), pitch_levels, PITCH_HOLD)

    start = InnerLoop(
        Kp_theta=P_START * UNIT,
        Ki_theta=0.0,
        Kd_theta=D_START["pitch"] * UNIT,
        Kp_roll=P_START * UNIT,
        Ki_roll=0.0,
        Kd_roll=D_START["roll"] * UNIT,
    )
    gains, roll_log = tune("roll", run_roll, start)
    gains, pitch_log = tune("pitch", run_pitch, gains)
    if report is not None:
        report.update(trim=trimmed, roll=roll_log, pitch=pitch_log)
    return gains


def main() -> int:
    report = {}
    gains = tuned_gains(report)
    theta, stick, lever, _ = report["trim"]
    print(f"trim: pitch {np.rad2deg(theta):.3f} deg, stick {stick:.4f} rad, ", end="")
    print(f"lever {lever:.3f}")
    print(f"tests: {len(report['roll'])} roll, {len(report['pitch'])} pitch")
    print(json.dumps({k: round(float(getattr(gains, k)), 4) for k in GAINS}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
