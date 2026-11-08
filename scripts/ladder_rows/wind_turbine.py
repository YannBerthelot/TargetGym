"""The reference ladder's row for ``wind_turbine`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score. The
plant is version 5 (energy/wind_turbine/PHYSICS.md): observation ``[omega_rpm,
pitch_deg, torque_pct, P_MW, rating_P_MW, a_fa, v_nacelle]``, action ``[pitch,
torque]`` raw over ``[pitch_min, pitch_max]`` deg and ``[0, torque_max]`` N m.
"""

from __future__ import annotations

import math

import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.energy.wind_turbine.env import (
        aerodynamic_torque,
        balance_pitch,
        drivetrain_inertia,
        induction,
        omega_rated,
    )

    spec = REGISTRY["wind_turbine"]
    p = spec.make_test_params()
    rated = float(omega_rated(p))  # rad/s
    rpm = float(p.omega_rated_rpm)
    eta, N = float(p.eta_gen), float(p.N_gear)
    tau_max, pitch_max = float(p.torque_max), float(p.pitch_max)
    P_rated = float(p.P_rated)

    def pitch_raw(pitch_deg):
        return 2.0 * pitch_deg / pitch_max - 1.0

    def torque_raw(tau):
        return 2.0 * tau / tau_max - 1.0

    def held(q):
        # The reset's own operating point (env_jax.reset_env): rated speed, the
        # rating's torque there, and the pitch that balances it in the mean wind.
        v, P = float(q.v_mean_range[0]), float(q.rating) * P_rated
        pitch = float(balance_pitch(rated, v, P / eta, q))
        tau = min(P / (eta * N * rated), tau_max)
        return pitch_raw(pitch), torque_raw(tau)

    def rest_at(q):
        action = np.asarray(held(q), np.float32)
        return lambda o, xp: xp.asarray(action)

    def balance(o, xp):
        # Rung 2, the rotor's steady balances: the pitch the Cp table balances at
        # rated speed for the rating's power in the nacelle anemometer's wind
        # (channel 6), and the torque that makes the rating at the measured speed.
        P = o[4] * 1.0e6
        pitch = balance_pitch(rated, o[6], P / eta, p)
        omega = xp.maximum(o[0] * 2.0 * np.pi / 60.0, 1e-3)
        tau = xp.clip(P / (eta * N * omega), 0.0, tau_max)
        return xp.stack([pitch_raw(pitch), torque_raw(tau)])

    # The test's turbine sits in a steady wind: no turbulence, the mean wind
    # pinned. The base point is the middle of the normal 13 to 25 m/s.
    v0 = 0.5 * sum(p.v_mean_range)
    test = {"turbulence_iref": 0.0, "v_mean_range": (v0, v0)}
    # The ends of the winds and ratings the suite visits (scenarios.py): near
    # rated from 11.4 m/s, near cut-out to 25 m/s, and the curtailment's 0.9.
    points = tuple(
        {"v_mean_range": (v, v), "rating": r} for v in (11.4, 25.0) for r in (0.9, 1.0)
    )

    def lag(q):
        # The rotor's lag under a held torque, J / (-d tau_aero / d omega), from
        # the Cp table, plus Oye's inflow lag: 1.8 to 20 s and 3 to 9 s.
        v, P = float(q.v_mean_range[0]), float(q.rating) * P_rated
        pitch = balance_pitch(rated, v, P / eta, q)
        d = 1e-3 * rated
        hi = aerodynamic_torque(rated + d, v, pitch, q)[0]
        lo = aerodynamic_torque(rated - d, v, pitch, q)[0]
        damping = -float(hi - lo) / (2.0 * d)
        inflow = float(induction(rated, v, pitch, q)[1])
        return float(drivetrain_inertia(q)) / damping + inflow

    at = [spec.make_test_params(**{**test, **change}) for change in ({}, *points)]
    dt = float(p.delta_t)
    speed_window = math.ceil(5.0 * max(lag(q) for q in at) / dt)
    # 0.1 deg of pitch; the test sizes it up to 5 noise bands where the rotor is
    # least sensitive.
    speed = Cv(
        "speed on pitch",
        0,
        lambda o, xp: o[0],
        lambda o, xp: xp.asarray(rpm, dtype=xp.float32),
        "omega",
        2.0 * 0.1 / pitch_max,
        speed_window,
        valve=False,
    )
    # Power on torque is algebraic within a step at the torque's rate (15 kN m/s,
    # S1 Table 7-2) before the rotor's lag, at least 1.8 s, pulls it back: the
    # test reads 2 s. 0.02 raw is 474 N m, about 55 kW.
    grid = Cv(
        "power on torque",
        1,
        lambda o, xp: o[3],
        lambda o, xp: o[4],
        "power",
        0.02,
        8,
        valve=False,
    )

    def published(s):
        # S1's baseline controller, the family's published one (experts.py).
        return s.make_pid_controller()

    return Row(
        480,
        rest_at(at[0]),
        (speed, grid),
        test=test,
        points=points,
        rest_at=rest_at,
        balance=balance,
        decouple=True,
        published=published,
        why={
            "cascade": "S5's tower damper feeds the accelerometer (channel 5) to "
            "pitch in parallel, not as a cascade, and the acceleration's steady gain "
            "on pitch is 0, so the script's inner loop can't hold it; the pitch and "
            "torque reach their commands at their rates, with no lag to cascade over",
            "override": "the one limit is the overspeed trip, instantaneous at 1.20 "
            "of rated: a select on it acts only past the trip, and no source gives "
            "a margin inside it",
        },
    )
