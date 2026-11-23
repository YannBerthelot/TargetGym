"""The wind turbine expert's pitch gains, by Jonkman's recipe on our Cp surface.

Run with ``uv run python scripts/wind_turbine_gains.py [--write]``.

Jonkman, Butterfield, Musial and Scott (2009), NREL/TP-500-38060, section 7.3:
linearise the aerodynamic power's sensitivity to pitch, ``dP/dtheta``, along
the region-3 operating curve (rated speed, the pitch that holds rated
mechanical power at each wind speed), fit it with a line in pitch, and place
the poles of the speed loop at ``wn`` 0.6 rad/s and ``zeta`` 0.7:

    Kp = 2 I W0 zeta wn / (N (-dP/dtheta(0)))
    Ki = I W0 wn^2 / (N (-dP/dtheta(0)))

with ``I`` the drivetrain inertia on the rotor shaft, ``W0`` rated rotor
speed and ``N`` the gearbox ratio; the gains act on generator speed in rad/s
and give pitch in rad. The line's doubling pitch ``theta_K`` sets the gain
schedule ``GK = 1 / (1 + theta / theta_K)``.

The sensitivity grows about linearly with pitch. Jonkman's has the wake frozen,
-25.52e6 W/rad at 0 deg with theta_K 6.3 deg, and so has this one: the plant's
induced velocity lags a pitch move by Oye's 3 to 9 s (``env.compute_velocity``),
far slower than the 0.6 rad/s loop, so the loop meets the tables read with the
induced velocity held where it was (``frozen_wake_power``). On ROSCO's tables
that is within about 20 % of Jonkman's line. With ``dynamic_inflow`` off the
plant has only its steady table, whose line runs almost through the origin
(-0.07 MW/rad, 0.015 deg), and the recipe reads that instead: until 2026-11-23 it
read the steady table whatever the plant did, which gave gains 2.5 times
Jonkman's at 12 m/s. The line is fitted on the region-3 curve and says nothing
off it, so the expert holds the schedule between the curve's first and last
pitch (docs/proposals/expert-protocols-process.md, section 10).

The speed filter and the limits are fixed, and nothing here reads noise, so
``--write`` stores the same values at every noise level.

The pairing is pitch on rotor speed and torque on power. ``pairing`` reads its
RGA from the two balances, with ``a`` and ``b`` the aerodynamic torque's slopes in
speed and pitch: at steady state ``a dw + b dtheta - N dtau = 0`` and ``dP = eta N
(tau dw + w dtau)``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

jax.config.update("jax_enable_x64", True)

from target_gym.energy.wind_turbine.env import (  # noqa: E402
    WindTurbineParams,
    aerodynamic_torque,
    drivetrain_inertia,
    induction,
    power_coefficient,
    rotor_thrust,
)
from target_gym.identification import rga  # noqa: E402
from target_gym.provenance import GAIN_LEVELS, gains_path  # noqa: E402

ZETA, WN = 0.7, 0.6
#: Peak shaving's share of the peak thrust it leaves: ROSCO's ``ps_percent``
#: default (Abbas et al. 2022, sect. 5.3; rosco.readthedocs.io, toolbox input).
SHAVE = 0.8
#: The pairing's operating points, dispatch in W and mean wind in m/s.
POWERS, WINDS = (3.5e6, 4.25e6, 5.0e6), (13.5, 16.75, 20.0)


def frozen_wake_power(p, w0, v, th, held):
    """Aerodynamic power (W) at pitch ``th`` (deg) while the induced velocity is
    still ``held`` (m/s): the tables read at the wind that gives the rotor plane
    its lagging inflow, as ``env.compute_velocity`` reads them. With
    ``dynamic_inflow`` 0 this is the steady table."""
    quasi_steady, _ = induction(w0, v, th, p)
    v_eff = v - p.dynamic_inflow * (held - quasi_steady)
    return aerodynamic_torque(w0, v_eff, th, p)[1]


def sensitivity_curve(p):
    """``(theta_deg, dP/dtheta in W/rad)`` along the region-3 operating curve,
    with the wake frozen at each point's own equilibrium."""
    w0 = float(p.omega_rated_rpm) * 2 * np.pi / 60
    P0 = float(p.P_rated) / float(p.eta_gen)  # mechanical, Jonkman's 5.296610 MW

    def power(v, th):
        return float(aerodynamic_torque(w0, v, th, p)[1])

    dP = jax.grad(lambda th, v, held: frozen_wake_power(p, w0, v, th, held))
    rows = []
    for v in np.arange(12.0, 25.01, 0.5):
        if power(v, 0.0) < P0:
            continue
        th = brentq(lambda t: power(v, t) - P0, 0.0, 45.0)
        held = induction(w0, v, th, p)[0]
        rows.append((th, float(dP(th, v, held)) * 180.0 / np.pi))
    return np.array(rows)


def shaving_schedule(p, percent: float = SHAVE):
    """Peak shaving's minimum pitch against wind, as ROSCO's toolbox computes it
    from the Ct table: ``(winds in m/s, minimum pitch in deg)``.

    The operating curve is the best tip-speed ratio at fine pitch up to rated
    speed, then rated speed with the pitch that leaves rated power. Its largest
    thrust, at rated wind, times ``percent`` is the cap; at each wind the
    minimum is the smallest pitch, at that wind's operating speed, whose thrust
    is under the cap (0 where fine pitch already is)."""
    w0 = float(p.omega_rated_rpm) * 2 * np.pi / 60
    P0 = float(p.P_rated) / float(p.eta_gen)
    lam = np.arange(4.0, 12.0, 0.05)
    best = float(lam[int(np.argmax(np.asarray(power_coefficient(lam, 0.0, p))))])

    def operating(v):
        speed = min(w0, best * v / float(p.R))
        power = lambda th: float(aerodynamic_torque(speed, v, th, p)[1])  # noqa: E731
        pitch = brentq(lambda t: power(t) - P0, 0.0, 45.0) if power(0.0) > P0 else 0.0
        return speed, pitch

    peak = max(
        float(rotor_thrust(*operating(v)[:1], v, operating(v)[1], p))
        for v in np.arange(8.0, 14.01, 0.1)
    )
    grid = np.arange(0.0, 30.0, 0.05)
    winds = np.arange(3.0, 25.01, 0.5)
    floors = []
    for v in winds:
        thrust = np.asarray(rotor_thrust(operating(v)[0], v, grid, p))
        floors.append(float(grid[int(np.argmax(thrust <= percent * peak))]))
    return winds, np.array(floors)


def pairing(p, P_e: float, v: float) -> float:
    """lambda_11, of pitch on rotor speed, at dispatch ``P_e`` and wind ``v``,
    from the speed and power balances at rated speed and the balancing pitch."""
    w0 = float(p.omega_rated_rpm) * 2 * np.pi / 60
    eta, N = float(p.eta_gen), float(p.N_gear)
    torque = P_e / (eta * N * w0)
    balance = lambda th: float(aerodynamic_torque(w0, v, th, p)[0]) - N * torque
    theta = brentq(balance, 0.0, 30.0)
    wind_step = 1e-4 * w0
    pitch_step = 0.5  # deg, half a table cell

    def tau(w, th):
        return float(aerodynamic_torque(w, v, th, p)[0])

    a = (tau(w0 + wind_step, theta) - tau(w0 - wind_step, theta)) / (2 * wind_step)
    b = (tau(w0, theta + pitch_step) - tau(w0, theta - pitch_step)) / (2 * pitch_step)
    G = np.array(
        [
            [-b / a, N / a],
            [-eta * N * torque * b / a, eta * N * (torque * N / a + w0)],
        ]
    )
    return float(rga(G)[0, 0])


def gains(p) -> dict:
    curve = sensitivity_curve(p)
    slope, intercept = np.polyfit(np.radians(curve[:, 0]), curve[:, 1], 1)
    I = float(drivetrain_inertia(p))
    W0 = float(p.omega_rated_rpm) * 2 * np.pi / 60
    den = float(p.N_gear) * -intercept
    print(
        "dP/dtheta along the curve (deg: MW/rad): "
        + " ".join(f"{t:.1f}:{s / 1e6:.1f}" for t, s in curve[::3])
    )
    theta_K = np.degrees(intercept / slope)
    print(
        f"best-fit line: dP/dtheta(0) {intercept / 1e6:.3f} MW/rad, theta_K {theta_K:.2f} deg"
    )
    print("lambda_11, pitch on speed, by dispatch (MW) and wind (m/s):")
    for P_e in POWERS:
        row = "  ".join(f"{v}: {pairing(p, P_e, v):.3f}" for v in WINDS)
        print(f"  {P_e / 1e6:.2f}: {row}")
    shave_wind, shave_pitch = shaving_schedule(p)
    print(
        "peak shaving's minimum pitch (m/s: deg): "
        + " ".join(
            f"{v:.0f}:{b:.2f}" for v, b in zip(shave_wind[::4], shave_pitch[::4])
        )
    )
    return {
        "Kp": 2 * I * W0 * ZETA * WN / den,
        "Ki": I * W0 * WN**2 / den,
        "theta_K_deg": float(theta_K),
        "schedule_lo_deg": float(curve[0, 0]),
        "schedule_hi_deg": float(curve[-1, 0]),
        "shave_percent": SHAVE,
        "shave_wind": [round(float(v), 2) for v in shave_wind],
        "shave_pitch_deg": [round(float(b), 2) for b in shave_pitch],
        "model": {
            "dP_dtheta": float(intercept),
            "theta_K_deg": float(theta_K),
            "zeta": ZETA,
            "wn": WN,
        },
        "tuning_rule": "jonkman_pole_placement",
        "note": (
            "NREL 5 MW baseline (Jonkman et al. 2009, ch. 7): pitch PI on the "
            "filtered generator speed, Kp and Ki by pole placement (wn 0.6, "
            "zeta 0.7) on the best-fit dP/dtheta along the region-3 curve, "
            "read from the Cp and Ct tables with the wake frozen as the plant's "
            "inflow lag holds it, with his schedule read at the last pitch "
            "command and held on that curve (scripts/wind_turbine_gains.py). "
            "Generator speed in rad/s to pitch in rad."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    out = gains(WindTurbineParams())
    print(json.dumps(out, indent=2))
    if args.write:
        for level in GAIN_LEVELS:
            path = gains_path(level)
            stored = json.loads(path.read_text())
            stored["wind_turbine"] = out
            path.write_text(json.dumps(stored, indent=2) + "\n")
            print(f"wrote wind_turbine to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
