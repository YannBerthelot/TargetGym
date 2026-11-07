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
-25.52e6 W/rad at 0 deg with theta_K 6.3 deg. On ROSCO's steady table the line
runs almost through the origin, -0.07 MW/rad and 0.015 deg, so ``Kp`` and ``Ki``
at 0 deg are huge and mean nothing alone: only ``GK`` times them is flown, read
at the last pitch command. The line is fitted on the region-3 curve and says
nothing off it, so the expert holds the schedule between the curve's first and
last pitch (docs/proposals/expert-protocols-process.md, section 10).

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
)
from target_gym.identification import rga  # noqa: E402
from target_gym.provenance import GAIN_LEVELS, GAINS_PATH, gains_path  # noqa: E402

ZETA, WN = 0.7, 0.6
#: The pairing's operating points, dispatch in W and mean wind in m/s.
POWERS, WINDS = (3.5e6, 4.25e6, 5.0e6), (13.5, 16.75, 20.0)


def sensitivity_curve(p):
    """``(theta_deg, dP/dtheta in W/rad)`` along the region-3 operating curve."""
    w0 = float(p.omega_rated_rpm) * 2 * np.pi / 60
    P0 = float(p.P_rated) / float(p.eta_gen)  # mechanical, Jonkman's 5.296610 MW

    def power(v, th):
        return float(aerodynamic_torque(w0, v, th, p)[1])

    dP = jax.grad(lambda th, v: aerodynamic_torque(w0, v, th, p)[1])
    rows = []
    for v in np.arange(12.0, 25.01, 0.5):
        if power(v, 0.0) < P0:
            continue
        th = brentq(lambda t: power(v, t) - P0, 0.0, 45.0)
        rows.append((th, float(dP(th, v)) * 180.0 / np.pi))
    return np.array(rows)


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
    b = (tau(w0, theta + pitch_step) - tau(w0, theta - pitch_step)) / (
        2 * pitch_step
    )
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
        f"best-fit line: dP/dtheta(0) {intercept / 1e6:.3f} MW/rad, theta_K {theta_K:.0f} deg"
    )
    print("lambda_11, pitch on speed, by dispatch (MW) and wind (m/s):")
    for P_e in POWERS:
        row = "  ".join(f"{v}: {pairing(p, P_e, v):.3f}" for v in WINDS)
        print(f"  {P_e / 1e6:.2f}: {row}")
    return {
        "Kp": 2 * I * W0 * ZETA * WN / den,
        "Ki": I * W0 * WN**2 / den,
        "theta_K_deg": float(theta_K),
        "schedule_lo_deg": float(curve[0, 0]),
        "schedule_hi_deg": float(curve[-1, 0]),
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
            "zeta 0.7) on the best-fit dP/dtheta of the Cp table along the "
            "region-3 curve, with his schedule read at the last pitch command "
            "and held on that curve (scripts/wind_turbine_gains.py). Generator "
            "speed in rad/s to pitch in rad."
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
