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

On the real blade the sensitivity grows about linearly with pitch (Jonkman's
-25.52e6 W/rad at 0 deg, theta_K 6.3 deg). On this plant's analytic Cp fit it
is not monotone: about -66 MW/rad near 1 deg, -4 at 9 deg, -58 at 33 deg. The
best-fit line is flat (theta_K over a thousand degrees), so the schedule is 1
and the gains are fixed at the line's intercept. A schedule read off the exact
sensitivity instead multiplies the gains where the surface is almost flat in
pitch, and trips the turbine.
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
from target_gym.provenance import GAINS_PATH  # noqa: E402

ZETA, WN = 0.7, 0.6


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
    return {
        "Kp": 2 * I * W0 * ZETA * WN / den,
        "Ki": I * W0 * WN**2 / den,
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
            "zeta 0.7) on the best-fit dP/dtheta of this plant's Cp surface "
            "along the region-3 curve; that line is flat, so no gain schedule "
            "(scripts/wind_turbine_gains.py). Generator speed in rad/s to "
            "pitch in rad."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    out = gains(WindTurbineParams())
    print(json.dumps(out, indent=2))
    if args.write:
        stored = json.loads(GAINS_PATH.read_text())
        stored["wind_turbine"] = out
        GAINS_PATH.write_text(json.dumps(stored, indent=2) + "\n")
        print(f"wrote wind_turbine to {GAINS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
