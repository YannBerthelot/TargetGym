"""The reference ladder's row for ``compressor_surge`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score
(PHYSICS.md, version 4, "What the reference ladder row needs"). The header pressure
is a two-sided zone, 24 +- 1.2 kPa, with no cheap side of its own, so its loop on
the drive speed holds the zone's centre. The cost lever is the recycle, cheapest
shut: it rests where the steady balance keeps the machine on its control line, and
the anti-surge override opens it as the margin nears that line.
"""

from __future__ import annotations

import math

import numpy as np
from reference_ladder import (  # noqa: E402
    Cv,
    Limit,
    Row,
)

from target_gym.registry import REGISTRY

#: The anti-surge control line's margin over the surge line, in flow at speed:
#: about 10 % [C6], the low end of C1's 10 to 15 %.
CONTROL_MARGIN = 0.10


def row() -> Row:
    from target_gym.compressor_surge import experts as X
    from target_gym.compressor_surge.env import (
        characteristic,
        check_valve_sqrt,
        suction_density,
    )

    spec = REGISTRY["compressor_surge"]
    p = spec.make_test_params()
    sp, band = float(p.header_setpoint), float(p.header_band)
    rho = float(suction_density(p))
    A, U_r = float(p.A_c), float(p.U_r)

    def head(dp, xp):
        """The consumers' valve term at header pressure ``dp`` (Pa)."""
        return check_valve_sqrt(dp - p.dp_out, p, xp)

    # The control line at the setpoint: Phi = 2W (1 + margin) on the map, at the
    # speed whose head there is the setpoint. Along the setpoint's isobar the
    # compressor flow rises with Phi, so a flow above this one keeps the margin.
    phi_line = 2.0 * float(p.W) * (1.0 + CONTROL_MARGIN)
    psi_line = float(characteristic(phi_line, p, np))
    m_line = rho * A * phi_line * math.sqrt(sp / (rho * psi_line))
    recycle_per_flow = 1.0 / (float(p.k_r) * float(check_valve_sqrt(sp, p, np)))

    def speed_for(m_c, xp):
        """The speed whose map head at compressor flow ``m_c`` is the setpoint, by
        bisection: at fixed flow the head rises with speed."""
        lo = xp.full(xp.shape(m_c), float(p.N_min))
        hi = xp.full(xp.shape(m_c), float(p.N_max))
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            U = mid * U_r
            low = rho * U**2 * characteristic(m_c / (rho * A * U), p, xp) < sp
            lo, hi = xp.where(low, mid, lo), xp.where(low, hi, mid)
        return 0.5 * (lo + hi)

    def commands(delivered, xp):
        """The steady balance at the setpoint for a delivered flow (kg/s at the
        setpoint): the compressor passes at least the control line's flow, the
        recycle the rest, and the speed holds the setpoint at that flow."""
        m_c = xp.maximum(delivered, m_line)
        x = (m_c - delivered) * recycle_per_flow
        return X.raw_from_commands(speed_for(m_c, xp), x, p, xp)

    def rest_at(q):
        opening = float(q.demand_range[0])
        delivered = float(p.k_d) * opening * float(head(sp, np))
        u = np.asarray(commands(np.asarray(delivered), np), np.float32)
        return lambda o, xp: xp.asarray(u)

    def balance(o, xp):
        """Feedforward from the metered delivered flow (obs 2) and the header
        pressure (obs 0): the consumers' opening it implies, their flow at the
        setpoint, and the balance's commands for it."""
        dp = o[0] * 1000.0
        delivered = o[2] * head(sp, xp) / head(dp, xp)
        return xp.clip(commands(delivered, xp), -1.0, 1.0)

    # The drive answers through its 3 %/s rate and 1 s lag, the plenum within its
    # 0.83 s Helmholtz cycle: a 10 s window is ten drive lags. A speed bump of 1 %
    # of rated; a recycle bump of 5 % of its opening (0.1 raw), its 0.15 s dead
    # time and 2 s stroke inside the same window. The noise band sizes them up.
    window = math.ceil(10.0 / float(p.delta_t))
    du_speed = 0.01 * 2.0 / float(p.N_max - p.N_min)
    pressure = Cv(
        "dp on speed",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[6],
        "dp",
        du_speed,
        window,
        valve=False,
        zone=((sp - band) / 1000.0, (sp + band) / 1000.0, None),
    )
    margin = Limit(
        "margin on recycle",
        lambda o, xp: X.surge_margin_from_obs(o, p),
        CONTROL_MARGIN,
        True,
        1,
        "m_c",
        0.1,
        window,
    )
    # Bump tests with the consumers' opening pinned and its drift at 0, at the
    # normal day's middle and at the ends of every scenario's demand range
    # (scenarios.py): 0.2, the fast approach's floor, to 0.95.
    low, high = 0.2, 0.95
    base = 0.5 * (0.7 + 0.95)
    test = {"demand_sigma": 0.0, "demand_range": (base, base)}
    return Row(
        300,
        rest_at(spec.make_test_params(**test)),
        (pressure,),
        test=test,
        points=({"demand_range": (low, low)}, {"demand_range": (high, high)}),
        rest_at=rest_at,
        balance=balance,
        limits=(margin,),
        why={
            "cascade": "the drive's own speed loop is the inner loop: the command "
            "is its setpoint, through the 3 %/s rate and the 1 s lag, with no other "
            "reading between it and the pressure",
            "decoupler": "the second MV serves the surge limit, not a target",
        },
    )
