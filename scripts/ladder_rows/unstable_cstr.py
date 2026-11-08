"""The reference ladder's row for ``unstable_cstr`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score
(PHYSICS.md section 10). The LDPE autoclave's zone temperature on the initiator
feed [L1, p. 2], tracking its setpoint. The operating point is open-loop unstable
(a mode growing over 175 s), so the loop is tuned by a published rule for such a
plant: SIMC for an integrating process (Skogestad 2003), which holds where the
unstable time constant is long against the test, read as an integrating lag over a
brief open-loop bump.
"""

from __future__ import annotations

import math

from reference_ladder import (  # noqa: E402
    Cv,
    Row,
)

from target_gym.registry import REGISTRY

#: The unstable mode's time constant at 513 K, s (PHYSICS.md, named property).
UNSTABLE_TAU = 175.0


def row() -> Row:
    from target_gym.pc_gym.unstable_cstr.env import steady_state

    spec = REGISTRY["unstable_cstr"]
    p = spec.make_test_params()
    i_min, i_span = float(p.I_f_min), float(p.I_f_max - p.I_f_min)

    def pump(o, xp):
        """The raw command holding the reset's initiator: the reset puts the
        reactor on its steady state, and the pump's rate reads exactly (obs 1)."""
        return xp.reshape(o[1] / 50.0 - 1.0, (1,))

    def balance(o, xp):
        """The initiator the balances need at the setpoint the operator reads
        (obs 2), at the nominal feed (``env.steady_state``)."""
        I_f = steady_state(o[2], p)[2]
        return xp.reshape(xp.clip(2.0 * (I_f - i_min) / i_span - 1.0, -1.0, 1.0), (1,))

    # The bump: 0.02 raw, 1 % of the pump's range, as the expert's test (PHYSICS.md
    # section 9), sized up by the thermocouple's noise band where it reads too
    # small; over a quarter of the unstable time constant, 43 s, so the response
    # stays near its first ramp. The pump has no valve's dead band.
    window = math.floor(UNSTABLE_TAU / 4 / float(p.delta_t))
    T = Cv(
        "T on initiator",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[2],
        "T",
        0.02,
        window,
        model="integrating_lag",
        valve=False,
    )
    # Bump tests from the steady state with the events off (PHYSICS.md section
    # 10), at every corner of the ranges the loop visits: the setpoint and its
    # grade change [L1], the feed 5 K warmer, the initiator 20 % less efficient,
    # the feed 10 % more (scenarios.py). The reset puts each on its own steady
    # state. The hold before a bump is open loop on an unstable plant, so it is
    # short: two 60 s episodes and the baseline grow a start error about fourfold.
    points = tuple(
        {
            "T_setpoint": float(p.T_setpoint) + dT,
            "T_feed": float(p.T_feed) + dTf,
            "f_eff": float(p.f_eff) * efficiency,
            "feed_mass": float(p.feed_mass) * flow,
        }
        for dT in (0.0, 24.0)
        for dTf in (0.0, 5.0)
        for efficiency in (1.0, 1.0 - 0.2)
        for flow in (1.0, 1.0 + 0.1)
    )
    return Row(
        60,
        pump,
        (T,),
        points=points,
        balance=balance,
        why={
            "cascade": "no inner measurement: the zone thermocouple is the only "
            "reading [L1, p. 2]",
            "decoupler": "one MV",
            "override": "the trip reads the loop's own temperature and the "
            "initiator is its only MV: an override would push the way the loop "
            "already does",
        },
    )
