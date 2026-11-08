"""The reference ladder's row for ``cstr`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score
(PHYSICS.md section 10). One loop, the reactor temperature on the fresh cooling
water, tracking its setpoint; then the jacket outlet as the cascade's inner reading.
"""

from __future__ import annotations

import math

from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.pc_gym.cstr.env import steady_state

    spec = REGISTRY["cstr"]
    p = spec.make_test_params()
    w_min, w_span = float(p.Fw_min), float(p.Fw_max - p.Fw_min)
    # The feed's temperature is hidden: the balances take the middle of its range
    # [S1, Tables 3, 4], as the expert does.
    T_f = 0.5 * (float(p.feed_T_range[0]) + float(p.feed_T_range[1]))

    def water(o, xp):
        """The raw command holding the reset's water: the reset puts the reactor on
        the steady state of its point (``env.steady_state``), and the valve reads
        exactly (obs 2)."""
        return xp.reshape(o[2] / 50.0 - 1.0, (1,))

    def balance(o, xp):
        """The water the three balances need at the setpoint the operator reads
        (obs 3), at the middle feed temperature, design activity and feed."""
        F_w = steady_state(o[3], T_f, p)[2]
        return xp.reshape(xp.clip(2.0 * (F_w - w_min) / w_span - 1.0, -1.0, 1.0), (1,))

    def jacket_rest(o, xp):
        """The jacket temperature at that steady state."""
        return steady_state(o[3], T_f, p)[1]

    dt = float(p.delta_t)
    # The reactor: a cooling step settles 95 % in 1.15 residence times open loop
    # (PHYSICS.md, T8), 39 min of the 34 min residence; the window is twice that.
    residence = float(p.V) / float(p.q)
    window = math.ceil(2 * 1.15 * residence / dt)
    # The jacket: its time constant M_j / (F_w + UA / cp_w) is 2.2 min at the design
    # water, read 2 min late [S6]; its window is that dead time and five of it.
    _, _, F_design = steady_state(float(p.T_setpoint), T_f, p)
    tau_j = float(p.M_j) / (float(F_design) + float(p.UA) / 4.18)
    inner_window = math.ceil((2.0 + 5 * tau_j) / dt)
    # Each water bump is 5 % of the valve's span (0.1 raw), the outer test's jacket
    # setpoint step 0.5 K; the noise band sizes them up where they read too small.
    T = Cv("T on water", 0, lambda o, xp: o[0], lambda o, xp: o[3], "T", 0.1, window)
    jacket = Inner(
        0,
        "T_j on water",
        lambda o, xp: o[1],
        jacket_rest,
        "T_j",
        0.1,
        inner_window,
        0.5,
        window,
    )

    # Bump tests from the reset steady state with the events off (PHYSICS.md
    # section 10), at every corner of the ranges the loop visits: the setpoint and
    # its 2 K step [S1, Fig. 2], the feed temperature's range [S1, Tables 3, 4],
    # the catalyst at full and 40 % lost activity [S1, Fig. 3], and the feed's
    # reactant at nominal and 10 % up. The reset puts each on its own steady state.
    # A corner whose steady water lies past the valve's range has no rest to test
    # at: the reactant's 10 % rise at full activity needs more than 110 g/s.
    def fits(q) -> bool:
        T_f = float(q["feed_T_range"][0])
        F_w = float(steady_state(q["T_setpoint"], T_f, p.replace(**q))[2])
        return float(p.Fw_min) <= F_w <= float(p.Fw_max)

    corners = (
        {
            "T_setpoint": float(p.T_setpoint) + dT,
            "feed_T_range": (T_feed, T_feed),
            "k0": float(p.k0) * activity,
            "Caf": float(p.Caf) * feed,
        }
        for dT in (0.0, 2.0)
        for T_feed in (float(p.feed_T_range[0]), float(p.feed_T_range[1]))
        for activity in (1.0, 1.0 - 0.4)
        for feed in (1.0, 1.0 + 0.1)
    )
    points = tuple(q for q in corners if fits(q))
    return Row(
        240,
        water,
        (T,),
        points=points,
        balance=balance,
        cascades=(jacket,),
        why={
            "decoupler": "one loop",
            "override": "the interlock reads the loop's own temperature and the "
            "water is its only MV: an override would push the way the loop already "
            "does",
        },
    )
