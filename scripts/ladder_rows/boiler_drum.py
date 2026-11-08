"""The reference ladder's row for ``boiler_drum`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score. The
plant is version 5 (boiler_drum/PHYSICS.md): observation ``[level, pressure,
q_steam, fuel_pct, feed_pct, target_level, target_pressure, load_now,
load_planned, s_to_ramp, fuel_cap_pct, feed_cap_pct]``, action ``[fuel, feed]``
raw over ``[0, Q_max]`` and ``[0, q_feed_max]``.
"""

from __future__ import annotations

from reference_ladder import (  # noqa: E402
    Cv,
    Inner,
    Limit,
    Row,
)

from target_gym.identification import Loop
from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.boiler_drum.env import steam_enthalpy

    p = REGISTRY["boiler_drum"].make_test_params()
    q_max, Q_max = float(p.q_feed_max), float(p.Q_max)
    # Firing balance at the pressure the boiler master holds: Q = q_s (h_s - h_f),
    # the reset's own steady state (env.steady_state), 1.71 MJ/kg at 85 bar.
    per_kg_fired = float(steam_enthalpy(float(p.p_nominal), p)) - float(p.h_feedwater)

    def firing_raw(q, xp):
        return 2.0 * q * per_kg_fired / Q_max - 1.0

    def feed_raw(q, xp):
        return 2.0 * q / q_max - 1.0

    def firing_pct(o, xp):
        return 100.0 * o[7] * per_kg_fired / Q_max

    # The bump tests' P-only level hold: the feed's span over the window between
    # the trips, -330 to +254 mm (S9, S20). 2 raw over 0.584 m is 342 kg/s per m,
    # a closed-loop lag of about (rho_w - rho_s) A_d / 342 = 43 s.
    window_m = float(p.level_trip_high) - float(p.level_trip_low)
    level_gain = 2.0 / window_m  # raw per m

    def rest(o, xp):
        # The load plan's steady state (channel 7, exact: the plan is known), the
        # level held by the P-only hold.
        feed = feed_raw(o[7], xp) + level_gain * (o[5] - o[0])
        return xp.stack([firing_raw(o[7], xp), feed])

    def balance(o, xp):
        # Rung 2, PHYSICS.md sec. 8: steam to feedwater (three-element, the
        # measured steam, channel 2) and q_s (h_s - h_f) to firing on the plan.
        return xp.stack([firing_raw(o[7], xp), feed_raw(o[2], xp)])

    # Pressure on firing is self-regulating through the valve, q_s = q_d p / p_n:
    # its lag is the drum's stored energy per bar (about 3.7e8 J/bar from S1's
    # e-coefficients at 85 bar) over the steam's energy per bar, q_d / 85 x 1.71
    # MJ/kg: about 190 s at full load and 780 s at 25 %. 600 steps (1200 s) cover
    # 1.5 of the slowest lag. 0.02 raw is 2.4 MW, 1.5 % of nominal firing.
    pressure = Cv(
        "pressure on firing",
        0,
        lambda o, xp: o[1],
        lambda o, xp: o[6],
        "pressure",
        0.02,
        600,
    )
    # Level on feed integrates, (rho_w - rho_s) A_d = 15 t per m, behind the
    # shrink of cold feed: 120 steps (240 s) are several of the hold's 43 s.
    # 0.05 raw is 5 kg/s, 5 % of nominal steam.
    level = Cv(
        "level on feed",
        1,
        lambda o, xp: o[0],
        lambda o, xp: o[5],
        "level",
        0.05,
        120,
        model="integrating_lag",
    )
    # Rung 3: the boiler master sets the firing rate, held by an inner loop on
    # the firing reading (channel 3), behind the burners' 20 s lag (PHYSICS.md,
    # Actuators): 50 steps are 5 lags. The outer step is 1 % of the span.
    firing = Inner(
        0,
        "firing on fuel",
        lambda o, xp: o[3],
        firing_pct,
        "",
        0.02,
        50,
        1.0,
        600,
    )
    # Rung 4, PHYSICS.md sec. 8: the fan and pump caps (channels 10, 11) read by
    # an override that holds each command at its cap, so no loop winds up
    # against a capped fan or pump. Each reads its position less its cap, in
    # % of span, kept below 0; the firing's lag is 20 s, the feed's 1 s.
    fuel_cap = Limit(
        "firing at the fan cap",
        lambda o, xp: o[3] - o[10],
        0.0,
        False,
        0,
        du=0.02,
        window=50,
    )
    feed_cap = Limit(
        "feed at the pump cap",
        lambda o, xp: o[4] - o[11],
        0.0,
        False,
        1,
        du=0.02,
        window=20,
    )
    return Row(
        300,
        rest,
        (pressure, level),
        # The base point holds the middle of the normal day's 40 to 100 % load;
        # a range pinned at one end holds that load, with no ramp.
        test={"load_range": (0.70, 0.70)},
        # Every end of the loads the suite visits (scenarios.py): normal 40 to
        # 100 %, low load 25 to 50 %, the upsets from 75 to 100 % to the
        # runback's 50 %. The gains move with load (PHYSICS.md sec. 8).
        points=tuple(
            {"load_range": (load, load)} for load in (0.25, 0.40, 0.50, 0.75, 1.0)
        ),
        balance=balance,
        cascades=(firing,),
        limits=(fuel_cap, feed_cap),
        holds=(Loop(1, 1, level_gain, 0),),
        why={
            "decoupler": "the level integrates, so it has no steady gain for a "
            "static decoupler to invert; the steam feedforward is the three-element "
            "cancellation of the load's effect on it",
        },
    )
