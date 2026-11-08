"""The reference ladder's row for ``battery`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score. The
plant is version 5 (energy/battery/PHYSICS.md): observation ``[reported_soc,
V_cell, T_cell, P_MW, required_MW, signal_pu, baseline_now_MW, baseline_next_MW,
baseline_after_MW, export_requirement_MWh, import_requirement_MWh,
seconds_to_next_period, period_lost, price_now, price_next, price_after,
price_open]``, action the Operational Baseline raw over +-``baseline_limit``
(positive exports) for the first period past gate closure.
"""

from __future__ import annotations

import math

from reference_ladder import (  # noqa: E402
    Cv,
    Limit,
    Row,
)

from target_gym.registry import REGISTRY


def row() -> Row:
    from target_gym.energy.battery.env import (
        baseline_limit,
        contracted_power,
        response_volume,
    )

    p = REGISTRY["battery"].make_test_params()
    capacity = float(p.energy_capacity)  # J
    capacity_MWh = capacity / 3.6e9
    limit_W = float(baseline_limit(p))
    # PHYSICS.md sec. 4: the zone is the reported charge between the export and
    # the import requirements, the response volume from each end, 0.357 to
    # 0.643 with no deficit. Its cheap side is low: energy sold is valued as
    # energy stored, energy bought back loses 8 %.
    low = float(response_volume(p)) / capacity
    # A baseline moves the charge at d soc / dt = -baseline / capacity, and
    # reaches the meter after gate closure, 60 min, plus up to one 30 min
    # period (NESO-GUIDE p. 36): at most 5400 s.
    slope = -limit_W / capacity  # 1/s per raw
    delay = float(p.gate_closure_seconds + p.period_seconds)
    # The test reads that delay and three periods of the ramp after it. 0.2 raw
    # is 0.11 MW, 4.3 % of the charge over the three periods.
    window = math.ceil((delay + 3 * float(p.period_seconds)) / float(p.delta_t))
    charge = Cv(
        "charge on baseline",
        0,
        lambda o, xp: o[0],
        lambda o, xp: xp.asarray(0.5, dtype=xp.float32),
        "soc_reported",
        0.2,
        window,
        model="integrating_lag",
        valve=False,
        zone=(low, 1.0 - low, "low"),
    )

    def rest(o, xp):
        # The test's still block: no response, so a zero baseline holds the charge.
        return xp.zeros((1,), dtype=xp.float32)

    # Rung 2, the stored energy's steady balance: with an energy-neutral signal
    # the one standing draw is the round trip's loss, lumped on charge (CAISO's
    # form). The response charges at its mean, contracted x signal_std /
    # sqrt(2 pi) for the Gaussian signal (the clip at 2.5 sigma moves it 0.5 %),
    # and loses (1 - eta) of it; the baseline imports that over eta.
    eta = float(p.round_trip_efficiency)
    spread = float(p.signal_std) / math.sqrt(2 * math.pi)
    charging_W = float(contracted_power(p)) * spread
    hold_raw = -((1.0 - eta) * charging_W / eta) / limit_W

    def balance(o, xp):
        return xp.full((1,), hold_raw, dtype=xp.float32)

    # Rung 4: overrides on the requirement window itself (channels 9, 10), as
    # its edges move with the deficits: the reported charge's margin over the
    # export requirement kept above 0, and under the import one below 0. Their
    # slope and delay are the datasheet's and the market's, above.
    floor = Limit(
        "export requirement",
        lambda o, xp: o[0] - o[9] / capacity_MWh,
        0.0,
        True,
        0,
        slope=slope,
        theta=delay,
    )
    ceiling = Limit(
        "import requirement",
        lambda o, xp: o[0] + o[10] / capacity_MWh - 1.0,
        0.0,
        False,
        0,
        slope=slope,
        theta=delay,
    )

    def published(s):
        # PHYSICS.md sec. 10 asks for the expert's recipe as a rung: NESO's
        # guidance, return the stored energy by the submitted baselines
        # (NESO-GUIDE p. 37, item 4), declared over one period after the locked
        # ones (energy/battery/experts.py).
        return s.make_pid_controller()

    return Row(
        int(p.max_steps_in_episode),
        rest,
        (charge,),
        # The test's block has the signal still, so the baseline alone moves
        # the charge.
        test={"signal_std": 0.0},
        balance=balance,
        limits=(floor, ceiling),
        published=published,
        why={
            "cascade": "no faster reading of the charge: the meter sees a baseline "
            "only when its period starts, as the charge does",
            "decoupler": "one loop",
            "points": "the charge's slope per baseline is the capacity's at any "
            "charge, price or signal, and its delay is the market's gate closure",
        },
    )
