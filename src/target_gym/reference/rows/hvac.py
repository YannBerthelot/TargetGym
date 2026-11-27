"""The reference ladder's row for ``hvac`` (docs/proposals/task-charter.md, J1).

What the plant's datasheet and balances say, and nothing read from a score. The
plant is version 5 (hvac/PHYSICS.md): observation ``[T_air, T_out, T_out_24h,
T_core, T_return, T_supply, sin_h, cos_h, band_centre, T_out_forecast_24h,
price_now, h_to_price_change]``, action the supply water raw over
``[T_supply_min, T_supply_max]``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from target_gym.experts.pid import Controller
from target_gym.reference.row import (
    Cv,
    Inner,
    Row,
)
from target_gym.registry import REGISTRY

#: The night charging window, 00:00 to 07:00 (Madrid's night setpoint [14, p. 14]).
NIGHT_END_H = 7.0
#: Day-to-day room correction [9, p. 3]: 1 K of curve per K of room for a floor
#: [11, p. 2], limited to about 5 K [9, part 2].
TRIM_PER_K, TRIM_MAX = 1.0, 5.0


def row() -> Row:
    from target_gym.hvac.env import (
        DAY_STEPS,
        internal_gain,
        slab_conductances,
        solar_gain,
        steady_state,
    )

    spec = REGISTRY["hvac"]
    scored = spec.make_test_params()
    lo, hi = float(scored.T_supply_min), float(scored.T_supply_max)
    T_set = float(scored.band_low)  # the band's heating edge [9], as the curve runs

    # The bump tests run on a still building: no daily swing, no weather
    # deviation, no sun, and the gains at their unoccupied level all day, so a
    # supply step is all that moves the room.
    test = {
        "T_out_amplitude": 0.0,
        "T_out_noise_std": 0.0,
        "solar_peak": 0.0,
        "gain_occupied": float(scored.gain_unoccupied),
    }
    still = spec.make_test_params(**test)
    still_gains = float(still.gain_unoccupied * still.A_floor)

    def raw_of(T_supply):
        return 2.0 * (T_supply - lo) / (hi - lo) - 1.0

    def held_supply(q):
        # The reset's own steady state (env.steady_state) at the point's outdoor
        # mean: the supply that holds the room at the band's low edge.
        return float(steady_state(q, float(q.T_out_mean), T_set, still_gains, 0.0)[3])

    def rest_at(q):
        raw = raw_of(held_supply(q))
        return lambda o, xp: xp.full((1,), raw, dtype=xp.float32)

    # Rung 2, PHYSICS.md sec. 8: the heating curve from the plant's balance, on
    # the next 24 h's forecast outdoor mean (channel 9), with the normal day's
    # mean gains and sun. The balance is linear in the outdoor temperature, so
    # two solves give it exactly.
    day = jnp.arange(DAY_STEPS)
    day_int = float(jnp.mean(internal_gain(day, scored)))
    day_sol = float(jnp.mean(solar_gain(day, scored)))
    ends = (-10.0, 20.0)
    sup = [float(steady_state(scored, T, T_set, day_int, day_sol)[3]) for T in ends]
    core = [float(steady_state(scored, T, T_set, day_int, day_sol)[1]) for T in ends]
    sup_slope = (sup[1] - sup[0]) / (ends[1] - ends[0])
    core_slope = (core[1] - core[0]) / (ends[1] - ends[0])

    def balance(o, xp):
        T = sup[0] + sup_slope * (o[9] - ends[0])
        return xp.reshape(xp.clip(raw_of(T), -1.0, 1.0), (1,))

    def core_rest(o, xp):
        return core[0] + core_slope * (o[9] - ends[0])

    # Time constants from the slab's datasheet numbers [9]: its core's capacity
    # over its couplings to the water and the room, C_slab / (H_water + H_slab),
    # 9.2 h. The room follows through the core and then the building's mass, so
    # its test reads 5 of those lags (about two days); the core's own reads 3.
    s = slab_conductances(scored)
    tau = float(s["C_slab"] / (s["H_water"] + s["H_slab"]))
    dt = float(scored.delta_t)
    room_window = math.ceil(5 * tau / dt)
    core_window = math.ceil(3 * tau / dt)
    # 0.1 raw is 1.7 K of supply, about 1.4 K of room at the balance's 0.81 K
    # per K: ten of the BMS's 0.1 K steps.
    air = Cv(
        "T_air on supply",
        0,
        lambda o, xp: o[0],
        lambda o, xp: o[8],
        "T_air",
        0.1,
        room_window,
        valve=False,
        # PHYSICS.md sec. 4: the comfort band is the zone, its cheap side low.
        zone=(float(scored.band_low), float(scored.band_high), "low"),
        # Leaving it is charged as lost work (the score's ``comfort`` term), so
        # each rung holds the lowest set point of the zone's grid at which none
        # of its seeds leaves (charter J1, owner, 2026-11-23).
        exit_terms=("comfort",),
    )
    # Rung 3, PHYSICS.md sec. 8: the core's sensor (channel 3) [7, 12] as the
    # inner loop; the room loop sets its setpoint around the balance's core. The
    # outer step is 0.5 K of core.
    slab = Inner(
        0,
        "T_core on supply",
        lambda o, xp: o[3],
        core_rest,
        "T_core",
        0.1,
        core_window,
        0.5,
        room_window,
        valve=False,
    )

    # Rung "night charging" (owner, 2026-11-25, the review's P2): Madrid charges
    # its floors at night on the cheap tariff and lets them release heat by day
    # [14, p. 14]. The water runs only from 00:00 to 07:00, a lift above the core
    # it reads that puts the day's heat in those 7 hours: 24 / 7 times the
    # balance's lift at the forecast's outdoor mean, for the set point. Each new
    # day the lift is trimmed on the last day's occupied room mean, a kelvin of
    # curve per kelvin of room [11, p. 2], held to 5 K [9, part 2]. Ours: the
    # 24 / 7 scaling, read from the balance, and the trim's once-a-day update.
    occupied = (float(scored.occupied_start_h), float(scored.occupied_end_h))

    def night(T_room):
        bal = [
            [float(steady_state(scored, T, T_room, day_int, day_sol)[k]) for T in ends]
            for k in (3, 1)
        ]
        lift0 = bal[0][0] - bal[1][0]
        lift_slope = ((bal[0][1] - bal[1][1]) - lift0) / (ends[1] - ends[0])
        scale = 24.0 / NIGHT_END_H

        def step(_p, state, o, xp):
            trim, total, count, last = state
            hour = (xp.arctan2(o[6], o[7]) * 24.0 / (2.0 * math.pi)) % 24.0
            new_day = hour < last
            mean = total / xp.maximum(count, 1.0)
            moved = xp.clip(trim + TRIM_PER_K * (T_room - mean), -TRIM_MAX, TRIM_MAX)
            trim = xp.where(new_day & (count > 0), moved, trim)
            total, count = xp.where(new_day, 0.0, total), xp.where(new_day, 0.0, count)
            in_hours = (hour >= occupied[0]) & (hour < occupied[1])
            total = total + xp.where(in_hours, o[0], 0.0)
            count = count + xp.where(in_hours, 1.0, 0.0)
            lift = lift0 + lift_slope * (o[9] - ends[0]) + trim
            T_sup = xp.where(hour < NIGHT_END_H, o[3] + scale * lift, lo)
            raw = xp.clip(raw_of(T_sup), -1.0, 1.0)
            return xp.reshape(raw, (1,)), (trim, total, count, hour)

        return Controller(None, (0.0, 0.0, 0.0, 24.0), step)

    def published(s):
        # The heating curve the family runs (hvac/experts.py, [9, 13]).
        return s.make_pid_controller()

    return Row(
        2 * DAY_STEPS * 7,
        rest_at(still),
        (air,),
        test=test,
        # The ends of the outdoor means the suite visits (scenarios.py): the cold
        # snap's -5 C and the shoulder season's 12 C; the normal 5 C is the base.
        points=({"T_out_mean": -5.0}, {"T_out_mean": 12.0}),
        rest_at=rest_at,
        balance=balance,
        cascades=(slab,),
        published=published,
        schedules={"night charging": night},
        why={
            "decoupler": "one loop",
            "override": "no limit a select could hold: the supply's 49 C and frost "
            "are the plant's own protection, and the floor's 29 C never binds in "
            "heating (22.2 C at design)",
        },
    )
