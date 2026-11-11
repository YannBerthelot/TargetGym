"""Control-room rendering for the slab-heated building.

The schematic is a section through the zone. Its floor is the thermally
activated concrete slab, tinted by its core temperature, with the supply and
return water at its end. Air, structure and slab are tinted separately,
because the whole difficulty of the task is that they are different
temperatures: the air responds in minutes, the slab in hours, and the
structure the controller is really fighting is never measured.
"""

from target_gym import render_kit as rk
from target_gym.hvac.env import (
    compute_reward,
    electricity_price,
    hour_of_day,
    is_occupied,
    slab_conductances,
    solar_gain,
)

HISTORY_KEYS = (
    "t",
    "T_air",
    "T_mass",
    "T_core",
    "T_floor",
    "T_out",
    "lo",
    "hi",
    "reward",
)

# Wide enough for a winter outdoor temperature as well as zone air.
_T_LO, _T_HI = -12.0, 35.0


def _draw_zone(ax, state, params):
    T_air = float(state.T_air)
    T_mass = float(state.T_mass)
    T_out = float(state.T_out)
    h = float(hour_of_day(state.time, params))
    occupied = bool(is_occupied(state.time, params))
    sol = float(solar_gain(state.time, params))
    sol_max = params.solar_peak * params.A_window * params.g_window + 1e-9

    air_c = rk.duty_hex((T_air - _T_LO) / (_T_HI - _T_LO), "#1565c0", rk.ORANGE)
    mass_c = rk.duty_hex((T_mass - _T_LO) / (_T_HI - _T_LO), "#1a3550", "#b06030")
    out_c = rk.duty_hex((T_out - _T_LO) / (_T_HI - _T_LO), "#0d47a1", rk.ORANGE)

    # Outdoors.
    ax.add_patch(
        rk.patches.Rectangle((0, 0), 1, 1, fc=out_c, alpha=0.13, ec="none", zorder=0)
    )
    rk.label(ax, 0.06, 0.94, f"OUTDOOR  {T_out:+.1f} C", color=out_c, size=8, ha="left")

    # Structure: the walls are the thermal mass, tinted by T_mass.
    wx, wy, ww, wh = 0.20, 0.16, 0.60, 0.60
    tw = 0.045
    for rx, ry, rw, rh in (
        (wx, wy, ww, tw),
        (wx, wy + wh - tw, ww, tw),
        (wx, wy, tw, wh),
        (wx + ww - tw, wy, tw, wh),
    ):
        ax.add_patch(
            rk.patches.Rectangle(
                (rx, ry), rw, rh, fc=mass_c, ec=rk.FRAME, lw=1.2, alpha=0.85, zorder=3
            )
        )
    # Zone air.
    ax.add_patch(
        rk.patches.Rectangle(
            (wx + tw, wy + tw),
            ww - 2 * tw,
            wh - 2 * tw,
            fc=air_c,
            ec="none",
            alpha=0.30,
            zorder=2,
        )
    )

    # Window on the south wall, with solar gain.
    win_y = wy + wh * 0.45
    ax.add_patch(
        rk.patches.Rectangle(
            (wx + ww - tw, win_y),
            tw,
            0.16,
            fc="#8fd3f4",
            ec=rk.FRAME,
            lw=1,
            alpha=0.55,
            zorder=4,
        )
    )
    s = rk.clamp01(sol / sol_max)
    if s > 0.02:
        rk.glow(ax, 0.93, win_y + 0.20, 0.075, color=rk.AMBER, strength=s, zorder=1)
        for i in range(3):
            rk.flow_arrow(
                ax,
                0.93,
                win_y + 0.16 - i * 0.035,
                wx + ww - tw + 0.01,
                win_y + 0.11 - i * 0.035,
                color=rk.AMBER,
                lw=0.8 + 1.6 * s,
                alpha=0.35 + 0.5 * s,
            )
    rk.label(ax, 0.93, win_y + 0.255, f"SOLAR {s * 100:.0f}%", color=rk.AMBER, size=7)

    # The slab: the floor, tinted by its core, with the water loop at its end.
    T_core = float(state.T_core)
    core_c = rk.duty_hex((T_core - _T_LO) / (_T_HI - _T_LO), "#1a3550", rk.RED)
    sx, sy, sw, sh = wx + tw, wy + tw, ww - 2 * tw, 0.06
    ax.add_patch(
        rk.patches.Rectangle(
            (sx, sy), sw, sh, fc=core_c, ec=rk.FRAME, lw=1.0, alpha=0.9, zorder=4
        )
    )
    q = (float(state.T_supply) - params.T_supply_min) / (
        params.T_supply_max - params.T_supply_min
    )
    sup_c = rk.duty_hex(q, rk.DIM, rk.RED)
    for i in range(7):
        ax.plot(
            [sx + 0.03 + i * (sw - 0.06) / 6] * 2,
            [sy + 0.012, sy + sh - 0.012],
            color=sup_c,
            lw=1.6,
            alpha=0.8,
            zorder=5,
        )
    rk.label(
        ax,
        sx + sw / 2,
        sy + sh / 2,
        f"SLAB CORE {T_core:.1f} C",
        color=rk.TEXT,
        size=7,
        weight="bold",
        zorder=7,
    )
    floor_hot = float(state.T_floor) > params.floor_surface_max
    rk.label(
        ax,
        sx + sw / 2,
        sy + sh + 0.022,
        f"floor surface {float(state.T_floor):.1f} C"
        f"  (max {params.floor_surface_max:.0f})",
        color=rk.RED if floor_hot else rk.DIM,
        size=6.5,
        zorder=7,
    )
    # Supply and return water from the heat pump, outside the west wall.
    rk.pipe(ax, [(0.03, sy + sh * 0.7), (sx, sy + sh * 0.7)], color=sup_c, lw=2.5)
    rk.pipe(ax, [(sx, sy + sh * 0.3), (0.03, sy + sh * 0.3)], color=rk.BLUE, lw=2.0)
    frost = float(state.frost) > 0.5
    rk.label(
        ax,
        0.015,
        sy + sh + 0.10,
        f"SUPPLY {float(state.T_supply):.1f} C",
        color=rk.RED if frost else sup_c,
        size=7,
        weight="bold",
        ha="left",
    )
    rk.label(
        ax,
        0.015,
        sy + sh + 0.065,
        f"return {float(state.T_return):.1f} C",
        color=rk.BLUE,
        size=6.5,
        ha="left",
    )
    if frost:
        rk.label(
            ax,
            0.015,
            sy + sh + 0.145,
            "FROST PROTECT",
            color=rk.RED,
            size=7,
            weight="bold",
            ha="left",
        )

    # Occupancy.
    occ_c = rk.GREEN if occupied else rk.DIM
    rk.disc(ax, wx + ww - 0.10, wy + wh - 0.10, 0.022, fc=occ_c, ec=occ_c, alpha=0.8)
    rk.label(
        ax,
        wx + ww - 0.10,
        wy + wh - 0.145,
        "OCCUPIED" if occupied else "SETBACK",
        color=occ_c,
        size=6.5,
    )

    rk.label(
        ax,
        wx + ww / 2,
        wy + wh * 0.60,
        f"{T_air:.1f} C",
        color=rk.TEXT,
        size=15,
        weight="bold",
    )
    rk.label(ax, wx + ww / 2, wy + wh * 0.47, "zone air", color=rk.DIM, size=7)
    rk.label(
        ax,
        wx + ww / 2,
        wy - 0.045,
        f"structure {T_mass:.1f} C  ● hidden",
        color=rk.PURPLE,
        size=7,
    )
    price = float(electricity_price(state.time, params))
    rk.label(
        ax,
        0.06,
        0.06,
        f"{int(h):02d}:{int((h % 1) * 60):02d}   EUR {price:.3f}/kWh",
        color=rk.DIM,
        size=9,
        ha="left",
    )
    rk.caption(ax, "ISO 13790 5R1C + TABS slab  ·  the slab answers in hours")


def render_hvac(state, params, step, history):
    p = params
    history["t"].append(step * p.delta_t / 3600.0)
    history["T_air"].append(float(state.T_air))
    history["T_mass"].append(float(state.T_mass))
    history["T_core"].append(float(state.T_core))
    history["T_floor"].append(float(state.T_floor))
    history["T_out"].append(float(state.T_out))
    history["lo"].append(p.band_low)
    history["hi"].append(p.band_high)
    history["reward"].append(float(compute_reward(state, p)))

    T_air = float(state.T_air)
    err = max(p.band_low - T_air, T_air - p.band_high, 0.0)
    occupied = bool(is_occupied(state.time, p))
    if float(state.frost) > 0.5:
        status, status_text = rk.ALARM, "FROST PROTECTION"
    elif float(state.T_floor) > p.floor_surface_max:
        status, status_text = rk.WATCH, "FLOOR TOO HOT"
    elif err > 0.0:
        status = rk.WATCH if occupied else rk.NOMINAL
        status_text = "OUT OF BAND" + ("" if occupied else " (unoccupied)")
    else:
        status, status_text = rk.NOMINAL, "IN BAND"
    span = _T_HI - _T_LO
    f = lambda T: (float(T) - _T_LO) / span  # noqa: E731
    # The slab's heat at full supply from its present core.
    q_max = max(
        float(slab_conductances(p)["H_water"]) * (p.T_supply_max - float(state.T_core)),
        1.0,
    )

    gauges = [
        rk.Gauge(
            "T AIR",
            f"{T_air:.2f} C",
            f(T_air),
            rk.CYAN,
            limit_frac=f(p.band_high),
            target_frac=f(p.band_low),
        ),
        rk.Gauge("T CORE", f"{float(state.T_core):.2f} C", f(state.T_core), rk.RED),
        rk.Gauge(
            "FLOOR",
            f"{float(state.T_floor):.2f} C",
            f(state.T_floor),
            rk.ORANGE,
            limit_frac=f(p.floor_surface_max),
        ),
        rk.Gauge(
            "T MASS",
            f"{float(state.T_mass):.2f} C",
            f(state.T_mass),
            rk.PURPLE,
            hidden=True,
        ),
        rk.Gauge("T OUT", f"{float(state.T_out):+.2f} C", f(state.T_out), rk.BLUE),
        rk.Gauge(
            "SUPPLY",
            f"{float(state.T_supply):.1f} C",
            (float(state.T_supply) - p.T_supply_min)
            / (p.T_supply_max - p.T_supply_min),
            rk.RED,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "HEAT",
            f"{float(state.Q_slab) / 1e3:.1f} kW",
            float(state.Q_slab) / q_max,
            rk.AMBER,
        ),
        rk.cost_gauge(history["reward"], p, "EUR"),
    ]

    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["T_air"], "T air", rk.CYAN),
            rk.Series(history["T_core"], "slab core", rk.RED, lw=1.2, alpha=0.85),
            rk.Series(history["T_floor"], "floor", rk.ORANGE, lw=1.0, alpha=0.7),
            rk.Series(
                history["T_mass"], "T mass (hidden)", rk.PURPLE, lw=1.2, alpha=0.8
            ),
            rk.Series(history["T_out"], "outdoor", rk.BLUE, lw=1.0, alpha=0.6),
        ],
        ylabel="temperature  [C]",
        xlabel="hours",
        bands=[(p.band_low, p.band_high, rk.GREEN)],
        lines=[(p.floor_surface_max, rk.RED, "floor max")],
    )
    fig = rk.frame(
        title="BUILDING  HVAC  —  CONCRETE  SLAB",
        step=step,
        elapsed_s=step * p.delta_t,
        schematic=lambda ax: _draw_zone(ax, state, p),
        schematic_title="ZONE  SECTION",
        gauges=gauges,
        strips=[strip],
        status=status,
        status_text=status_text,
        subtitle="supply water set a day ahead  ·  comfort band all hours",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_hvac, HISTORY_KEYS)
