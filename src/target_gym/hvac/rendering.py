"""Control-room rendering for the single-zone building.

The schematic is a section through the zone. Air and structure are tinted
separately, because the whole difficulty of the task is that they are different
temperatures: the air responds in minutes and the mass the controller is really
fighting has a ~43 h time constant and is never measured.
"""

from target_gym import render_kit as rk
from target_gym.hvac.env import (
    compute_reward,
    hour_of_day,
    is_occupied,
    solar_gain,
)

HISTORY_KEYS = ("t", "T_air", "T_mass", "T_out", "target", "heat", "reward")

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

    # Emitter.
    q = float(state.Q_emitter) / max(params.Q_heat_max, 1e-9)
    ex, ey = wx + tw + 0.03, wy + tw + 0.03
    ax.add_patch(
        rk.patches.Rectangle(
            (ex, ey), 0.10, 0.11, fc=rk.WELL, ec=rk.FRAME, lw=1.2, zorder=5
        )
    )
    for i in range(4):
        ax.plot(
            [ex + 0.012 + i * 0.025] * 2,
            [ey + 0.012, ey + 0.098],
            color=rk.duty_hex(q, rk.DIM, rk.RED),
            lw=2.4,
            zorder=6,
        )
    if q > 0.02:
        rk.glow(ax, ex + 0.05, ey + 0.055, 0.09, color=rk.RED, strength=q, zorder=1)
    rk.label(
        ax,
        ex + 0.05,
        ey + 0.128,
        f"HEAT {q * 100:.0f}%",
        color=rk.duty_hex(q, rk.DIM, rk.RED),
        size=7,
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
        f"structure {T_mass:.1f} C  ● hidden, tau ~43 h",
        color=rk.PURPLE,
        size=7,
    )
    rk.label(
        ax,
        0.06,
        0.06,
        f"{int(h):02d}:{int((h % 1) * 60):02d}",
        color=rk.DIM,
        size=9,
        ha="left",
    )
    rk.caption(ax, "ISO 13790 5R1C  ·  air responds in minutes, the mass in days")


def render_hvac(state, params, step, history):
    history["t"].append(step * params.delta_t / 3600.0)
    history["T_air"].append(float(state.T_air))
    history["T_mass"].append(float(state.T_mass))
    history["T_out"].append(float(state.T_out))
    history["target"].append(float(state.target_T))
    history["heat"].append(100.0 * float(state.Q_emitter) / params.Q_heat_max)
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.T_air) - float(state.target_T))
    status = (
        rk.NOMINAL
        if err < params.comfort_band
        else (rk.WATCH if err < 2 * params.comfort_band else rk.ALARM)
    )
    span = _T_HI - _T_LO
    f = lambda T: (float(T) - _T_LO) / span

    gauges = [
        rk.Gauge(
            "T AIR",
            f"{float(state.T_air):.2f} C",
            f(state.T_air),
            rk.CYAN,
            target_frac=f(state.target_T),
        ),
        rk.Gauge(
            "SETPOINT", f"{float(state.target_T):.2f} C", f(state.target_T), rk.DIM
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
            "HEAT",
            f"{history['heat'][-1]:.0f} %",
            history["heat"][-1] / 100.0,
            rk.RED,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "REWARD",
            f"{history['reward'][-1]:+.3f}",
            rk.clamp01(history["reward"][-1]),
            rk.GREEN,
        ),
    ]

    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["T_air"], "T air", rk.CYAN, fill_to=history["target"]),
            rk.Series(
                history["target"], "setpoint", "white", ls="--", lw=1.0, alpha=0.6
            ),
            rk.Series(
                history["T_mass"], "T mass (hidden)", rk.PURPLE, lw=1.2, alpha=0.8
            ),
            rk.Series(history["T_out"], "outdoor", rk.BLUE, lw=1.0, alpha=0.6),
        ],
        ylabel="temperature  [C]",
        xlabel="hours",
    )
    fig = rk.frame(
        title="BUILDING  HVAC  —  SINGLE  ZONE",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_zone(ax, state, params),
        schematic_title="ZONE  SECTION",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="scheduled comfort setpoint  ·  weather and solar are disturbances",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_hvac, HISTORY_KEYS, stride=4)
