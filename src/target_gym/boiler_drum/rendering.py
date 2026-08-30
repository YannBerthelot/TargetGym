"""Control-room rendering for the natural-circulation boiler drum.

The schematic is the circulation loop: drum at the top with its water level
against the trip bands, downcomer on one side, heated riser on the other. The
riser is tinted by void fraction, which is the state that drives the inverse
response and which no plant instrument reads -- when the level swells, the
picture shows the bubbles that did it.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.boiler_drum.env import compute_reward, void_fraction

HISTORY_KEYS = (
    "t",
    "level",
    "pressure",
    "target_p",
    "q_steam",
    "q_feed",
    "fuel",
    "alpha",
    "reward",
)


def _draw_loop(ax, state, params):
    level = float(state.level)
    p_bar = float(state.pressure)
    alpha = float(void_fraction(state.m_sr, state.pressure, params))

    # Drum.
    dx, dy, dw, dh = 0.24, 0.60, 0.52, 0.26
    rk.vessel(ax, dx, dy, dw, dh, fc="#0b1420", ec=rk.FRAME, lw=2.4)
    # Water level: normal water level sits mid-drum, trips at +/- level_trip.
    lf = 0.5 + 0.5 * float(np.clip(level / params.level_trip, -1.3, 1.3)) * 0.72
    rk.fill_level(ax, dx, dy, dw, dh, lf, color=rk.BLUE, alpha=0.55)
    # Bubbles below the surface -- the swell made visible.
    n_b = int(4 + 26 * rk.clamp01(float(state.m_sd) / 200.0))
    rng = np.random.default_rng(int(state.time) // 4)
    for _ in range(n_b):
        bx = dx + 0.03 + rng.random() * (dw - 0.06)
        by = dy + 0.02 + rng.random() * max(dh * lf - 0.04, 0.01)
        rk.disc(
            ax,
            bx,
            by,
            0.004 + 0.004 * rng.random(),
            fc="#bfe6ff",
            ec="none",
            alpha=0.45,
            zorder=4,
        )
    # Trip lines and normal water level.
    for sgn, name in ((+1, "HIGH  carryover"), (-1, "LOW  dryout")):
        yy = dy + dh * (0.5 + sgn * 0.36)
        ax.plot(
            [dx, dx + dw], [yy, yy], color=rk.RED, lw=1.2, ls="--", alpha=0.75, zorder=6
        )
        rk.label(ax, dx + dw + 0.02, yy, name, color=rk.RED, size=6, ha="left")
    ax.plot(
        [dx, dx + dw],
        [dy + dh * 0.5] * 2,
        color="white",
        lw=1.0,
        ls=":",
        alpha=0.7,
        zorder=6,
    )
    rk.label(ax, dx - 0.02, dy + dh * 0.5, "NWL", color=rk.TEXT, size=6.5, ha="right")
    rk.label(
        ax,
        dx + dw / 2,
        dy + dh + 0.030,
        f"LEVEL {level * 100:+.1f} cm   ·   {p_bar:.2f} bar",
        color=rk.TEXT,
        size=9.5,
        weight="bold",
    )

    # Steam out.
    rk.flow_arrow(
        ax,
        dx + dw * 0.72,
        dy + dh,
        dx + dw * 0.72,
        dy + dh + 0.10,
        color=rk.TEXT,
        lw=1.2 + 2.4 * rk.clamp01(float(state.q_steam) / (2 * params.q_steam_nominal)),
    )
    rk.label(
        ax,
        dx + dw + 0.10,
        dy + dh + 0.085,
        f"STEAM {float(state.q_steam):.1f} kg/s",
        color=rk.TEXT,
        size=7,
        ha="left",
    )
    # Feedwater in.
    rk.flow_arrow(
        ax,
        dx - 0.12,
        dy + dh * 0.62,
        dx - 0.005,
        dy + dh * 0.62,
        color=rk.CYAN,
        lw=1.2 + 2.4 * rk.clamp01(float(state.q_feed) / params.q_feed_max),
    )
    rk.label(
        ax,
        dx - 0.13,
        dy + dh * 0.62,
        f"FEED\n{float(state.q_feed):.1f} kg/s",
        color=rk.CYAN,
        size=7,
        ha="right",
    )

    # Downcomer (all water) and riser (two-phase).
    lxc, rxc = 0.30, 0.70
    y0 = 0.14
    rk.pipe(ax, [(lxc, dy), (lxc, y0), (rxc, y0), (rxc, dy)], color=rk.FRAME, lw=6)
    rk.flow_arrow(ax, lxc, dy - 0.06, lxc, y0 + 0.06, color=rk.BLUE, lw=1.8)
    rk.label(
        ax,
        lxc - 0.035,
        (dy + y0) / 2,
        "DOWNCOMER",
        color=rk.BLUE,
        size=6.5,
        rotation=90,
    )

    # Riser voidage: bubbles fill more of the tube as alpha rises.
    rh = dy - y0
    rk.vessel(
        ax, rxc - 0.035, y0, 0.07, rh, fc="#0b1420", ec=rk.FRAME, lw=1.4, pad=0.006
    )
    ax.add_patch(
        rk.patches.Rectangle(
            (rxc - 0.030, y0 + 0.004),
            0.06,
            rh - 0.008,
            fc=rk.duty_hex(alpha, "#1565c0", rk.AMBER),
            ec="none",
            alpha=0.8,
            zorder=3,
        )
    )
    rk.flow_arrow(ax, rxc, y0 + 0.06, rxc, dy - 0.04, color=rk.AMBER, lw=1.8)
    rk.label(
        ax,
        rxc + 0.055,
        (dy + y0) / 2,
        f"RISER\nvoid {alpha * 100:.0f}%\n● hidden",
        color=rk.PURPLE,
        size=6.5,
        ha="left",
    )

    # Burner.
    fuel_f = rk.clamp01(float(state.Q_fuel) / params.Q_max)
    rk.glow(ax, 0.5, y0, 0.13, color=rk.RED, strength=fuel_f)
    rk.label(
        ax,
        0.5,
        y0 - 0.055,
        f"FIRING  {float(state.Q_fuel) / 1e6:.1f} MW",
        color=rk.duty_hex(fuel_f, rk.DIM, rk.RED),
        size=8,
        weight="bold",
    )
    rk.caption(
        ax,
        "swell: falling pressure expands the bubbles, so level rises " "as mass leaves",
    )


def render_boiler_drum(state, params, step, history):
    alpha = float(void_fraction(state.m_sr, state.pressure, params))
    history["t"].append(step * params.delta_t / 60.0)
    history["level"].append(float(state.level) * 100.0)
    history["pressure"].append(float(state.pressure))
    history["target_p"].append(float(state.target_pressure))
    history["q_steam"].append(float(state.q_steam))
    history["q_feed"].append(float(state.q_feed))
    history["fuel"].append(float(state.Q_fuel) / 1e6)
    history["alpha"].append(alpha * 100.0)
    history["reward"].append(float(compute_reward(state, params)))

    lvl_err = abs(float(state.level))
    status = (
        rk.NOMINAL
        if lvl_err < params.level_band
        else (rk.WATCH if lvl_err < 0.6 * params.level_trip else rk.ALARM)
    )
    trip_cm = params.level_trip * 100.0

    gauges = [
        rk.Gauge(
            "DRUM LEVEL",
            f"{float(state.level) * 100:+.2f} cm",
            float(state.level) / params.level_trip,
            rk.CYAN,
            bipolar=True,
            neg_color=rk.CYAN,
            target_frac=None,
        ),
        rk.Gauge(
            "PRESSURE",
            f"{float(state.pressure):.2f} bar",
            (float(state.pressure) - params.pressure_min)
            / (params.pressure_max - params.pressure_min),
            rk.ORANGE,
            target_frac=(float(state.target_pressure) - params.pressure_min)
            / (params.pressure_max - params.pressure_min),
        ),
        rk.Gauge(
            "STEAM OUT",
            f"{float(state.q_steam):.1f} kg/s",
            float(state.q_steam) / (2 * params.q_steam_nominal),
            rk.TEXT,
        ),
        rk.Gauge(
            "FEEDWATER",
            f"{float(state.q_feed):.1f} kg/s",
            float(state.q_feed) / params.q_feed_max,
            rk.BLUE,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "FIRING",
            f"{float(state.Q_fuel) / 1e6:.1f} MW",
            float(state.Q_fuel) / params.Q_max,
            rk.RED,
            limit_frac=1.0,
        ),
        rk.Gauge("RISER VOID", f"{alpha * 100:.1f} %", alpha, rk.PURPLE, hidden=True),
        rk.Gauge(
            "REWARD",
            f"{history['reward'][-1]:+.3f}",
            rk.clamp01(history["reward"][-1]),
            rk.GREEN,
        ),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [rk.Series(history["level"], "level", rk.CYAN)],
            ylabel="level  [cm]",
            ylim=(-trip_cm * 1.15, trip_cm * 1.15),
            bands=[(-params.level_band * 100, params.level_band * 100, rk.GREEN)],
            lines=[
                (trip_cm, rk.RED, "trip"),
                (-trip_cm, rk.RED, "trip"),
                (0.0, "white", "NWL"),
            ],
        ),
        rk.Strip(
            history["t"],
            [
                rk.Series(
                    history["pressure"],
                    "pressure",
                    rk.ORANGE,
                    fill_to=history["target_p"],
                ),
                rk.Series(
                    history["target_p"], "target", "white", ls="--", lw=1.0, alpha=0.6
                ),
            ],
            ylabel="pressure  [bar]",
            xlabel="minutes",
        ),
    ]
    fig = rk.frame(
        title="BOILER  DRUM  —  NATURAL  CIRCULATION",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_loop(ax, state, params),
        schematic_title="CIRCULATION  LOOP",
        gauges=gauges,
        strips=strips,
        status=status,
        subtitle="non-minimum phase  ·  level is an integrator  ·  both trips are fatal",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_boiler_drum, HISTORY_KEYS, stride=4)
