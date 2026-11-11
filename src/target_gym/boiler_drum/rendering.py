"""Control-room rendering for the natural-circulation boiler drum.

The schematic is the circulation loop: drum at the top with its water level
against the trip bands, downcomer on one side, heated riser on the other. The
riser is tinted by void fraction, which is the state that drives the inverse
response and which no plant instrument reads -- when the level swells, the
picture shows the bubbles that did it. The drum's trip lines sit at their own
heights, low-low lower than high-high, and an upset (an ID fan runback or a
feed pump trip) is announced over the drum with the cap it puts on firing or
feedwater.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.boiler_drum.env import actuator_caps, compute_reward, void_fraction

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


_EVENTS = {1: "ID FAN RUNBACK", 2: "FEED PUMP TRIP"}


def _level_span(params):
    """Half the drum's drawn height, in metres of level."""
    return 1.3 * max(abs(params.level_trip_low), params.level_trip_high)


def _event_text(state, params):
    """The upset's name and timing, or None in a normal hour."""
    kind = int(params.event_kind)
    if kind not in _EVENTS:
        return None, False
    now = float(state.time) * params.delta_t
    t_ev = float(state.event_time)
    if now < t_ev:
        return f"{_EVENTS[kind]} in {rk.format_clock(t_ev - now)}", False
    if kind == 2:
        back = t_ev + float(state.standby_delay) - now
        if back > 0:
            return f"{_EVENTS[kind]}  ·  standby pump in {back:.0f} s", True
        return f"{_EVENTS[kind]}  ·  standby pump on", False
    return f"{_EVENTS[kind]}  ·  firing held to {params.runback_load:.0%} load", True


def _draw_loop(ax, state, params):
    level = float(state.level)
    p_bar = float(state.pressure)
    alpha = float(void_fraction(state.alpha_r, state.pressure, params))

    # Drum.
    dx, dy, dw, dh = 0.24, 0.60, 0.52, 0.26
    rk.vessel(ax, dx, dy, dw, dh, fc="#0b1420", ec=rk.FRAME, lw=2.4)
    # Water level: normal water level sits mid-drum, each trip at its height.
    span = _level_span(params)
    frac = lambda lv: 0.5 + 0.5 * float(np.clip(lv / span, -1.0, 1.0))  # noqa: E731
    lf = frac(level)
    rk.fill_level(ax, dx, dy, dw, dh, lf, color=rk.BLUE, alpha=0.55)
    # Bubbles below the surface -- the swell made visible.
    n_b = int(4 + 26 * rk.clamp01(float(state.V_sd) / 10.0))
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
    trips = (
        (params.level_trip_high, "HH", float(state.time_high)),
        (params.level_trip_low, "LL", float(state.time_low)),
    )
    for lv, name, held in trips:
        yy = dy + dh * frac(lv)
        ax.plot(
            [dx, dx + dw], [yy, yy], color=rk.RED, lw=1.2, ls="--", alpha=0.75, zorder=6
        )
        timer = f"  {held:.0f}/{params.level_trip_delay:.0f} s" if held > 0 else ""
        rk.label(
            ax,
            dx + dw + 0.02,
            yy,
            f"{name} {lv * 1000:+.0f} mm{timer}",
            color=rk.RED,
            size=6,
            ha="left",
        )
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
        dx - 0.04,  # left of the steam line at 0.72 of the drum
        dy + dh + 0.030,
        f"LEVEL {level * 100:+.1f} cm  ·  {p_bar:.2f} bar",
        color=rk.TEXT,
        size=9,
        weight="bold",
        ha="left",
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
    text, active = _event_text(state, params)
    if text:
        rk.label(
            ax,
            0.5,
            0.985,
            text,
            color=rk.RED if active else rk.AMBER,
            size=8,
            weight="bold",
        )
    rk.caption(
        ax,
        "swell: falling pressure expands the bubbles, so level rises as mass leaves",
    )


def render_boiler_drum(state, params, step, history):
    alpha = float(void_fraction(state.alpha_r, state.pressure, params))
    history["t"].append(step * params.delta_t / 60.0)
    history["level"].append(float(state.level) * 100.0)
    history["pressure"].append(float(state.pressure))
    history["target_p"].append(float(state.target_pressure))
    history["q_steam"].append(float(state.q_steam))
    history["q_feed"].append(float(state.q_feed))
    history["fuel"].append(float(state.Q_fuel) / 1e6)
    history["alpha"].append(alpha * 100.0)
    history["reward"].append(float(compute_reward(state, params)))

    level = float(state.level)
    trip = params.level_trip_high if level >= 0 else params.level_trip_low
    near = level / trip  # 1 at the trip on the level's side
    timing = max(float(state.time_high), float(state.time_low)) > 0
    if timing or near >= 1.0:
        status, status_text = rk.ALARM, "PAST TRIP LEVEL"
    elif near >= 0.6:
        status, status_text = rk.WATCH, None
    else:
        status, status_text = rk.NOMINAL, None
    text, active = _event_text(state, params)
    if active and status == rk.NOMINAL:
        status, status_text = rk.WATCH, _EVENTS[int(params.event_kind)]
    Q_cap, q_cap = actuator_caps(state, float(state.time) * params.delta_t, params)
    p_lo, p_hi = 0.75 * params.p_nominal, 1.25 * params.p_nominal
    pf = lambda v: (float(v) - p_lo) / (p_hi - p_lo)  # noqa: E731

    gauges = [
        rk.Gauge(
            "DRUM LEVEL",
            f"{level * 100:+.2f} cm",
            level / abs(trip),
            rk.CYAN,
            bipolar=True,
            neg_color=rk.CYAN,
            target_frac=None,
        ),
        rk.Gauge(
            "PRESSURE",
            f"{float(state.pressure):.2f} bar",
            pf(state.pressure),
            rk.ORANGE,
            target_frac=pf(state.target_pressure),
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
            limit_frac=float(q_cap) / params.q_feed_max,
        ),
        rk.Gauge(
            "FIRING",
            f"{float(state.Q_fuel) / 1e6:.1f} MW",
            float(state.Q_fuel) / params.Q_max,
            rk.RED,
            limit_frac=float(Q_cap) / params.Q_max,
        ),
        rk.Gauge("RISER VOID", f"{alpha * 100:.1f} %", alpha, rk.PURPLE, hidden=True),
        rk.cost_gauge(history["reward"], params, "GBP"),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [rk.Series(history["level"], "level", rk.CYAN)],
            ylabel="level  [cm]",
            ylim=(-_level_span(params) * 100, _level_span(params) * 100),
            bands=[(-params.level_band * 100, params.level_band * 100, rk.GREEN)],
            lines=[
                (params.level_trip_high * 100, rk.RED, "HH trip"),
                (params.level_trip_low * 100, rk.RED, "LL trip"),
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
        status_text=status_text,
        subtitle="inverse response  ·  level trips "
        f"{params.level_trip_delay:.0f} s past HH or LL",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_boiler_drum, HISTORY_KEYS)
