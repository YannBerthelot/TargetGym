"""Control-room rendering for the cement rotary kiln.

The schematic is the kiln laid out along its axis, which is the only honest way
to draw it: the charge band is tinted by temperature and the band beneath it by
free lime, so the burning zone and the burn-out front are visible as positions
in the kiln rather than as two numbers. That whole profile is hidden from the
controller, and the half-hour it takes material to cross the picture is the
task.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.cement_kiln.env import (
    burning_zone_temperature,
    compute_reward,
    discharge_lime,
    residence_time,
    specific_heat_consumption,
)

HISTORY_KEYS = ("t", "lime", "target", "T_bz", "T_exh", "fuel", "rpm", "feed", "reward")


def _draw_kiln(ax, state, params):
    T_solid = np.asarray(state.T_solid)
    T_gas = np.asarray(state.T_gas)
    lime = np.asarray(state.lime)
    tau_min = float(residence_time(state.rpm, params)) / 60.0

    x0, x1 = 0.10, 0.90
    w = x1 - x0

    # Kiln shell, drawn on the real 3.5 % slope: feed end high, burner end low.
    y_hi, y_lo, th = 0.66, 0.44, 0.20
    shell = [(x0, y_hi), (x1, y_lo), (x1, y_lo - th), (x0, y_hi - th)]
    ax.add_patch(
        rk.patches.Polygon(
            shell, closed=True, fc="#0b1420", ec=rk.FRAME, lw=2.2, zorder=2
        )
    )

    n = len(T_solid)
    for i in range(n):
        fa, fb = i / n, (i + 1) / n
        xa, xb = x0 + w * fa, x0 + w * fb
        ya, yb = y_hi + (y_lo - y_hi) * fa, y_hi + (y_lo - y_hi) * fb
        # Gas above the charge.
        ax.add_patch(
            rk.patches.Polygon(
                [
                    (xa, ya - 0.004),
                    (xb, yb - 0.004),
                    (xb, yb - th * 0.42),
                    (xa, ya - th * 0.42),
                ],
                closed=True,
                fc=rk.incandescence(T_gas[i], 900.0, 2100.0),
                ec="none",
                alpha=0.45,
                zorder=3,
            )
        )
        # Charge bed.
        ax.add_patch(
            rk.patches.Polygon(
                [
                    (xa, ya - th * 0.42),
                    (xb, yb - th * 0.42),
                    (xb, yb - th * 0.80),
                    (xa, ya - th * 0.80),
                ],
                closed=True,
                fc=rk.incandescence(T_solid[i], 700.0, 1650.0),
                ec="none",
                alpha=0.92,
                zorder=4,
            )
        )
        # Free lime beneath, so the burn-out front reads as a position.
        ax.add_patch(
            rk.patches.Polygon(
                [
                    (xa, ya - th * 0.80),
                    (xb, yb - th * 0.80),
                    (xb, yb - th),
                    (xa, ya - th),
                ],
                closed=True,
                fc=rk.duty_hex(float(lime[i]), rk.GREEN, rk.PINK),
                ec="none",
                alpha=0.85,
                zorder=4,
            )
        )

    # Tyres, to say it rotates.
    for f in (0.25, 0.62):
        xa = x0 + w * f
        ya = y_hi + (y_lo - y_hi) * f
        ax.plot([xa, xa], [ya + 0.012, ya - th - 0.012], color=rk.DIM, lw=3, zorder=5)

    # Feed end.
    rk.flow_arrow(
        ax,
        x0 - 0.075,
        y_hi - th * 0.55,
        x0 - 0.005,
        y_hi - th * 0.55,
        color=rk.CYAN,
        lw=1.8,
    )
    rk.label(
        ax,
        x0 + 0.01,
        y_hi + 0.045,
        f"HOT MEAL  {float(state.raw_meal):.1f} kg/s   ·   800 C, 92 % calcined",
        color=rk.CYAN,
        size=7,
        ha="left",
    )

    # Burner end.
    fuel_f = rk.clamp01(
        (float(state.fuel) - params.fuel_min) / (params.fuel_max - params.fuel_min)
    )
    rk.glow(
        ax, x1 - 0.06, y_lo - th * 0.5, 0.10, color=rk.AMBER, strength=fuel_f, zorder=1
    )
    rk.flow_arrow(
        ax,
        x1 + 0.075,
        y_lo - th * 0.5,
        x1 - 0.02,
        y_lo - th * 0.5,
        color=rk.AMBER,
        lw=1.4 + 2.4 * fuel_f,
    )
    rk.label(
        ax,
        x1 + 0.08,
        y_lo - th * 0.5 + 0.055,
        f"BURNER\n{float(state.fuel):.2f} kg/s",
        color=rk.AMBER,
        size=7,
        ha="left",
    )
    rk.flow_arrow(
        ax,
        x1 - 0.02,
        y_lo - th - 0.035,
        x1 + 0.06,
        y_lo - th - 0.055,
        color=rk.ORANGE,
        lw=1.8,
    )
    rk.label(
        ax,
        x1 + 0.02,
        y_lo - th - 0.10,
        f"CLINKER\nlime {float(discharge_lime(state)) * 100:.2f} %",
        color=rk.ORANGE,
        size=7,
        ha="left",
    )

    # Counter-current exhaust.
    rk.flow_arrow(
        ax,
        x0 + 0.10,
        y_hi + 0.030,
        x0 - 0.02,
        y_hi + 0.045,
        color=rk.ORANGE,
        lw=1.6,
        alpha=0.8,
    )
    rk.label(
        ax,
        x0 + 0.13,
        y_hi + 0.085,
        f"exhaust {float(state.T_exhaust) - 273:.0f} C -> precalciner",
        color=rk.ORANGE,
        size=6.5,
        ha="left",
    )

    rk.label(
        ax,
        0.5,
        0.94,
        f"BURNING ZONE {float(burning_zone_temperature(state)) - 273:.0f} C"
        f"   ·   {float(state.rpm):.2f} rpm   ·   residence {tau_min:.1f} min",
        color=rk.TEXT,
        size=9,
        weight="bold",
    )
    rk.label(
        ax,
        0.5,
        0.895,
        "● the whole axial profile is HIDDEN from the controller",
        color=rk.PURPLE,
        size=6.5,
    )
    rk.label(
        ax, x0, y_lo - th - 0.075, "charge temperature", color=rk.DIM, size=6, ha="left"
    )
    rk.label(
        ax,
        x0,
        y_lo - th - 0.115,
        "free lime  (green = burnt out)",
        color=rk.DIM,
        size=6,
        ha="left",
    )
    rk.caption(ax, "material takes half an hour to cross this picture")


def render_cement_kiln(state, params, step, history):
    lime = float(discharge_lime(state)) * 100.0
    history["t"].append(step * params.delta_t / 60.0)
    history["lime"].append(lime)
    history["target"].append(float(state.target_lime) * 100.0)
    history["T_bz"].append(float(burning_zone_temperature(state)) - 273.0)
    history["T_exh"].append(float(state.T_exhaust) - 273.0)
    history["fuel"].append(float(state.fuel))
    history["rpm"].append(float(state.rpm))
    history["feed"].append(float(state.raw_meal))
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(lime - float(state.target_lime) * 100.0)
    status = rk.NOMINAL if err < 0.3 else (rk.WATCH if err < 1.0 else rk.ALARM)
    T_bz = float(burning_zone_temperature(state))
    bz_span = params.T_bz_max - params.T_bz_min

    gauges = [
        rk.Gauge(
            "FREE LIME",
            f"{lime:.3f} %",
            rk.clamp01(lime / 6.0),
            rk.PINK,
            target_frac=float(state.target_lime) * 100.0 / 6.0,
        ),
        rk.Gauge(
            "TARGET",
            f"{float(state.target_lime) * 100:.3f} %",
            float(state.target_lime) * 100.0 / 6.0,
            rk.DIM,
        ),
        rk.Gauge(
            "BURNING ZONE",
            f"{T_bz - 273:.0f} C",
            (T_bz - params.T_bz_min) / bz_span,
            rk.ORANGE,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "BACK END",
            f"{float(state.T_exhaust) - 273:.0f} C",
            (float(state.T_exhaust) - 900.0) / 500.0,
            rk.AMBER,
        ),
        rk.Gauge(
            "FUEL",
            f"{float(state.fuel):.3f} kg/s",
            (float(state.fuel) - params.fuel_min) / (params.fuel_max - params.fuel_min),
            rk.RED,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "KILN SPEED",
            f"{float(state.rpm):.2f} rpm",
            (float(state.rpm) - params.rpm_min) / (params.rpm_max - params.rpm_min),
            rk.CYAN,
        ),
        rk.Gauge(
            "FEED",
            f"{float(state.raw_meal):.1f} kg/s",
            float(state.raw_meal) / (1.5 * params.raw_meal_nominal),
            rk.TEAL,
        ),
        rk.Gauge(
            "HEAT",
            f"{float(specific_heat_consumption(state, params)):.2f} MJ/kg",
            float(specific_heat_consumption(state, params)) / 5.0,
            rk.GREEN,
        ),
    ]

    strips = [
        rk.Strip(
            history["t"],
            [
                rk.Series(
                    history["lime"], "free lime", rk.PINK, fill_to=history["target"]
                ),
                rk.Series(
                    history["target"], "target", "white", ls="--", lw=1.0, alpha=0.6
                ),
            ],
            ylabel="free lime  [%]",
            bands=[(0.5, 2.0, rk.GREEN)],
        ),
        rk.Strip(
            history["t"],
            [
                rk.Series(history["T_bz"], "burning zone", rk.ORANGE),
                rk.Series(history["T_exh"], "back end", rk.AMBER, lw=1.1, alpha=0.75),
            ],
            ylabel="temperature  [C]",
            xlabel="minutes",
            lines=[
                (params.T_bz_max - 273, rk.RED, "rings"),
                (params.T_bz_min - 273, rk.BLUE, "cold"),
            ],
        ),
    ]
    fig = rk.frame(
        title="CEMENT  ROTARY  KILN  —  3000 t/day",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_kiln(ax, state, params),
        schematic_title="KILN  AXIS",
        gauges=gauges,
        strips=strips,
        status=status,
        subtitle="transport delay ~25 min  ·  64 states behind 8 measurements",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_cement_kiln, HISTORY_KEYS, stride=4)
