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


def _cyclone(ax, x, y, w, h, colour):
    """One preheater cyclone: a cylinder over a cone."""
    ax.add_patch(
        rk.patches.Polygon(
            [
                (x, y + h),
                (x + w, y + h),
                (x + w, y + 0.45 * h),
                (x + 0.6 * w, y),
                (x + 0.4 * w, y),
                (x, y + 0.45 * h),
            ],
            closed=True,
            fc=colour,
            ec=rk.DIM,
            lw=1.0,
            alpha=0.85,
            zorder=4,
        )
    )


def _draw_plant(ax, state, params, y_hi):
    """Decor around the kiln: preheater, precalciner, stack and ground."""
    ax.axhspan(0.0, 0.12, color=rk.PANEL, zorder=0)
    ax.axhline(0.12, color=rk.FRAME, lw=1.2, zorder=1)
    # Preheater tower: steel frame and four cyclone stages, gas cooling upward
    # from the kiln inlet's exhaust to the top stage (the top's 350 C is decor).
    fx0, fx1 = 0.03, 0.17
    for xx in (fx0, fx1):
        ax.plot([xx, xx], [0.12, 0.90], color=rk.FRAME, lw=1.6, zorder=2)
    for yy in np.linspace(0.20, 0.86, 5):
        ax.plot([fx0, fx1], [yy, yy], color=rk.FRAME, lw=1.0, zorder=2)
    T_ex = float(state.T_exhaust)
    for k, (yy, xx) in enumerate(
        ((0.74, 0.045), (0.58, 0.095), (0.42, 0.045), (0.26, 0.095))
    ):
        T_k = 623.0 + (T_ex - 623.0) * k / 3.0
        _cyclone(ax, xx, yy, 0.055, 0.12, rk.incandescence(T_k, 500.0, 1700.0))
    rk.label(ax, 0.10, 0.93, "PREHEATER", color=rk.DIM, size=6.5, weight="bold")
    # Precalciner and the riser from the kiln inlet.
    ax.add_patch(
        rk.patches.FancyBboxPatch(
            (0.19, 0.62),
            0.035,
            0.24,
            boxstyle=rk.patches.BoxStyle("Round", pad=0.008),
            fc=rk.incandescence(T_ex, 500.0, 1700.0),
            ec=rk.DIM,
            lw=1.0,
            alpha=0.8,
            zorder=3,
        )
    )
    rk.label(ax, 0.25, 0.74, "PRECALCINER", color=rk.DIM, size=5.5, rotation=90)
    rk.pipe(
        ax, [(0.235, y_hi - 0.01), (0.235, 0.62)], color=rk.ORANGE, lw=3.0, alpha=0.6
    )
    # Stack and ID fan over the tower.
    ax.add_patch(
        rk.patches.Rectangle(
            (0.155, 0.90), 0.02, 0.09, fc=rk.WELL, ec=rk.DIM, lw=1.0, zorder=3
        )
    )
    rk.disc(ax, 0.165, 0.905, 0.012, ec=rk.DIM)


def _draw_kiln(ax, state, params):
    T_solid = np.asarray(state.T_solid)
    T_gas = np.asarray(state.T_gas)
    lime = np.asarray(state.lime)
    tau_min = float(residence_time(state.rpm, params)) / 60.0

    x0, x1 = 0.27, 0.80
    w = x1 - x0
    # Kiln shell, on a slope (exaggerated): feed end high, burner end low.
    y_hi, y_lo, th = 0.66, 0.50, 0.15
    _draw_plant(ax, state, params, y_hi)
    along = lambda f: (x0 + w * f, y_hi + (y_lo - y_hi) * f)  # noqa: E731

    # Support piers with their rollers under each tyre, and the tyres.
    for f in (0.15, 0.55, 0.90):
        xa, ya = along(f)
        base = ya - th - 0.035
        ax.add_patch(
            rk.patches.Polygon(
                [
                    (xa - 0.03, 0.12),
                    (xa + 0.03, 0.12),
                    (xa + 0.018, base),
                    (xa - 0.018, base),
                ],
                closed=True,
                fc="#2a3440",
                ec=rk.FRAME,
                lw=1.0,
                zorder=1,
            )
        )
        rk.disc(ax, xa - 0.012, base + 0.012, 0.011, fc=rk.WELL, ec=rk.DIM, zorder=2)
        rk.disc(ax, xa + 0.012, base + 0.012, 0.011, fc=rk.WELL, ec=rk.DIM, zorder=2)
        ax.add_patch(
            rk.patches.Rectangle(
                (xa - 0.009, ya - th - 0.016),
                0.018,
                th + 0.032,
                fc="#56606a",
                ec=rk.DIM,
                lw=1.0,
                zorder=6,
            )
        )

    # Girth gear and the drive on the ground.
    xg, yg = along(0.35)
    ax.add_patch(
        rk.patches.Rectangle(
            (xg - 0.006, yg - th - 0.022),
            0.012,
            th + 0.044,
            fc="#6a5a3a",
            ec=rk.DIM,
            lw=1.0,
            hatch="||",
            zorder=6,
        )
    )
    ax.add_patch(
        rk.patches.Rectangle(
            (xg - 0.035, 0.12), 0.07, 0.05, fc=rk.WELL, ec=rk.DIM, lw=1.0, zorder=2
        )
    )
    ax.plot([xg, xg], [0.17, yg - th - 0.022], color=rk.DIM, lw=1.2, zorder=2)
    rk.label(ax, xg, 0.145, "DRIVE", color=rk.DIM, size=5.5, zorder=7)

    shell = [(x0, y_hi), (x1, y_lo), (x1, y_lo - th), (x0, y_hi - th)]
    ax.add_patch(
        rk.patches.Polygon(
            shell, closed=True, fc="#0b1420", ec="#56606a", lw=3.0, zorder=2
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

    # Firing hood at the discharge, with the burner pipe and its flame.
    fuel_f = rk.clamp01(
        (float(state.fuel) - params.fuel_min) / (params.fuel_max - params.fuel_min)
    )
    ax.add_patch(
        rk.patches.Rectangle(
            (x1 - 0.005, 0.30),
            0.075,
            y_lo + 0.04 - 0.30,
            fc="#1a2430",
            ec=rk.DIM,
            lw=1.4,
            zorder=1,
        )
    )
    rk.label(ax, x1 + 0.033, y_lo + 0.06, "HOOD", color=rk.DIM, size=5.5)
    yb = y_lo - th * 0.5
    flame = w * params.flame_length / params.length
    ax.add_patch(
        rk.patches.Polygon(
            [
                (x1 - 0.01, yb + 0.014),
                (x1 - 0.01 - flame, yb + flame * (y_hi - y_lo) / w),
                (x1 - 0.01, yb - 0.014),
            ],
            closed=True,
            fc=rk.AMBER,
            ec="none",
            alpha=0.25 + 0.5 * fuel_f,
            zorder=5,
        )
    )
    rk.glow(ax, x1 - 0.05, yb, 0.08, color=rk.AMBER, strength=fuel_f, zorder=5)
    rk.pipe(ax, [(0.99, yb), (x1 - 0.008, yb)], color="#8a96a2", lw=3.2, zorder=6)
    rk.label(
        ax,
        0.985,
        yb + 0.05,
        f"BURNER\n{float(state.fuel):.2f} kg/s",
        color=rk.AMBER,
        size=7,
        ha="right",
    )

    # Grate cooler under the hood: clinker in, air up through the grate,
    # secondary air back to the kiln.
    cx0, cx1, cy0, cy1 = x1 - 0.005, 0.99, 0.15, 0.29
    ax.add_patch(
        rk.patches.Rectangle(
            (cx0, cy0), cx1 - cx0, cy1 - cy0, fc="#1a2430", ec=rk.DIM, lw=1.4, zorder=1
        )
    )
    clinker_c = rk.incandescence(float(T_solid[-1]), 700.0, 1650.0)
    ax.add_patch(
        rk.patches.Rectangle(
            (cx0 + 0.005, cy0 + 0.035),
            cx1 - cx0 - 0.01,
            0.025,
            fc=clinker_c,
            ec="none",
            alpha=0.9,
            zorder=2,
        )
    )
    ax.plot(
        [cx0 + 0.005, cx1 - 0.005],
        [cy0 + 0.033] * 2,
        color=rk.DIM,
        lw=1.0,
        ls=(0, (2, 1)),
        zorder=3,
    )
    for xf in np.linspace(cx0 + 0.03, cx1 - 0.03, 3):
        rk.flow_arrow(
            ax, xf, cy0 + 0.005, xf, cy0 + 0.03, color=rk.BLUE, lw=1.0, mutation=7
        )
    rk.flow_arrow(
        ax,
        x1 + 0.03,
        cy1 - 0.01,
        x1 + 0.03,
        y_lo - th + 0.01,
        color=rk.ORANGE,
        lw=1.2,
        alpha=0.7,
    )
    rk.label(
        ax,
        (cx0 + cx1) / 2,
        cy1 - 0.035,
        "GRATE COOLER",
        color=rk.DIM,
        size=5.5,
        zorder=7,
    )
    rk.label(
        ax,
        0.985,
        0.035,
        f"CLINKER  lime {float(discharge_lime(state)) * 100:.2f} %",
        color=rk.ORANGE,
        size=7,
        ha="right",
    )

    # Hot meal into the kiln inlet, exhaust up the riser.
    rk.flow_arrow(
        ax, 0.17, y_hi - th * 0.55, x0 - 0.005, y_hi - th * 0.55, color=rk.CYAN, lw=1.8
    )
    rk.label(
        ax,
        x0 + 0.02,
        y_hi + 0.05,
        f"HOT MEAL {float(state.raw_meal):.1f} kg/s  ·  800 C, 92 % calcined",
        color=rk.CYAN,
        size=6.5,
        ha="left",
    )
    rk.label(
        ax,
        x0 + 0.02,
        y_hi + 0.09,
        f"exhaust {float(state.T_exhaust) - 273:.0f} C -> precalciner",
        color=rk.ORANGE,
        size=6.5,
        ha="left",
    )

    rk.label(
        ax,
        0.58,
        0.94,
        f"BURNING ZONE {float(burning_zone_temperature(state)) - 273:.0f} C"
        f"  ·  {float(state.rpm):.2f} rpm  ·  residence {tau_min:.1f} min",
        color=rk.TEXT,
        size=8.5,
        weight="bold",
    )
    rk.label(
        ax,
        0.58,
        0.895,
        "● the whole axial profile is HIDDEN from the controller",
        color=rk.PURPLE,
        size=6.5,
    )
    rk.label(
        ax,
        0.25,
        0.075,
        "bands: gas · charge T · free lime (green = burnt out)",
        color=rk.DIM,
        size=6,
        ha="left",
    )
    rk.caption(ax, "material takes half an hour to cross the kiln")


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
    band = params.lime_band * 100.0
    lime_max = params.lime_max * 100.0
    if lime >= 0.8 * lime_max:
        status, status_text = rk.ALARM, "RUNNING RAW"
    else:
        status = (
            rk.NOMINAL if err < 0.5 * band else (rk.WATCH if err < band else rk.ALARM)
        )
        status_text = None
    T_bz = float(burning_zone_temperature(state))
    bz_span = params.T_bz_max - params.T_bz_min
    lime_span = 1.2 * lime_max  # the gauge's full scale, past the trip

    gauges = [
        rk.Gauge(
            "FREE LIME",
            f"{lime:.3f} %",
            rk.clamp01(lime / lime_span),
            rk.PINK,
            limit_frac=lime_max / lime_span,
            target_frac=float(state.target_lime) * 100.0 / lime_span,
        ),
        rk.Gauge(
            "TARGET",
            f"{float(state.target_lime) * 100:.3f} %",
            float(state.target_lime) * 100.0 / lime_span,
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
            bands=[tuple(100.0 * v for v in params.target_lime_range) + (rk.GREEN,)],
            lines=[(lime_max, rk.RED, "raw trip")],
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
        status_text=status_text,
        subtitle="transport delay ~25 min  ·  64 states behind 8 measurements",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_cement_kiln, HISTORY_KEYS)
