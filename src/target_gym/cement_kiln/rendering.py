"""Control-room rendering for the cement rotary kiln.

The schematic is an isometric view of the kiln line: the preheater tower and
precalciner at the feed end, the kiln on its tyres and piers with its girth gear
and drive, the firing hood with the burner pipe, and the grate cooler. The kiln
is drawn to its model's length and diameter, and its axial profile is painted on
the shell in three bands: the gas, the charge by its temperature, and free lime
(green once burnt out). That way the burning zone and the burn-out front read
as positions along the kiln rather than as two numbers. The whole profile is
hidden from the controller, and the half-hour it takes material to cross the
kiln is the task.
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

_STEEL = "#56606a"
_CONCRETE = "#3a4450"
_ZX = 1.4  # heights exaggerated, as a stylised plant drawing
_XS = 0.55  # lengths shortened, so the line fills the panel
_SLOPE = 0.035  # the kiln's fall along its length, the usual 3.5 %, decor
_AXIS0 = 8.0  # m, the kiln's axis over the ground at its feed end, decor
_TOWER = (-16.0, -4.0, 26.0)  # m: the preheater's x span and height, decor


def _axis_z(x):
    return _AXIS0 - _SLOPE * x


def _draw_preheater(ax, iso, state, R):
    """Preheater tower (frame and four cyclones) and the precalciner."""
    x0, x1, top = _TOWER
    T_ex = float(state.T_exhaust)
    for xx in (x0, x1):
        for yy in (-4.0, 4.0):
            iso.line(
                ax, [(xx, yy, 0.0), (xx, yy, top)], color=rk.FRAME, lw=1.2, zorder=1
            )
    for zz in np.linspace(5.0, top, 5):
        iso.line(
            ax,
            [(x0, 4.0, zz), (x1, 4.0, zz), (x1, -4.0, zz)],
            color=rk.FRAME,
            lw=0.9,
            zorder=1,
        )
    # Precalciner and the riser from the kiln inlet to it.
    pc = rk.incandescence(T_ex, 500.0, 1700.0)
    iso.revolve(
        ax,
        (x1 + 1.0, -3.0, _axis_z(0.0) + R),
        (x1 + 1.0, -3.0, top - 4.0),
        1.6,
        1.6,
        lambda t: pc,
        n=20,
        zorder=2,
        cap=True,
    )
    iso.text(ax, x1 + 6.0, -3.0, top - 3.0, "PRECALCINER", color=rk.DIM, ha="left")
    # Four stages, gas cooling upward from the kiln inlet's exhaust to the top
    # stage (the top's 350 C is decor).
    for k, (zb, yy) in enumerate(
        ((top - 6.0, -1.5), (top - 11.0, 1.5), (top - 16.0, -1.5), (top - 21.0, 1.5))
    ):
        T_k = 623.0 + (T_ex - 623.0) * k / 3.0
        col = rk.incandescence(T_k, 500.0, 1700.0)
        cx = 0.5 * (x0 + x1) + (-2.0 if yy < 0 else 2.0)
        iso.revolve(
            ax,
            (cx, yy, zb),
            (cx, yy, zb + 2.5),
            0.3,
            2.0,
            lambda t: col,
            n=20,
            zorder=3,
        )
        iso.revolve(
            ax,
            (cx, yy, zb + 2.5),
            (cx, yy, zb + 5.0),
            2.0,
            2.0,
            lambda t: col,
            n=20,
            zorder=3,
            cap=True,
        )
    iso.text(
        ax,
        0.5 * (x0 + x1),
        0.0,
        top + 3.0,
        "PREHEATER",
        color=rk.DIM,
        size=6.5,
        weight="bold",
    )
    iso.box(
        ax,
        (x1 - 1.0, 0.0),
        (-1.5, 1.5),
        (0.0, _axis_z(0.0) + R + 1.0),
        color=_CONCRETE,
        zorder=4,
    )


def _draw_kiln(ax, state, params):
    T_solid = np.asarray(state.T_solid)
    T_gas = np.asarray(state.T_gas)
    lime = np.asarray(state.lime)
    tau_min = float(residence_time(state.rpm, params)) / 60.0
    L, R = float(params.length), 0.5 * float(params.diameter)
    _, _, top = _TOWER
    iso = rk.Iso(
        (_TOWER[0], L + 22.0),
        (-5.0, 5.0),
        (0.0, top + 4.0),
        zx=_ZX,
        box=(0.0, 1.0, 0.12, 0.88),
        x_deg=12.0,
        y_deg=35.0,
        x_scale=_XS,
    )

    # Ground.
    iso.poly(
        ax,
        [
            (_TOWER[0] - 2, -5, 0),
            (L + 22, -5, 0),
            (L + 22, 6, 0),
            (_TOWER[0] - 2, 6, 0),
        ],
        fc=rk.PANEL,
        ec=rk.FRAME,
        lw=0.8,
        zorder=0,
    )
    _draw_preheater(ax, iso, state, R)

    # Piers with their rollers under the three tyres, and the drive.
    tyres = (0.15 * L, 0.55 * L, 0.90 * L)
    for xt in tyres:
        zt = _axis_z(xt) - R - 0.6
        iso.box(
            ax, (xt - 1.2, xt + 1.2), (-2.5, 2.5), (0.0, zt), color=_CONCRETE, zorder=1
        )
    xg = 0.35 * L
    iso.box(ax, (xg - 1.5, xg + 1.5), (3.0, 5.5), (0.0, 1.8), color=rk.WELL, zorder=1)
    iso.text(ax, xg, 5.5, -1.2, "DRIVE", color=rk.DIM, size=5.5)

    # The kiln, segment by segment, its profile painted on the shell: from the
    # top, steel, the gas, the charge by its temperature, free lime, steel.
    n = len(T_solid)
    dx = L / n

    def bands(i):
        gas = rk.incandescence(T_gas[i], 900.0, 2100.0)
        bed = rk.incandescence(T_solid[i], 700.0, 1650.0)
        lim = rk.duty_hex(float(lime[i]), rk.GREEN, rk.PINK)

        def colour(t):
            d = np.degrees(t)
            if d > 105.0 or d < -55.0:
                return _STEEL
            if d > 55.0:
                return gas
            if d > 5.0:
                return bed
            return lim

        return colour

    for i in range(n):
        xa, xb = i * dx, (i + 1) * dx
        iso.revolve(
            ax,
            (xa, 0.0, _axis_z(xa)),
            (xb, 0.0, _axis_z(xb)),
            R,
            R,
            bands(i),
            n=36,
            zorder=4,
        )
        for xt in tyres:
            if xa <= xt < xb:
                iso.revolve(
                    ax,
                    (xt - 0.6, 0.0, _axis_z(xt - 0.6)),
                    (xt + 0.6, 0.0, _axis_z(xt + 0.6)),
                    R * 1.15,
                    R * 1.15,
                    lambda t: rk.shade(_STEEL, 1.25),
                    n=36,
                    zorder=5,
                )
        if xa <= xg < xb:
            iso.revolve(
                ax,
                (xg - 0.4, 0.0, _axis_z(xg - 0.4)),
                (xg + 0.4, 0.0, _axis_z(xg + 0.4)),
                R * 1.25,
                R * 1.25,
                lambda t: "#6a5a3a",
                n=36,
                zorder=5,
            )
    fuel_f = rk.clamp01(
        (float(state.fuel) - params.fuel_min) / (params.fuel_max - params.fuel_min)
    )

    # Firing hood, with the burner pipe through it.
    zL = _axis_z(L)
    iso.box(
        ax,
        (L - 0.5, L + 3.5),
        (-3.5, 3.5),
        (0.0, zL + R + 2.0),
        color="#2a3440",
        zorder=6,
    )
    u, v = iso.p(L + 3.5, 0.0, zL)
    rk.glow(ax, float(u), float(v), 0.06, color=rk.AMBER, strength=fuel_f, zorder=7)
    iso.line(
        ax,
        # Coaxial with the kiln, out through the hood.
        [(L + 3.5, 0.0, zL), (L + 14.0, 0.0, _axis_z(L + 14.0))],
        color="#8a96a2",
        lw=3.0,
        zorder=7,
    )
    iso.text(
        ax,
        L + 14.0,
        0.0,
        zL + 4.5,
        f"BURNER\n{float(state.fuel):.2f} kg/s",
        color=rk.AMBER,
        size=7,
        zorder=8,
    )
    iso.text(ax, L + 1.5, -3.5, zL + R + 3.2, "HOOD", color=rk.DIM, size=5.5)

    # Grate cooler past the hood: the clinker bed on its top, air from below.
    clinker = rk.incandescence(float(T_solid[-1]), 700.0, 1650.0)
    iso.box(ax, (L + 3.5, L + 20.0), (-3.0, 3.0), (0.0, 3.0), color="#2a3440", zorder=6)
    iso.poly(
        ax,
        [
            (L + 3.5, -2.6, 3.01),
            (L + 20.0, -2.6, 3.01),
            (L + 20.0, 2.6, 3.01),
            (L + 3.5, 2.6, 3.01),
        ],
        fc=rk.shade(clinker, 0.75),
        ec="none",
        zorder=7,
    )
    iso.text(ax, L + 12.0, 3.0, -1.4, "GRATE COOLER", color=rk.DIM, size=5.5)

    # Flows: hot meal in, exhaust up the riser, clinker out.
    u0, v0 = iso.p(_TOWER[1], 0.0, _axis_z(0.0) + 4.0)
    u1, v1 = iso.p(0.5, 0.0, _axis_z(0.0) + 0.5)
    rk.flow_arrow(
        ax, float(u0), float(v0), float(u1), float(v1), color=rk.CYAN, lw=1.8, zorder=8
    )

    rk.label(
        ax,
        0.98,
        0.97,
        f"BURNING ZONE {float(burning_zone_temperature(state)) - 273:.0f} C"
        f"  ·  {float(state.rpm):.2f} rpm  ·  residence {tau_min:.1f} min",
        color=rk.TEXT,
        size=8,
        weight="bold",
        ha="right",
    )
    rk.label(
        ax,
        0.98,
        0.93,
        f"HOT MEAL {float(state.raw_meal):.1f} kg/s, 800 C  ·  exhaust "
        f"{float(state.T_exhaust) - 273:.0f} C -> precalciner",
        color=rk.CYAN,
        size=6.5,
        ha="right",
    )
    rk.label(
        ax,
        0.98,
        0.0,
        f"CLINKER  lime {float(discharge_lime(state)) * 100:.2f} %",
        color=rk.ORANGE,
        size=7,
        ha="right",
    )
    rk.label(
        ax,
        0.02,
        0.07,
        "shell bands: gas · charge T · free lime (green = burnt out) ●",
        color=rk.PURPLE,
        size=6,
        ha="left",
    )
    rk.label(
        ax,
        0.02,
        0.035,
        f"lengths ×{_XS:g}, heights ×{_ZX:g}  ·  axial profile hidden",
        color=rk.PURPLE,
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
        schematic_title="KILN  LINE",
        gauges=gauges,
        strips=strips,
        status=status,
        status_text=status_text,
        subtitle="transport delay ~25 min  ·  64 states behind 8 measurements",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_cement_kiln, HISTORY_KEYS)
