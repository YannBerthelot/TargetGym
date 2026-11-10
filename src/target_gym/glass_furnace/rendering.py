"""Control-room rendering for the float glass furnace.

The schematic is an isometric cutaway of a cross-fired regenerative furnace,
drawn to the model's own lengths and cut along its centreline so the melt shows
in section: the batch charger at the back wall, the melter under its crown with
a burner port per bay, the unfired zone, the waist, the working end and the canal
that feeds the float bath. The crown is tinted by each bay's hot face, the flames
by each bay's gas, the melt cell by cell by its temperature, and the batch
blanket lies on the melt as far as it reaches. The agent reads only the
instruments marked so: the crown couple, not the hottest face that the plant's
override protects against the crown limit.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.glass_furnace import physical as P

HISTORY_KEYS = ("t", "crown", "canal", "fining", "pass", "fuel", "reward")

_MELT = (1300.0, 1900.0)  # K, the colour scale of the melt and the crown
_BRICK = "#5a3f36"  # refractory, muted for the dark console
_STEEL = "#4a5866"
_BATCH = "#b8b0a4"
_ZX = 3.0  # heights exaggerated, as a stylised plant drawing
_XS = 0.45  # lengths shortened, so the long plant fills the panel
_WALL = 0.8  # m of refractory, decor
_CANAL = (5.0, 1.0)  # m long and half-wide, decor
_SLAB = 0.8  # m of floor under the bath, decor


def _glow(T):
    return rk.incandescence(T, *_MELT)


def _draw_furnace(ax, state, params):
    g = params.grid
    a = P.layout(g)
    plant = state.plant
    T = np.asarray(plant.T)
    roof = np.asarray(plant.T_roof)[0]
    gas = np.asarray(plant.T_gas)
    dx, dz = np.asarray(a.dx), np.asarray(a.dz)
    half = 0.5 * np.asarray(a.wc)  # each column's half width
    W, D = float(a.width), float(dz.sum())
    H = 2.4  # m of gas space over the melter (physical._layout)
    xs = np.concatenate([[0.0], np.cumsum(dx)])
    end = xs[-1]
    total = end + _CANAL[0]
    yc = 0.5 * W  # the cut, along the centreline
    nm, nr = a.nm, a.nm + g.n_refiner
    nw = nr + g.n_waist
    iso = rk.Iso(
        (-4.0, total + 6.0),
        (-_WALL - 3.0, yc),
        (-D - _SLAB, H + 5.0),
        zx=_ZX,
        box=(0.0, 1.0, 0.12, 0.95),
        x_deg=10.0,
        x_scale=_XS,
        y_deg=25.0,
        y_scale=0.8,
    )
    zones = (  # (first column, last column + 1, wall height, name)
        (0, nm, H, "MELTER"),
        (nm, nr, H, "UNFIRED"),
        (nr, nw, 1.3, "WAIST"),
        (nw, len(dx), 1.6, "WORKING END"),
    )

    # The stack, behind the unfired zone (decor), and the back walls.
    iso.box(
        ax,
        (xs[nm] + 6.0, xs[nm] + 8.5),
        (-_WALL - 3.0, -_WALL - 0.5),
        (-D, H + 4.0),
        color=_BRICK,
        zorder=1,
    )
    iso.text(ax, xs[nm] + 7.2, -_WALL - 1.7, H + 4.6, "STACK", color=rk.DIM, size=6)
    for j0, j1, h, _ in zones:
        yb = yc - half[j0]
        iso.box(ax, (xs[j0], xs[j1]), (yb - _WALL, yb), (-D, h), color=_BRICK, zorder=1)
    # The charger's hopper behind the back end wall, standing over it.
    iso.box(
        ax,
        (-4.0, -_WALL - 0.3),
        (yc - 3.5, yc - 0.5),
        (0.0, H + 1.6),
        color=_STEEL,
        zorder=1,
    )
    iso.text(ax, -2.2, yc - 2.0, H + 2.6, "CHARGER", color=rk.DIM, size=6, zorder=8)
    # The back end wall, in front of the charger.
    iso.box(
        ax, (-_WALL, 0.0), (yc - half[0] - _WALL, yc), (-D, H), color=_BRICK, zorder=1
    )

    # A burner port per melter bay in the back wall.
    bay_w = xs[nm] / g.n_bays
    yb = yc - half[0]
    for k in range(g.n_bays):
        xc = (k + 0.5) * bay_w
        iso.poly(
            ax,
            [
                (xc - 1.0, yb, 0.5),
                (xc + 1.0, yb, 0.5),
                (xc + 1.0, yb, 1.5),
                (xc - 1.0, yb, 1.5),
            ],
            fc="#0a0a0a",
            ec=rk.DIM,
            lw=0.8,
            zorder=2,
        )

    # Floor slab, cut at the centreline.
    for j in range(len(dx)):
        iso.box(
            ax,
            (xs[j], xs[j + 1]),
            (yc - half[j] - _WALL, yc),
            (-D - _SLAB, -D),
            color=_BRICK,
            ec="none",
            zorder=2,
        )

    # The melt: its surface, and its section on the cut, cell by cell.
    zs = -np.concatenate([[0.0], np.cumsum(dz)])
    for j in range(len(dx)):
        x0, x1, y0 = xs[j], xs[j + 1], yc - half[j]
        iso.poly(
            ax,
            [(x0, y0, 0), (x1, y0, 0), (x1, yc, 0), (x0, yc, 0)],
            fc=_glow(T[0, j]),
            ec="none",
            zorder=3,
        )
        for i in range(T.shape[0]):
            iso.poly(
                ax,
                [
                    (x0, yc, zs[i + 1]),
                    (x1, yc, zs[i + 1]),
                    (x1, yc, zs[i]),
                    (x0, yc, zs[i]),
                ],
                fc=_glow(T[i, j]),
                ec="none",
                zorder=3,
            )
    # The batch blanket, as far as it reaches.
    cover = np.asarray(P.cover(plant.m))
    for j, c in enumerate(cover):
        if c > 0.02:
            x1 = xs[j] + (xs[j + 1] - xs[j]) * c
            iso.box(
                ax,
                (xs[j], x1),
                (yc - half[j], yc),
                (0.0, 0.12),
                color=_BATCH,
                ec="none",
                zorder=4,
            )
    iso.text(ax, 2.0, yc + 1.2, 0.3, "batch", color=_BATCH, size=6, zorder=8)

    # Flames from each port across the bath, by the bay's gas and its share.
    fire = rk.clamp01(
        (float(state.fuel) - params.fuel_min) / (params.fuel_max - params.fuel_min)
    )
    shares = np.asarray(params.plant.port_shares, dtype=float)
    for k in range(g.n_bays):
        xc, f = (k + 0.5) * bay_w, fire * shares[k] / shares.max()
        reach = yb + (0.4 + 0.5 * f) * half[0]
        iso.poly(
            ax,
            [
                (xc - 1.0, yb, 1.0),
                (xc + 1.0, yb, 1.0),
                (xc + 0.3, reach, 0.5),
                (xc - 0.3, reach, 0.5),
            ],
            fc=_glow(gas[k]),
            ec="none",
            alpha=0.35 + 0.55 * f,
            zorder=4,
        )

    # The crown's back half, translucent, each melter bay tinted by its hot face.
    ys = np.linspace(yb, yc, 12)
    rise = 0.45 * np.sqrt(np.clip(1.0 - ((yc - ys) / half[0]) ** 2, 0.0, 1.0))
    for k in range(g.n_bays + 1):
        x0 = k * bay_w if k < g.n_bays else xs[nm]
        x1 = (k + 1) * bay_w if k < g.n_bays else xs[nr]
        col = _glow(roof[k]) if k < g.n_bays else _BRICK
        arch = [(x0, y, H + r) for y, r in zip(ys, rise)]
        arch += [(x1, y, H + r) for y, r in zip(ys[::-1], rise[::-1])]
        iso.poly(
            ax, arch, fc=col, ec=rk.shade(_BRICK, 1.3), lw=0.6, alpha=0.55, zorder=5
        )
    iso.line(
        ax,
        [(0.0, yc, H + 0.45), (xs[nr], yc, H + 0.45)],
        color=rk.shade(_BRICK, 1.5),
        lw=1.6,
        zorder=6,
    )
    hot = int(np.argmax(roof))
    iso.text(
        ax,
        (hot + 0.5) * bay_w,
        yc - 2.0,
        H + 2.2,
        "▼ hottest face ●",
        color=rk.RED,
        size=6.5,
        zorder=8,
    )
    tc = (P.COUPLE_BAY + 0.5) * bay_w
    iso.line(
        ax,
        [(tc, yc - 1.5, H + 0.4), (tc, yc - 1.5, H + 1.3)],
        color=rk.CYAN,
        lw=1.6,
        zorder=7,
    )
    iso.text(ax, tc + 1.5, yc - 1.5, H + 1.3, "TC", color=rk.CYAN, size=6.5, zorder=8)

    # Waist and working end: flat roofs over the back half, the working end
    # cooled by air blown in through its roof.
    for j0, j1, h, _ in zones[2:]:
        iso.box(
            ax,
            (xs[j0], xs[j1]),
            (yc - half[j0] - _WALL, yc),
            (h, h + 0.35),
            color=_BRICK,
            alpha=0.9,
            zorder=5,
        )
    cool = rk.clamp01(float(state.cooling) / params.cooling_max)
    xm = 0.5 * (xs[nw] + end)
    for dxs in (-4.0, 0.0, 4.0):
        u0, v0 = iso.p(xm + dxs, yc - 2.0, 4.6)
        u1, v1 = iso.p(xm + dxs, yc - 2.0, 2.1)
        rk.flow_arrow(
            ax,
            float(u0),
            float(v0),
            float(u1),
            float(v1),
            color=rk.BLUE,
            lw=0.8 + 1.8 * cool,
            alpha=0.4 + 0.5 * cool,
            zorder=7,
        )
    iso.text(
        ax,
        xm,
        yc - 2.0,
        5.3,
        f"cooling {float(state.cooling) / 1e6:.2f} MW",
        color=rk.BLUE,
        size=6.5,
        zorder=8,
    )

    # The canal to the float bath.
    cw = _CANAL[1]
    iso.box(
        ax,
        (end, total),
        (yc - cw - _WALL, yc),
        (-D - _SLAB, -0.3),
        color=_BRICK,
        ec="none",
        zorder=5,
    )
    iso.poly(
        ax,
        [
            (end, yc - cw, -0.3),
            (total, yc - cw, -0.3),
            (total, yc, -0.3),
            (end, yc, -0.3),
        ],
        fc=_glow(T[0, -1]),
        ec="none",
        zorder=6,
    )
    # The canal's back wall, standing over its glass.
    iso.box(
        ax,
        (end, total),
        (yc - cw - _WALL, yc - cw),
        (-0.3, 0.6),
        color=_BRICK,
        zorder=6,
    )
    # Along the line, out of the canal's end.
    u0, v0 = iso.p(total - 2.0, yc - 0.5, -0.3)
    u1, v1 = iso.p(total + 6.0, yc - 0.5, -0.3)
    rk.flow_arrow(
        ax,
        float(u0),
        float(v0),
        float(u1),
        float(v1),
        color=rk.AMBER,
        lw=1.4,
        zorder=8,
    )
    iso.text(
        ax,
        total + 3.0,
        yc - 0.5,
        -1.6,
        f"to float bath\n{float(state.pull) * 86.4:.0f} t/d",
        color=rk.AMBER,
        size=6.5,
        zorder=8,
    )

    # Zone names on the cut, under the floor.
    for j0, j1, _, name in zones:
        iso.text(
            ax,
            0.5 * (xs[j0] + xs[j1]),
            yc + 1.0,
            -D - _SLAB - 1.2,
            name,
            color=rk.DIM,
            size=6.5,
            weight="bold",
            zorder=8,
        )
    rk.label(
        ax,
        0.02,
        0.98,
        f"cut on the centreline  ·  melt ● hidden  ·  lengths ×{_XS:g}, heights ×{_ZX:g}",
        color=rk.PURPLE,
        size=6.5,
        ha="left",
    )
    rk.caption(
        ax,
        f"melt {T.min() - 273.15:.0f} to {T.max() - 273.15:.0f} C"
        f"  ·  {end:.0f} m charger to canal",
    )


def render_glass_furnace(state, params, step, history):
    read = P.sensors(state.plant, params.grid, params.plant)
    hours = step * params.delta_t / 3600.0
    from target_gym.glass_furnace.env_jax import compute_reward

    history["t"].append(hours)
    history["crown"].append(float(read["crown"]) - 273.15)
    history["canal"].append(float(read["canal"]) - 273.15)
    history["fining"].append(float(read["fining"]))
    history["pass"].append(float(params.fining_pass))
    history["fuel"].append(float(state.fuel))
    history["reward"].append(float(compute_reward(state, params)))

    fining, canal = float(read["fining"]), float(read["canal"])
    short = fining < params.fining_pass or canal < params.canal_target
    status = rk.ALARM if short else rk.NOMINAL
    status_text = None
    if not short and float(read["hot"]) >= params.crown_high - params.override_margin:
        status, status_text = rk.WATCH, "CROWN OVERRIDE"
    to_frac = lambda T: (float(T) - _MELT[0]) / (_MELT[1] - _MELT[0])
    span = params.fuel_max - params.fuel_min
    gauges = [
        rk.Gauge(
            "CROWN",
            f"{float(read['crown']) - 273.15:.0f} C",
            to_frac(read["crown"]),
            rk.ORANGE,
        ),
        rk.Gauge(
            "HOTTEST FACE",
            f"{float(read['hot']) - 273.15:.0f} C",
            to_frac(read["hot"]),
            rk.RED,
            limit_frac=to_frac(params.crown_high),
            target_frac=to_frac(params.crown_high - params.override_margin),
            hidden=True,
        ),
        rk.Gauge(
            "CANAL",
            f"{canal - 273.15:.0f} C",
            to_frac(canal),
            rk.AMBER,
            target_frac=to_frac(params.canal_target),
        ),
        rk.Gauge(
            "FINING",
            f"{fining:.3f}",
            rk.clamp01(fining),
            rk.CYAN,
            target_frac=float(params.fining_pass),
        ),
        rk.Gauge(
            "REFINER COOLING",
            f"{float(state.cooling) / 1e6:.2f} MW",
            float(state.cooling) / params.cooling_max,
            rk.BLUE,
        ),
        rk.Gauge(
            "FUEL",
            f"{float(state.fuel):.3f} kg/s",
            (float(state.fuel) - params.fuel_min) / span,
            rk.GREEN,
        ),
        rk.Gauge(
            "PULL",
            f"{float(state.pull) * 86.4:.0f} t/d",
            rk.clamp01(float(state.pull) / (2.0 * params.pull)),
            rk.DIM,
        ),
    ]
    strips = [
        rk.Strip(
            history["t"],
            [
                rk.Series(history["fining"], "fining share", rk.CYAN),
                rk.Series(history["pass"], "pass mark", "white", ls="--", lw=1.0),
            ],
            ylabel="share fined",
            xlabel="hours",
        )
    ]
    fig = rk.frame(
        title="GLASS  FURNACE  —  FLOAT",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_furnace(ax, state, params),
        schematic_title="CUTAWAY",
        gauges=gauges,
        strips=strips,
        status=status,
        status_text=status_text,
        subtitle="float glass  ·  fining read a transit late  ·  hours-long lags",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_glass_furnace, HISTORY_KEYS)
