"""Control-room rendering for the float glass furnace.

The schematic is a side section of a cross-fired regenerative furnace, drawn to
the model's own lengths: the batch charger at the back wall, the melter under its
crown with a burner port per bay, the unfired zone, the waist, the working end
and the canal that feeds the float bath. The crown is tinted by each bay's hot
face, the flames by each bay's gas, the melt cell by cell by its temperature, and
the batch blanket lies on the melt as far as it reaches. The agent reads only the
instruments marked so: the crown couple, not the hottest face that the plant's
override protects against the crown limit.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.glass_furnace import physical as P

HISTORY_KEYS = ("t", "crown", "canal", "fining", "pass", "fuel", "reward")

_MELT = (1300.0, 1900.0)  # K, the colour scale of the melt and the crown
_BRICK, _BRICK_EDGE = "#3a2824", "#5c4038"  # refractory, muted for the dark console
_CANAL_M = 4.0  # m of canal drawn past the working end, decor
_X0, _X1 = 0.07, 0.95  # the furnace's extent in the schematic
_FLOOR, _SURF = 0.26, 0.42  # the bath's floor and surface (depth exaggerated)


def _brick(ax, x, y, w, h, zorder=2):
    """A block of refractory, hatched in courses."""
    ax.add_patch(
        rk.patches.Rectangle(
            (x, y),
            w,
            h,
            fc=_BRICK,
            ec=_BRICK_EDGE,
            lw=0.8,
            hatch="--",
            zorder=zorder,
        )
    )


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
    edges = np.concatenate([[0.0], np.cumsum(dx)])
    total = edges[-1] + _CANAL_M
    X = lambda m: _X0 + (_X1 - _X0) * m / total  # noqa: E731
    zone_ends = np.cumsum([a.nm, g.n_refiner, g.n_waist, g.n_end])
    melter_end, refiner_end, waist_end, end_end = (edges[k] for k in zone_ends)

    # Ground, floor and back wall.
    ax.axhspan(0.0, 0.17, color=rk.PANEL, zorder=0)
    _brick(ax, X(0) - 0.02, 0.17, X(end_end) - X(0) + 0.02, _FLOOR - 0.17)
    _brick(ax, X(0) - 0.02, _FLOOR, 0.02, 0.38)

    # The melt, cell by cell.
    zs = _SURF - (_SURF - _FLOOR) * np.concatenate([[0.0], np.cumsum(dz)]) / dz.sum()
    for j in range(T.shape[1]):
        for i in range(T.shape[0]):
            ax.add_patch(
                rk.patches.Rectangle(
                    (X(edges[j]), zs[i + 1]),
                    X(edges[j + 1]) - X(edges[j]),
                    zs[i] - zs[i + 1],
                    fc=_glow(T[i, j]),
                    ec="none",
                    zorder=3,
                )
            )

    # The batch blanket floating on the melter's surface, and the foam.
    cover = np.asarray(P.cover(plant.m))
    for j, c in enumerate(cover):
        if c > 0.02:
            ax.add_patch(
                rk.patches.Rectangle(
                    (X(edges[j]), _SURF),
                    (X(edges[j + 1]) - X(edges[j])) * c,
                    0.014,
                    fc="#b8b0a4",
                    ec="none",
                    zorder=4,
                )
            )
    rk.label(
        ax,
        X(0) + 0.01,
        _SURF + 0.03,
        "batch",
        color="#b8b0a4",
        size=6,
        ha="left",
        zorder=7,
    )

    # Charger at the back wall, with its hopper.
    ax.add_patch(
        rk.patches.Polygon(
            [
                (X(0) - 0.065, 0.60),
                (X(0) - 0.015, 0.60),
                (X(0) - 0.03, 0.50),
                (X(0) - 0.05, 0.50),
            ],
            fc=rk.WELL,
            ec=rk.DIM,
            lw=1.0,
            zorder=4,
        )
    )
    rk.flow_arrow(
        ax, X(0) - 0.04, 0.49, X(0) + 0.01, _SURF + 0.02, color="#b8b0a4", lw=1.4
    )
    rk.label(ax, X(0) - 0.04, 0.63, "CHARGER", color=rk.DIM, size=6)

    # Breast walls and the crown arch over the melter and the unfired zone,
    # each melter bay tinted by its hot face.
    wall_top = 0.60
    _brick(ax, X(0) - 0.02, _SURF, 0.02, wall_top - _SURF)
    bay_w = melter_end / g.n_bays
    for k in range(g.n_bays):
        xa, xb = X(k * bay_w), X((k + 1) * bay_w)
        ax.add_patch(
            rk.patches.Rectangle(
                (xa, wall_top),
                xb - xa,
                0.035,
                fc=_glow(roof[k]),
                ec=_BRICK_EDGE,
                lw=0.8,
                alpha=0.9,
                zorder=4,
            )
        )
    arch_x = np.linspace(X(0) - 0.02, X(refiner_end), 60)
    mid, half = 0.5 * (arch_x[0] + arch_x[-1]), 0.5 * (arch_x[-1] - arch_x[0])
    rise = 0.14 * np.sqrt(np.clip(1.0 - ((arch_x - mid) / half) ** 2, 0.0, 1.0))
    top = wall_top + 0.035 + rise
    ax.fill_between(
        arch_x,
        wall_top + 0.035,
        top,
        color=_BRICK,
        ec=_BRICK_EDGE,
        lw=0.8,
        hatch="--",
        zorder=3,
    )
    ax.plot(arch_x, top + 0.02, color=_BRICK_EDGE, lw=1.0, zorder=3)
    hot = int(np.argmax(roof))
    rk.label(
        ax,
        X((hot + 0.5) * bay_w),
        top.max() + 0.045,
        "▼ hottest face ●",
        color=rk.RED,
        size=6.5,
    )
    tc = X((P.COUPLE_BAY + 0.5) * bay_w)
    ax.plot(
        [tc, tc], [wall_top + 0.04, wall_top + 0.10], color=rk.CYAN, lw=1.6, zorder=6
    )
    rk.label(ax, tc, wall_top + 0.115, "TC", color=rk.CYAN, size=6.5, zorder=7)
    # The flue gas leaves through the regenerators to the stack (decor).
    sx = X(refiner_end) + 0.012
    _brick(ax, sx, 0.57, 0.03, 0.36, zorder=1)
    rk.label(ax, sx + 0.015, 0.95, "STACK", color=rk.DIM, size=6)

    # A burner port per bay, its flame across the melt by the bay's gas and share.
    fire = rk.clamp01(
        (float(state.fuel) - params.fuel_min) / (params.fuel_max - params.fuel_min)
    )
    shares = np.asarray(params.plant.port_shares, dtype=float)
    for k in range(g.n_bays):
        xc = X((k + 0.5) * bay_w)
        pw = 0.55 * (X(bay_w) - X(0))
        ax.add_patch(
            rk.patches.FancyBboxPatch(
                (xc - pw / 2, wall_top - 0.045),
                pw,
                0.03,
                boxstyle=rk.patches.BoxStyle("Round", pad=0.006),
                fc="#0a0a0a",
                ec=rk.DIM,
                lw=1.0,
                zorder=5,
            )
        )
        strength = fire * shares[k] / shares.max()
        rk.glow(
            ax,
            xc,
            _SURF + 0.06,
            0.06,
            color=rk.ORANGE,
            strength=1.4 * strength,
            zorder=5,
        )
        ax.add_patch(
            rk.patches.Polygon(
                [
                    (xc - 0.4 * pw, wall_top - 0.05),
                    (xc + 0.4 * pw, wall_top - 0.05),
                    (xc + 0.15 * pw, _SURF + 0.025 + 0.06 * (1.0 - strength)),
                    (xc - 0.15 * pw, _SURF + 0.025 + 0.06 * (1.0 - strength)),
                ],
                closed=True,
                fc=_glow(gas[k]),
                ec="none",
                alpha=0.3 + 0.55 * strength,
                zorder=5,
            )
        )
    rk.label(
        ax,
        X(0.5 * (melter_end + refiner_end)),
        0.53,
        "◄ BURNER PORTS",
        color=rk.DIM,
        size=6,
        zorder=7,
    )

    # Waist and working end: lower, flat roofs, the working end cooled by air.
    for x0m, x1m, roof_y in (
        (refiner_end, waist_end, 0.50),
        (waist_end, end_end, 0.54),
    ):
        _brick(ax, X(x0m), roof_y, X(x1m) - X(x0m), 0.035)
        _brick(ax, X(x0m), _SURF + 0.03, 0.006, roof_y - _SURF - 0.03)
    cool = rk.clamp01(float(state.cooling) / params.cooling_max)
    xm = X(0.5 * (waist_end + end_end))
    for dxs in (-0.03, 0.0, 0.03):
        rk.flow_arrow(
            ax,
            xm + dxs,
            0.64,
            xm + dxs,
            0.585,
            color=rk.BLUE,
            lw=0.8 + 1.8 * cool,
            alpha=0.4 + 0.5 * cool,
        )
    rk.label(
        ax,
        xm,
        0.665,
        f"cooling {float(state.cooling) / 1e6:.2f} MW",
        color=rk.BLUE,
        size=6.5,
    )
    _brick(ax, X(end_end) - 0.006, _SURF + 0.03, 0.006, 0.54 - _SURF - 0.03)

    # Canal to the float bath.
    canal_y = _SURF - 0.04
    _brick(ax, X(end_end), 0.17, X(total) - X(end_end), canal_y - 0.17)
    ax.add_patch(
        rk.patches.Rectangle(
            (X(end_end), canal_y),
            X(total) - X(end_end),
            0.018,
            fc=_glow(T[0, -1]),
            ec="none",
            zorder=3,
        )
    )
    rk.flow_arrow(
        ax,
        X(end_end) + 0.005,
        canal_y + 0.03,
        X(total) + 0.03,
        canal_y + 0.03,
        color=rk.AMBER,
        lw=1.4,
    )
    rk.label(
        ax,
        X(total) + 0.005,
        canal_y + 0.075,
        f"to float bath\n{float(state.pull) * 86.4:.0f} t/d",
        color=rk.AMBER,
        size=6.5,
    )

    # Zone names along the floor.
    for x0m, x1m, name in (
        (0.0, melter_end, "MELTER"),
        (melter_end, refiner_end, "UNFIRED"),
        (refiner_end, waist_end, "WAIST"),
        (waist_end, end_end, "WORKING END"),
        (end_end, total, "CANAL"),
    ):
        rk.label(
            ax, X(0.5 * (x0m + x1m)), 0.135, name, color=rk.DIM, size=6.5, weight="bold"
        )
    stretch = ((_SURF - _FLOOR) / dz.sum()) / ((_X1 - _X0) / total)
    rk.label(
        ax,
        X(0.5 * end_end),
        0.09,
        f"melt field ● hidden  ·  depth ×{stretch:.0f}",
        color=rk.PURPLE,
        size=6.5,
    )
    rk.caption(
        ax,
        f"melt {T.min() - 273.15:.0f} to {T.max() - 273.15:.0f} C"
        f"  ·  {edges[-1]:.0f} m charger to canal",
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
        schematic_title="LONG  SECTION",
        gauges=gauges,
        strips=strips,
        status=status,
        status_text=status_text,
        subtitle="float glass  ·  fining read a transit late  ·  hours-long lags",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_glass_furnace, HISTORY_KEYS)
