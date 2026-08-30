"""Control-room rendering for the grid battery.

The schematic is the pack: a stack of cells filled to state of charge, with the
grid tie above it. Current direction and colour say whether the battery is
being paid to deliver or to absorb, and the fill is the budget that makes this
task different from every thermal plant here -- tracking now spends the ability
to track later.
"""

from target_gym import render_kit as rk
from target_gym.energy.battery.env import compute_reward, open_circuit_voltage

HISTORY_KEYS = ("t", "power", "target", "soc", "T_cell", "v_cell", "reward")


def _draw_pack(ax, state, params):
    soc = float(state.soc)
    P = float(state.power) / 1e6
    T = float(state.T_cell)
    v_cell = float(open_circuit_voltage(state.soc, params)) / params.n_series

    # Grid tie.
    rk.label(ax, 0.5, 0.955, "GRID", color=rk.DIM, size=8, weight="bold")
    ax.plot([0.18, 0.82], [0.915, 0.915], color=rk.FRAME, lw=3, zorder=1)
    for x in (0.30, 0.50, 0.70):
        ax.plot([x, x], [0.915, 0.86], color=rk.FRAME, lw=2, zorder=1)

    discharging = P >= 0
    col = rk.AMBER if discharging else rk.GREEN
    mag = rk.clamp01(abs(P) / (params.power_max / 1e6))
    if mag > 0.01:
        for x in (0.30, 0.50, 0.70):
            y0, y1 = (0.80, 0.905) if discharging else (0.905, 0.80)
            rk.flow_arrow(
                ax, x, y0, x, y1, color=col, lw=1.0 + 2.4 * mag, mutation=9 + 9 * mag
            )
    rk.label(
        ax,
        0.5,
        0.795,
        f"{'DISCHARGING' if discharging else 'CHARGING'}  {abs(P):.2f} MW",
        color=col,
        size=8.5,
        weight="bold",
    )

    # Cell stack. Each module fills in turn, so state of charge reads as a
    # quantity of stored energy rather than a percentage.
    n_mod = 8
    x0, y0, w, h, gap = 0.14, 0.20, 0.72, 0.50, 0.012
    mw = (w - (n_mod - 1) * gap) / n_mod
    filled = soc * n_mod
    fill_c = rk.duty_hex(soc, "#b03a2e", rk.GREEN)
    for i in range(n_mod):
        mx = x0 + i * (mw + gap)
        rk.vessel(ax, mx, y0, mw, h, fc="#0b1420", ec=rk.FRAME, lw=1.2, pad=0.006)
        f = rk.clamp01(filled - i)
        if f > 0:
            rk.fill_level(
                ax,
                mx + 0.004,
                y0 + 0.006,
                mw - 0.008,
                h - 0.012,
                f,
                color=fill_c,
                alpha=0.72,
            )

    # Usable window. The stack fills left to right, so the limits are vertical
    # marks along the row -- a horizontal line would correspond to nothing.
    for lim, name in ((params.soc_min, "min"), (params.soc_max, "max")):
        lxx = x0 + w * lim
        ax.plot(
            [lxx, lxx],
            [y0 - 0.028, y0 + h + 0.028],
            color=rk.RED,
            lw=1.2,
            ls="--",
            alpha=0.75,
            zorder=7,
        )
        rk.label(ax, lxx, y0 + h + 0.048, name, color=rk.RED, size=6.5)

    rk.label(
        ax,
        0.5,
        y0 + h * 0.5,
        f"{soc * 100:.1f} %",
        color=rk.TEXT,
        size=17,
        weight="bold",
        zorder=8,
    )
    rk.label(
        ax, 0.5, y0 + h * 0.5 - 0.065, "state of charge", color=rk.DIM, size=7, zorder=8
    )

    # Thermal strip along the bottom of the pack.
    t_frac = rk.clamp01((T - params.T_ambient) / (params.T_max - params.T_ambient))
    ax.add_patch(
        rk.patches.Rectangle(
            (x0, 0.13),
            w,
            0.035,
            fc=rk.duty_hex(t_frac, "#1a3550", rk.RED),
            ec=rk.FRAME,
            lw=1,
            alpha=0.9,
            zorder=3,
        )
    )
    rk.label(
        ax,
        x0 + w / 2,
        0.147,
        f"CELL  {T:.1f} C   ·   {v_cell:.3f} V/cell",
        color=rk.TEXT,
        size=7.5,
        zorder=6,
    )
    rk.caption(ax, "2 MWh / 1 MW pack  ·  losses grow with the square of current")


def render_battery(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["power"].append(float(state.power) / 1e6)
    history["target"].append(float(state.target_power) / 1e6)
    history["soc"].append(float(state.soc))
    history["T_cell"].append(float(state.T_cell))
    history["v_cell"].append(
        float(open_circuit_voltage(state.soc, params)) / params.n_series
    )
    history["reward"].append(float(compute_reward(state, params)))

    err = abs(float(state.power) - float(state.target_power)) / 1e6
    status = rk.NOMINAL if err < 0.05 else (rk.WATCH if err < 0.2 else rk.ALARM)

    gauges = [
        rk.Gauge(
            "POWER",
            f"{float(state.power) / 1e6:+.3f} MW",
            float(state.power) / params.power_max,
            rk.AMBER,
            bipolar=True,
            neg_color=rk.GREEN,
        ),
        rk.Gauge(
            "DISPATCH",
            f"{float(state.target_power) / 1e6:+.3f} MW",
            float(state.target_power) / params.power_max,
            rk.DIM,
            bipolar=True,
            neg_color=rk.DIM,
        ),
        rk.Gauge(
            "CHARGE",
            f"{float(state.soc) * 100:.1f} %",
            float(state.soc),
            rk.CYAN,
            limit_frac=params.soc_max,
        ),
        rk.Gauge(
            "CELL V",
            f"{history['v_cell'][-1]:.3f} V",
            (history["v_cell"][-1] - params.V_cell_min)
            / (params.V_cell_max - params.V_cell_min),
            rk.BLUE,
        ),
        rk.Gauge(
            "CELL T",
            f"{float(state.T_cell):.1f} C",
            (float(state.T_cell) - params.T_ambient)
            / (params.T_max - params.T_ambient),
            rk.RED,
            limit_frac=1.0,
        ),
        rk.Gauge(
            "FADE",
            f"{float(state.q_loss) * 100:.4f} %",
            rk.clamp01(float(state.q_loss) * 100),
            rk.PURPLE,
            hidden=True,
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
            rk.Series(history["power"], "power", rk.AMBER, fill_to=history["target"]),
            rk.Series(
                history["target"], "dispatch", "white", ls="--", lw=1.0, alpha=0.6
            ),
        ],
        ylabel="power  [MW]",
        xlabel="minutes",
        lines=[(0.0, rk.DIM, "")],
    )
    fig = rk.frame(
        title="GRID  BATTERY  —  2 MWh / 1 MW",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_pack(ax, state, params),
        schematic_title="PACK",
        gauges=gauges,
        strips=[strip],
        status=status,
        subtitle="dispatch tracking from a finite, degrading store",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_battery, HISTORY_KEYS, stride=4)
