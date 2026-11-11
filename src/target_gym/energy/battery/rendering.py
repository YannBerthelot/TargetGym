"""Control-room rendering for the grid battery.

The schematic is the pack: a stack of cells filled to the true state of
charge, with the grid tie above it. The dashed mark is the BMS's report, which
is what the market checks, and the green window is the period's requirement it
must sit inside. Below the pack, the block's settlement periods: each one's
declared baseline and day-ahead price, the periods already locked by gate
closure, and the ones whose availability was lost.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.energy.battery.env import (
    baseline_limit,
    compute_reward,
    period_of,
    requirements,
    submitted_period,
)

HISTORY_KEYS = ("t", "power", "target", "baseline", "soc", "rep", "reward", "lost")


def _n_periods(params):
    """Settlement periods in the episode's block."""
    block = params.max_steps_in_episode * params.delta_t
    return max(1, int(np.ceil(block / params.period_seconds)))


def _window(state, params):
    """The reported state of charge the period's requirement allows."""
    need_out, need_in = requirements(state.deficit_export, state.deficit_import, params)
    cap = params.energy_capacity
    return float(need_out) / cap, 1.0 - float(need_in) / cap


def _draw_pack(ax, state, params, lost):
    soc = float(state.soc)
    rep = float(state.soc_reported)
    P = float(state.power) / 1e6

    # Grid tie.
    discharging = P >= 0
    col = rk.AMBER if discharging else rk.GREEN
    rk.label(
        ax,
        0.5,
        0.985,
        f"GRID  ·  {'EXPORT' if discharging else 'IMPORT'} {abs(P):.2f} MW",
        color=col,
        size=9,
        weight="bold",
    )
    ax.plot([0.18, 0.82], [0.945, 0.945], color=rk.FRAME, lw=3, zorder=1)
    for x in (0.30, 0.50, 0.70):
        ax.plot([x, x], [0.945, 0.905], color=rk.FRAME, lw=2, zorder=1)
    discharging = P >= 0
    col = rk.AMBER if discharging else rk.GREEN
    mag = rk.clamp01(abs(P) / (params.power_max / 1e6))
    if mag > 0.01:
        for x in (0.30, 0.50, 0.70):
            y0, y1 = (0.855, 0.94) if discharging else (0.94, 0.855)
            rk.flow_arrow(
                ax, x, y0, x, y1, color=col, lw=1.0 + 2.4 * mag, mutation=9 + 9 * mag
            )

    # Cell stack, filled to the true charge.
    n_mod = 8
    x0, y0, w, h, gap = 0.14, 0.45, 0.72, 0.33, 0.012
    mw = (w - (n_mod - 1) * gap) / n_mod
    fill_c = rk.duty_hex(soc, "#b03a2e", rk.GREEN)
    for i in range(n_mod):
        mx = x0 + i * (mw + gap)
        rk.vessel(ax, mx, y0, mw, h, fc="#0b1420", ec=rk.FRAME, lw=1.2, pad=0.006)
        f = rk.clamp01(soc * n_mod - i)
        if f > 0:
            rk.fill_level(
                ax, mx + 0.004, y0 + 0.006, mw - 0.008, h - 0.012, f, color=fill_c
            )

    # The requirement window the report must sit inside, and the report.
    lo, hi = _window(state, params)
    ax.add_patch(
        rk.patches.Rectangle(
            (x0 + w * lo, y0 - 0.032),
            w * max(hi - lo, 0.0),
            0.018,
            fc=rk.GREEN,
            alpha=0.45,
            ec="none",
            zorder=7,
        )
    )
    rk.label(ax, x0 - 0.012, y0 - 0.023, "window", color=rk.GREEN, size=6, ha="right")
    inside = lo <= rep <= hi
    rcol = rk.TEXT if inside else rk.RED
    lx = x0 + w * rep
    ax.plot([lx, lx], [y0 - 0.04, y0 + h + 0.02], color=rcol, lw=1.3, ls="--")
    rk.label(ax, lx, y0 + h + 0.04, f"BMS {rep * 100:.1f}%", color=rcol, size=6.5)
    rk.label(
        ax,
        0.5,
        y0 + h * 0.55,
        f"{soc * 100:.1f} %",
        color=rk.TEXT,
        size=16,
        weight="bold",
        zorder=8,
    )
    rk.label(
        ax,
        0.5,
        y0 + h * 0.55 - 0.06,
        "true charge ●",
        color=rk.TEXT,
        size=7,
        zorder=8,
    )

    _draw_periods(ax, state, params, lost)
    E = params.energy_capacity / 3.6e9
    rk.caption(
        ax,
        f"{E:.2f} MWh / {params.power_max / 1e6:.2f} MW LFP unit"
        "  ·  baseline set 60-90 min ahead",
    )


def _draw_periods(ax, state, params, lost):
    """The block's settlement periods: baseline bars, prices, gate closure."""
    n = _n_periods(params)
    t = int(state.time)
    now = int(period_of(t, params))
    first_open = int(submitted_period(t, params))
    limit = float(baseline_limit(params))
    x0, w, ymid, hh = 0.06, 0.88, 0.22, 0.085
    pw = w / n
    rk.label(ax, 0.5, 0.37, "SETTLEMENT PERIODS", color=rk.DIM, size=7, weight="bold")
    ax.plot([x0, x0 + w], [ymid, ymid], color=rk.DIM, lw=0.8, zorder=2)
    prices = np.asarray(state.prices)
    base = np.asarray(state.baselines)
    for k in range(n):
        px = x0 + k * pw
        if k < first_open:  # past gate closure: the baseline is fixed
            ax.add_patch(
                rk.patches.Rectangle(
                    (px, ymid - hh),
                    pw,
                    2 * hh,
                    fc=rk.FRAME,
                    alpha=0.35,
                    ec=rk.DIM,
                    lw=0,
                    hatch="///" if k >= now else None,
                    zorder=1,
                )
            )
        b = float(base[k]) / limit
        ax.add_patch(
            rk.patches.Rectangle(
                (px + 0.15 * pw, ymid),
                0.7 * pw,
                hh * b,
                fc=rk.AMBER if b >= 0 else rk.GREEN,
                alpha=0.8,
                ec="none",
                zorder=3,
            )
        )
        if k in lost:
            rk.label(ax, px + pw / 2, ymid, "✕", color=rk.RED, size=13, zorder=6)
        price = float(prices[k])
        pc = rk.TEXT if price == params.price_base else rk.ORANGE
        rk.label(ax, px + pw / 2, ymid - hh - 0.03, f"£{price:.0f}", color=pc, size=6)
        ax.plot([px, px], [ymid - hh, ymid + hh], color=rk.FRAME, lw=0.8)
    nx = x0 + w * min(t * params.delta_t / (n * params.period_seconds), 1.0)
    ax.plot([nx, nx], [ymid - hh - 0.01, ymid + hh + 0.01], color=rk.CYAN, lw=1.6)
    rk.label(ax, nx, ymid + hh + 0.025, "now", color=rk.CYAN, size=6)
    gx = nx + w * params.gate_closure_seconds / (n * params.period_seconds)
    if gx <= x0 + w:
        ax.plot([gx, gx], [ymid - hh, ymid + hh], color=rk.RED, lw=1.2, ls="--")
        rk.label(ax, gx, ymid + hh + 0.025, "gate", color=rk.RED, size=6)
    rk.label(
        ax,
        0.5,
        ymid - hh - 0.07,
        "bar: baseline (export up)  ·  hatched: locked  ·  £/MWh day-ahead"
        "  ·  ✕ availability lost",
        color=rk.DIM,
        size=6,
    )


def render_battery(state, params, step, history):
    k = int(period_of(int(state.time), params))
    history["t"].append(step * params.delta_t / 60.0)
    history["power"].append(float(state.power) / 1e6)
    history["target"].append(float(state.target_power) / 1e6)
    history["baseline"].append(float(state.baselines[k]) / 1e6)
    history["soc"].append(float(state.soc))
    history["rep"].append(float(state.soc_reported))
    history["reward"].append(float(compute_reward(state, params)))
    if float(state.unavailable) > 0.5 and k not in history["lost"]:
        history["lost"].append(k)

    lo, hi = _window(state, params)
    rep = float(state.soc_reported)
    short = float(state.short_seconds) > 0.0
    if float(state.unavailable) > 0.5:
        status, status_text = rk.ALARM, "PERIOD LOST"
    elif short or not lo <= rep <= hi:
        status, status_text = rk.WATCH, "SHORT" if short else "OUTSIDE WINDOW"
    else:
        status, status_text = rk.NOMINAL, None

    pmax = params.power_max
    limit = float(baseline_limit(params))
    price = float(state.prices[k])
    gauges = [
        rk.Gauge(
            "POWER",
            f"{float(state.power) / 1e6:+.3f} MW",
            float(state.power) / pmax,
            rk.AMBER,
            bipolar=True,
            neg_color=rk.GREEN,
        ),
        rk.Gauge(
            "REQUIRED",
            f"{float(state.target_power) / 1e6:+.3f} MW",
            float(state.target_power) / pmax,
            rk.DIM,
            bipolar=True,
            neg_color=rk.DIM,
        ),
        rk.Gauge(
            "BASELINE",
            f"{float(state.baselines[k]) / 1e6:+.3f} MW",
            float(state.baselines[k]) / limit,
            rk.AMBER,
            bipolar=True,
            neg_color=rk.GREEN,
        ),
        rk.Gauge(
            "BMS SoC",
            f"{rep * 100:.1f} %",
            rep,
            rk.CYAN,
            limit_frac=hi,
            target_frac=0.5 * (lo + hi),
        ),
        rk.Gauge(
            "TRUE SoC",
            f"{float(state.soc) * 100:.1f} %",
            float(state.soc),
            rk.PURPLE,
            hidden=True,
        ),
        rk.Gauge(
            "CELL V",
            f"{float(state.v_cell):.3f} V",
            (float(state.v_cell) - params.V_cell_min)
            / (params.V_cell_max - params.V_cell_min),
            rk.BLUE,
        ),
        rk.Gauge(
            "PRICE",
            f"£{price:.0f}/MWh",
            price / max(float(np.max(np.abs(np.asarray(state.prices)))), 1.0),
            rk.ORANGE if price != params.price_base else rk.DIM,
            bipolar=True,
        ),
        rk.cost_gauge(history["reward"], params, "GBP"),
    ]

    strip = rk.Strip(
        history["t"],
        [
            rk.Series(history["power"], "power", rk.AMBER, fill_to=history["target"]),
            rk.Series(
                history["target"], "required", "white", ls="--", lw=1.0, alpha=0.6
            ),
            rk.Series(history["baseline"], "baseline", rk.CYAN, lw=1.4),
        ],
        ylabel="power  [MW]",
        xlabel="minutes",
        lines=[(0.0, rk.DIM, "")],
    )
    E = params.energy_capacity / 3.6e9
    fig = rk.frame(
        title=f"GRID  BATTERY  —  {E:.2f} MWh / {pmax / 1e6:.2f} MW",
        step=step,
        elapsed_s=step * params.delta_t,
        schematic=lambda ax: _draw_pack(ax, state, params, history["lost"]),
        schematic_title="PACK",
        gauges=gauges,
        strips=[strip],
        status=status,
        status_text=status_text,
        subtitle="baseline is the only lever  ·  market checks the BMS estimate",
    )
    return rk.finish(fig), history


_render = rk.make_render_hook(render_battery, HISTORY_KEYS)
