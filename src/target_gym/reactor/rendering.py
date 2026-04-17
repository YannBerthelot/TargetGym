"""Control-room-style reactor visualisation.

Left panel  -- reactor pressure-vessel cross-section (schematic):
    * fuel assemblies coloured by fuel temperature (incandescence)
    * control rods moving physically in/out of the core
    * Cherenkov-blue glow whose intensity tracks neutron power
    * purple xenon-135 fog overlay whose opacity tracks Xe_hat
    * coolant channels coloured by coolant temperature

Right panel -- horizontal instrument-gauge bars:
    POWER, ROD WDN, T FUEL, T COOL, Xe-135, I-135, REWARD

Bottom      -- rolling strip chart of power vs target.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.reactor.env import BETA_TOT, compute_revenue_rate, compute_reward

# -- Dark control-room palette ------------------------------------------------
_BG = "#080c14"
_PANEL = "#0e1824"
_FRAME = "#1e3248"
_TEXT = "#c4d8ec"
_DIM = "#4a6678"
_CYAN = "#00bcd4"
_BLUE = "#42a5f5"
_GREEN = "#66bb6a"
_RED = "#ef5350"
_AMBER = "#ffca28"
_PURPLE = "#ab47bc"

_STRIP_LEN = 400  # timesteps shown in the strip chart


# -- Colour helpers ------------------------------------------------------------
def _fuel_rgb(T, T_lo=600.0, T_hi=2800.0):
    """Fuel incandescence: cool gray -> dull red -> orange -> yellow-white."""
    f = float(np.clip((T - T_lo) / (T_hi - T_lo), 0, 1))
    if f < 0.25:
        g = 0.35 + f * 1.0
        return (g, g * 0.9, g * 0.85)
    if f < 0.50:
        t = (f - 0.25) / 0.25
        return (0.6 + 0.4 * t, 0.25 + 0.25 * t, 0.10)
    if f < 0.75:
        t = (f - 0.50) / 0.25
        return (1.0, 0.50 + 0.30 * t, 0.10)
    t = (f - 0.75) / 0.25
    return (1.0, 0.80 + 0.20 * t, 0.10 + 0.85 * t)


def _coolant_hex(T, T_lo=550.0, T_hi=700.0):
    """Coolant colour for arrows: blue (cold) -> red (hot)."""
    rgba = plt.cm.coolwarm(float(np.clip((T - T_lo) / (T_hi - T_lo), 0, 1)))
    return matplotlib.colors.rgb2hex(rgba[:3])


# -- Reactor vessel drawing ----------------------------------------------------
def _draw_vessel(ax, state, params):
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.12, 1.18)
    ax.axis("off")

    # Vessel outline (pressure vessel body)
    vx, vy, vw, vh = 0.10, 0.0, 0.80, 1.0
    vessel = mpatches.FancyBboxPatch(
        (vx, vy),
        vw,
        vh,
        boxstyle=mpatches.BoxStyle("Round", pad=0.045),
        fc="#111e2c",
        ec=_FRAME,
        lw=2.5,
    )
    ax.add_patch(vessel)

    # Dome top
    dome = mpatches.Wedge(
        (vx + vw / 2, vy + vh - 0.01),
        vw * 0.43,
        0,
        180,
        fc="#111e2c",
        ec=_FRAME,
        lw=2.5,
    )
    ax.add_patch(dome)

    # -- Core region --
    cx, cy, cw, ch = 0.20, 0.10, 0.60, 0.72
    ax.add_patch(mpatches.Rectangle((cx, cy), cw, ch, fc="#060e18", ec="none"))

    # -- Coolant channels (tinted fill between rods) --
    T_cool = float(state.T_coolant)
    cool_rgba = plt.cm.coolwarm(float(np.clip((T_cool - 550) / 150, 0, 1)))
    ax.add_patch(
        mpatches.Rectangle(
            (cx, cy),
            cw,
            ch,
            fc=cool_rgba,
            alpha=0.18,
            ec="none",
            zorder=1,
        )
    )

    # -- Cherenkov glow --
    n = float(state.n)
    glow_i = float(np.clip(n / params.n_max, 0, 1))
    for k in range(8):
        r = 0.05 + k * 0.04
        a = glow_i * 0.20 * (1 - k / 8)
        ax.add_patch(
            mpatches.Ellipse(
                (0.50, cy + ch * 0.45),
                r * 2.2,
                r * 3.5,
                fc=(0.20, 0.50, 1.0),
                alpha=a,
                ec="none",
                zorder=1,
            )
        )

    # -- Fuel assemblies --
    n_rods = 7
    rod_xs = np.linspace(cx + 0.045, cx + cw - 0.045, n_rods)
    fc = _fuel_rgb(float(state.T_fuel))
    rod_w = 0.030
    for x in rod_xs:
        ax.add_patch(
            mpatches.Rectangle(
                (x - rod_w / 2, cy + 0.02),
                rod_w,
                ch - 0.04,
                fc=fc,
                ec="#1a1a1a",
                lw=0.5,
                zorder=3,
            )
        )

    # -- Control rods --
    rho = float(state.rho_ext)
    rod_frac = float(
        np.clip(
            (rho - params.rho_ext_min) / (params.rho_ext_max - params.rho_ext_min),
            0,
            1,
        )
    )
    # tip_y: frac=0 -> fully inserted (bottom of core), frac=1 -> withdrawn
    tip_y = cy + rod_frac * (ch + 0.08)
    shaft_top = vy + vh + 0.12

    ctrl_xs = rod_xs[1::2]  # interleaved with fuel assemblies
    cw_rod = 0.020
    for x in ctrl_xs:
        # Shaft
        ax.add_patch(
            mpatches.Rectangle(
                (x - cw_rod / 2, tip_y),
                cw_rod,
                shaft_top - tip_y,
                fc="#282828",
                ec="#444",
                lw=0.6,
                zorder=4,
            )
        )
        # Absorber tip (darker, wider)
        ax.add_patch(
            mpatches.Rectangle(
                (x - cw_rod * 0.8, tip_y),
                cw_rod * 1.6,
                0.025,
                fc="#0a0a0a",
                ec="#555",
                lw=0.6,
                zorder=5,
            )
        )

    # -- Xenon-135 fog --
    xe = float(state.Xe_hat)
    xe_a = float(np.clip((xe - 0.5) * 0.35, 0, 0.45))
    if xe_a > 0.005:
        for dy in (0.0, 0.15, -0.15, 0.30, -0.08):
            ax.add_patch(
                mpatches.Ellipse(
                    (0.50, cy + ch * 0.50 + dy),
                    cw * 0.80,
                    ch * 0.22,
                    fc="#7b1fa2",
                    alpha=xe_a * 0.45,
                    ec="none",
                    zorder=2,
                )
            )

    # -- Coolant flow arrows --
    cool_hex = _coolant_hex(T_cool)
    arr_in = dict(arrowstyle="->", color=_CYAN, lw=1.8)
    arr_out = dict(arrowstyle="->", color=cool_hex, lw=1.8)
    ax.annotate(
        "",
        xy=(0.28, cy + 0.04),
        xytext=(0.28, cy - 0.06),
        arrowprops=arr_in,
    )
    ax.annotate(
        "",
        xy=(0.72, cy + ch + 0.06),
        xytext=(0.72, cy + ch - 0.04),
        arrowprops=arr_out,
    )
    ax.text(
        0.28,
        cy - 0.08,
        "IN",
        ha="center",
        va="top",
        fontsize=6,
        color=_DIM,
        family="monospace",
    )
    ax.text(
        0.72,
        cy + ch + 0.08,
        "OUT",
        ha="center",
        va="bottom",
        fontsize=6,
        color=_DIM,
        family="monospace",
    )

    # -- Labels --
    ax.text(
        0.50,
        1.14,
        "REACTOR  VESSEL",
        ha="center",
        va="bottom",
        fontsize=11,
        color=_TEXT,
        fontweight="bold",
        family="monospace",
    )
    ax.text(
        0.93,
        0.94,
        f"RODS\n{rod_frac * 100:.0f}% WDN",
        ha="center",
        va="top",
        fontsize=7,
        color=_DIM,
        family="monospace",
    )


# -- Horizontal gauge bar ------------------------------------------------------
def _hbar(
    ax,
    y,
    frac,
    color,
    label,
    value_str,
    *,
    limit_frac=None,
    target_frac=None,
    bar_h=0.065,
    x0=0.30,
    bw=0.48,
    bipolar=False,
    neg_color=None,
):
    """Draw a horizontal gauge bar.

    If *bipolar* is True, *frac* is in [-1, 1] with 0 at the bar center.
    Positive fills right in *color*, negative fills left in *neg_color*.
    """
    # Background
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0, y),
            bw,
            bar_h,
            boxstyle=mpatches.BoxStyle("Round", pad=0.004),
            fc="#0a1018",
            ec=_FRAME,
            lw=1,
        )
    )
    pad_x, pad_y = 0.005, 0.007
    inner_h = bar_h - 2 * pad_y

    if bipolar:
        mid = x0 + bw / 2
        clamped = float(np.clip(frac, -1, 1))
        if clamped >= 0:
            fw = (bw / 2) * clamped
            if fw > 0.002:
                ax.add_patch(
                    mpatches.Rectangle(
                        (mid, y + pad_y),
                        fw,
                        inner_h,
                        fc=color,
                        alpha=0.82,
                        ec="none",
                    )
                )
        else:
            fw = (bw / 2) * (-clamped)
            if fw > 0.002:
                ax.add_patch(
                    mpatches.Rectangle(
                        (mid - fw, y + pad_y),
                        fw,
                        inner_h,
                        fc=neg_color or _RED,
                        alpha=0.82,
                        ec="none",
                    )
                )
        # Center zero line
        ax.plot(
            [mid, mid],
            [y - 0.003, y + bar_h + 0.003],
            color=_DIM,
            lw=0.8,
            alpha=0.6,
        )
    else:
        fw = bw * float(np.clip(frac, 0, 1))
        if fw > 0.002:
            ax.add_patch(
                mpatches.Rectangle(
                    (x0 + pad_x, y + pad_y),
                    fw - 2 * pad_x,
                    inner_h,
                    fc=color,
                    alpha=0.82,
                    ec="none",
                )
            )

    # SCRAM / limit line
    if limit_frac is not None:
        lx = x0 + bw * float(np.clip(limit_frac, 0, 1))
        ax.plot(
            [lx, lx],
            [y - 0.006, y + bar_h + 0.006],
            color=_RED,
            lw=1.4,
            ls="--",
            alpha=0.75,
        )
    # Target triangle
    if target_frac is not None:
        tx = x0 + bw * float(np.clip(target_frac, 0, 1))
        s = 0.014
        ax.fill(
            [tx - s, tx + s, tx],
            [y + bar_h + 0.010, y + bar_h + 0.010, y + bar_h - 0.003],
            color="white",
            alpha=0.90,
            zorder=5,
        )
    # Label (left)
    ax.text(
        x0 - 0.02,
        y + bar_h / 2,
        label,
        ha="right",
        va="center",
        fontsize=8,
        color=_DIM,
        family="monospace",
        fontweight="bold",
    )
    # Value (right)
    ax.text(
        x0 + bw + 0.02,
        y + bar_h / 2,
        value_str,
        ha="left",
        va="center",
        fontsize=8.5,
        color=_TEXT,
        family="monospace",
    )


# -- Instrument panel ----------------------------------------------------------
def _draw_gauges(ax, state, params, cumul_revenue_M=0.0):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = float(state.n)
    target = float(state.target_n)
    T_fuel = float(state.T_fuel)
    T_cool = float(state.T_coolant)
    xe = float(state.Xe_hat)
    iod = float(state.I_hat)
    rho = float(state.rho_ext)
    rho_dol = rho / BETA_TOT
    reward = float(compute_reward(state, params))
    # Revenue is tracked cumulatively in the history dict by render_reactor.

    rod_frac = float(
        np.clip(
            (rho - params.rho_ext_min) / (params.rho_ext_max - params.rho_ext_min),
            0,
            1,
        )
    )

    # Max possible revenue: full power, perfect tracking → P_GW * 1000 * spot / 1000
    max_rev = params.P_electric_GW * params.spot_price_per_MWh  # k$/h

    ax.text(
        0.50,
        0.97,
        "INSTRUMENTS",
        ha="center",
        va="top",
        fontsize=11,
        color=_TEXT,
        fontweight="bold",
        family="monospace",
    )

    y0, dy = 0.88, 0.098

    # 1. Power -- cyan, target triangle, overpower limit
    _hbar(
        ax,
        y0,
        np.clip(n / params.n_max, 0, 1),
        _CYAN,
        "POWER",
        f"{n:.3f}",
        limit_frac=1.0,
        target_frac=float(np.clip(target / params.n_max, 0, 1)),
    )

    # 2. Rod withdrawal %
    _hbar(ax, y0 - dy, rod_frac, _BLUE, "ROD WDN", f"{rho_dol:+.2f} $")

    # 3. Fuel temperature
    tf_frac = float(np.clip((T_fuel - 500) / (params.T_fuel_max - 500), 0, 1))
    _hbar(ax, y0 - 2 * dy, tf_frac, _AMBER, "T FUEL", f"{T_fuel:.0f} K", limit_frac=1.0)

    # 4. Coolant temperature
    tc_frac = float(np.clip((T_cool - 500) / (params.T_coolant_max - 500), 0, 1))
    _hbar(
        ax, y0 - 3 * dy, tc_frac, "#ff7043", "T COOL", f"{T_cool:.0f} K", limit_frac=1.0
    )

    # 5. Xe-135 (scale 0-3x equilibrium; eq marker at 1/3)
    _hbar(
        ax,
        y0 - 4 * dy,
        float(np.clip(xe / 3.0, 0, 1)),
        _PURPLE,
        "Xe-135",
        f"{xe:.2f}",
        target_frac=1.0 / 3.0,
    )

    # 6. I-135
    _hbar(
        ax,
        y0 - 5 * dy,
        float(np.clip(iod / 3.0, 0, 1)),
        "#7e57c2",
        "I-135",
        f"{iod:.2f}",
        target_frac=1.0 / 3.0,
    )

    # 7. Reward
    _hbar(
        ax, y0 - 6 * dy, float(np.clip(reward, 0, 1)), _GREEN, "REWARD", f"{reward:.2f}"
    )

    # 8. Cumulative revenue — bipolar bar, green right / red left.
    # Scale: max_rev * 24h = perfect-tracking 24-hour revenue in M$.
    max_cumul_M = max_rev * 24.0 / 1000.0  # k$/h * 24h / 1000 = M$
    rev_frac = float(np.clip(cumul_revenue_M / max(max_cumul_M, 1e-6), -1, 1))
    _hbar(
        ax,
        y0 - 7 * dy,
        rev_frac,
        _GREEN,
        "CUM. REV",
        f"{cumul_revenue_M:+.2f} M$",
        bipolar=True,
        neg_color=_RED,
    )

    # Status dot
    scram = (
        n > params.n_max or T_fuel > params.T_fuel_max or T_cool > params.T_coolant_max
    )
    sc, st = (_RED, "SCRAM") if scram else (_GREEN, "NOMINAL")
    ax.text(
        0.50,
        y0 - 8 * dy + 0.02,
        f"\u25cf {st}",
        ha="center",
        va="center",
        fontsize=10,
        color=sc,
        fontweight="bold",
        family="monospace",
    )


# -- Strip chart ---------------------------------------------------------------
def _draw_strip(ax, history, params):
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_color(_FRAME)
    ax.tick_params(colors=_DIM, labelsize=7)
    ax.grid(color=_FRAME, alpha=0.5, lw=0.5)

    t_all = history["t"][-_STRIP_LEN:]
    n_all = history["n"][-_STRIP_LEN:]
    tgt_all = history["target_n"][-_STRIP_LEN:]
    if len(t_all) < 2:
        return

    t_sec = [s * params.delta_t for s in t_all]
    use_h = t_sec[-1] > 600
    t_plot = [s / 3600 for s in t_sec] if use_h else t_sec

    ax.plot(t_plot, n_all, color=_CYAN, lw=1.5, label="n", zorder=3)
    ax.plot(
        t_plot,
        tgt_all,
        color="white",
        lw=1,
        ls="--",
        alpha=0.55,
        label="target",
        zorder=2,
    )
    ax.fill_between(t_plot, n_all, tgt_all, color=_CYAN, alpha=0.07)

    ax.set_ylabel("n", fontsize=7, color=_DIM, family="monospace")
    ax.set_xlabel(
        "hours" if use_h else "seconds",
        fontsize=7,
        color=_DIM,
        family="monospace",
    )
    ax.legend(
        loc="upper right",
        fontsize=6,
        framealpha=0.4,
        facecolor=_PANEL,
        edgecolor=_FRAME,
        labelcolor=_TEXT,
    )


# -- Main entry point ----------------------------------------------------------
def render_reactor(state, params, step, history):
    """Render one control-room frame and append data to *history*."""
    history["t"].append(step)
    history["n"].append(float(state.n))
    history["target_n"].append(float(state.target_n))
    history["T_fuel"].append(float(state.T_fuel))
    history["T_coolant"].append(float(state.T_coolant))
    history["I_hat"].append(float(state.I_hat))
    history["Xe_hat"].append(float(state.Xe_hat))
    history["rho_ext"].append(float(state.rho_ext))
    history["reward"].append(float(compute_reward(state, params)))
    rev_rate_kph = float(compute_revenue_rate(state, params))
    history["revenue"].append(rev_rate_kph)
    # Cumulative revenue in M$ (integrate rate in k$/h over dt in seconds).
    dt_h = params.delta_t / 3600.0
    prev_cumul = history["cumul_revenue_M"][-1] if history["cumul_revenue_M"] else 0.0
    history["cumul_revenue_M"].append(prev_cumul + rev_rate_kph * dt_h / 1000.0)

    fig = plt.figure(figsize=(14, 7.5), facecolor=_BG, dpi=100)

    # -- Header --
    elapsed = step * params.delta_t
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    fig.text(
        0.50,
        0.975,
        "NUCLEAR  REACTOR  \u2014  CONTROL  ROOM",
        ha="center",
        va="top",
        fontsize=14,
        color=_TEXT,
        fontweight="bold",
        family="monospace",
    )
    fig.text(
        0.97,
        0.975,
        f"T+{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}",
        ha="right",
        va="top",
        fontsize=10,
        color=_DIM,
        family="monospace",
    )
    fig.text(
        0.03,
        0.975,
        f"STEP {step}",
        ha="left",
        va="top",
        fontsize=10,
        color=_DIM,
        family="monospace",
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 1.1],
        height_ratios=[2.8, 1.0],
        hspace=0.22,
        wspace=0.08,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )

    ax_vessel = fig.add_subplot(gs[0, 0], facecolor=_BG)
    ax_gauges = fig.add_subplot(gs[0, 1], facecolor=_BG)
    ax_strip = fig.add_subplot(gs[1, :])

    _draw_vessel(ax_vessel, state, params)
    cumul_M = history["cumul_revenue_M"][-1] if history["cumul_revenue_M"] else 0.0
    _draw_gauges(ax_gauges, state, params, cumul_revenue_M=cumul_M)
    _draw_strip(ax_strip, history, params)

    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
        ..., :3
    ]
    plt.close(fig)
    return image, history


# -- save_video hook -----------------------------------------------------------
def _render(cls, screen, state, params, frames, clock, stride: int = 48):
    """Render hook used by ``save_video`` -- subsamples frames by *stride*.

    stride=48 because episodes are 86 400 steps (24 hours at 1.0 s/step).
    Produces ~1 800 frames at 60 fps → ~30 s video.
    """
    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    if not hasattr(cls, "history") or state.time == 1:
        cls.history = {
            "t": [],
            "n": [],
            "target_n": [],
            "T_fuel": [],
            "T_coolant": [],
            "I_hat": [],
            "Xe_hat": [],
            "rho_ext": [],
            "reward": [],
            "revenue": [],
            "cumul_revenue_M": [],
        }

    step = state.time
    if step % stride == 0 or step == 1:
        frame, cls.history = render_reactor(state, params, step, cls.history)
        frames.append(frame)
        cls.frames = frames

    return frames, screen, clock
