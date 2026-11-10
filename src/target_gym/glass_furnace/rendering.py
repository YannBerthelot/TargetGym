"""Control-room rendering for the float glass furnace.

The schematic is a long section: the crown's hot face over each bay, and the melt's
temperature field from the charger to the canal, with the fining pass mark beside the
share the glass leaving it has done. The agent reads only the instruments marked so.
The crown couple is what the agent reads; the plant's override protects the hottest
face, which the agent does not see, against the crown limit.
"""

import numpy as np

from target_gym import render_kit as rk
from target_gym.glass_furnace import physical as P

HISTORY_KEYS = ("t", "crown", "canal", "fining", "pass", "fuel", "reward")

_MELT = (1300.0, 1900.0)  # K, the colour scale of the melt and the crown


def _draw_furnace(ax, state, params):
    T = np.asarray(state.plant.T)
    roof = np.asarray(state.plant.T_roof)[0]
    x, w = 0.05, 0.90
    colour = lambda v: rk.thermal_hex(v, *_MELT)
    rk.gradient_bar(ax, x, 0.78, w * 0.5, 0.06, roof, colour)
    rk.label(ax, x + 0.25 * w, 0.88, "CROWN FACE, BY BAY")
    hot = int(np.argmax(roof))
    hx = x + w * 0.5 * (hot + 0.5) / len(roof)
    rk.label(ax, hx, 0.755, "▲ hottest", color=rk.ORANGE, size=6.5)
    rows = T.shape[0]
    h = 0.40 / rows
    for i in range(rows):
        rk.gradient_bar(ax, x, 0.62 - (i + 1) * h, w, h, T[i], colour, lw=0.0)
    rk.label(ax, x + 0.5 * w, 0.66, "MELT, CHARGER TO CANAL (HIDDEN)")
    rk.caption(ax, f"melt {T.min() - 273.15:.0f} to {T.max() - 273.15:.0f} C")


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
