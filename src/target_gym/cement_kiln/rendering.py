"""Time-series rendering for the cement kiln environment.

The bottom panel is the axial profile -- the thing the controller cannot see.
Everything above it is what the control room actually has.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.cement_kiln.env import (
    burning_zone_temperature,
    compute_reward,
    discharge_lime,
    residence_time,
    zone_centres,
)


def render_kiln(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["lime"].append(100.0 * float(discharge_lime(state)))
    history["target"].append(100.0 * float(state.target_lime))
    history["T_bz"].append(float(burning_zone_temperature(state)) - 273.0)
    history["T_exh"].append(float(state.T_exhaust) - 273.0)
    history["fuel"].append(float(state.fuel))
    history["rpm"].append(float(state.rpm))
    history["feed"].append(float(state.raw_meal))
    history["tau"].append(float(residence_time(state.rpm, params)) / 60.0)
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(4, 1, figsize=(7, 9), dpi=100)
    fig.suptitle(
        "Cement rotary kiln — 3000 t/day, preheater + precalciner",
        fontsize=13,
        weight="bold",
    )

    axs[0].plot(history["t"], history["lime"], color="crimson", lw=2, label="free lime")
    axs[0].plot(
        history["t"], history["target"], color="black", ls="--", lw=1.5, label="target"
    )
    axs[0].axhspan(0.5, 2.0, color="seagreen", alpha=0.12, label="saleable band")
    axs[0].set_ylabel("free lime [%]")
    axs[0].set_ylim(0, 6)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(
        history["t"], history["T_bz"], color="darkorange", lw=2, label="burning zone"
    )
    axs[1].axhline(
        params.T_bz_max - 273.0, color="firebrick", lw=1.2, label="rings / melt"
    )
    axs[1].axhline(
        params.T_bz_min - 273.0, color="steelblue", lw=1.2, label="kiln goes cold"
    )
    axs[1].plot(
        history["t"], history["T_exh"], color="gray", lw=1.5, label="back-end gas"
    )
    axs[1].set_ylabel("temperature [C]")
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(alpha=0.3)

    ax2b = axs[2].twinx()
    axs[2].plot(
        history["t"], history["fuel"], color="dimgray", lw=2, label="fuel [kg/s]"
    )
    axs[2].plot(
        history["t"],
        history["feed"],
        color="seagreen",
        lw=1.5,
        alpha=0.7,
        label="raw meal [kg/s]",
    )
    ax2b.plot(
        history["t"],
        history["tau"],
        color="mediumpurple",
        lw=1.5,
        ls=":",
        label="residence [min]",
    )
    axs[2].set_ylabel("flow [kg/s]")
    ax2b.set_ylabel("residence [min]")
    h1, l1 = axs[2].get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    axs[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    axs[2].set_xlabel("time [min]")
    axs[2].grid(alpha=0.3)

    z = np.asarray(zone_centres(params))
    axs[3].plot(
        z, np.asarray(state.T_gas) - 273.0, color="orangered", lw=2, label="gas"
    )
    axs[3].plot(
        z, np.asarray(state.T_wall) - 273.0, color="peru", lw=1.5, label="refractory"
    )
    axs[3].plot(
        z, np.asarray(state.T_solid) - 273.0, color="black", lw=2, label="charge"
    )
    ax3b = axs[3].twinx()
    ax3b.plot(
        z,
        np.asarray(state.lime) * 100.0,
        color="crimson",
        lw=1.5,
        ls="--",
        label="free lime [%]",
    )
    ax3b.set_ylabel("free lime [%]")
    axs[3].set_xlabel("distance from feed end [m]   (burner at the right)")
    axs[3].set_ylabel("temperature [C]")
    axs[3].set_title("axial profile — HIDDEN from the controller", fontsize=9)
    h1, l1 = axs[3].get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    axs[3].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    axs[3].grid(alpha=0.3)

    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
        ..., :3
    ]
    plt.close(fig)
    return image, history


def _render(cls, screen, state, params, frames, clock, stride: int = 4):
    if state is None:
        raise ValueError("No state provided")
    if not hasattr(cls, "history") or state.time == 1:
        cls.history = {
            k: []
            for k in (
                "t",
                "lime",
                "target",
                "T_bz",
                "T_exh",
                "fuel",
                "rpm",
                "feed",
                "tau",
                "reward",
            )
        }
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_kiln(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
