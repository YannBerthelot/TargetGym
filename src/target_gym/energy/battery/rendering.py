"""Time-series rendering for the grid battery environment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.energy.battery.env import compute_reward


def render_battery(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["P"].append(float(state.power) / 1e6)
    history["target"].append(float(state.target_power) / 1e6)
    history["soc"].append(float(state.soc))
    history["T"].append(float(state.T_cell))
    history["fade"].append(float(state.q_loss) * 100.0)
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True, dpi=100)
    fig.suptitle("Grid Battery — 2 MWh / 1 MW", fontsize=14, weight="bold")

    axs[0].plot(history["t"], history["P"], color="seagreen", lw=2, label="delivered")
    axs[0].plot(
        history["t"],
        history["target"],
        color="black",
        ls="--",
        lw=1.5,
        label="dispatch",
    )
    axs[0].axhline(0.0, color="grey", lw=0.8)
    axs[0].set_ylabel("power (MW)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["t"], history["soc"], color="navy", lw=2)
    axs[1].axhline(params.soc_min, color="red", ls=":", lw=1.2)
    axs[1].axhline(params.soc_max, color="red", ls=":", lw=1.2)
    axs[1].set_ylabel("state of charge")
    axs[1].set_ylim(0, 1)
    axs[1].grid(alpha=0.3)

    ax2 = axs[2]
    ax2.plot(history["t"], history["T"], color="crimson", lw=2, label="T (degC)")
    ax2.set_ylabel("temperature (degC)")
    ax2.set_xlabel("time (min)")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(
        history["t"],
        history["fade"],
        color="darkorange",
        lw=1.5,
        label="capacity fade % (HIDDEN)",
    )
    ax3.set_ylabel("capacity fade (%)")
    lines = ax2.get_lines() + ax3.get_lines()
    ax2.legend(
        lines, [line.get_label() for line in lines], loc="upper left", fontsize=8
    )

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
            k: [] for k in ("t", "P", "target", "soc", "T", "fade", "reward")
        }
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_battery(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
