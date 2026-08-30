"""Rendering for the distillation column: profile plus product compositions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.pc_gym.distillation.env import N_FEED, compute_reward


def render_column(state, params, step, history):
    history["t"].append(step * params.delta_t)
    history["yD"].append(float(state.x[-1]))
    history["xB"].append(float(state.x[0]))
    history["tyD"].append(float(state.target_yD))
    history["txB"].append(float(state.target_xB))
    history["L"].append(float(state.L))
    history["V"].append(float(state.V))
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), dpi=100)
    fig.suptitle("Distillation — Column A (40 trays)", fontsize=14, weight="bold")

    profile = np.asarray(state.x)
    stages = np.arange(1, len(profile) + 1)
    axs[0].plot(profile, stages, color="crimson", lw=2)
    axs[0].axhline(N_FEED, color="grey", ls=":", lw=1.2, label="feed stage")
    axs[0].set_xlabel("liquid composition x")
    axs[0].set_ylabel("stage (1 = reboiler)")
    axs[0].set_title("Column profile (HIDDEN)", fontsize=10)
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["t"], history["yD"], color="seagreen", lw=2, label="yD")
    axs[1].plot(history["t"], history["tyD"], color="seagreen", ls="--", lw=1.2)
    axs[1].plot(history["t"], history["xB"], color="darkorange", lw=2, label="xB")
    axs[1].plot(history["t"], history["txB"], color="darkorange", ls="--", lw=1.2)
    axs[1].set_xlabel("time (min)")
    axs[1].set_ylabel("product composition")
    axs[1].set_title("Products (measured)", fontsize=10)
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    axs[2].plot(history["t"], history["L"], color="navy", lw=2, label="L (reflux)")
    axs[2].plot(history["t"], history["V"], color="purple", lw=2, label="V (boilup)")
    axs[2].set_xlabel("time (min)")
    axs[2].set_ylabel("flow (kmol/min)")
    axs[2].set_title("Manipulated variables", fontsize=10)
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.3)

    plt.tight_layout()
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
            k: [] for k in ("t", "yD", "xB", "tyD", "txB", "L", "V", "reward")
        }
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_column(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
