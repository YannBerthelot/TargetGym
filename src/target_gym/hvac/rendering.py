"""Time-series rendering for the building HVAC environment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.hvac.env import compute_reward, hour_of_day


def render_hvac(state, params, step, history):
    history["t"].append(step * params.delta_t / 3600.0)
    history["T_air"].append(float(state.T_air))
    history["T_mass"].append(float(state.T_mass))
    history["T_out"].append(float(state.T_out))
    history["target"].append(float(state.target_T))
    history["heat"].append(100.0 * float(state.Q_emitter) / params.Q_heat_max)
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True, dpi=100)
    fig.suptitle("Building HVAC — single zone (5R1C)", fontsize=14, weight="bold")

    axs[0].plot(history["t"], history["T_air"], color="crimson", lw=2, label="T_air")
    axs[0].plot(
        history["t"],
        history["target"],
        color="black",
        ls="--",
        lw=1.5,
        label="setpoint",
    )
    axs[0].plot(
        history["t"],
        history["T_mass"],
        color="darkorange",
        lw=1.5,
        alpha=0.6,
        label="T_mass (HIDDEN)",
    )
    axs[0].plot(
        history["t"], history["T_out"], color="steelblue", lw=1.5, label="T_out"
    )
    axs[0].set_ylabel("temperature (°C)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["t"], history["heat"], color="navy", lw=2)
    axs[1].set_ylabel("heating (% of max)")
    axs[1].set_ylim(-5, 105)
    axs[1].grid(alpha=0.3)

    axs[2].plot(history["t"], history["reward"], color="purple", lw=2)
    axs[2].set_ylabel("reward")
    axs[2].set_xlabel("time (hours)")
    axs[2].grid(alpha=0.3)

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
            k: [] for k in ("t", "T_air", "T_mass", "T_out", "target", "heat", "reward")
        }
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_hvac(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
