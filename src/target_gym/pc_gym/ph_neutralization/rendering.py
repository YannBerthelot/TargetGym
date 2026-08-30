"""Time-series rendering for the pH neutralisation environment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.pc_gym.ph_neutralization.env import compute_reward


def render_ph(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["pH"].append(float(state.pH))
    history["target"].append(float(state.target_pH))
    history["q3"].append(float(state.q3))
    history["q2"].append(float(state.q2))
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True, dpi=100)
    fig.suptitle("pH Neutralisation — CSTR", fontsize=14, weight="bold")

    axs[0].plot(history["t"], history["pH"], color="crimson", lw=2, label="pH")
    axs[0].plot(
        history["t"],
        history["target"],
        color="black",
        ls="--",
        lw=1.5,
        label="setpoint",
    )
    axs[0].axhline(7.0, color="grey", ls=":", lw=1, alpha=0.6)
    axs[0].set_ylabel("pH")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["t"], history["q3"], color="navy", lw=2)
    axs[1].set_ylabel("base flow q3 (mL/s)")
    axs[1].grid(alpha=0.3)

    axs[2].plot(
        history["t"], history["q2"], color="darkorange", lw=2, label="buffer (HIDDEN)"
    )
    axs[2].set_ylabel("buffer flow q2 (mL/s)")
    axs[2].set_xlabel("time (min)")
    axs[2].legend(loc="upper right", fontsize=8)
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
        cls.history = {k: [] for k in ("t", "pH", "target", "q3", "q2", "reward")}
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_ph(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
