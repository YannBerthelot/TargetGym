import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.pc_gym.first_order.env import compute_reward


def render_first_order(state, params, step, history):
    history["t"].append(step)
    history["x"].append(float(state.x))
    history["target_x"].append(float(state.target_x))
    history["u"].append(float(state.u))
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True, dpi=100)
    fig.subplots_adjust(hspace=0.35)
    fig.suptitle("First Order System Evolution Over Time", fontsize=16, weight="bold")

    axs[0].plot(history["t"], history["x"], color="blue", lw=2, label="x")
    axs[0].plot(
        history["t"],
        history["target_x"],
        color="orange",
        lw=1.5,
        ls="--",
        label="target",
    )
    axs[0].axhline(params.x_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[0].axhline(params.x_min, color="red", ls="--", lw=1, alpha=0.6)
    axs[0].set_ylabel("x")
    axs[0].set_title("System State", fontsize=12, pad=8)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["t"], history["u"], color="green", lw=2)
    axs[1].axhline(params.u_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[1].axhline(params.u_min, color="blue", ls="--", lw=1, alpha=0.6)
    axs[1].set_ylabel("u")
    axs[1].set_title("Control Input", fontsize=12, pad=8)
    axs[1].grid(alpha=0.3)

    axs[2].plot(history["t"], history["reward"], color="purple", lw=2)
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Time step")
    axs[2].set_title("Reward Signal", fontsize=12, pad=8)
    axs[2].grid(alpha=0.3)

    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
        ..., :3
    ]
    plt.close(fig)
    return image, history


def _render(cls, screen, state, params, frames, clock, stride: int = 10):
    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    if not hasattr(cls, "history") or state.time == 1:
        cls.history = {"t": [], "x": [], "target_x": [], "u": [], "reward": []}

    step = state.time
    if step % stride == 0 or step == 1:
        frame, cls.history = render_first_order(state, params, step, cls.history)
        frames.append(frame)
        cls.frames = frames

    return frames, screen, clock
