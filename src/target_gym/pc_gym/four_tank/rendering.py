import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.pc_gym.four_tank.env import compute_reward


def render_four_tank(state, params, step, history):
    history["t"].append(step)
    history["h1"].append(float(state.h1))
    history["h2"].append(float(state.h2))
    history["h3"].append(float(state.h3))
    history["h4"].append(float(state.h4))
    history["target_h1"].append(float(state.target_h1))
    history["target_h2"].append(float(state.target_h2))
    history["v1"].append(float(state.v1))
    history["v2"].append(float(state.v2))
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True, dpi=100)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle("Four-Tank System Evolution Over Time", fontsize=16, weight="bold")

    # Lower tanks (controlled)
    axs[0].plot(history["t"], history["h1"], color="blue", lw=2, label="h1")
    axs[0].plot(
        history["t"],
        history["target_h1"],
        color="blue",
        lw=1.5,
        ls="--",
        label="target h1",
    )
    axs[0].plot(history["t"], history["h2"], color="teal", lw=2, label="h2")
    axs[0].plot(
        history["t"],
        history["target_h2"],
        color="teal",
        lw=1.5,
        ls="--",
        label="target h2",
    )
    axs[0].axhline(params.h_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[0].axhline(params.h_min, color="red", ls="--", lw=1, alpha=0.6)
    axs[0].set_ylabel("Level (m)")
    axs[0].set_title("Lower Tanks (h1, h2)", fontsize=12, pad=8)
    axs[0].legend(loc="upper right", fontsize=7)
    axs[0].grid(alpha=0.3)

    # Upper tanks (uncontrolled)
    axs[1].plot(history["t"], history["h3"], color="orange", lw=2, label="h3")
    axs[1].plot(history["t"], history["h4"], color="brown", lw=2, label="h4")
    axs[1].axhline(params.h_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[1].axhline(params.h_min, color="red", ls="--", lw=1, alpha=0.6)
    axs[1].set_ylabel("Level (m)")
    axs[1].set_title("Upper Tanks (h3, h4)", fontsize=12, pad=8)
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(alpha=0.3)

    # Pump inputs
    axs[2].plot(history["t"], history["v1"], color="green", lw=2, label="v1")
    axs[2].plot(history["t"], history["v2"], color="lime", lw=2, label="v2")
    axs[2].axhline(params.v_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[2].axhline(params.v_min, color="blue", ls="--", lw=1, alpha=0.6)
    axs[2].set_ylabel("Voltage (V)")
    axs[2].set_title("Pump Inputs (v1, v2)", fontsize=12, pad=8)
    axs[2].legend(loc="upper right", fontsize=8)
    axs[2].grid(alpha=0.3)

    axs[3].plot(history["t"], history["reward"], color="purple", lw=2)
    axs[3].set_ylabel("Reward")
    axs[3].set_xlabel("Time step")
    axs[3].set_title("Reward Signal", fontsize=12, pad=8)
    axs[3].grid(alpha=0.3)

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
        cls.history = {
            "t": [],
            "h1": [],
            "h2": [],
            "h3": [],
            "h4": [],
            "target_h1": [],
            "target_h2": [],
            "v1": [],
            "v2": [],
            "reward": [],
        }

    step = state.time
    if step % stride == 0 or step == 1:
        frame, cls.history = render_four_tank(state, params, step, cls.history)
        frames.append(frame)
        cls.frames = frames

    return frames, screen, clock
