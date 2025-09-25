import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.pc_gym.cstr.env import compute_reward, convert_raw_action_to_range


def render_cstr(state, params, step, history):
    """
    Render CSTR values (C_a, T, T_c, Reward) as time series graphs.
    Returns an RGB image (numpy array).
    """
    # Update history
    history["t"].append(step)
    history["C_a"].append(float(state.C_a))
    history["T"].append(float(state.T))
    history["T_c"].append(float(state.T_c))
    history["reward"].append(float(compute_reward(state, params)))

    # Create figure
    fig, axs = plt.subplots(4, 1, figsize=(7, 9), sharex=True, dpi=100)
    fig.subplots_adjust(hspace=0.35)
    fig.suptitle("CSTR Evolution Over Time", fontsize=16, weight="bold")

    # Plot C_a
    axs[0].plot(history["t"], history["C_a"], color="green", lw=2)
    axs[0].set_ylabel("C_a (mol/L)")
    axs[0].set_title("Concentration of A", fontsize=12, pad=8)
    axs[0].grid(alpha=0.3)

    # Plot T
    axs[1].plot(history["t"], history["T"], color="red", lw=2)
    axs[1].set_ylabel("T (K)")
    axs[1].set_title("Reactor Temperature", fontsize=12, pad=8)
    axs[1].grid(alpha=0.3)
    axs[1].axhline(params.T_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[1].axhline(params.T_min, color="blue", ls="--", lw=1, alpha=0.6)
    converted_history = convert_raw_action_to_range(
        np.array(history["T_c"]), params.T_c_min, params.T_c_max
    )
    # Plot T_c
    axs[2].plot(
        history["t"],
        converted_history,
        color="blue",
        lw=2,
    )
    axs[2].set_ylabel("T_c (K)")
    axs[2].set_title("Cooling Jacket Temperature", fontsize=12, pad=8)
    axs[2].grid(alpha=0.3)
    axs[2].axhline(params.T_c_max, color="red", ls="--", lw=1, alpha=0.6)
    axs[2].axhline(params.T_c_min, color="blue", ls="--", lw=1, alpha=0.6)

    # Plot Reward
    axs[3].plot(history["t"], history["reward"], color="purple", lw=2)
    axs[3].set_ylabel("Reward")
    axs[3].set_xlabel("Time step")
    axs[3].set_title("Reward Signal", fontsize=12, pad=8)
    axs[3].grid(alpha=0.3)
    # axs[3].set_ylim(0, 1.05)

    # Convert to numpy image
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
        ..., :3
    ]

    plt.close(fig)  # avoid memory leak

    return image, history


def _render(cls, screen, state, params, frames, clock, stride: int = 10):
    """Render function for CSTR environment using matplotlib graphs.

    Args:
        cls: Environment class reference.
        screen: Unused (for Gymnax compatibility).
        state: Current environment state.
        params: Environment parameters.
        frames: List of rendered frames.
        clock: Unused (for Gymnax compatibility).
        stride: Only render every N steps (default=1 = render every step).
    """

    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    # Initialize histories
    if not hasattr(cls, "history"):
        cls.history = {"t": [], "C_a": [], "T": [], "T_c": [], "reward": []}

    step = state.time
    # Only render every `stride` steps
    if step % stride == 0 or step == 1:
        frame, cls.history = render_cstr(state, params, step, cls.history)

        frames.append(frame)
        cls.frames = frames

    return frames, screen, clock
