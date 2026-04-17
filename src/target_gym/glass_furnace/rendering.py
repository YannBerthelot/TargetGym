import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.glass_furnace.env import N_SETPOINTS, compute_reward


def render_glass_furnace(state, params, step, history):
    """
    Render the glass furnace state as time-series graphs:
        - Crown temperature (with target + bounds)
        - Glass zone temperatures (hidden from agent, shown for debugging)
        - Fuel flow (the control input)
        - Reward

    Returns an RGB image (numpy array).
    """
    # Update history
    history["t"].append(step)
    history["T_crown"].append(float(state.T_crown))
    history["T_melt"].append(float(state.T_melt))
    history["T_work"].append(float(state.T_work))
    history["target_T_crown"].append(float(state.target_T_crown))
    # Fuel flow is rendered as a percentage of fuel_max (0–100) for consistency
    # with the observation vector.
    history["fuel_flow"].append(100.0 * float(state.fuel_flow) / params.fuel_max)
    history["reward"].append(float(compute_reward(state, params)))

    # Convert step index to simulated hours for x-axis readability
    t_hours = [s * params.delta_t / 3600.0 for s in history["t"]]

    fig, axs = plt.subplots(4, 1, figsize=(7, 9), sharex=True, dpi=100)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(
        "Glass Furnace (float process) — Evolution", fontsize=15, weight="bold"
    )

    # 1) Crown temperature + target + bounds
    axs[0].plot(t_hours, history["T_crown"], color="crimson", lw=2, label="T_crown")
    axs[0].plot(
        t_hours,
        history["target_T_crown"],
        color="black",
        ls="--",
        lw=1.5,
        label="target (schedule)",
    )
    # Schedule transition markers
    ep_hours = params.max_steps_in_episode * params.delta_t / 3600.0
    for i in range(1, N_SETPOINTS):
        axs[0].axvline(
            ep_hours * i / N_SETPOINTS, color="gray", ls=":", lw=0.8, alpha=0.6
        )
    axs[0].axhline(params.T_crown_max, color="red", ls=":", lw=1, alpha=0.6)
    axs[0].axhline(params.T_crown_min, color="blue", ls=":", lw=1, alpha=0.6)
    axs[0].set_ylabel("T_crown (°C)")
    axs[0].set_title(
        "Crown temperature (OBSERVED — controlled variable)", fontsize=11, pad=6
    )
    axs[0].grid(alpha=0.3)
    axs[0].legend(loc="upper right", fontsize=8)

    # 2) Glass zone temperatures (hidden from agent)
    axs[1].plot(t_hours, history["T_melt"], color="orangered", lw=2, label="T_melt")
    axs[1].plot(t_hours, history["T_work"], color="darkorange", lw=2, label="T_work")
    axs[1].set_ylabel("T_glass (°C)")
    axs[1].set_title("Glass zone temperatures (HIDDEN from agent)", fontsize=11, pad=6)
    axs[1].grid(alpha=0.3)
    axs[1].legend(loc="upper right", fontsize=8)

    # 3) Fuel flow (action) — expressed as a percentage of fuel_max
    axs[2].plot(t_hours, history["fuel_flow"], color="navy", lw=2)
    axs[2].axhline(100.0, color="red", ls=":", lw=1, alpha=0.6)
    axs[2].axhline(
        100.0 * params.fuel_min / params.fuel_max,
        color="blue",
        ls=":",
        lw=1,
        alpha=0.6,
    )
    axs[2].set_ylabel("fuel (% of max)")
    axs[2].set_title("Fuel flow (ACTION)", fontsize=11, pad=6)
    axs[2].set_ylim(-5.0, 105.0)
    axs[2].grid(alpha=0.3)

    # 4) Reward
    axs[3].plot(t_hours, history["reward"], color="purple", lw=2)
    axs[3].set_ylabel("reward")
    axs[3].set_xlabel("Time (hours)")
    axs[3].set_title("Reward signal", fontsize=11, pad=6)
    axs[3].grid(alpha=0.3)
    axs[3].set_ylim(-0.05, 1.05)

    # Convert figure to numpy image
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[
        ..., :3
    ]

    plt.close(fig)

    return image, history


def _render(cls, screen, state, params, frames, clock, stride: int = 10):
    """Render function for GlassFurnace environment using matplotlib graphs.

    Args:
        cls: Environment class reference.
        screen: Unused (for Gymnax compatibility).
        state: Current environment state.
        params: Environment parameters.
        frames: List of rendered frames.
        clock: Unused (for Gymnax compatibility).
        stride: Only render every N steps (episodes are long, so default=10).
    """

    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    # Initialize / reset histories on new episode
    if not hasattr(cls, "history") or state.time == 1:
        cls.history = {
            "t": [],
            "T_crown": [],
            "T_melt": [],
            "T_work": [],
            "target_T_crown": [],
            "fuel_flow": [],
            "reward": [],
        }

    step = state.time
    if step % stride == 0 or step == 1:
        frame, cls.history = render_glass_furnace(state, params, step, cls.history)
        frames.append(frame)
        cls.frames = frames

    return frames, screen, clock
