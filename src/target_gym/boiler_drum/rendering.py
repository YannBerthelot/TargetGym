"""Time-series rendering for the boiler drum environment.

The top panel is the one that matters: it shows drum level against the trip
limits, so the inverse response is visible as the level moving *away* from
where the feedwater is pushing it.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.boiler_drum.env import compute_reward, void_fraction


def render_boiler(state, params, step, history):
    history["t"].append(step * params.delta_t / 60.0)
    history["level"].append(100.0 * float(state.level))
    history["pressure"].append(float(state.pressure))
    history["target_p"].append(float(state.target_pressure))
    history["q_steam"].append(float(state.q_steam))
    history["q_feed"].append(float(state.q_feed))
    history["fuel"].append(100.0 * float(state.Q_fuel) / params.Q_max)
    history["alpha"].append(
        100.0 * float(void_fraction(state.m_sr, state.pressure, params))
    )
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(4, 1, figsize=(7, 9), sharex=True, dpi=100)
    fig.suptitle(
        "Boiler drum — natural circulation, 160 MW", fontsize=14, weight="bold"
    )

    trip = 100.0 * params.level_trip
    axs[0].plot(history["t"], history["level"], color="crimson", lw=2, label="level")
    axs[0].axhline(0.0, color="black", ls="--", lw=1.5, label="normal water level")
    axs[0].axhspan(trip, trip + 15, color="firebrick", alpha=0.18)
    axs[0].axhspan(-trip - 15, -trip, color="firebrick", alpha=0.18)
    axs[0].axhline(trip, color="firebrick", lw=1.2)
    axs[0].axhline(-trip, color="firebrick", lw=1.2, label="trip (carryover / dryout)")
    axs[0].set_ylim(-trip - 12, trip + 12)
    axs[0].set_ylabel("drum level [cm]")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].plot(history["t"], history["pressure"], color="darkorange", lw=2)
    axs[1].plot(history["t"], history["target_p"], color="black", ls="--", lw=1.5)
    axs[1].set_ylabel("drum pressure [bar]")
    axs[1].grid(alpha=0.3)

    axs[2].plot(
        history["t"], history["q_steam"], color="steelblue", lw=2, label="steam out"
    )
    axs[2].plot(
        history["t"], history["q_feed"], color="seagreen", lw=2, label="feedwater in"
    )
    axs[2].set_ylabel("flow [kg/s]")
    axs[2].legend(loc="upper right", fontsize=8)
    axs[2].grid(alpha=0.3)

    axs[3].plot(
        history["t"], history["fuel"], color="dimgray", lw=2, label="firing [%]"
    )
    axs[3].plot(
        history["t"],
        history["alpha"],
        color="mediumpurple",
        lw=1.5,
        label="riser void [%] (HIDDEN)",
    )
    axs[3].set_ylabel("[%]")
    axs[3].set_xlabel("time [min]")
    axs[3].legend(loc="upper right", fontsize=8)
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
                "level",
                "pressure",
                "target_p",
                "q_steam",
                "q_feed",
                "fuel",
                "alpha",
                "reward",
            )
        }
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_boiler(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
