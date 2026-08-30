"""Time-series rendering for the wind turbine environment."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from target_gym.energy.wind_turbine.env import (
    compute_reward,
    electrical_power,
    omega_rated,
)


def render_turbine(state, params, step, history):
    history["t"].append(step * params.delta_t)
    history["P"].append(
        float(electrical_power(state.omega, state.torque, params)) / 1e6
    )
    history["target"].append(float(state.target_power) / 1e6)
    history["rpm"].append(float(state.omega) * 60.0 / (2.0 * np.pi))
    history["pitch"].append(float(state.pitch))
    history["wind"].append(float(state.v_wind))
    history["reward"].append(float(compute_reward(state, params)))

    fig, axs = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True, dpi=100)
    fig.suptitle("Wind Turbine — NREL 5 MW", fontsize=14, weight="bold")

    axs[0].plot(history["t"], history["P"], color="seagreen", lw=2, label="power")
    axs[0].plot(
        history["t"],
        history["target"],
        color="black",
        ls="--",
        lw=1.5,
        label="setpoint",
    )
    axs[0].set_ylabel("electrical power (MW)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(alpha=0.3)

    rated_rpm = float(omega_rated(params)) * 60.0 / (2.0 * np.pi)
    axs[1].plot(history["t"], history["rpm"], color="crimson", lw=2, label="rotor")
    axs[1].axhline(rated_rpm, color="grey", ls=":", lw=1.2, label="rated")
    axs[1].axhline(
        rated_rpm * params.overspeed_factor, color="red", ls=":", lw=1.2, label="trip"
    )
    axs[1].set_ylabel("rotor speed (rpm)")
    axs[1].legend(loc="upper right", fontsize=8)
    axs[1].grid(alpha=0.3)

    axs[2].plot(history["t"], history["pitch"], color="navy", lw=2, label="pitch (deg)")
    axs[2].plot(
        history["t"],
        history["wind"],
        color="darkorange",
        lw=1.5,
        label="wind (m/s, HIDDEN)",
    )
    axs[2].set_ylabel("pitch / wind")
    axs[2].set_xlabel("time (s)")
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


def _render(cls, screen, state, params, frames, clock, stride: int = 8):
    if state is None:
        raise ValueError("No state provided")
    if not hasattr(cls, "history") or state.time == 1:
        cls.history = {
            k: [] for k in ("t", "P", "target", "rpm", "pitch", "wind", "reward")
        }
    if state.time % stride == 0 or state.time == 1:
        frame, cls.history = render_turbine(state, params, state.time, cls.history)
        frames.append(frame)
        cls.frames = frames
    return frames, screen, clock
