import os
from typing import Any, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

EnvState = Any


def compute_norm_from_coordinates(coordinates: jnp.ndarray) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(coordinates, axis=0)


def plot_curve(data, name, folder="figs"):
    fig, ax = plt.subplots()
    ax.plot(data)
    title = f"{name} vs time"
    plt.title(f"{name} vs time")
    plt.savefig(os.path.join(folder, title))
    plt.close()


def plot_features_from_trajectory(states: Sequence[EnvState], folder: str):
    for feature_name in states[0].__dataclass_fields__.keys():
        if "__dataclass_fields__" in dir(states[0].__dict__[feature_name]):
            plot_features_from_trajectory(
                [state.__dict__[feature_name] for state in states], folder
            )
        else:
            feature_values = [state.__dict__[feature_name] for state in states]
            plot_curve(feature_values, feature_name, folder=folder)


def list_to_array(list):
    cls = type(list[0])
    return cls(
        **{
            k: jnp.array([getattr(v, k) for v in list])
            for k in cls.__dataclass_fields__
        }
    )


def array_to_list(array):
    cls = type(array)
    size = len(getattr(array, cls._fields[0]))
    return [
        cls(**{k: v(getattr(array, k)[i]) for k, v in cls._field_types.items()})
        for i in range(size)
    ]


def convert_frames_from_gym_to_wandb(frames: list) -> np.ndarray:
    """Convert frames from gym format (time, width, height, channel) to wandb format (time, channel, height, width)"""
    return np.array(frames).swapaxes(1, 3).swapaxes(2, 3)
