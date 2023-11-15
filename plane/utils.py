from typing import Sequence
import jax.numpy as jnp


def compute_norm_from_coordinates(coordinates: Sequence[float]) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(jnp.array(coordinates))
