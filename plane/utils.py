from typing import Sequence

import jax.numpy as jnp


def compute_norm_from_coordinates(coordinates: jnp.ndarray) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(coordinates, axis=0)


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
