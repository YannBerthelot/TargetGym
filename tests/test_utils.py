import jax.numpy as jnp
import numpy as np
import pytest

from plane_env.utils import compute_norm_from_coordinates


def test_compute_norm_from_coordinates():
    coordinates = jnp.array([1, 2])
    norm = compute_norm_from_coordinates(coordinates)
    assert norm == pytest.approx(np.sqrt(1 + 4))
    assert norm.shape == ()

    # test it still works when vectorized
    A = jnp.array([1, 2])
    B = jnp.array([3, 4])
    coordinates = jnp.array([A, B])
    norm = compute_norm_from_coordinates(coordinates)
    assert jnp.allclose(
        norm,
        jnp.array([np.sqrt(1 + 9), np.sqrt(4 + 16)]),
    )

    assert norm.shape == (2,)
