import jax.numpy as jnp


def compute_norm_from_coordinates(coordinates: jnp.ndarray) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(coordinates, axis=0)


def log_scaled_reward(error, floor, envelope, xp=jnp):
    """Tracking reward that keeps paying for precision.

    Returns 1.0 at zero error and 0.0 at ``envelope``.

    A reward normalised by the state-space envelope (``1 - e/E``, however
    sharply exponentiated) spends nearly all of its dynamic range on errors the
    controller has already eliminated: with ``E`` = 12 km, the gap between a
    10 m and a 1 m altitude error is under 0.001, so the reward cannot tell a
    good controller from a great one.  Scaling in ``log(error)`` instead pays a
    *constant* increment for every halving of the error, all the way down to
    ``floor`` -- the finest error the corresponding sensor can resolve, below
    which further "improvement" is measuring noise.

    Args:
        error: absolute tracking error, in the same units as the other two.
        floor: sensor resolution; the error scale below which reward saturates.
        envelope: the error at which the reward reaches 0.
    """
    return xp.clip(1.0 - xp.log1p(error / floor) / xp.log1p(envelope / floor), 0.0, 1.0)


def stationary_innovation(std, decay, xp=jnp):
    """The innovation's spread ``s`` for which ``x' = decay * x + s * xi``, with
    ``xi ~ N(0, 1)``, keeps the stationary standard deviation ``std``:
    ``std * sqrt(1 - decay**2)``. An Euler step's ``std * sqrt(2 theta dt)``
    leaves the process ``1 / sqrt(1 - theta dt / 2)`` wider than stated."""
    return std * xp.sqrt(1.0 - decay**2)


def convert_raw_action_to_range(raw_action, min_action, max_action):
    """
    Assuming the action is roughly in (-1,1), we rescale to it (min_action,max_action).
    """
    action = min_action + 0.5 * (jnp.clip(raw_action, -1, 1) + 1) * (
        max_action - min_action
    )
    return action
