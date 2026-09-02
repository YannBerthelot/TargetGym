import functools

import jax
import jax.numpy as jnp
from flax import struct


def canonical_reset(reset_env):
    """Give ``reset_env`` the same state dtypes a stepped state has.

    A state built from Python floats comes out *weakly typed*: JAX prints it as
    ``~float32[]`` rather than ``float32[]``. One ``step_env`` promotes it, so a
    freshly reset state and a stepped state are different abstract values -- and
    anything jitted over the state compiles **twice**, once for each.

    Measured, all eighteen environments did this, 187 leaves in total. It cost a
    full second compile of the gradient MPC's optimiser: 10.35 s then another
    10.24 s for the 3D aircraft, against 9.65 s and nothing once canonicalised.
    Any user jitting their own loop over ``reset_env`` output paid the same
    twice-over.

    Passing an explicit dtype is what removes the weakness; on an already-strong
    leaf it folds away, so this is trace-time only and costs nothing per step.
    """

    @functools.wraps(reset_env)
    def wrapper(self, key, params=None):
        obs, state = reset_env(self, key, params)
        return obs, jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.asarray(x).dtype), state
        )

    return wrapper


@struct.dataclass
class EnvParams:
    delta_t: float = 1.0
    max_steps_in_episode: int = 1_000


@struct.dataclass
class EnvState:
    time: int
