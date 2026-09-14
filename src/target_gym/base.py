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


NO_RESTART = 2**30  # ``restart_steps`` for a plant that never comes back


def _select(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def failure_kernel(env, key, state, new_state, params):
    """Trips and downtime, as part of the transition kernel.

    A reach-and-hold task is continuing, so leaving the operating envelope is
    not the end of an episode: the plant trips, is down at the failure cost
    for ``params.restart_steps`` steps, and restarts as ``reset_env`` would,
    on the same window clock. A plant with ``restart_steps = NO_RESTART``
    stays down to the end of the window -- the absorbing state of the
    Target-MDP note, lived through rather than asserted.

    ``terminated`` is therefore never raised by a plant. Raising it would
    tell a discounted learner the crashed state is worth zero -- the cheapest
    state in the plant, and the bootstrap behind every agent that learns to
    crash -- and would cut the chain an average-reward learner estimates its
    gain on. The event is reported as ``info["tripped"]`` for the evaluator,
    which counts it; nothing an agent runs has to read it.

    Given the state before the step, the physics' proposal for after it and
    the step's key, returns ``(state, tripped, down)``:

    - healthy before, proposal inside the envelope: the proposal;
    - healthy before, proposal outside: a trip -- the proposal, frozen, with
      ``downtime = restart_steps`` (this step included);
    - down before with more than one step of downtime: the frozen state,
      one step less;
    - down before on its last step of downtime: a fresh draw of ``reset_env``.

    ``downtime`` on a returned state is the number of down steps up to and
    including the one just taken, so it is at least 1 on every down step and
    the reward -- a function of the returned state -- charges the failure
    cost on each of them and on none other.
    """
    countdown = state.downtime
    was_down = jnp.logical_or(env.is_terminated(state, params), countdown > 0)
    restart = jnp.logical_and(was_down, countdown <= 1)
    stay_down = jnp.logical_and(was_down, countdown > 1)
    tripped = jnp.logical_and(
        jnp.logical_not(was_down), env.is_terminated(new_state, params)
    )
    fresh = env.reset_env(key, params)[1].replace(time=new_state.time, downtime=0)
    frozen = state.replace(time=new_state.time, downtime=jnp.maximum(countdown - 1, 1))
    entered = new_state.replace(downtime=jnp.asarray(params.restart_steps, jnp.int32))
    healthy = new_state.replace(downtime=jnp.zeros((), jnp.int32))
    out = _select(
        restart, fresh, _select(stay_down, frozen, _select(tripped, entered, healthy))
    )
    return out, tripped, jnp.logical_or(stay_down, tripped)
