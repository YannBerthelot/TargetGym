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


def _select(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


def failure_kernel(env, key, state, new_state, params):
    """Trips, as part of the transition kernel: a lump cost and a restart.

    A reach-and-hold task is continuing, so leaving the operating envelope is
    not the end of an episode. When the physics' proposal for the next state
    is outside the envelope the plant trips: the step is charged the trip
    cost -- ``restart_steps * failure_cost``, the downtime a real restart
    would take priced at the per-step failure cost (``reward.trip_cost``) --
    and the plant restarts at once as ``reset_env`` would, on the same
    window clock. Nothing is frozen: a frozen plant paid the same total over
    a dead stretch of identical steps, which taught a learner nothing and,
    inside a test window shorter than the restart, priced a trip by when it
    happened rather than by what it was.

    ``terminated`` is therefore never raised by a plant. Raising it would
    tell a discounted learner the crashed state is worth zero -- the cheapest
    state in the plant, and the bootstrap behind every agent that learns to
    crash -- and would cut the chain an average-reward learner estimates its
    gain on. The event is reported as ``info["tripped"]`` for the evaluator,
    which counts it; nothing an agent runs has to read it.

    A plant whose params carry ``restart_in_place = True`` restarts from
    where it tripped instead of from a fresh draw: every step outside the
    envelope is charged, and the plant stays under control.

    Given the state before the step, the physics' proposal for after it and
    the step's key, returns ``(state, tripped, scored)``: the state the
    window continues from (the proposal, or a fresh draw when it tripped),
    whether it tripped, and the state the step's reward is computed on --
    the proposal when it tripped, so that the reward, a function of a state,
    charges the trip cost exactly on the step that left the envelope.
    """
    tripped = env.is_terminated(new_state, params)
    # A plant that does not restart cold (a building: its thermal mass
    # persists through a lockout) keeps its state. Every step outside the
    # envelope is then a trip -- charged the trip cost -- until the
    # controller brings it back, which is the physical picture and closes
    # the shortcut a fresh draw would offer (a free cool-down).
    in_place = jnp.asarray(getattr(params, "restart_in_place", False), bool)
    fresh = env.reset_env(key, params)[1].replace(time=new_state.time)
    out = _select(jnp.logical_and(tripped, jnp.logical_not(in_place)), fresh, new_state)
    return out, tripped, new_state
