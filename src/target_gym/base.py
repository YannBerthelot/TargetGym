import functools

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.reward import trip_cost


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
    #: Leaving the envelope restarts the plant, the normal setting
    #: (``failure_kernel``); ``False`` ends the episode instead.
    continuing: bool = True


@struct.dataclass
class EnvState:
    time: int


def _select(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


#: Salt for the restart draw's key (``failure_kernel``).
_RESTART = 0x52535452


def restart_key(key, time):
    """The key a trip at step ``time`` restarts the plant from.

    Rollouts pass one constant key, so the raw key restarted every trip at
    the episode's initial state; the step index makes each restart a fresh
    draw, and the salt keeps it apart from the noise keys
    (``fold_in(key, time)``).
    """
    return jax.random.fold_in(jax.random.fold_in(key, time), _RESTART)


def failure_kernel(env, key, state, new_state, params):
    """Failure, as part of the transition kernel: a trip cost, then a restart.

    Operation is continuing and only ever truncated. When the physics'
    proposal for the next state is outside the operating envelope the plant
    trips: the step is charged the trip cost -- ``restart_steps *
    failure_cost``, the downtime a real restart would take priced at the
    per-step failure cost (``reward.trip_cost``) -- and the plant restarts
    at once as ``reset_env`` would, on the same window clock. ``terminated``
    stays false: the restart is an ordinary transition with a cost, as
    continuing and average-reward methods want it (the chain they estimate a
    gain on is never cut), and every window is the same length. Each restart
    is a fresh draw from a key of its own step (``restart_key``), so a run of
    trips under a constant rollout key does not replay one initial state.

    Episodic use is optional: with ``params.continuing = False`` the plant is
    lost instead, ``terminated`` is raised and the state is the proposal. The
    lump then also keeps a discounted learner from crashing on purpose, as
    long as ``restart_steps >= 1 / (2 (1 - gamma))`` (``failure_cost`` is
    twice the largest tracking cost a plant reaches): 50 steps at
    ``gamma = 0.99``.

    A plant whose params carry ``restart_in_place = True`` neither ends nor
    restarts in either mode: it keeps its state, every step outside the
    envelope is charged, and the plant stays under control.

    The event is reported as ``info["tripped"]``, and the cost it charged as
    ``info["trip_cost"]`` (``charged_trip_cost``), in both modes.

    Given the state before the step, the physics' proposal for after it and
    the step's key, returns ``(state, tripped, scored, terminated)``: the
    state the window continues from (the proposal, or a fresh draw when it
    restarted), whether it tripped, the state the step's reward is computed
    on -- the proposal when it tripped, so that the reward, a function of a
    state, charges the trip cost exactly on the step that left the envelope
    -- and whether the episode ends here.
    """
    tripped = env.is_terminated(new_state, params)
    # A plant that does not restart cold (a building: its thermal mass
    # persists through a lockout) keeps its state. Every step outside the
    # envelope is then a trip -- charged the trip cost -- until the
    # controller brings it back, which is the physical picture and closes
    # the shortcut a fresh draw would offer (a free cool-down).
    in_place = jnp.asarray(getattr(params, "restart_in_place", False), bool)
    continuing = jnp.asarray(getattr(params, "continuing", False), bool)
    lost = jnp.logical_and(tripped, jnp.logical_not(in_place))
    fresh = env.reset_env(restart_key(key, new_state.time), params)[1]
    fresh = fresh.replace(time=new_state.time)
    out = _select(jnp.logical_and(lost, continuing), fresh, new_state)
    return out, tripped, new_state, jnp.logical_and(lost, jnp.logical_not(continuing))


def charged_trip_cost(tripped, params):
    """The trip cost a step charged, for ``info["trip_cost"]``: ``reward.trip_cost``
    when it tripped, else 0. The reward already contains it; this is so a
    learner can normalise or subtract it separately."""
    return jnp.where(tripped, trip_cost(params), 0.0)
