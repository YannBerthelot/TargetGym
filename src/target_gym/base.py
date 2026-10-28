import functools

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.reward import trip_cost


def canonical_reset(reset_env):
    """Give ``reset_env`` the same state dtypes a stepped state has.

    A state built from Python floats is weakly typed, and one ``step_env``
    promotes it, so without this anything jitted over the state compiles twice:
    once for a reset state and once for a stepped one. The cast is trace-time
    only.
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


#: Unit of each numeric params field, read by the generated environment pages.
#: Every module that defines a params class keeps one of these for the fields
#: it adds; "-" marks a dimensionless value.
PARAM_UNITS = {
    "delta_t": "s",
    "max_steps_in_episode": "steps",
}


@struct.dataclass
class EnvState:
    time: int


def _select(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


#: Salt for the restart draw's key (``failure_kernel``).
_RESTART = 0x52535452


def restart_key(key, time):
    """The key a trip at step ``time`` restarts the plant from.

    Folding in the step makes each restart a fresh draw under a constant
    rollout key, and the salt keeps it apart from the noise keys
    (``fold_in(key, time)``).
    """
    return jax.random.fold_in(jax.random.fold_in(key, time), _RESTART)


def failure_kernel(env, key, state, new_state, params):
    """Failure, as part of the transition kernel: a trip cost, then a restart.

    When the physics' proposal for the next state leaves the operating
    envelope the plant trips: the step is charged ``reward.trip_cost``
    (``restart_steps * failure_cost``) and the plant restarts at once, as
    ``reset_env`` would, on the same window clock. ``terminated`` stays false,
    so the restart is an ordinary transition with a cost and every window has
    the same length. Each restart is a fresh draw (``restart_key``).

    - ``params.continuing = False``: a trip ends the episode instead;
      ``terminated`` is raised and the state is the proposal.
    - ``params.restart_in_place = True``: the plant keeps its state in either
      mode, and every step outside the envelope is charged.

    The trip is reported as ``info["tripped"]`` and its charge as
    ``info["trip_cost"]`` (``charged_trip_cost``).

    Returns ``(state, tripped, scored, terminated)``: the state the window
    continues from, whether it tripped, the state the step's reward is
    computed on (the proposal, so the trip is charged on the step that left
    the envelope), and whether the episode ends here.
    """
    tripped = env.is_terminated(new_state, params)
    # A plant that does not restart cold (a building's thermal mass) keeps its
    # state, and is charged on every step outside until brought back.
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


def finish_step(env, key, state, new_state, params, **info):
    """The end of every ``step_env``: the failure kernel, the reward, the info.

    ``new_state`` is the physics' proposal. It goes through
    ``failure_kernel``, and the reward is ``env.compute_reward`` on the state
    the kernel scores, so an environment that overrides ``compute_reward``
    (one per 3D aircraft task) is scored by its override. The caller builds
    the observation from the returned state.

    Returns ``(state, reward, terminated, info)``. ``info`` holds the keyword
    arguments (``metrics``, say), plus:

    - ``last_state``: the returned state. gymnax's ``step`` resets a finished
      episode and keeps its final observation, not its final state, so this
      keeps the state. After a trip in continuing mode it is the restart
      state, not the state that tripped.
    - ``tripped`` and ``trip_cost`` (``charged_trip_cost``).
    """
    new_state, tripped, scored, terminated = failure_kernel(
        env, key, state, new_state, params
    )
    info = {
        **info,
        "last_state": new_state,
        "tripped": tripped,
        "trip_cost": charged_trip_cost(tripped, params),
    }
    return new_state, env.compute_reward(scored, params), terminated, info
