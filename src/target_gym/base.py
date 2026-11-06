import dataclasses
import functools
import operator

import jax
import jax.numpy as jnp
from flax import struct

from target_gym.reward import select, total, trip_cost

#: Salt for the instruments' key, kept apart from every draw ``reset_env`` makes.
_SENSORS = 0x53454E53
#: Salt for the actuator's noise, kept apart from the readings' streams.
_ACTUATOR = 0x41435455


def canonical_reset(reset_env):
    """Give ``reset_env`` the same state dtypes a stepped state has, and store
    the episode's ``sensor_key``.

    A state built from Python floats is weakly typed, and one ``step_env``
    promotes it, so without this anything jitted over the state compiles twice:
    once for a reset state and once for a stepped one. The cast is trace-time
    only.

    The key is folded from the episode's, so it takes nothing from the draws
    ``reset_env`` made, and those keep their values. The observation is taken
    again because ``reset_env`` took its own before the key was there.

    Late instruments start out reading the reset plant, and a task with an
    ``initial_action(state, params)`` starts its actuator where the plant is.
    """

    @functools.wraps(reset_env)
    def wrapper(self, key, params=None):
        obs, state = reset_env(self, key, params)
        state = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=jnp.asarray(x).dtype), state
        )
        state = state.replace(
            sensor_key=jax.random.key_data(jax.random.fold_in(key, _SENSORS))
        )
        p = self.default_params if params is None else params
        if _delayed(self):
            state = state.replace(sensor_history=_fresh_history(self, state, p))
        if hasattr(self, "initial_action"):
            state = state.replace(actuator_held=self.initial_action(state, p))
        return (self.get_obs(state, params) if _readings(self) else obs), state

    return wrapper


@struct.dataclass
class EnvParams:
    delta_t: float = 1.0
    max_steps_in_episode: int = 1_000
    #: Leaving the envelope restarts the plant, the normal setting
    #: (``failure_kernel``); ``False`` ends the episode instead.
    continuing: bool = True
    #: Multiplies every reading's ``sigma`` (``Reading``). 0 is the noise-free
    #: task, rounding included; other values test how far a result depends on
    #: the figures.
    sensor_noise_scale: float = 1.0
    #: Opt-in actuator defects, as fractions of the action's span (``actuate``).
    actuator_dead_band: float = 0.0
    actuator_stiction: float = 0.0
    action_noise: float = 0.0
    #: The actuator's own dynamics (``actuate``): the fastest it moves, as a
    #: fraction of the span per ``delta_t`` unit, and its first-order lag, in
    #: ``delta_t`` units. 0 is none. A task's params set these from its hardware,
    #: as one number or a tuple with one per action.
    actuator_rate: float = 0.0
    actuator_lag: float = 0.0


#: Unit of each numeric params field, read by the generated environment pages.
#: Every module that defines a params class keeps one of these for the fields
#: it adds; "-" marks a dimensionless value.
PARAM_UNITS = {
    "delta_t": "s",
    "max_steps_in_episode": "steps",
    "sensor_noise_scale": "-",
    "actuator_dead_band": "fraction of span",
    "actuator_stiction": "fraction of span",
    "action_noise": "fraction of span",
    "actuator_rate": "fraction of span per delta_t unit",
    "actuator_lag": "delta_t units",
}


@struct.dataclass
class EnvState:
    time: int
    #: Where the episode's reading noise comes from (``canonical_reset``). A
    #: state built by hand gets a fixed key, so it has the same tree as a reset
    #: one.
    sensor_key: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(2, jnp.uint32), kw_only=True
    )
    #: ``(late readings, steps)``, newest first: what ``Reading.dead_time`` and
    #: ``Reading.sampled`` read back from. A placeholder on a task with none:
    #: one element and not zero, because whatever reshapes every field of a
    #: batched state cannot reshape an empty one.
    sensor_history: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((1, 1)), kw_only=True
    )


# ---------------------------------------------------------------------------
# Sensors (docs/proposals/measurement-noise.md)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, kw_only=True)
class Reading:
    """One instrument: ``value + scale * sigma * epsilon``, rounded to ``step``.

    ``sigma`` is the short-term scatter of one reading, in the value's units.
    ``step`` is the digital step, 0 for none. ``interval`` is the time between
    updates in ``delta_t``'s units, 0 for every step: seconds on most tasks,
    but minutes on ``distillation``, ``cstr`` and ``unstable_cstr``. ``epsilon``
    is held that long.

    A reading can be late. ``dead_time``, in the same units, is the age of what
    it reports, and with ``sampled`` it reports the same value from one update
    to the next, instead of a current value with held noise. At step ``t``, with
    both in steps, it reports the value at ``interval * (t // interval) -
    dead_time``. ``lag`` is the sensor's first-order time constant, in the same
    units, applied to what the instrument sees, before its dead time. A late
    reading must be a scalar, and none of these goes at ``sensor_noise_scale`` 0,
    which removes noise only: ``true_obs`` is the plant now.
    """

    sigma: float | tuple
    step: float = 0.0
    interval: float = 0.0
    dead_time: float = 0.0
    sampled: bool = False
    lag: float = 0.0


@dataclasses.dataclass(frozen=True)
class FieldReading(Reading):
    """A reading applied to a measured copy of the state, before ``get_obs``,
    so what is derived from the field inherits it.

    ``field`` is a state field, dotted if nested (``"lead.x"``). On an array
    field, ``index`` reads one element, so each element can have its own
    instrument, and without it ``sigma`` applies to every element or gives one
    value per element. ``shared`` takes one draw for the whole array, which
    puts the noise after a ``max`` over it.
    """

    field: str
    index: int | None = None
    shared: bool = False


@dataclasses.dataclass(frozen=True)
class ChannelReading(Reading):
    """A reading of ``channel`` of the observation, for a quantity computed
    from several others. It is taken on the true state's observation, so it
    inherits no field reading's noise."""

    channel: int


def _readings(env):
    return getattr(env, "readings", ())


def _noisy(reading, stream, value, state, params):
    """``reading`` of ``value`` at ``state``: exactly ``value`` at scale 0.

    ``epsilon`` depends only on the episode's key, the reading's ``stream`` and
    the update index, so every policy on one episode key meets the same noise.
    """
    scale = params.sensor_noise_scale
    period = jnp.maximum(jnp.round(reading.interval / params.delta_t), 1)
    update = jnp.asarray(state.time, jnp.int32) // period.astype(jnp.int32)
    key = jax.random.fold_in(jax.random.fold_in(state.sensor_key, stream), update)
    shape = () if getattr(reading, "shared", False) else jnp.shape(value)
    epsilon = jax.random.normal(key, shape, jnp.result_type(value))
    read = value + scale * jnp.asarray(reading.sigma, epsilon.dtype) * epsilon
    if reading.step:
        # Rounded forward, and the instrument's own slope backward: a rounded
        # value is flat, which would cut a gradient tuner's feedback path.
        rounded = jnp.round(read / reading.step) * reading.step
        read = read + jax.lax.stop_gradient(rounded - read)
    return jnp.where(scale > 0, read, value)


def _delayed(env):
    return tuple(r for r in _readings(env) if r.dead_time > 0 or r.sampled or r.lag > 0)


def _history_steps(env) -> int:
    """How far back a late reading reaches. From the default step, because the
    history's shape cannot depend on a traced ``delta_t``. At least 2, so a
    history is never the shape of the placeholder on a state built by hand."""
    dt = float(_control_step(env, env.default_params))
    return 1 + max(
        1,
        *(
            round(r.dead_time / dt) + max(round(r.interval / dt) - 1, 0)
            for r in _delayed(env)
        ),
    )


def true_obs(env, state, params):
    """The observation with no instrument in the way: the plant now, for what
    is a claim about the plant (the time-in-band KPI), never for a controller."""
    get_obs = type(env).get_obs
    return getattr(get_obs, "__wrapped__", get_obs)(env, state, params)


def _late_values(env, state, params):
    """What each late reading's instrument sees now, true and undelayed."""
    obs = true_obs(env, state, params)
    values = []
    for reading in _delayed(env):
        if isinstance(reading, ChannelReading):
            values.append(obs[reading.channel])
        else:
            value = operator.attrgetter(reading.field)(state)
            values.append(value if reading.index is None else value[reading.index])
    if any(jnp.ndim(v) != 0 for v in values):
        raise ValueError("a late reading must be a scalar: give its array an index")
    return jnp.stack(values)


def _fresh_history(env, state, params):
    return jnp.tile(_late_values(env, state, params)[:, None], (1, _history_steps(env)))


def _has_history(env, state) -> bool:
    return state.sensor_history.shape == (len(_delayed(env)), _history_steps(env))


def remember(env, state, new_state, params):
    """``new_state`` with its late readings recorded. The history comes from
    ``state``, so a task that rebuilds its state cannot drop it. A state built
    by hand has none, and starts one here."""
    if not _delayed(env):
        return new_state
    if not _has_history(env, state):
        return new_state.replace(sensor_history=_fresh_history(env, new_state, params))
    now = _late_values(env, new_state, params)
    last = state.sensor_history[:, 0]
    lags = jnp.asarray([r.lag for r in _delayed(env)], last.dtype)
    closes = 1.0 - jnp.exp(-_control_step(env, params) / jnp.where(lags > 0, lags, 1.0))
    now = jnp.where(lags > 0, last + closes * (now - last), now)
    return new_state.replace(
        sensor_history=jnp.concatenate(
            [now[:, None], state.sensor_history[:, :-1]], axis=1
        )
    )


def _aged(env, reading, value, state, params):
    """``value`` as a late ``reading`` reports it at ``state``, or ``value`` if
    the reading is not late (``Reading``)."""
    delayed = _delayed(env)
    if reading not in delayed or not _has_history(env, state):
        return value
    dt = _control_step(env, params)
    age = jnp.round(reading.dead_time / dt).astype(jnp.int32)
    if reading.sampled:
        period = jnp.maximum(jnp.round(reading.interval / dt), 1)
        age = age + jnp.asarray(state.time, jnp.int32) % period.astype(jnp.int32)
    last = state.sensor_history.shape[1] - 1
    return state.sensor_history[delayed.index(reading), jnp.minimum(age, last)]


def _replace_at(state, path, value):
    head, _, rest = path.partition(".")
    if rest:
        value = _replace_at(getattr(state, head), rest, value)
    return state.replace(**{head: value})


def measured_state(env, state, params):
    """The copy of ``state`` the instruments report, for ``get_obs`` and the
    measured reward. It never goes back into the dynamics."""
    measured = state
    for stream, reading in enumerate(_readings(env)):
        if not isinstance(reading, FieldReading):
            continue
        field = operator.attrgetter(reading.field)
        if reading.index is None:
            true = _aged(env, reading, field(state), state, params)
            value = _noisy(reading, stream, true, state, params)
        else:
            true = _aged(env, reading, field(state)[reading.index], state, params)
            value = (
                field(measured)
                .at[reading.index]
                .set(_noisy(reading, stream, true, state, params))
            )
        measured = _replace_at(measured, reading.field, value)
    return measured


def measured_obs(get_obs):
    """Make an environment's ``get_obs`` return readings.

    A reading is a pure function of the state, so experts, tuners and Ajax's
    ``get_raw_obs`` see what the step returned, with no key to pass.
    """

    @functools.wraps(get_obs)
    def wrapper(self, state, params=None):
        params = self.default_params if params is None else params
        obs = get_obs(self, measured_state(self, state, params), params)
        channels = [
            (stream, reading)
            for stream, reading in enumerate(_readings(self))
            if isinstance(reading, ChannelReading)
        ]
        if channels:
            true = get_obs(self, state, params)
            for stream, reading in channels:
                value = _aged(self, reading, true[reading.channel], state, params)
                read = _noisy(reading, stream, value, state, params)
                obs = obs.at[reading.channel].set(read)
        return obs

    return wrapper


def measured_reward(env, scored, tripped, reward, params):
    """The reward a plant can compute from its own sensors, for
    ``info["measured_reward"]``: the costs on the measured copy of ``scored``,
    with the trip the true state took (``tripped``). ``reward`` keeps the true
    cost. Version 1 has no terms, so there it is the true reward."""
    if not _readings(env):
        return reward
    terms = env.reward_terms(
        measured_state(env, scored, params), params, tripped=tripped
    )
    return select(params.reward_version, reward, total(terms))


# ---------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------


def actuator_field(actions: int):
    """The state field ``actuator_held`` that ``actuate`` needs, for a task
    with ``actions`` actions: ``actuator_held: jax.Array = actuator_field(2)``."""
    return struct.field(default_factory=lambda: jnp.zeros(actions), kw_only=True)


def _control_step(env, params):
    """A step's length in ``delta_t`` units, sub-steps included."""
    return getattr(env, "control_period", 1) * params.delta_t


def _scatter(state, action, params, span):
    key = jax.random.fold_in(
        jax.random.fold_in(state.sensor_key, _ACTUATOR), state.time
    )
    epsilon = jax.random.normal(key, action.shape, action.dtype)
    return params.action_noise * span * epsilon


def applied_action(env, state, params):
    """What ``actuate`` sent the plant, for ``info["applied_action"]``. ``state``
    is the one ``actuate`` returned."""
    space = env.action_space(params)
    held = state.actuator_held
    scatter = _scatter(state, held, params, space.high - space.low)
    return jnp.clip(held + scatter, space.low, space.high)


def actuate(env, state, action, params):
    """The actuator between the action and ``compute_next_state``: returns
    ``(applied action, state)``. At the default of 0 for every field the action
    passes as it is, and ``actuator_held`` records it clipped to the box.

    Dead band and stiction are He, Wang, Pottmann and Qin's model (Ind. Eng.
    Chem. Res. 46, 2007), which needs only the held output y: with
    ``e = u[k] - y[k-1]``, the output moves to ``u[k] - sign(e) * f_D`` when
    ``|e| > f_S`` and otherwise holds. ``actuator_dead_band`` D and
    ``actuator_stiction`` S are fractions of the action's span, with
    ``f_D = D / 2`` and ``f_S = D / 2 + S``. So D alone is a pure dead band of
    width D, S alone sticks through S and slips onto the command, and in
    Choudhury, Thornhill and Shah's terms (ADCHEM 2003) deadband plus stickband
    is ``D + S`` and the slip jump ``S``. Their stuck point resets at each
    reversal, so a command that zigzags as it climbs may not break free; here the
    force follows the command's distance from the stem. A command at the end
    of its range reaches it, as in their model.

    ``action_noise`` is positioner repeatability: the output's scatter as a
    fraction of the span, on top of the held output and not stored in it. It
    comes from the state, so it is the same for every policy on an episode key.

    ``actuator_lag`` tau then closes ``1 - exp(-dt / tau)`` of the gap to that
    output each step, and ``actuator_rate`` caps the move at the rate times the
    span and the step. So a rate-limited actuator reaches the end of its range at
    its rate, not at once.
    """
    space = env.action_space(params)
    low, high = space.low, space.high
    span = high - low
    command = jnp.clip(action, low, high)
    f_d = 0.5 * params.actuator_dead_band * span
    f_s = f_d + params.actuator_stiction * span
    last = state.actuator_held
    gap = command - last
    # With no band the command is the output even at gap 0, where taking ``last``
    # gives the same value but no slope, and a planner needs the slope.
    moves = (jnp.abs(gap) > f_s) | (f_s == 0)
    wanted = jnp.where(moves, command - jnp.sign(gap) * f_d, last)
    wanted = jnp.where((command <= low) | (command >= high), command, wanted)
    dt = _control_step(env, params)
    lag, rate = jnp.asarray(params.actuator_lag), jnp.asarray(params.actuator_rate)
    closes = jnp.where(lag > 0, 1.0 - jnp.exp(-dt / jnp.where(lag > 0, lag, 1.0)), 1.0)
    most = jnp.where(rate > 0, rate * span * dt, jnp.inf)
    held = last + jnp.clip(closes * (wanted - last), -most, most)
    applied = jnp.clip(held + _scatter(state, action, params, span), low, high)
    active = (
        (params.actuator_dead_band > 0)
        | (params.actuator_stiction > 0)
        | (params.action_noise > 0)
        | (rate > 0)
        | (lag > 0)
    )
    return jnp.where(active, applied, action), state.replace(actuator_held=held)


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
    - The instruments stay: a restart keeps ``sensor_key``, so a policy that
      trips meets the noise of one that did not.
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
    fresh = fresh.replace(time=new_state.time, sensor_key=new_state.sensor_key)
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
    - ``measured_reward`` (``measured_reward``), what a learner trains on.
      ``reward`` stays the true cost, and the score reads it.
    - ``applied_action`` (``applied_action``), what the actuator did, from the
      ``state`` ``actuate`` returned.
    """
    new_state = remember(env, state, new_state, params)
    new_state, tripped, scored, terminated = failure_kernel(
        env, key, state, new_state, params
    )
    reward = env.compute_reward(scored, params)
    info = {
        **info,
        "last_state": new_state,
        "tripped": tripped,
        "trip_cost": charged_trip_cost(tripped, params),
        "measured_reward": measured_reward(env, scored, tripped, reward, params),
        "applied_action": applied_action(env, state, params),
    }
    return new_state, reward, terminated, info
