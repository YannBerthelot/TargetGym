"""The state estimator an observation-only MPC plans from.

Item 5 of the task protocol, as the owner read it on 2026-11-28: the room
comes from an MPC on the true model that estimates the hidden state from the
observations, never from one that reads it. This module is that estimator,
the same for every task: an extended Kalman filter on the task's own JAX
model.

- **Model.** The env's ``step_env`` predicts on the planner's params, the
  process noise zeroed (``experts.mpc.plan_params``), which the caller passes
  as ``model``, with no restart. Its observation, read through its
  instruments' ages but with no noise, is the measurement.
- **Noise.** The measurement noise comes from the env's readings
  (``env.readings``): each reading's ``sigma`` at ``sensor_noise_scale``, its
  rounding's ``step**2 / 12``, and a held reading counted once per update. The
  process noise is the env's own: the spread of the next state over draws of
  its step's noise. The prior is the spread of the env's reset.
- **Interface.** ``belief = est.reset(obs)``, then after each step
  ``belief = est.update(belief, action, obs)``. Nothing takes a state, so
  nothing can read a hidden field. ``belief.state`` is an env state, the
  filter's mean, which an MPC reads in place of the true one (see
  :class:`ObservationOnly`); ``belief.cov`` is the covariance of the fields
  the filter estimates.

A task says which state fields the filter estimates (``fields``): its plant's
hidden and measured states, and the sensor history if a reading is late. Every
other field is carried by the model from the filter's own start, the reset's
mean, so a disturbance the env draws once at its reset, a schedule say, is
something the filter cannot know. A task estimates such a disturbance as a
field with a ``random_walk``, and writes what the controller knows exactly,
such as a metered input, with ``known``.

A solve whose JAX derivative is wrong, such as a bisection (pH's), takes
``linearize="secant"``: central differences across one standard deviation of
the belief, in place of ``jax.jacfwd``.
"""

from __future__ import annotations

import operator
import re
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.base import ChannelReading, FieldReading, true_obs

#: Keys of the filter's own draws: the prior's, the process noise's, and the
#: one its model steps on, none of them an episode's.
_PRIOR, _NOISE, _MODEL = 0x50524952, 0x4E4F4953, 0x4D4F444C
#: The smallest measurement variance, relative to the observation's size, so
#: an exact channel keeps the innovation's covariance invertible.
_FLOOR = 1e-5


class Belief(NamedTuple):
    """What the filter knows after an observation. ``state`` is an env state
    holding the mean, ``cov`` the covariance of ``Estimator.fields``, each
    field's values flattened in their order."""

    state: Any
    cov: jax.Array


def _parse(field: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"([\w.]+)(?:\[(\d+)\])?", field)
    if match is None:
        raise ValueError(f"{field!r} is not a field, nor a field[index]")
    index = match.group(2)
    return match.group(1), None if index is None else int(index)


def _get(state, field: str):
    name, index = _parse(field)
    value = jnp.asarray(operator.attrgetter(name)(state))
    return value if index is None else value[index]


def _replace(state, path: str, value):
    head, _, rest = path.partition(".")
    if rest:
        value = _replace(getattr(state, head), rest, value)
    return state.replace(**{head: value})


def _set(state, field: str, value):
    name, index = _parse(field)
    old = jnp.asarray(operator.attrgetter(name)(state))
    if index is None:
        new = jnp.reshape(value, old.shape).astype(old.dtype)
    else:
        new = old.at[index].set(jnp.reshape(value, old[index].shape).astype(old.dtype))
    return _replace(state, name, new)


def _cov(samples):
    """The covariance of the rows of ``samples``, with every correlation under
    four standard errors of a sample correlation, 4 / sqrt(n), set to 0.

    Fields drawn apart are correlated by a few hundredths in any finite sample,
    and a filter would learn a field no reading sees through that correlation.
    A real one, such as a sensor history's with its field, is far above it."""
    n = samples.shape[0]
    centred = samples - jnp.mean(samples, axis=0)
    cov = centred.T @ centred / (n - 1)
    spread = jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0))
    both = jnp.outer(spread, spread)
    corr = jnp.where(both > 0, cov / jnp.where(both > 0, both, 1.0), 0.0)
    keep = (jnp.abs(corr) >= 4.0 / np.sqrt(n)) | jnp.eye(cov.shape[0], dtype=bool)
    return jnp.where(keep, cov, 0.0)


class Estimator:
    """An extended Kalman filter on task ``env``'s own model (the module's
    docstring says what it is built from).

    ``params`` is the configuration the plant runs, noise included: the filter
    zeroes the noise for its model and reads the noise's size from it.
    ``fields`` are the state fields it estimates, by name (``"T_mass"``),
    dotted if nested, or one element of an array (``"upset_levels[0]"``).
    ``channels`` are the observation's entries it reads, every one by default.

    - ``random_walk``: ``{field: std per step}``, process noise added to the
      env's own, for a disturbance its model holds constant or drives by a
      schedule the filter cannot see.
    - ``known(state, obs) -> state``: writes what the controller knows exactly
      into the mean before each correction, for fields it does not estimate.
    - ``model``: the params the filter's model steps on, the planner's with the
      process noise zeroed (``experts.mpc.plan_params(spec, params)``). The
      caller passes them, so the filter imports neither the registry nor any
      expert code, and an oracle may build on it.
    - ``noise_fields``: in place of ``model``, the params fields that carry
      process noise, zeroed in ``params`` with the sensor and actuator noise.
    - ``linearize``: ``"jax"`` (``jax.jacfwd``) or ``"secant"``.
    - ``draws``: the samples the prior and the process noise are measured on.

    ``prior`` is the belief before any observation, ``process_noise`` and
    ``measurement_noise`` the covariances it uses (the latter at the prior).
    """

    def __init__(
        self,
        env,
        params,
        fields,
        *,
        channels=None,
        random_walk: dict | None = None,
        known: Callable | None = None,
        model=None,
        noise_fields=(),
        linearize: str = "jax",
        draws: int = 1024,
    ):
        if linearize not in ("jax", "secant"):
            raise ValueError(f"linearize is 'jax' or 'secant', not {linearize!r}")
        self.env, self.params, self.fields = env, params, tuple(fields)
        self.linearize, self._known = linearize, known
        if model is None:
            zeroed = (*noise_fields, "sensor_noise_scale", "action_noise")
            model = params.replace(**dict.fromkeys(zeroed, 0.0))
        #: What the filter's model steps on: the plant without noise, and no
        #: restart, so a predicted trip stays the physics' proposal.
        self.model = model.replace(continuing=False)

        keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(_PRIOR), i))(
            jnp.arange(draws)
        )
        resets = jax.vmap(lambda k: env.reset_env(k, params)[1])(keys)
        template = env.reset_env(jax.random.PRNGKey(_PRIOR), self.model)[1]
        for field in self.fields:
            if not jnp.issubdtype(_get(template, field).dtype, jnp.floating):
                raise TypeError(
                    f"{field} is not a float field, so no filter estimates it"
                )
        self._shapes = [_get(template, f).shape for f in self.fields]
        self._sizes = [int(np.prod(s)) for s in self._shapes]
        samples = jax.vmap(self.vector)(resets)
        start = self.embed(template, jnp.mean(samples, axis=0))
        prior_cov = _cov(samples)
        #: The scale the filter works in: the prior's spread, or a thousandth of
        #: the value where the reset fixes a field.
        mean = jnp.abs(self.vector(start))
        std = jnp.sqrt(jnp.diag(prior_cov))
        self._scale = jnp.where(
            std > 1e-6 * jnp.maximum(mean, 1.0), std, 1e-3 * jnp.maximum(mean, 1.0)
        )
        self.prior = Belief(start, prior_cov)
        self._check_deterministic(start)

        self.process_noise = self._step_noise(start, keys, draws) + self._walks(
            random_walk
        )
        n_obs = int(jnp.size(env.get_obs(start, self.model)))
        self.channels = tuple(range(n_obs)) if channels is None else tuple(channels)
        self._readings = self._reading_variances()
        self.measurement_noise = self._measurement_noise(start)
        self.reset = jax.jit(self.reset)
        self.update = jax.jit(self.update)

    # -- the filter's vector and the env state ---------------------------------

    def vector(self, state):
        """The estimated fields of ``state``, flattened in order."""
        return jnp.concatenate([jnp.ravel(_get(state, f)) for f in self.fields])

    def embed(self, state, x):
        """``state`` with its estimated fields set from vector ``x``."""
        for field, piece in zip(self.fields, jnp.split(x, np.cumsum(self._sizes)[:-1])):
            state = _set(state, field, piece)
        return state

    def std(self, belief: Belief) -> dict:
        """Each estimated field's standard deviation, in its own shape."""
        stds = jnp.split(jnp.sqrt(jnp.diag(belief.cov)), np.cumsum(self._sizes)[:-1])
        return {
            f: jnp.reshape(s, shape)
            for f, s, shape in zip(self.fields, stds, self._shapes)
        }

    # -- the filter --------------------------------------------------------------

    def reset(self, obs) -> Belief:
        """The belief after the episode's first observation."""
        return self._correct(self.prior, obs)

    def update(self, belief: Belief, action, obs) -> Belief:
        """The belief after ``action`` was sent and ``obs`` came back."""
        key = jax.random.PRNGKey(_MODEL)

        def step(state):
            return self.env.step_env(key, state, action, self.model)[1]

        mean, cov = belief
        ahead = step(mean)
        jac = self._jacobian(
            lambda x: self.vector(step(self.embed(mean, x))), mean, cov
        )
        return self._correct(Belief(ahead, jac @ cov @ jac.T + self.process_noise), obs)

    def _correct(self, belief: Belief, obs) -> Belief:
        mean, cov = belief
        if self._known is not None:
            mean = self._known(mean, obs)
        idx = jnp.asarray(self.channels)

        def observe(x):
            return jnp.ravel(self.env.get_obs(self.embed(mean, x), self.model))[idx]

        x = self.vector(mean)
        h = self._jacobian(observe, mean, cov) * self._scale[None, :]
        r = self._measurement_noise(mean)
        w = jnp.sqrt(jnp.diag(r))
        # In units of the prior's spread and of each reading's noise, which
        # keeps float32 well away from its limits on every task.
        hw, rw = h / w[:, None], r / jnp.outer(w, w)
        pz = cov / jnp.outer(self._scale, self._scale)
        innovation = (jnp.ravel(jnp.asarray(obs))[idx] - observe(x)) / w
        gain = jnp.linalg.solve(hw @ pz @ hw.T + rw, hw @ pz).T
        keep = jnp.eye(x.size, dtype=pz.dtype) - gain @ hw
        pz = keep @ pz @ keep.T + gain @ rw @ gain.T
        pz = 0.5 * (pz + pz.T)
        x = x + self._scale * (gain @ innovation)
        return Belief(self.embed(mean, x), pz * jnp.outer(self._scale, self._scale))

    def _jacobian(self, fn, mean, cov):
        x = self.vector(mean)
        if self.linearize == "jax":
            return jax.jacfwd(fn)(x)
        # Central differences across one standard deviation of the belief, or
        # the filter's scale where the belief is sharper than a thousandth of it.
        step = jnp.maximum(jnp.sqrt(jnp.clip(jnp.diag(cov), 0.0)), 1e-3 * self._scale)
        moves = jnp.diag(step)
        ahead = jax.vmap(fn)(x + moves)
        behind = jax.vmap(fn)(x - moves)
        return ((ahead - behind) / (2.0 * step[:, None])).T

    # -- the noise ---------------------------------------------------------------

    def _check_deterministic(self, start):
        """Refuse a model that still draws noise: it would predict along one
        invented noise path, the same for every episode, which biases them all."""
        held = getattr(start, "actuator_held", None)
        if held is None:
            held = jnp.zeros(self.env.action_space(self.params).shape)
        a, b = (
            self.env.step_env(jax.random.PRNGKey(k), start, held, self.model)[1]
            for k in (1, 2)
        )
        same = jax.tree.map(lambda x, y: bool(jnp.array_equal(x, y)), a, b)
        if not all(jax.tree.leaves(same)):
            raise ValueError(
                f"{type(self.env).__name__}'s step still draws noise on the "
                "filter's model: pass the planner's params as model, or name the "
                "noise's params fields in noise_fields"
            )

    def _step_noise(self, start, keys, draws):
        """The spread of one step's state over the env's own noise draws."""
        if draws < 2:
            return jnp.zeros((self.prior.cov.shape[0],) * 2)
        noisy = self.params.replace(continuing=False)
        held = getattr(start, "actuator_held", None)
        if held is None:
            held = jnp.zeros(self.env.action_space(self.params).shape)
        keys = jax.vmap(lambda k: jax.random.fold_in(k, _NOISE))(keys)
        ahead = jax.vmap(lambda k: self.env.step_env(k, start, held, noisy)[1])(keys)
        return _cov(jax.vmap(self.vector)(ahead))

    def _walks(self, random_walk):
        variances = [jnp.zeros(n) for n in self._sizes]
        for field, std in (random_walk or {}).items():
            if field not in self.fields:
                raise KeyError(f"{field} has a random walk but is not estimated")
            i = self.fields.index(field)
            variances[i] = variances[i] + jnp.asarray(std, jnp.float32) ** 2
        return jnp.diag(jnp.concatenate(variances))

    def _reading_variances(self):
        """Each reading's variance per element, counted once per update."""
        p, out = self.params, []
        scale = float(p.sensor_noise_scale)
        step_s = getattr(self.env, "control_period", 1) * float(p.delta_t)
        for reading in getattr(self.env, "readings", ()):
            var = (scale * np.asarray(reading.sigma, float)) ** 2
            if reading.step and scale > 0:
                var = var + reading.step**2 / 12.0
            # A held reading repeats one draw, so its updates carry what one
            # reading does.
            var = var * max(round(reading.interval / step_s), 1)
            out.append((reading, jnp.asarray(var, jnp.float32)))
        return out

    def _measurement_noise(self, state):
        """The observation's noise covariance at ``state``, on ``channels``."""
        n_obs = int(jnp.size(true_obs(self.env, state, self.model)))
        r = jnp.zeros((n_obs, n_obs))
        for reading, var in self._readings:
            if isinstance(reading, ChannelReading):
                r = r.at[reading.channel, reading.channel].add(var)
                continue
            if not isinstance(reading, FieldReading):
                continue
            field = reading.field + (
                "" if reading.index is None else f"[{reading.index}]"
            )
            value = _get(state, field)
            jac = jax.jacfwd(
                lambda v: jnp.ravel(
                    true_obs(self.env, _set(state, field, v), self.model)
                )
            )(value)
            jac = jnp.reshape(jac, (n_obs, -1))
            if reading.shared:
                jac = jnp.sum(jac, axis=1, keepdims=True)
            var = jnp.broadcast_to(var, (jac.shape[1],))
            r = r + (jac * var[None, :]) @ jac.T
        idx = jnp.asarray(self.channels)
        r = r[idx][:, idx]
        size = jnp.maximum(
            jnp.abs(jnp.ravel(true_obs(self.env, state, self.model))[idx]), 1.0
        )
        return r + jnp.diag((_FLOOR * size) ** 2)


class ObservationOnly:
    """A full-state policy fed an :class:`Estimator`'s belief in place of the
    true state: ``ObservationOnly(tg.make_oracle(name), estimator)`` is an MPC
    that reads the observation only.

    ``policy.reset()`` at each episode start, then ``action = policy(obs)``.
    It keeps the belief and its last action in Python, so it runs on the host
    one episode at a time, as an oracle does.
    """

    def __init__(self, policy, estimator: Estimator):
        self.policy, self.estimator = policy, estimator
        self.reset()

    def reset(self):
        self.belief, self._action = None, None
        if callable(getattr(self.policy, "reset", None)):
            self.policy.reset()

    def __call__(self, obs):
        obs = jnp.asarray(obs)
        if self.belief is None:
            self.belief = self.estimator.reset(obs)
        else:
            self.belief = self.estimator.update(self.belief, self._action, obs)
        step = getattr(self.policy, "step", None)
        action = (
            step(obs, self.belief.state)
            if step
            else self.policy(obs, self.belief.state)
        )
        self._action = jnp.asarray(action, jnp.float32)
        return action
