"""The stream: a task run as a continuing plant (docs/proposals/long-strata.md, Part A).

A task's ``step_env`` is a window's step; ``StreamEnv`` runs it with no window. It
never signals ``terminated`` (a trip restarts the plant inside the task), and signals
``truncated`` only at an end it is given: a stream length, or a scoring run's end.
Between those, the world keeps time on its own:

- scenarios switch as a memoryless chain whose long-run time shares are their weights;
- each scenario's events start at its rate, through the task's ``start_event`` hook;
- every draw of the world, the task's own step keys included, comes from the stream's
  seed and the step, so the policy and the learner's key never move it (common
  random numbers, S2);
- the observation is the task's own: no clock and no scenario label.

A scoring run (R2) pins one scenario, burns in for a hidden ``B`` drawn in
``[b_min, 2 b_min)`` with the scenario's events at their rate, then places exactly
``count`` events at distinct uniform steps of the scored part ``S``. ``info["scored"]``
marks the scored steps.

Draws and keys::

    stream = StreamEnv(env, params, scenarios.suite(spec))
    obs, state = stream.reset(key, params)
    obs, state, reward, terminated, truncated, info = stream.step(key, state, a, params)
"""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment

__all__ = ["StreamEnv", "StreamState", "event_counts", "make_stream"]

#: Salt for a start's draws, apart from the per-step keys ``fold_in(key, t)``.
_START = 0x53545254


@struct.dataclass
class StreamState:
    env: Any  # the task's own state
    key: jax.Array  # the stream's seed: every draw of the world comes from it and t
    time: jax.Array  # steps since the stream started
    scenario: jax.Array  # the current scenario's index, never observed
    burn_in: jax.Array  # a scoring run's hidden burn-in; 0 in a plain stream
    planned: jax.Array  # a scoring run's event steps in S, -1 where unused


def _per_step(rate_per_h: float, dt_h: float) -> float:
    """The chance that a memoryless clock of ``rate_per_h`` rings within one step."""
    return float(1.0 - np.exp(-rate_per_h * dt_h))


def make_stream(name: str, **kwargs) -> "StreamEnv":
    """Registry task ``name`` as a stream over its scenario suite, at its scored
    params. A task opts in with ``STREAM_READY = True`` on its env, once its events
    end, its cycles wrap and its rates are set (long-strata draft 4, S1)."""
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    spec = REGISTRY[name]
    env = spec.make_env()
    if not getattr(env, "STREAM_READY", False):
        raise ValueError(f"{name} is not wired for the stream yet")
    suite = scenarios.suite(spec)
    return StreamEnv(env, spec.make_test_params(), suite, **kwargs)


def event_counts(expected: float, n: int) -> np.ndarray:
    """Events per scoring seed for ``n`` seeds: the floor and the ceiling of
    ``expected``, as many ceilings as bring the mean nearest to it (R2)."""
    low = int(np.floor(expected))
    high = int(round((expected - low) * n))
    return np.array([low + 1] * high + [low] * (n - high))


class StreamEnv(environment.Environment):
    """``env`` as a continuing plant over ``scenarios`` (``scenarios.Scenario``)."""

    def __init__(
        self,
        env,
        params,
        scenarios: Sequence,
        *,
        length: int | None = None,
        pin: int | None = None,
        b_min: int | None = None,
        scored: int | None = None,
        max_events: int = 0,
    ):
        if not bool(getattr(params, "continuing", True)):
            raise ValueError("a stream needs params.continuing: a trip restarts")
        scenarios = tuple(scenarios)
        weights = np.array([s.weight for s in scenarios], float)
        if not (weights > 0).all():
            raise ValueError("every scenario in a stream needs a positive weight")
        weights /= weights.sum()
        switching = pin is None and len(scenarios) > 1
        if switching and any(s.dwell_h is None for s in scenarios):
            raise ValueError("switching scenarios need a dwell_h each")
        from target_gym.registry import control_step_seconds

        # Plant hours per step: ``delta_t`` is in minutes on some tasks.
        dt_h = control_step_seconds(env, params) / 3600.0
        leave = np.array(
            [
                0.0 if s.dwell_h is None else _per_step(1 / s.dwell_h, dt_h)
                for s in scenarios
            ]
        )
        rates = np.array([_per_step(s.events_per_h, dt_h) for s in scenarios])
        self._hook = getattr(env, "start_event", None)
        if rates.any() and not callable(self._hook):
            raise ValueError(f"{type(env).__name__} has events but no start_event hook")
        if (pin is None) != (b_min is None) or (pin is None) != (scored is None):
            raise ValueError("a scoring run needs pin, b_min and scored together")
        if max_events > (scored or 0) and pin is not None:
            raise ValueError("more planned events than scored steps")
        self.env, self._base, self._length = env, params, length
        self._pin, self._b_min, self._scored = pin, b_min, scored
        self._max_events = int(max_events) if pin is not None else 0
        self._weights = jnp.asarray(weights)
        self._leave = jnp.asarray(leave)
        # Leaving s with chance h_s per step and entering s' with chance proportional
        # to w_s' h_s', the current scenario included, gives time shares w exactly.
        entry = weights * leave if switching else weights
        self._entry = jnp.log(jnp.asarray(entry / entry.sum()))
        self._event_p = jnp.asarray(rates)
        self._expected = float(rates[pin] * scored) if pin is not None else 0.0
        self.scenarios = scenarios
        sets = [
            {**s.overrides, **getattr(s, "stream_overrides", {})} for s in scenarios
        ]
        self._stacked = {
            f: jax.tree_util.tree_map(
                lambda *v: jnp.stack([jnp.asarray(x) for x in v]),
                *[o.get(f, getattr(params, f)) for o in sets],
            )
            for f in sorted({f for o in sets for f in o})
        }

    @property
    def default_params(self):
        return self._base

    @property
    def name(self) -> str:
        return f"Stream({getattr(self.env, 'name', type(self.env).__name__)})"

    def action_space(self, params=None):
        return self.env.action_space(params)

    def observation_space(self, params=None):
        return self.env.observation_space(params)

    def scenario_params(self, params, scenario):
        """``params`` as the stream runs scenario ``scenario`` (an index): its
        ``overrides``, then its ``stream_overrides``."""
        picked = {
            f: jax.tree_util.tree_map(lambda a: a[scenario], v)
            for f, v in self._stacked.items()
        }
        return params.replace(**picked) if picked else params

    _params = scenario_params

    def start(self, key, count=0, params=None):
        """A stream from seed ``key``; a scoring run with ``count`` events in S."""
        params = self._base if params is None else params
        k_scenario, k_env, k_burn, k_plan = jax.random.split(
            jax.random.fold_in(key, _START), 4
        )
        if self._pin is None:
            # A stationary start: the scenario drawn from its long-run share.
            scenario = jax.random.categorical(k_scenario, jnp.log(self._weights))
            burn_in = jnp.int32(0)
        else:
            scenario = jnp.int32(self._pin)
            burn_in = jax.random.randint(k_burn, (), self._b_min, 2 * self._b_min)
        planned = jnp.full((self._max_events,), -1, jnp.int32)
        if self._max_events:
            steps = jax.random.choice(
                k_plan, self._scored, (self._max_events,), replace=False
            )
            planned = jnp.where(
                jnp.arange(self._max_events) < count, burn_in + steps, -1
            )
        obs, env_state = self.env.reset_env(k_env, self._params(params, scenario))
        state = StreamState(
            env_state,
            key,
            jnp.int32(0),
            jnp.int32(scenario),
            jnp.int32(burn_in),
            jnp.sort(planned).astype(jnp.int32),
        )
        return obs, state

    def reset_env(self, key, params):
        # A scoring run reset without a count draws one with the right mean.
        low = np.floor(self._expected)
        count = low + (
            jax.random.uniform(jax.random.fold_in(key, ~_START & 0x7FFFFFFF))
            < self._expected - low
        )
        return self.start(key, count.astype(jnp.int32), params)

    def step_env(self, key, state, action, params):
        # ``key`` is the learner's and is not used: the world is the stream's own.
        t = state.time
        k_env, k_leave, k_next, k_event, k_hook = jax.random.split(
            jax.random.fold_in(state.key, t), 5
        )
        p = self._params(params, state.scenario)
        obs, env_state, reward, _, info = self.env.step_env(k_env, state.env, action, p)
        scenario = state.scenario
        if self._pin is None:
            leave = jax.random.uniform(k_leave) < self._leave[scenario]
            entered = jax.random.categorical(k_next, self._entry).astype(jnp.int32)
            scenario = jnp.where(leave, entered, scenario)
        event = jax.random.uniform(k_event) < self._event_p[state.scenario]
        if self._pin is not None:
            event = jnp.where(t >= state.burn_in, jnp.any(state.planned == t), event)
        if callable(self._hook):
            # An event starts after the step that draws it, and that step returns
            # the reading of the state the event leaves: a new sortie's aircraft
            # shows at once, not one step late.
            started = self._hook(env_state, k_hook, p)
            env_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(event, a, b), started, env_state
            )
            obs = jnp.where(event, self.env.get_obs(started, p), obs)
        info = {
            **info,
            "scenario": state.scenario,
            "event": event,
            "scored": t >= state.burn_in,
        }
        new = state.replace(env=env_state, time=t + 1, scenario=scenario)
        return obs, new, reward, jnp.array(False), info

    def is_truncated(self, state, params) -> jax.Array:
        if self._pin is not None:
            return state.time >= state.burn_in + self._scored
        if self._length is not None:
            return state.time >= self._length
        return jnp.array(False)

    @partial(jax.jit, static_argnames=("self",))
    def step(self, key, state, action, params=None):
        if self._pin is not None or self._length is not None:
            return super().step(key, state, action, params)
        # Nothing can end, so no reset is computed: gymnax computes one every step.
        params = self._base if params is None else params
        obs, state, reward, terminated, info = self.step_env(key, state, action, params)
        no = jnp.array(False)
        info = {**info, "terminated": no, "truncated": no, "final_observation": obs}
        return obs, state, reward, terminated, no, info
