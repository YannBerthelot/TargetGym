"""The front door: every task by its registry name.

``env, params = target_gym.make("plane")`` mirrors ``gymnax.make``, and
``params`` is the scored configuration, the one the shipped baselines and the
protocol are measured on. Everything here resolves through the registry.

Kept out of ``base.py`` and the environment packages, which are hashed into
every provenance fingerprint: how a task is looked up is not part of what it is.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np

from target_gym import registry


def list_envs(group: str | None = None) -> list[str]:
    """Registry names of every task, or of one group (``"aircraft"``, ...)."""
    specs = registry.all_specs() if group is None else registry.specs_in_group(group)
    return [spec.name for spec in specs]


def make(name: str, scenario: str | None = None, **overrides):
    """``(env, params)`` for task ``name``, like ``gymnax.make``.

    ``params`` is the scored configuration. ``scenario`` takes one of the
    task's scenarios instead (:func:`list_scenarios`): the scored
    configuration with that scenario's overrides, e.g.
    ``make("ph_neutralization", scenario="bad_day")``. Keyword ``overrides``
    then replace single fields, e.g. ``make("plane", max_steps_in_episode=500)``.

    A task is scored on all its scenarios, weighted by their share of
    operating time. To train on that mix, step :func:`make_stream`, which
    rotates the scenarios by weight, or draw ``scenario`` by
    :func:`list_scenarios`'s shares for each episode.

    The environment is shared between calls: environments carry no episode
    state (it all lives in the ``state`` passed through ``reset`` and
    ``step``), and a new instance would be compiled again by JAX.
    """
    spec = registry.get(name)
    if scenario is None:
        return spec.make_env(), spec.make_test_params(**overrides)
    from target_gym import scenarios

    suite = {s.name: s for s in scenarios.suite(spec)}
    if scenario not in suite:
        raise KeyError(f"{name} has no scenario {scenario!r}; it has {list(suite)}")
    return spec.make_env(), scenarios.params_for(spec, suite[scenario], **overrides)


def list_scenarios(name: str) -> dict[str, float]:
    """Task ``name``'s scenarios, each with its share of operating time, the
    weight its score gives it. The shares sum to 1; any name is a ``scenario``
    for :func:`make`."""
    from target_gym import scenarios

    return scenarios.weights(registry.get(name))


class FullState:
    """A policy that reads the simulator's true state: ``FullState(policy)``,
    with ``policy(obs, state) -> action``.

    A deployed controller reads only its observation, and that is what every
    runner hands a policy unless it is wrapped in this. Wrap an oracle, an MPC
    on the true state or a debugging probe, never a policy you compare with
    the reference: a scored result records whether its policy read the state.
    """

    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        if callable(getattr(self.policy, "reset", None)):
            self.policy.reset()

    def __call__(self, obs, state):
        return self.policy(obs, state)


class Expert:
    """The shipped expert for one task, used like the environment.

    ``memory = expert.reset()`` at each episode start, then
    ``action, memory = expert.step(memory, obs)``: the memory is the PID's
    integral and last error, carried by you as the environment's ``state``
    is. The same ``step`` runs in a Python loop, under ``jit`` and
    ``lax.scan``, and on a batch as ``jax.vmap(expert.step)`` from
    ``expert.reset(n_envs)``. Under an auto-resetting ``env.step``, clear
    the memory of finished episodes with
    ``target_gym.experts.episodes.reset_where_done``.

    It reads the observation only, as a deployed controller does.
    ``expert.params`` holds its gains.
    """

    def __init__(self, controller):
        self._controller = controller
        self._step = jax.jit(controller.act)

    @property
    def params(self):
        return self._controller.params

    def reset(self, n_envs: int | None = None):
        """Fresh memory for one episode, or ``n_envs`` copies stacked."""
        fresh = jax.tree.map(jnp.asarray, self._controller.init_state)
        if n_envs is None:
            return fresh
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n_envs,) + x.shape), fresh)

    def step(self, memory, obs):
        """``(action, memory)``: the expert's action on ``obs``, in the
        environment's units and bounds, and its memory after it."""
        return self._step(memory, obs)

    # Deprecated in 0.8.0: one form replaces the three of 0.7.0.

    def host(self):
        _deprecated("make_expert(name).host()")
        return self._controller.host()

    def batched(self):
        _deprecated("make_expert(name).batched()")
        return self._controller.batched()

    @property
    def init_state(self):
        _deprecated("make_expert(name).init_state")
        return self._controller.init_state

    def act(self, state, obs):
        _deprecated("make_expert(name).act()")
        return self._controller.act(state, obs)


class _Carried:
    """An :class:`Expert` as ``policy(obs) -> action``, holding its memory,
    for the helpers that drive a policy in their own loop."""

    def __init__(self, expert: Expert):
        self.expert = expert
        self.reset()

    def reset(self):
        self.memory = self.expert.reset()

    def __call__(self, obs):
        action, self.memory = self.expert.step(self.memory, obs)
        return np.asarray(action)


def as_policy(policy):
    """``policy`` for a loop that calls ``policy(obs)``: an :class:`Expert`
    gets fresh memory it carries itself, anything else is returned as is."""
    return _Carried(policy) if isinstance(policy, Expert) else policy


def _deprecated(old: str) -> None:
    warnings.warn(
        f"{old} is deprecated; use memory = expert.reset() and "
        "action, memory = expert.step(memory, obs)",
        DeprecationWarning,
        stacklevel=3,
    )


def make_expert(name: str, sensor_noise_scale: float | None = None) -> Expert:
    """The shipped expert for ``name``: see :class:`Expert`.

    ``sensor_noise_scale`` takes the gains tuned at that noise level, for an
    environment made with the same level. None and 1 mean the shipped gains,
    and a level with no stored gains raises."""
    return Expert(registry.get(name).make_pid_controller(sensor_noise_scale))


class Reference(Expert):
    """The charter's reference controller for one scored task, used like
    :class:`Expert`: ``memory = ref.reset()``, then
    ``action, memory = ref.step(memory, obs)``.

    It is the rung of the reference ladder the charter kept for the task (a
    PI by SIMC from bump tests, with feedforward, a cascade or an override
    where they pay), frozen as package data, or the shipped expert on the
    reactor. ``rung`` names it. Like the expert it reads the observation only.

    Its line runs on the scored configuration with a few changes, such as the
    setpoint its episodes start on: ``ref.params(scenario)`` gives that
    configuration, the one to run a policy on to compare with it seed by seed.
    """

    def __init__(self, controller, frozen: dict):
        from target_gym.reference import changes

        super().__init__(controller)
        self.task, self.rung = frozen["task"], frozen["rung"]
        self._changes = changes(frozen)

    def params(self, scenario: str | None = None):
        """The configuration the reference's line runs on, in ``scenario``
        (:func:`make`'s), or in the scored configuration."""
        return make(self.task, scenario=scenario, **self._changes)[1]


def make_reference(name: str) -> Reference:
    """The charter's reference for scored task ``name``: see :class:`Reference`.

    Built from its frozen numbers with no bump test. A task with none raises
    ``KeyError``, and a reference frozen on another plant raises
    ``target_gym.reference.StaleReference``. Beating it is the charter's
    headline claim; beating :func:`make_expert` is a weaker one on most tasks,
    since the shipped expert is not the reference there.
    """
    from target_gym import reference

    controller, frozen = reference.build(name)
    return Reference(controller, frozen)


def score(name: str, policy, seeds=None, jit: bool = True) -> dict:
    """The charter's comparison of ``policy`` with scored task ``name``'s
    reference: the saving ``1 - J_policy / J_reference`` over every scenario,
    weighted by its share of operating time, with its 95 % interval, and one row
    per scenario beside it.

    ``policy`` is ``policy(obs) -> action``, compiled with JAX, or an object
    stepped as :func:`make_expert`'s is, also compiled. One that keeps its own
    Python memory, or any with ``jit=False``, runs on the host, one episode at a
    time, and so does a :class:`FullState`, which the result records. A list of
    policies is one per training run, and the interval then covers the runs.

    The episodes are the charter's: the scoring seeds (``seeds`` replaces them;
    choose checkpoints on ``charter.VALIDATION_SEEDS``, never on these), each
    scenario at :meth:`Reference.params`, each read over the task's scored window
    with its edge charge, and the reactor on its pinned 7-day streams. The
    reference runs beside the policy on the same seeds and worlds. See
    ``target_gym.scorecard.score`` for every key of the result.
    """
    from target_gym.scorecard import score as run

    return run(name, policy, seeds, jit)


def make_oracle(name: str, params=None):
    """The full-state MPC for ``name``, built as the recorded baselines build it.

    ``action = oracle.step(obs, state)``. It reads the true simulator state,
    including what the observation withholds, so it bounds what a perfect
    model could do rather than competing with a policy. ``params`` defaults to
    the scored configuration; the planner's copy has its process noise zeroed,
    so it plans on the mean disturbance. Returned reset, ready for an episode.
    """
    from target_gym.experts import mpc

    spec = registry.get(name)
    if not spec.has_mpc:
        raise NotImplementedError(spec.baselines_note or f"{name} ships no oracle")
    params = spec.make_test_params() if params is None else params
    oracle = spec.make_mpc(
        spec.make_env(), mpc.plan_params(spec, params), disturbances=params
    )
    oracle.reset()
    return oracle


def make_stream(name: str, **kwargs):
    """Task ``name`` as a continuing plant: the stream.

    It steps like :func:`make`'s environment and never ends: a trip restarts
    the plant inside the task, the task's scenarios take turns in proportion to
    their weights, and their events arrive at their rates. The observation is
    the task's own, with no clock and no scenario label. The world's draws come
    from the reset key alone, so two policies on one key meet the same world::

        stream = tg.make_stream("hvac")
        obs, state = stream.reset(key)
        obs, state, reward, terminated, truncated, info = stream.step(key, state, a)

    ``length=n`` truncates after ``n`` steps. It runs the task's scored
    parameters, and refuses a task not wired for it yet.
    """
    from target_gym.stream import make_stream as build

    return build(name, **kwargs)


def make_gymnasium(
    name: str, render_mode: str | None = None, stream: bool = False, **overrides
):
    """Task ``name`` as a Gymnasium environment, on the scored configuration.

    For libraries that expect the classic stateful API. It steps on the host
    and returns NumPy, so it cannot sit inside ``jax.jit``; prefer
    :func:`make` for JAX training. ``overrides`` work as in :func:`make`.

    ``stream=True`` gives the continuing plant of :func:`make_stream`: it
    never ends, and the seed of ``reset`` fixes its world. It runs the scored
    parameters, so it takes no ``overrides``.
    """
    from target_gym.wrapper import GymnasiumEnv

    if stream:
        if overrides:
            raise ValueError(
                f"a stream runs {name}'s scored parameters and takes no "
                f"overrides: {sorted(overrides)}"
            )
        if render_mode is not None:
            raise ValueError("a Gymnasium stream does not render yet")
        plant = make_stream(name)
        return GymnasiumEnv(plant, plant.default_params, render_mode=render_mode)
    env, params = make(name, **overrides)
    return GymnasiumEnv(env, params, render_mode=render_mode)
