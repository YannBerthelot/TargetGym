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


def make(name: str, **overrides):
    """``(env, params)`` for task ``name``, like ``gymnax.make``.

    ``params`` is the scored configuration. Keyword ``overrides`` replace
    single fields of it, e.g. ``make("plane", max_steps_in_episode=500)``.

    The environment is shared between calls: environments carry no episode
    state (it all lives in the ``state`` passed through ``reset`` and
    ``step``), and a new instance would be compiled again by JAX.
    """
    spec = registry.get(name)
    return spec.make_env(), spec.make_test_params(**overrides)


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
    oracle = spec.make_mpc(spec.make_env(), mpc.plan_params(spec, params))
    oracle.reset()
    return oracle


def make_gymnasium(name: str, render_mode: str | None = None, **overrides):
    """Task ``name`` as a Gymnasium environment, on the scored configuration.

    For libraries that expect the classic stateful API. It steps on the host
    and returns NumPy, so it cannot sit inside ``jax.jit``; prefer
    :func:`make` for JAX training. ``overrides`` work as in :func:`make`.
    """
    from target_gym.wrapper import GymnasiumEnv

    env, params = make(name, **overrides)
    return GymnasiumEnv(env, params, render_mode=render_mode)
