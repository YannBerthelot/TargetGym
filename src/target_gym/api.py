"""The front door: every task by its registry name.

``env, params = target_gym.make("plane")`` mirrors ``gymnax.make``, and
``params`` is the scored configuration, the one the shipped baselines and the
protocol are measured on. Everything here resolves through the registry.

Kept out of ``base.py`` and the environment packages, which are hashed into
every provenance fingerprint: how a task is looked up is not part of what it is.
"""

from __future__ import annotations

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


def make_expert(name: str):
    """The shipped expert for ``name``, as a pure controller.

    Pick its form where you use it, so the two kinds of memory are never
    confused:

    - ``.host()`` keeps its own memory, for a Python loop on NumPy
      observations: ``pid.reset()`` at each episode start, then
      ``action = pid(obs)``. This is the form the baselines are scored with.
    - ``.batched()`` is for ``jit``, ``vmap`` and GPU, and the caller carries
      the memory: ``state = policy.init_state(n_envs)``, then
      ``actions, state = policy(state, obs)``.
    - ``.act(state, obs)`` is one traced step for a single environment,
      starting from ``.init_state``.

    The expert reads the observation only, as a deployed controller does.
    """
    return registry.get(name).make_pid_controller()


def make_oracle(name: str, params=None):
    """The full-state MPC for ``name``, built as the recorded baselines build it.

    ``action = oracle.step(obs, state)``. It reads the true simulator state,
    including what the observation withholds, so it bounds what a perfect
    model could do rather than competing with a policy. ``params`` defaults to
    the scored configuration; the planner's copy has its process noise zeroed,
    so it plans on the mean disturbance. Returned reset, ready for an episode.

    Nine tasks plan with CasADi: install ``target-gym[mpc]`` for those.
    """
    from target_gym.experts import mpc

    spec = registry.get(name)
    if not spec.has_mpc:
        raise NotImplementedError(spec.baselines_note or f"{name} ships no oracle")
    params = spec.make_test_params() if params is None else params
    try:
        oracle = spec.make_mpc(spec.make_env(), mpc.plan_params(spec, params))
    except (ImportError, NameError) as err:
        # Without CasADi some planners fail on the missing module name before
        # reaching their own import check; either way the cure is the extra.
        if mpc._CASADI_AVAILABLE:
            raise
        raise ImportError(
            f"the {name} oracle needs CasADi and do-mpc: "
            "pip install 'target-gym[mpc]'"
        ) from err
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
