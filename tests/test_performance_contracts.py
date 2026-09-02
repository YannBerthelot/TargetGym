"""Performance properties that are easy to lose by copying an existing file.

Nothing here times anything -- a timing assertion on shared CI is a flaky test.
These check for shapes of code whose cost is structural and was measured once.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import jax
import pytest

from target_gym.registry import REGISTRY

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "target_gym"

# ``@partial(jax.jit, static_argnames=["params"])`` on a helper that takes an
# environment's params dataclass. Matches the decorator, not prose about it.
STATIC_PARAMS_JIT = re.compile(
    r"^@partial\(\s*jax\.jit\s*,\s*static_argnames=\[\s*[\"']params[\"']\s*\]\s*\)",
    re.MULTILINE,
)


def test_no_helper_is_jitted_on_a_static_params_object():
    """Keying a compilation cache on the params dataclass recompiles per instance.

    ``static_argnames=["params"]`` makes the params object part of the cache
    key. The dataclasses are frozen and hashable, so this is legal and silent --
    and every ``Params(...)`` a sweep, a tuner or an MPC constructs is then a
    cache miss and a full recompile. Measured at roughly 1600x the cost of a
    cached call before it was removed.

    It had spread to twelve files by copying, which is why this is a test
    rather than a comment: the next environment will be written by copying one
    of these too.
    """
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if STATIC_PARAMS_JIT.search(path.read_text())
    ]
    assert not offenders, (
        "jit keyed on the params object recompiles for every distinct params "
        f"instance; found in: {offenders}. Let the caller jit step_env instead."
    )


@pytest.mark.parametrize("name", ["cstr", "hvac", "reactor", "boiler_drum"])
def test_get_obs_is_not_a_compiled_wrapper(name):
    """The same property, checked on the object rather than by timing.

    A ``jax.jit``-wrapped function carries a compilation cache and exposes
    ``_cache_size``; a plain function does not. Asserting on that is
    deterministic, unlike a wall-clock threshold on a shared runner.
    """
    module = importlib.import_module(type(REGISTRY[name].make_env()).__module__)
    env_module = importlib.import_module(module.__name__.replace(".env_jax", ".env"))
    get_obs = getattr(env_module, "get_obs", None)

    assert get_obs is not None, f"{name}: no module-level get_obs to check"
    assert not hasattr(get_obs, "_cache_size"), (
        f"{name}: get_obs is jit-wrapped again. If its params argument is "
        "static, every fresh Params(...) recompiles; if it is traced, the "
        "wrapper only adds dispatch overhead to a function the caller's own "
        "jit would inline anyway."
    )


@pytest.mark.parametrize("name", list(REGISTRY))
def test_reset_returns_strongly_typed_state(name):
    """A reset state must have the dtypes a stepped state has.

    A state built from Python floats is *weakly typed* -- JAX shows it as
    ``~float32[]``. One ``step_env`` promotes it, so a freshly reset state and a
    stepped state are different abstract values and anything jitted over the
    state compiles twice, once for each. All eighteen environments did this, 187
    leaves in total; it cost the gradient MPC a second full compile of its
    optimiser (10.35 s, then another 10.24 s, against 9.65 s and nothing after).

    ``base.canonical_reset`` is what fixes it. This asserts the decorator has not
    been dropped from a new environment, which is easy to do and silent.
    """
    spec = REGISTRY[name]
    env = spec.make_env()
    _, state = env.reset_env(jax.random.PRNGKey(0), spec.params_cls())
    weak = [
        leaf
        for leaf in jax.tree_util.tree_leaves(state)
        if getattr(jax.api_util.shaped_abstractify(leaf), "weak_type", False)
    ]
    assert not weak, (
        f"{name}: {len(weak)} of {len(jax.tree_util.tree_leaves(state))} reset "
        f"state leaves are weakly typed. Decorate reset_env with "
        f"@canonical_reset from target_gym.base, or anything jitted over this "
        f"environment's state will compile once for the reset state and again "
        f"for every state after it."
    )


@pytest.mark.parametrize("name", list(REGISTRY))
def test_make_env_shares_one_instance(name):
    """Building an environment twice must not produce two objects.

    ``step_env`` is a bound method, so a new instance is a new callable to JAX,
    which keys its compilation cache on the callable and keeps the executable for
    the life of the process. Forty fresh instances of the 2D aircraft retained
    104 MB -- about 2.6 MB each -- and re-paid the compile every time. Re-wrapping
    the *same* instance in ``jax.jit`` costs nothing, so the instance is the thing
    that has to be shared.

    Safe only while these environments stay free of per-episode state: every
    attribute is configuration, and all real state travels in the ``EnvState``.
    An environment that starts keeping state across steps breaks this and must
    fix the sharing rather than the test.
    """
    spec = REGISTRY[name]
    assert spec.make_env() is spec.make_env(), (
        f"{name}: make_env returned two different objects. Each one costs JAX a "
        f"retained compiled executable, so building environments in a loop leaks."
    )
