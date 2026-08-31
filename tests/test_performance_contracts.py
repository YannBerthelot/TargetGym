"""Performance properties that are easy to lose by copying an existing file.

Nothing here times anything -- a timing assertion on shared CI is a flaky test.
These check for shapes of code whose cost is structural and was measured once.
"""

from __future__ import annotations

import importlib
import pathlib
import re

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
