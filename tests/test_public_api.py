"""What the package promises at its top level.

Written after the README's own quickstart failed to import. It reached for
``Airplane2D`` -- which is what ``type(env).__name__`` reports -- while the
package exported that class only as ``Plane``. Nothing caught it, because
nothing asserted anything about the public surface at all.

These are cheap, and each one pins a mistake that had actually been made.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

import target_gym
from target_gym.registry import REGISTRY


def test_everything_in_all_resolves():
    missing = [n for n in target_gym.__all__ if not hasattr(target_gym, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"


def test_no_public_name_is_missing_from_all():
    """A name without a leading underscore is a promise, listed or not."""
    listed = set(target_gym.__all__)
    leaked = [
        n
        for n in dir(target_gym)
        if not n.startswith("_")
        and n not in listed
        and not isinstance(getattr(target_gym, n), types.ModuleType)
    ]
    assert not leaked, (
        f"public attributes absent from __all__: {leaked}. Either add them, or "
        "import them under a private alias -- `from importlib.metadata import "
        "version` once put `target_gym.version` in the public namespace, where "
        "it shadowed the meaning of the name it borrowed."
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_environment_class_is_importable_under_its_own_name(name):
    """``type(env).__name__`` must be importable from the package root.

    The first thing anyone does with an unfamiliar object is print its type,
    and the second is import it. If those disagree the user gets an
    ImportError while reading the name the library told them. Friendly aliases
    are welcome; they just cannot be the *only* way in.
    """
    cls = type(REGISTRY[name].make_env()).__name__
    assert hasattr(target_gym, cls), (
        f"{name}: environments report themselves as {cls!r} but that name is "
        f"not importable from target_gym"
    )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_params_class_is_importable_under_its_own_name(name):
    cls = REGISTRY[name].params_cls().__class__.__name__
    assert hasattr(
        target_gym, cls
    ), f"{name}: parameters class {cls!r} is not importable from target_gym"


def test_version_is_a_string():
    """``__version__``, not ``version`` -- and a string, not the function that
    produced it."""
    assert isinstance(target_gym.__version__, str)
    assert target_gym.__version__


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_default_params_are_the_scored_configuration(name):
    """``env.default_params`` is the configuration the task is scored on, so
    an environment reset without parameters runs the benchmark task."""
    spec = REGISTRY[name]
    assert spec.make_env().default_params == spec.make_test_params()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_shared_env_methods_come_from_the_mixins(name):
    """``save_video`` and ``make_mpc`` live once, in unhashed mixins, so no
    environment package carries its own copy."""
    from target_gym.expert_access import RegistryExperts
    from target_gym.video import SavesVideo

    cls = type(REGISTRY[name].make_env())
    assert cls.save_video is SavesVideo.save_video
    assert cls.make_mpc is RegistryExperts.make_mpc


def test_env_make_mpc_is_deprecated():
    env = REGISTRY["battery"].make_env()
    with pytest.warns(DeprecationWarning, match="make_oracle"):
        env.make_mpc()


def test_building_every_environment_needs_no_render_extra():
    """pygame and moviepy are the ``[render]`` extra: importing the package
    and building every environment must not import them."""
    code = (
        "import sys\n"
        "from target_gym.registry import all_specs\n"
        "[s.make_env() for s in all_specs()]\n"
        "print(sorted(m for m in ('pygame', 'moviepy') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
