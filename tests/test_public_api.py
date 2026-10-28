"""What the package promises at its top level.

Written after the README's own quickstart failed to import. It reached for
``Airplane2D`` -- which is what ``type(env).__name__`` reports -- while the
package exported that class only as ``Plane``. Nothing caught it, because
nothing asserted anything about the public surface at all.

These are cheap, and each one pins a mistake that had actually been made.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

import target_gym
from target_gym.provenance import load_recorded_baselines
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


@pytest.mark.records
@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_default_params_are_the_scored_configuration(name):
    """``env.default_params`` is the configuration the baselines are recorded
    on (docs/api.md), so an environment run without parameters is the scored
    task. The oracle is the episode length each recorded row was measured
    over, and ``tg.make`` must hand out the same configuration."""
    env, params = target_gym.make(name)
    recorded_steps = load_recorded_baselines()[name]["steps"]

    assert env.default_params.max_steps_in_episode == recorded_steps
    assert env.default_params == params


@pytest.mark.parametrize("name", sorted(n for n, s in REGISTRY.items() if s.has_mpc))
def test_env_make_mpc_is_deprecated_on_every_environment(name):
    env = REGISTRY[name].make_env()
    with pytest.warns(DeprecationWarning, match="make_oracle"):
        env.make_mpc()


# A core install, simulated by refusing to import the optional extras.
_WITHOUT_EXTRAS = """
import json, sys

class NotInstalled:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("pygame", "moviepy", "casadi", "do_mpc"):
            raise ImportError(f"No module named {name!r}")

sys.meta_path.insert(0, NotInstalled())
import jax
import target_gym as tg
from target_gym import registry
from target_gym.experts.mpc import casadi_step_map

built = [tg.make(name)[0] for name in tg.list_envs()]
env, params = tg.make("plane")
_, state = env.reset(jax.random.PRNGKey(0), params)
errors = {}
for what, call in {
    "render": lambda: env.render(None, state, params, [], None),
    "step_map": lambda: casadi_step_map(lambda x, u: x, 1, 1, 1.0, "rk4_1"),
}.items():
    try:
        call()
    except ImportError as err:
        errors[what] = str(err)
oracles, oracle_errors = 0, {}
for name in tg.list_envs():
    if not registry.get(name).has_mpc:
        continue
    try:
        tg.make_oracle(name)
        oracles += 1
    except ImportError as err:
        oracle_errors[name] = str(err)
print(json.dumps({"built": len(built), "errors": errors,
                  "oracles": oracles, "oracle_errors": oracle_errors}))
"""


@pytest.mark.medium
def test_a_core_install_builds_every_task_and_names_the_missing_extra():
    """pygame and moviepy are the ``[render]`` extra and CasADi the ``[mpc]``
    one. Without them every task still builds, and drawing a frame or
    building a CasADi step map raises an ImportError naming the extra.

    ``make_oracle`` documents that nine tasks plan with CasADi: without it
    those nine raise an ImportError naming the extra, and every other oracle
    still builds."""
    out = subprocess.run(
        [sys.executable, "-c", _WITHOUT_EXTRAS],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])

    assert result["built"] == len(REGISTRY)
    assert "target-gym[render]" in result["errors"].get("render", "")
    assert "target-gym[mpc]" in result["errors"].get("step_map", "")
    with_oracle = sum(spec.has_mpc for spec in REGISTRY.values())
    assert len(result["oracle_errors"]) == 9
    assert result["oracles"] == with_oracle - 9
    for name, message in result["oracle_errors"].items():
        assert "target-gym[mpc]" in message, name
