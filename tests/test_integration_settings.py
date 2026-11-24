"""A plant has one integration setting, and says it the same way everywhere.

Contract (owner, 2026-11-24, "Tidy the setting"): the setting a plant's step
function takes when none is given is the setting the registered plant runs at.
A caller of ``compute_next_state`` that leaves it out then steps the plant the
benchmark scores, not a second one.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from target_gym import registry


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.name)
def test_the_step_functions_default_setting_is_the_plants(spec):
    env = spec.make_env()
    module = importlib.import_module(type(env).__module__)
    step = getattr(module, "compute_next_state", None)
    if step is None:
        pytest.skip("its module exposes no compute_next_state")
    setting = inspect.signature(step).parameters.get("integration_method")
    if setting is None:
        pytest.skip("its step function takes no integration setting")
    assert setting.default == env.integration_method
