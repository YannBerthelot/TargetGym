"""``scripts/measure_oracle_widths.py``: K11's noise term perturbs, in each noisy
task's planner data, the deviation the env's noise drives, by one innovation
of that noise (docs/proposals/oracle-smoothing-protocol.md, K11)."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

import jax
import pytest

from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The state field each env's step drives with its noise, and how the planner
#: data restates it (cited from each env's step).
NOISE_FIELD = {
    "reactor": ("load_wobble", 0.0),  # reactor/env.py, the load's OU wobble
    "compressor_surge": ("demand_dev", 0.0),  # the consumers' OU deviation
    "wind_turbine": ("v_turb", 0.0),  # the Kaimal turbulence about v_mean
    "cement_kiln": ("raw_meal", "raw_meal_nominal"),  # feed OU about nominal
    "hvac": ("weather_dev", 0.0),  # the weather's OU deviation
}


@pytest.fixture(scope="module")
def widths():
    spec = importlib.util.spec_from_file_location(
        "measure_oracle_widths_under_test",
        ROOT / "scripts" / "measure_oracle_widths.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_noisy_task_is_listed(widths):
    assert set(widths.NOISY) == set(NOISE_FIELD)


@pytest.mark.parametrize("name", sorted(NOISE_FIELD))
def test_the_noise_term_perturbs_the_deviation_the_env_drives(widths, name):
    where, decay = widths.NOISY[name]
    module, attr = where.split(":")
    spec = REGISTRY[name]
    env, p = spec.make_env(), spec.make_test_params()
    model = getattr(importlib.import_module(module), attr)(env, p)
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    field, base = NOISE_FIELD[name]
    base = getattr(p, base) if isinstance(base, str) else base
    marked = state.replace(**{field: base + 0.123})

    task = widths._with_deviation(name, model.data(marked), 1.0)

    i = widths.DEVIATION_AT.get(name, len(task) - 1)
    assert float(task[i]) == pytest.approx(1.123)
    a, sigma = decay(p)
    assert 0.0 < float(a) < 1.0 and float(sigma) > 0.0
