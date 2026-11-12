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
    "ph_neutralization": ("q2", 0.0),  # the buffer flow's OU, read raw at the solve
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


def test_a_hold_noise_task_is_noisy_and_has_a_hold(widths):
    """K11h (Amendment 4) sizes the term over T_c steps: the task must have a
    noise to size and a measured T_c."""
    from target_gym import records

    for name in widths.HOLD_NOISE:
        assert name in widths.NOISY
        assert records.load_row("oracle_horizons", name)["T_c_steps"] >= 1


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


@pytest.mark.parametrize(
    "a, sigma, j, want",
    [
        # One step is one innovation, sigma sqrt(1 - a^2): 2 sqrt(0.19).
        (0.9, 2.0, 1, 0.8717797887081347),
        # Two steps: s^2 (1 + a^2) = 0.75 x 1.25 = 0.9375.
        (0.5, 1.0, 2, 0.9682458365518543),
        # A white deviation (a = 0) is one innovation however long the hold.
        (0.0, 3.0, 7, 3.0),
        # A hold much longer than 1 / (1 - a) reaches the stationary std.
        (0.9, 2.0, 400, 2.0),
    ],
)
def test_the_hold_std_is_the_ar1_spread_over_the_hold(widths, a, sigma, j, want):
    assert widths.hold_std(a, sigma, j) == pytest.approx(want, rel=1e-12)


def test_the_ph_hold_back_off_uses_the_buffer_noise_over_eleven_steps(widths):
    """pH: theta 2e-3 /s and 5 s steps give a = 1 - theta dt = 0.99; the
    stationary std is 0.35 mL/s (env.py). Over T_c = 11 steps,
    0.35 sqrt(1 - 0.99^22) = 0.35 sqrt(1 - 0.801631) = 0.155886 mL/s, 3.16
    times one innovation, 0.35 sqrt(1 - 0.99^2) = 0.049373 mL/s."""
    from target_gym import records

    _, decay = widths.NOISY["ph_neutralization"]
    a, sigma = decay(REGISTRY["ph_neutralization"].make_test_params())
    tc = records.load_row("oracle_horizons", "ph_neutralization")["T_c_steps"]
    assert (float(a), float(sigma), tc) == (pytest.approx(0.99), 0.35, 11)
    assert widths.hold_std(a, sigma, tc) == pytest.approx(0.155886, rel=1e-5)
    assert widths.hold_std(a, sigma, 1) == pytest.approx(0.049373, rel=1e-4)
    assert widths.HOLD_Z == 2.0
