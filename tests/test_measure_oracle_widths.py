"""``scripts/measure_oracle_widths.py``: K11's noise term perturbs, in each noisy
task's planner data, the deviation the env's noise drives, by one innovation
of that noise (docs/proposals/oracle-smoothing-protocol.md, K11)."""

from __future__ import annotations

import importlib
import importlib.util
import pathlib

import jax
import numpy as np
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


def test_a_hold_noise_task_is_noisy_and_has_a_measured_dead_time(widths):
    """K11h (Amendment 4, owner 2026-11-13) sizes the term over the dead time
    of the actuator left to correct: the task must have a noise to size and a
    valid horizon test of that actuator."""
    for name in widths.HOLD_NOISE:
        assert name in widths.NOISY
        assert widths.uncorrected_steps(name) >= 1


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


def test_the_ph_back_off_covers_the_base_valves_measured_dead_time(widths):
    """pH (owner, 2026-11-13): with the trim at its limit only the base valve
    corrects, so the plan cannot correct the buffer for the base's dead time.
    The horizon protocol's one valid base test (seed 108,
    evidence/oracle_horizons/ph_neutralization.json) fits theta = 303.959 s:
    60.79 steps of 5 s, 61 to cover it whole. theta 2e-3 /s and 5 s steps give
    a = 1 - theta dt = 0.99, the stationary std is 0.35 mL/s (env.py); over
    61 steps 0.35 sqrt(1 - 0.99^122) = 0.35 sqrt(1 - 0.293387) = 0.294204
    mL/s, against 0.155886 over T_c = 11 steps."""
    _, decay = widths.NOISY["ph_neutralization"]
    a, sigma = decay(REGISTRY["ph_neutralization"].make_test_params())
    assert (float(a), float(sigma)) == (pytest.approx(0.99), 0.35)
    j = widths.uncorrected_steps("ph_neutralization")
    assert j == 61
    assert widths.hold_std(a, sigma, j) == pytest.approx(0.294204, rel=1e-5)
    assert widths.HOLD_Z == 2.0


def test_the_ph_floor_probe_discharges_at_the_permit_floor(widths):
    """The charge balance in env.py's docstring, solved by Brent's method (not
    the env's bisection or the closed-form inverse that sets the probe):
    Wa + 10^(pH-14) - 10^-pH + Wb (1 + 2 10^(pH-pK2)) / (1 + 10^(pK1-pH)
    + 10^(pH-pK2)) = 0, at the probe's discharge invariants."""
    from scipy.optimize import brentq

    (state, _), *rest = widths.ph_floor_probe()
    p = REGISTRY["ph_neutralization"].make_test_params()
    Wa, Wb = float(state.Wa), float(state.Wb)

    def balance(pH):
        carbonate = (1 + 2 * 10 ** (pH - p.pK2)) / (
            1 + 10 ** (p.pK1 - pH) + 10 ** (pH - p.pK2)
        )
        return Wa + 10 ** (pH - 14) - 10**-pH + Wb * carbonate

    assert rest == []
    # float32 invariants near 1e-3 M: about 1e-7 relative, and dpH/dWa ~ 5.4
    # per mM there, so about 1e-6 pH.
    assert brentq(balance, 0.0, 14.0, xtol=1e-14) == pytest.approx(6.0, abs=1e-5)


def test_the_ph_floor_probe_is_a_steady_state_of_the_plan(widths):
    """At the floor the plan holds still: one step of the planner's model
    under the probe's own move leaves every tank's invariants where they are,
    since each holds the streams' flow-weighted mean."""
    import jax.numpy as jnp

    from target_gym.oracles.controller import enable_x64, float64
    from target_gym.pc_gym.ph_neutralization.oracle import ph_model

    spec = REGISTRY["ph_neutralization"]
    (state, u), *_ = widths.ph_floor_probe()
    with enable_x64():
        p, s = float64(spec.make_test_params()), float64(state)
        model = ph_model(spec.make_env(), p)
        data = {"task": model.data(s), "end": 10**9, "widths": jnp.zeros(0)}
        x = model.state_x(s)
        x1 = model.step(data, x, jnp.asarray(u, jnp.float64), 0)[0]
    # The buffer holds its nominal; the invariants in mM: float32 inputs, 1e-6.
    np.testing.assert_allclose(x1[:7], x[:7], rtol=1e-5, atol=1e-6)
