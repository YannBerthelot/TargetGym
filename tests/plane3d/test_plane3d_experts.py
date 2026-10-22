"""The 3D aircraft oracle: a gradient MPC whose step decays over each solve.

At a fixed normalised step of 0.05 the planner could not place an action
closer than about that step to the optimum, which left steady altitude and
path offsets. Since the oracle audit (2026-10) the step decays from 0.05 to
0.002 over each solve, with 200 iterations on all four tasks (the racetrack
had 50 until it was measured at 200). The settings live in the package's
``experts.py``, which is in the four tasks' baseline fingerprint, so these
tests pin the registry's oracles to them. The floors were reset to the
instrument resolutions with it (the -v3 tasks), and the NEA reference follows
the oracle's recorded holds.
"""

import json
import pathlib

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.plane3d.experts import make_plane3d_mpc, task_settings

TASKS = ["plane3d_heading", "plane3d_circle", "plane3d_racetrack", "plane3d_figure8"]
N_ITER = {
    "plane3d_heading": 200,
    "plane3d_circle": 200,
    "plane3d_racetrack": 200,
    "plane3d_figure8": 200,
}
HOLDS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "target_gym"
    / "data"
    / "hold_measurements.json"
)


def _oracle(name, **kwargs):
    spec = registry.get(name)
    env = spec.make_env()
    params = spec.make_test_params()
    return spec.make_mpc(env, plan_params(spec, params), **kwargs)


@pytest.mark.parametrize("name", TASKS)
def test_the_registry_builds_each_task_with_its_settings(name):
    oracle = _oracle(name)
    assert oracle.horizon == 30
    assert oracle.n_iter == N_ITER[name]
    assert oracle.lr == pytest.approx(0.05)
    assert oracle.lr_end == pytest.approx(0.002)


def test_explicit_settings_override_the_table():
    spec = registry.get("plane3d_circle")
    env = spec.make_env()
    oracle = make_plane3d_mpc(env, spec.make_test_params(), n_iter=3, lr_end=None)
    assert oracle.n_iter == 3
    assert oracle.lr_end is None


def test_an_unknown_task_keeps_the_fixed_step():
    assert task_settings(object()) == {"n_iter": 50, "lr_end": None}


@pytest.mark.parametrize("name", TASKS)
def test_the_floors_are_the_instrument_resolutions(name):
    """Every floor is the resolution of the instrument that reads it, and the
    failure cost is twice the envelope's cost in those units (the altitude
    envelope's, or the path's on the figure-8, which scores no altitude)."""
    spec = registry.get(name)
    p = spec.make_test_params()
    assert spec.version == 3
    if name == "plane3d_figure8":
        assert p.e_floor_path == p.position_precision_floor == 3.0
        assert p.failure_cost == pytest.approx(2.0 * (20000.0 / 3.0) ** 2)
        return
    assert p.e_floor_altitude == p.precision_floor == 1.0
    assert p.failure_cost == pytest.approx(2.0 * (12192.0 / 1.0) ** 2)
    if name == "plane3d_heading":
        assert p.e_floor_heading == p.heading_precision_floor
    else:
        assert p.e_floor_path == p.position_precision_floor == 3.0


@pytest.mark.parametrize("name", TASKS)
def test_the_nea_reference_is_the_recorded_hold(name):
    """rho_floor is the oracle's lowest per-seed hold in floor units, summed
    over the tracked outputs (``scripts/measure_hold.py``)."""
    p = registry.get(name).make_test_params()
    holds = json.loads(HOLDS.read_text())[name]["mpc"]["e_hold_min"]
    floors = {
        "plane3d_heading": [p.e_floor_altitude, p.e_floor_heading],
        "plane3d_circle": [p.e_floor_altitude, p.e_floor_path],
        "plane3d_racetrack": [p.e_floor_altitude, p.e_floor_path],
        "plane3d_figure8": [p.e_floor_path],
    }[name]
    rho = sum((h / f) ** 2 for h, f in zip(holds, floors))
    assert p.rho_floor == p.rho_floor_tracking
    assert p.rho_floor == pytest.approx(rho, rel=0.02)


@pytest.mark.slow
def test_the_oracle_settles_on_the_heading_task():
    """On protocol seed 1 the shipped oracle is 37 m off its altitude on
    average over the first 20 steps and holds 0.74 m over steps 40 to 59; in
    the oracle audit (2026-10, under the -v2 floors) the fixed step held 2.9 m
    there."""
    spec = registry.get("plane3d_heading")
    env = spec.make_env()
    params = spec.make_test_params()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    oracle.reset()
    key = jax.random.PRNGKey(1)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    errors = []
    for _ in range(60):
        action = np.asarray(oracle.step(obs, state))
        obs, state, _, _, info = step(key, state, action, params)
        assert not bool(info["tripped"])
        errors.append(abs(float(state.target_altitude - state.z)))
    assert np.mean(errors[40:]) < 1.5
