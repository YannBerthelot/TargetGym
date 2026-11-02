"""The aircraft tuning script flies the protocol's episodes.

Contract (``target_gym.benchmark``'s docstring, the library's one key
convention): seed ``s`` of ``runners.rollout``, ``eval.run_episode`` and the
recorded baselines is the one-episode run on ``PRNGKey(s)``, which resets on
its own key and gives every step a fresh key, as gymnax does.
``scripts/tune_aircraft_guidance.py`` says its ``hold`` measures the
protocol's hold window, so its ``rollout`` of seed ``s`` must be that same
episode: the same reset, and on a task with turbulence the same gusts.
"""

import importlib.util
import pathlib

import numpy as np

import target_gym as tg
from target_gym import eval as E
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _script():
    # Loaded inside the test: the script silences warnings when imported, and
    # pytest restores the filters after each test.
    spec = importlib.util.spec_from_file_location(
        "_tune_aircraft_guidance", ROOT / "scripts" / "tune_aircraft_guidance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_tuning_rollout_flies_the_protocols_episode():
    tune = _script()
    spec = REGISTRY["plane"]
    params = spec.make_test_params()
    assert params.turbulence_sigma > 0  # the gusts are part of what must match

    _, tracking = tune.rollout(
        "plane", tune.P2.plane_tecs_step, tune.P2.PlaneTECSParams(), [100]
    )
    episode = E.run_episode(spec, params, tg.make_expert("plane").host(), seed=100)

    # The same float32 plant and TECS law, traced in one run and on the host
    # in the other: only the order of operations differs. Tracking is
    # (altitude error / 1 m)^2 at 3500 to 7500 m, where float32 spaces the
    # altitude 2^-11 m (0.0005 m) apart; a few of those, dz, move the cost by
    # 2 |e| dz, a few thousandths near the target. Hence atol 1e-2 there, and
    # rtol 1e-3 once the cost passes about 10.
    np.testing.assert_allclose(
        tracking[0], episode.terms["tracking"], rtol=1e-3, atol=1e-2
    )
