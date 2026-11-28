"""The scored suite's defaults are their heaviest scenarios (owner, 2026-11-27, the
RL research review's H1, decision "Heaviest scenario + scenario=").

Contract:

- ``tg.make(name)`` hands out ``spec.make_test_params()``, the environment's
  ``default_params``. For each of the five scored tasks it is the configuration of
  its heaviest-weight scenario: ``normal`` for hvac, cstr, ph_neutralization and
  compressor_surge, ``baseload`` for the reactor (the decision names each).
- Only the defaults moved. Every scenario resolves to exactly the parameters it
  resolved to before, both as an episode (``scenarios.params_for``) and as a
  regime of the task's stream (``StreamEnv.scenario_params``): pinned below.
"""

import hashlib

import numpy as np
import pytest

from target_gym import scenarios
from target_gym.registry import REGISTRY
from target_gym.stream import StreamEnv

#: The decision's heaviest scenario of each scored task.
HEAVIEST = {
    "hvac": "normal",
    "cstr": "normal",
    "ph_neutralization": "normal",
    "compressor_surge": "normal",
    "reactor": "baseload",
}


def _scenario(spec, name):
    return {s.name: s for s in scenarios.suite(spec)}[name]


@pytest.mark.parametrize("task", sorted(HEAVIEST))
def test_the_named_scenario_is_the_heaviest(task):
    weights = scenarios.weights(REGISTRY[task])
    assert max(weights, key=weights.get) == HEAVIEST[task]


@pytest.mark.parametrize("task", sorted(HEAVIEST))
def test_the_default_is_the_heaviest_scenario(task):
    spec = REGISTRY[task]
    heaviest = scenarios.params_for(spec, _scenario(spec, HEAVIEST[task]))
    assert spec.make_test_params() == heaviest
    assert spec.make_env().default_params == heaviest


def _digest(params) -> str:
    """Every field of ``params`` by name and value. Floats are read as float32, the
    precision the plants run in, so a stream's stacked fields (JAX arrays, float64
    where a test turned x64 on) read as the same numbers as an episode's floats."""

    def plain(v):
        a = np.asarray(v)
        return (a.astype(np.float32) if a.dtype.kind == "f" else a).tolist()

    rows = [f"{k}={plain(v)!r}" for k, v in sorted(vars(params).items())]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


# Characterization (TESTING.md section 7): pins the parameters each scenario
# resolved to on int/amendment3 at dceb2537, before the defaults moved, so the
# move can be shown to have changed no scenario. Not verified against a spec: a
# deliberate change to a scenario re-pins its row here, with a note. hvac is left
# out: its default did not move, and the same change removed its T_out_design
# (review H2), a field that priced the stored heat and nothing else.
PINNED = {
    # task: {scenario: (episode params, stream regime params)}
    "cstr": {
        "normal": ("fa9bef453dc0b6e7", "9bb9d70d5d08c64a"),
        "setpoint_move": ("c03d0bb05cc85b99", "33af14dd3e99a157"),
        "catalyst_loss": ("b60d98e0e0b938a3", "d14c3a12e0bc5dae"),
        "feed_buildup": ("fc427daa4327129f", "3a6e57c611c211ba"),
        "cooling_loss": ("228fc3689e999b7a", "384a121be391bbd6"),
    },
    # Re-pinned 2026-11-27: failure_cost rose from 9 to 13 L a step in every pH
    # scenario (owner, the trip-cost bound on the normal day). With it at 9 the
    # rows hashed as at dceb2537 (normal: 3cd97449cacdb87a), so the defaults' move
    # itself changed no pH scenario.
    "ph_neutralization": {
        "normal": ("4e538f469a48f1ea", "64208a761c3e5741"),
        "bad_day": ("9ecffe7f547fcd17", "8b8fc309455ea3a7"),
        "buffer_loss": ("98c4cb57f4771a1a", "c58d10ff758de030"),
    },
    "compressor_surge": {
        "normal": ("d66dc620b4b69509", "d66dc620b4b69509"),
        "slow_approach": ("c6076fb46d24096b", "c6076fb46d24096b"),
        "fast_approach": ("e72586c986ea62a3", "e72586c986ea62a3"),
    },
    # Re-pinned 2026-11-28 (the owner's picks of 2026-11-28): rod_step_price 1 to 0
    # and Tavg's lost output at spot (imbalance_multiple, rho_floor and
    # rho_floor_tracking at 1 x in place of 1.25 x spot) in every reactor scenario.
    # With those four at their old values the rows hash as before (baseload:
    # c916a62102471145, 63508e0d4fdc3dca).
    "reactor": {
        "edf_daily": ("6a226206a25b055c", "8ab7c7a7703c7224"),
        "edf_short_hold": ("d49a2f211541824f", "ecdf493618727c7a"),
        "krsko_slow": ("da88fbb263d19f64", "25fbc5bc9645682d"),
        "krsko_shallow": ("5eced93bc50a7787", "5cb049755f912641"),
        "fast_return": ("8d9c2fd7e11cf221", "37338561f3d7e673"),
        # In a stream a load-follow day is an event, so baseload's regime has the
        # params of edf_daily's: they differ by their rate of days alone.
        "baseload": ("05183a66127c1f78", "8ab7c7a7703c7224"),
    },
}


@pytest.mark.parametrize(
    "task, scenario",
    [(t, s) for t, rows in sorted(PINNED.items()) for s in sorted(rows)],
)
def test_characterization_every_scenario_resolves_as_before_the_defaults_moved(
    task, scenario
):
    spec = REGISTRY[task]
    suite = scenarios.suite(spec)
    names = [s.name for s in suite]
    episode = scenarios.params_for(spec, suite[names.index(scenario)])
    base = spec.make_test_params()
    regime = StreamEnv(spec.make_env(), base, suite).scenario_params(
        base, names.index(scenario)
    )
    assert (_digest(episode), _digest(regime)) == PINNED[task][scenario]


@pytest.mark.parametrize("task", sorted(PINNED))
def test_every_scenario_is_pinned(task):
    assert sorted(PINNED[task]) == sorted(
        s.name for s in scenarios.suite(REGISTRY[task])
    )
