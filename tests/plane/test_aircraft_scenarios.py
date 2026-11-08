"""The kept aircraft tasks' scenario suites (charter Amendment 3, rule 20).

Contract: each kept task (``plane``, ``plane3d_heading``, ``plane3d_racetrack``)
has 4 to 10 scenarios, each naming the source that shows it (plane/REALITY.md),
with a calm and a normal day among them; a wind scenario starts established in
that wind, flying the assigned airspeed while its ground speed differs by the
wind [rule 17; IFH ch. 10].
"""

import jax
import numpy as np
import pytest

from target_gym import registry, scenarios

KEPT = ("plane", "plane3d_heading", "plane3d_racetrack")


@pytest.mark.parametrize("name", KEPT)
def test_each_kept_task_has_a_sourced_suite(name):
    suite = scenarios.suite(registry.get(name))
    assert 4 <= len(suite) <= 10
    assert {"normal", "calm"} <= {s.name for s in suite}
    assert all(s.source for s in suite)


def test_a_hold_in_wind_starts_established_in_that_wind():
    spec = registry.get("plane3d_racetrack")
    (strong,) = [s for s in scenarios.suite(spec) if s.name == "low_hold_wind"]
    params = scenarios.params_for(spec, strong, sensor_noise_scale=0.0)
    env = spec.make_env()
    obs, state = env.reset_env(jax.random.PRNGKey(2), params)
    air = float(obs[env.obs_index["airspeed"]])
    assigned = float(obs[env.obs_index["target_speed"]])
    ground = float(np.hypot(state.x_dot, state.y_dot))
    heading = float(state.target_heading)
    assert air == pytest.approx(assigned, abs=0.05)
    assert ground == pytest.approx(
        np.sqrt(
            (assigned * np.cos(heading) + params.wind_x) ** 2
            + (assigned * np.sin(heading) + params.wind_y) ** 2
        ),
        rel=1e-3,
    )
