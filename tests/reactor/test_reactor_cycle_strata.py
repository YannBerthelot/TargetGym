"""The scoring seeds' cores spread evenly over the fuel cycle (owner, 2026-11-27,
picker, from a practitioner review's release plan: stratify the core's cycle point
across the scoring seeds, as compressor_surge stratifies its machine).

Contract:

- A reset draws the core's point in its cycle evenly over its range: the first
  90 % of the cycle less what the run burns (``test_reactor_cycle``). With
  ``cycle_stratum`` k in 0 to 9 it draws inside tenth k of that range.
- The protocol sets the stratum from the seed (``Reactor.params_for_seed``, read
  through ``benchmark.seed_params`` wherever a protocol run resets a seed): seed s
  draws in tenth s mod 10. So the scoring seeds 100 to 109 hold one core in each
  tenth, as do any ten consecutive seeds. Today they fall in tenths 3, 2, 0, 3, 4,
  7, 3, 5, 9 and 1, leaving two tenths empty.
- A scoring run's stream gives its plant the seed's stratum too, so the reactor's
  scored streams are spread as its episodes are.
- A reset with no stratum, a user's or a training run's, draws as before.

By hand, a 7-day run at 3.3 ppm a day burns 23.1 ppm, 23.1 / 1690 of the cycle,
so its range is [0, 0.9 - 0.013669] = [0, 0.886331] and a tenth is 0.0886331.
"""

import jax
import numpy as np
import pytest

from target_gym import scoring_runs
from target_gym.benchmark import seed_params
from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import cycle_point

TOP = 0.9 - 7 * 3.3 / 1690.0
WEEK = dict(burnup_range=(3.3, 3.3), max_steps_in_episode=7 * 2880)


def _tenth(point, top=TOP):
    return int(np.floor(10.0 * point / top))


def _point(env, params, seed):
    _, s = env.reset_env(jax.random.PRNGKey(seed), seed_params(env, params, seed))
    return float(cycle_point(s, params))


@pytest.mark.parametrize("first", [scoring_runs.seeds("reactor")[0], 37])
def test_ten_consecutive_seeds_hold_one_core_in_each_tenth_of_the_cycle(first):
    env, p = Reactor(), ReactorParams(**WEEK)
    points = [_point(env, p, s) for s in range(first, first + 10)]
    assert all(0.0 <= x <= TOP + 1e-6 for x in points)
    assert sorted(_tenth(x) for x in points) == list(range(10))
    assert [_tenth(x) for x in points] == [s % 10 for s in range(first, first + 10)]


def test_the_scoring_seeds_are_the_protocols_first_ten():
    assert scoring_runs.seeds("reactor") == list(range(100, 110))


def test_a_scoring_streams_seed_draws_its_plants_tenth():
    """The seed reaches the plant through the stream (``scoring_runs.stream``,
    what the reactor's lines run on), as it does through an episode."""
    stream, params = scoring_runs.stream("reactor", "baseload", steps=7 * 2880)
    params = params.replace(burnup_range=(3.3, 3.3))
    for seed in (103, 108):
        _, s = stream.reset_env(
            jax.random.PRNGKey(seed), seed_params(stream, params, seed)
        )
        assert _tenth(float(cycle_point(s.env, params))) == seed % 10


# Characterization (TESTING.md section 7): the cycle points a reset with no
# stratum drew on int/amendment3 at dceb2537, default params. Pins current
# behaviour, not a spec: the stratum must leave a user's or a training run's
# draw where it was.
UNSTRATIFIED = {0: 0.8081501722335815, 1: 0.2736378610134125, 104: 0.4203818738460541}


@pytest.mark.parametrize("seed", sorted(UNSTRATIFIED))
def test_characterization_a_reset_with_no_stratum_draws_as_before(seed):
    env, p = Reactor(), ReactorParams()
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    assert float(cycle_point(s, p)) == UNSTRATIFIED[seed]
