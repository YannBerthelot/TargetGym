"""The recipe's rod bump is sized by a written rule (owner, 2026-11-27).

Contract, written before the code, from practice: a bump test must not carry the
process past an operating limit (Instrumentation Tools, "Process Control Loop
Testing", 2019), and a test that trips the plant reads its restart, not its
response (the recipe's own rule since the v9 sweep). So the recipe's rod bump,
ROD_BUMP steps in with boron held for an hour, is watched against Tavg's floor
(``T_avg_min``, the limit that runs a clock): if Tavg crosses it, the bump is
stopped there and repeated from the lined-out plant at half its size, at most
three times, and refused past that. A bump that stays above the floor runs as it
always did, so a member far from the floor reads the same numbers.

Hand values, from the curve book and the rods' worth curve (the integral worth of
a cosine flux): at half load at the start of the cycle the floor is 16.5 K under
Tavg. On the sweep's draw 5 (MTC -9.2 pcm/K, the weakest member drawn) 30 steps
are predicted to take 16.4 K, and the plant takes a little more, so it crosses;
15 steps take about 8 K, so one halving. On the shipped member 30 steps take
13.3 K and it never crosses.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

#: The family sweep's draw 5 of 2026-11-26 (SEED 20261110), the member whose rod
#: bump tripped the plant.
DRAW_5 = dict(
    alpha_coolant=-9.155275561073893e-05, alpha_coolant_end=-0.0005248489172516515,
    boron_delay=22, boron_worth=-6.77246192850115e-05,
    floor_allowed=1005.7541451094116, loading_stop=10.61052765152564,
    restart_steps=6721,
)  # fmt: skip
QUIET = dict(demand_sigma=0.0, sensor_noise_scale=0.0, cycle_range=(0.0, 0.0))


def _half_load(member):
    import reactor_ladder as L

    env = Reactor()
    q, s = L.points(env, ReactorParams(**member, **QUIET), 100)[1]
    return L, env, q, s


@pytest.mark.medium
def test_a_bump_that_would_cross_the_floor_is_repeated_at_half_its_size():
    L, env, q, s = _half_load(DRAW_5)
    tr, size = L.rod_bump(env, q, s)
    assert size == L.ROD_BUMP / 2.0
    assert float(np.min(L.tavg(tr))) >= q.T_avg_min


@pytest.mark.medium
@pytest.mark.parametrize("load", [0, 1])
def test_a_bump_that_stays_above_the_floor_runs_at_its_full_size(load):
    import reactor_ladder as L

    env = Reactor()
    q, s = L.points(env, ReactorParams(**QUIET), 100)[load]
    _, size = L.rod_bump(env, q, s)
    assert size == L.ROD_BUMP


@pytest.mark.large
def test_the_recipe_runs_on_the_weakest_member_drawn():
    import reactor_ladder as L

    """The contract here is that the recipe completes: before the rule its rod
    bump tripped this member, and the family sweep refused it."""
    tune, _ = L.bumps(Reactor(), ReactorParams(**DRAW_5, cycle_range=(0.0, 0.0)))
    assert np.isfinite(tune.wd_Kc) and tune.wd_Kc > 0.0
