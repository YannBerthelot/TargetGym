"""The four-tank's oracle on the shared solver: its execution.

Contract (oracle-horizon-protocol.md, execution; ``oracles.controller``):
one solve every T_c, Cooper's sample time as measured in
``data/oracle_horizons/four_tank.json`` (4 steps until the 2026-11-06 rerun,
3 since), the move held in between, every move in the box. T_c is a measured
record, not a constant of the task, so the test reads it. Its planner problem and certificate are tested with the
other process oracles (tests/oracles/test_process_oracles.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import records
from target_gym.pc_gym.four_tank.oracle import make_four_tank_mpc
from target_gym.registry import REGISTRY

SPEC = REGISTRY["four_tank"]


@pytest.mark.medium  # compiles the solver
def test_a_short_closed_loop_solves_once_per_sample_and_holds_the_move():
    """Thirty steps: a solve at steps 0, T_c, 2 T_c, ... (ten at T_c = 3,
    eight at 4), and the same move over each sample."""
    tc = int(records.load_row("oracle_horizons", "four_tank")["T_c_steps"])
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=30)
    mpc = make_four_tank_mpc(env, p)
    assert mpc.tc == tc
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    moves = []
    for _ in range(30):
        u = np.asarray(mpc.step(None, state))
        assert u.shape == (2,) and np.all(np.abs(u) <= 1.0)
        moves.append(u)
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.asarray(u), p)
    assert mpc.solver_report()["solver_calls"] == len(range(0, 30, tc))
    for t in range(30):
        np.testing.assert_array_equal(moves[t], moves[t - t % tc])
