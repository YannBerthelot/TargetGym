"""The CSTR's oracle on the shared solver: its execution.

Contract (oracle-horizon-protocol.md, execution; ``oracles.controller``):
one solve every T_c (one step on the CSTR, ``data/oracle_horizons/cstr.json``),
every move in the box, and after ``reset`` the next move is a fresh
oracle's. Its planner problem and certificate are tested with the other
process oracles (tests/oracles/test_process_oracles.py).
"""

import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.cstr.oracle import make_cstr_mpc
from target_gym.registry import REGISTRY

SPEC = REGISTRY["cstr"]


@pytest.mark.medium  # compiles the solver, about 10 s
def test_a_short_closed_loop_solves_every_step_and_resets_cold():
    """Eight steps at T_c = 1, eight solves; after ``reset`` the next action
    is a fresh oracle's."""
    env = SPEC.make_env()
    p = SPEC.make_test_params(max_steps_in_episode=8)
    mpc = make_cstr_mpc(env, p)
    assert mpc.tc == 1
    _, start = env.reset_env(jax.random.PRNGKey(0), p)
    state = start
    step = jax.jit(env.step_env)
    for _ in range(8):
        u = mpc.step(None, state)
        assert -1.0 <= u <= 1.0
        _, state, *_ = step(jax.random.PRNGKey(0), state, jnp.asarray([u]), p)
    assert mpc.solver_report()["solver_calls"] == 8
    mpc.reset()
    fresh = make_cstr_mpc(env, p).step(None, start)
    assert mpc.step(None, start) == pytest.approx(fresh, abs=1e-6)
