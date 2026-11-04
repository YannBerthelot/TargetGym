"""
Tests for the runner module: generic rollouts and figure modes,
obs_value_index / obs_target_index / tracked_names attributes, and
MPC / make_pid / make_mpc env methods.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import (
    CSTR,
    CSTRParams,
    FirstOrderParams,
    FirstOrderSystem,
    FourTank,
    FourTankParams,
)
from target_gym.experts.mpc import GradientMPC, NewtonMPC
from target_gym.pc_gym.cstr.oracle import make_cstr_mpc
from target_gym.pc_gym.first_order.oracle import make_first_order_mpc
from target_gym.pc_gym.four_tank.oracle import make_four_tank_mpc
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D
from target_gym.plane.oracle import PlaneHandoverMPC, make_plane_mpc
from target_gym.registry import REGISTRY
from target_gym.runners import runners as R

N = 10  # minimal steps for speed


# ---------------------------------------------------------------------------
# Generic rollouts and figure modes
#
# These used to be four near-identical pairs of tests, one per per-environment
# runner module. The runner is now registry-driven, so the same coverage is a
# parametrisation -- and it extends to every environment rather than the four
# that happened to have a module.
# ---------------------------------------------------------------------------

RUNNER_ENVS = ["plane", "cstr", "first_order", "four_tank"]


@pytest.mark.parametrize("name", RUNNER_ENVS)
def test_rollout_with_pid_tracks_shapes_and_stays_finite(name):
    spec = REGISTRY[name]
    params = spec.make_test_params(max_steps_in_episode=N)
    values, targets, rewards = R.rollout(spec, params, R.pid_policy(spec), seed=0)

    n_tracked = len(R._as_tuple(spec.make_env().obs_value_index))
    assert values.shape == (N, n_tracked)
    assert targets.shape == (N, n_tracked)
    assert rewards.shape == (N,)
    assert np.all(np.isfinite(values)), f"{name}: non-finite tracked value"
    assert np.all(np.isfinite(rewards)), f"{name}: non-finite reward"


@pytest.mark.parametrize("name", RUNNER_ENVS)
def test_figure_modes_run_headless(name):
    """The figure modes must complete without a display or a writable figures/."""
    spec = REGISTRY[name]
    params = spec.make_test_params(max_steps_in_episode=N)
    assert R.figure_sweep(name, params=params, resolution=3, plot=False)
    assert R.figure_pid(name, params=params, n_seeds=2, plot=False)


@pytest.mark.medium
def test_importing_the_runners_keeps_the_callers_matplotlib_backend():
    """Only the command line picks a headless backend; a library import must
    leave the caller's own backend alone."""
    code = (
        "import matplotlib; import target_gym.runners.runners; "
        "print(matplotlib.get_backend())"
    )
    env = {**os.environ, "MPLBACKEND": "svg"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().lower() == "svg"


# ---------------------------------------------------------------------------
# obs_value_index / obs_target_index attributes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GradientMPC — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_cls,params_cls,make_fn,make_kwargs,action_dim",
    [
        (CSTR, CSTRParams, make_cstr_mpc, {}, 1),
        (FirstOrderSystem, FirstOrderParams, make_first_order_mpc, {"horizon": 3}, 1),
        (FourTank, FourTankParams, make_four_tank_mpc, {}, 2),
    ],
)
def test_mpc_step_returns_finite(env_cls, params_cls, make_fn, make_kwargs, action_dim):
    """MPC.step() returns a finite scalar (or array for MIMO)."""
    env = env_cls(integration_method="rk4_1")
    params = params_cls(max_steps_in_episode=N)
    mpc = make_fn(env, params, **make_kwargs)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = mpc.step(obs, state)

    if action_dim == 1:
        assert np.isfinite(float(action)), "MPC action is not finite"
    else:
        arr = np.array(action)
        assert arr.shape == (
            action_dim,
        ), f"Expected shape ({action_dim},), got {arr.shape}"
        assert np.all(np.isfinite(arr)), "MPC action contains non-finite values"


# Compiling an aircraft planner peaks at 2.5 to 4 GB and takes 25 to 95 s on a
# cold cache (measured 2026-10-28), more than a small test's share of a machine.
@pytest.mark.medium
def test_plane_mpc_step_returns_finite():
    """PlaneMPC.step() returns a finite [power, stick] array."""
    env = Airplane2D(integration_method="rk4_1")
    params = PlaneParams(max_steps_in_episode=N)
    mpc = make_plane_mpc(env, params, horizon=3, n_iter=3)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = mpc.step(obs, state)
    assert np.array(action).shape == (2,), "PlaneMPC action should have shape (2,)"
    assert np.all(np.isfinite(action)), "PlaneMPC action contains non-finite values"


@pytest.mark.medium  # an aircraft planner, as above
def test_mpc_reset_zeroes_actions():
    """GradientMPC.reset() zeros out the internal action sequence."""
    env = Airplane2D(integration_method="rk4_1")
    params = PlaneParams(max_steps_in_episode=N)
    mpc = make_plane_mpc(env, params, horizon=5, n_iter=3)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    mpc.step(obs, state)  # populates internal actions
    mpc.reset()
    assert jnp.all(mpc._actions == 0.0), "reset() did not zero out action sequence"


# ---------------------------------------------------------------------------
# make_pid / make_mpc env methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_cls,params_cls",
    [
        (CSTR, CSTRParams),
        (FirstOrderSystem, FirstOrderParams),
        (FourTank, FourTankParams),
    ],
)
def test_env_make_pid_returns_controller(env_cls, params_cls):
    """env.make_pid() still returns a usable PID controller, and warns."""
    env = env_cls()
    with pytest.warns(DeprecationWarning, match="make_expert"):
        pid = env.make_pid()
    assert pid is not None
    # A HostController: step(obs) -> action, plus reset()
    assert callable(getattr(pid, "step", None))


def test_plane_env_make_pid_returns_controller():
    env = Airplane2D()
    with pytest.warns(DeprecationWarning, match="make_expert"):
        pid = env.make_pid()
    assert pid is not None
    assert callable(getattr(pid, "step", None))


@pytest.mark.parametrize(
    "env_cls,params_cls,make_kwargs,expected_type",
    [
        (CSTR, CSTRParams, {}, NewtonMPC),
        (FirstOrderSystem, FirstOrderParams, {"horizon": 3}, NewtonMPC),
        (FourTank, FourTankParams, {}, NewtonMPC),
    ],
)
def test_env_make_mpc_returns_controller(
    env_cls, params_cls, make_kwargs, expected_type
):
    """env.make_mpc() returns the correct MPC controller type."""
    env = env_cls()
    params = params_cls(max_steps_in_episode=N)
    mpc = env.make_mpc(params=params, **make_kwargs)
    assert isinstance(mpc, expected_type)


def test_plane_env_make_mpc_returns_controller():
    env = Airplane2D()
    params = PlaneParams(max_steps_in_episode=N)
    mpc = env.make_mpc(params=params, horizon=3, n_iter=2)
    # The default pattern (plane's hold) flies two gradient planners in turn:
    # one for the capture of the altitude, one for the hold.
    assert isinstance(mpc, PlaneHandoverMPC)
    assert isinstance(mpc.capture, GradientMPC)
    assert isinstance(mpc.hold, GradientMPC)
    assert mpc.action_dim == 2
