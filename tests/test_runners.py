"""
Tests for the runner module: generic rollouts and figure modes, the
save_comparison_gif utility, obs_value_index / obs_target_index /
tracked_names attributes, and MPC / make_pid / make_mpc env methods.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import interp1d

from target_gym import (
    CSTR,
    CSTRParams,
    FirstOrderParams,
    FirstOrderSystem,
    FourTank,
    FourTankParams,
)
from target_gym.experts.mpc import (
    CasadiMPC,
    GradientMPC,
    make_cstr_mpc,
    make_first_order_mpc,
    make_four_tank_mpc,
    make_plane_mpc,
)
from target_gym.plane.env import PlaneParams
from target_gym.plane.env_jax import Airplane2D
from target_gym.registry import REGISTRY
from target_gym.runners import runners as R
from target_gym.runners.utils import run_constant_policy_final_value
from target_gym.utils import (
    load_or_build_interpolator,
    load_or_run_mpc_episode,
    save_comparison_gif,
)

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
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": N})
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
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": N})
    assert R.figure_sweep(name, params=params, resolution=3, plot=False)
    assert R.figure_pid(name, params=params, n_seeds=2, plot=False)


def test_every_registered_environment_exposes_a_tracked_label():
    """tracked_names is what labels a generated figure's axes.

    The runner covers the whole registry, so a new environment that forgets
    this ships plots labelled with an index instead of a quantity.
    """
    missing = [
        name
        for name, spec in REGISTRY.items()
        if not getattr(spec.make_env(), "tracked_names", None)
    ]
    assert not missing, f"environments without tracked_names: {missing}"


def test_tracked_names_matches_the_number_of_tracked_channels():
    for name, spec in REGISTRY.items():
        env = spec.make_env()
        n = len(R._as_tuple(env.obs_value_index))
        assert len(env.tracked_names) == n, (
            f"{name}: {len(env.tracked_names)} tracked_names for {n} tracked "
            "observation slots"
        )


# ---------------------------------------------------------------------------
# Comparison-GIF interpolator building
# ---------------------------------------------------------------------------


def test_plane_comparison_interpolator_valid_range(tmp_path):
    """The interpolator built from a stick sweep maps altitudes -> stick in [-1,1]."""
    env = Airplane2D(integration_method="rk4_1")
    params = PlaneParams(max_steps_in_episode=N)
    stick_levels = jnp.linspace(-1.0, 1.0, 8)

    final_alts = np.array(
        jax.vmap(
            lambda s: run_constant_policy_final_value(
                env, params, action=(0.5, s), state_attr="z", steps=N
            )
        )(stick_levels)
    )
    s_np = np.array(stick_levels)
    sort_idx = np.argsort(final_alts)
    cache = str(tmp_path / "plane_interp.pkl")
    interpolator = load_or_build_interpolator(
        cache,
        lambda: interp1d(
            final_alts[sort_idx],
            s_np[sort_idx],
            bounds_error=False,
            fill_value="extrapolate",
        ),
    )

    mid_alt = float(np.median(final_alts[np.isfinite(final_alts)]))
    best_stick = float(np.clip(interpolator(mid_alt), -1.0, 1.0))
    assert -1.0 <= best_stick <= 1.0


# ---------------------------------------------------------------------------
# obs_value_index / obs_target_index attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_cls,expected_value,expected_target",
    [
        (Airplane2D, 1, 6),
        (CSTR, 0, 2),
        (FirstOrderSystem, 0, 1),
    ],
)
def test_obs_indices_single(env_cls, expected_value, expected_target):
    env = env_cls()
    assert (
        env.obs_value_index == expected_value
    ), f"{env_cls.__name__}.obs_value_index should be {expected_value}"
    assert (
        env.obs_target_index == expected_target
    ), f"{env_cls.__name__}.obs_target_index should be {expected_target}"


def test_obs_indices_four_tank():
    """FourTank has two controlled variables; indices are tuples."""
    env = FourTank()
    assert env.obs_value_index == (0, 1), "FourTank obs_value_index should be (0, 1)"
    assert env.obs_target_index == (4, 5), "FourTank obs_target_index should be (4, 5)"


def test_obs_indices_accessible_on_class():
    """Attributes must be accessible on the class itself, not just instances."""
    assert Airplane2D.obs_value_index == 1
    assert Airplane2D.obs_target_index == 6
    assert FourTank.obs_value_index == (0, 1)
    assert FourTank.obs_target_index == (4, 5)


@pytest.mark.parametrize(
    "env_cls,params_cls,action",
    [
        (CSTR, CSTRParams, 0.0),
        (FirstOrderSystem, FirstOrderParams, 0.0),
    ],
)
def test_obs_value_index_matches_obs_array(env_cls, params_cls, action):
    """obs[obs_value_index] and obs[obs_target_index] must be finite scalars."""
    import jax

    env = env_cls()
    params = params_cls(max_steps_in_episode=N)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    vi = env.obs_value_index
    ti = env.obs_target_index
    assert np.isfinite(float(obs[vi])), "obs[obs_value_index] is not finite"
    assert np.isfinite(float(obs[ti])), "obs[obs_target_index] is not finite"


def test_plane_obs_indices_match_obs_array():
    import jax

    env = Airplane2D()
    params = PlaneParams(max_steps_in_episode=N)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    assert np.isfinite(float(obs[env.obs_value_index]))  # z
    assert np.isfinite(float(obs[env.obs_target_index]))  # target_altitude
    # sanity: target altitude must be positive
    assert float(obs[env.obs_target_index]) > 0


def test_four_tank_obs_indices_match_obs_array():
    import jax

    env = FourTank()
    params = FourTankParams(max_steps_in_episode=N)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)

    for vi, ti in zip(env.obs_value_index, env.obs_target_index):
        assert np.isfinite(float(obs[vi])), f"obs[{vi}] (value) is not finite"
        assert np.isfinite(float(obs[ti])), f"obs[{ti}] (target) is not finite"


# ---------------------------------------------------------------------------
# save_comparison_gif utility
# ---------------------------------------------------------------------------


def test_save_comparison_gif_creates_file(tmp_path):
    """save_comparison_gif produces a GIF at the requested output path."""
    env = CSTR(integration_method="rk4_1")
    params = CSTRParams(max_steps_in_episode=N)
    from target_gym.experts.pid import make_cstr_stateful_pid

    pid = make_cstr_stateful_pid()

    output_path = str(tmp_path / "comparison.gif")
    save_comparison_gif(
        env=env,
        const_select_action=lambda _: np.array([0.5]),
        pid_select_action=lambda obs: np.array([pid.step(obs)]),
        output_path=output_path,
        get_state_val=lambda s: float(s.C_a),
        get_target_val=lambda s: float(s.target_CA),
        ylabel="Concentration (mol/L)",
        const_label="Constant T_c=0.5",
        pid_label="PID",
        params=params,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_load_or_run_mpc_episode_caches(tmp_path):
    """load_or_run_mpc_episode saves on first call and loads on second (no re-run)."""
    env = CSTR(integration_method="rk4_1")
    params = CSTRParams(max_steps_in_episode=N)
    mpc = make_cstr_mpc(env, params, horizon=3)
    call_count = [0]

    def counting_action(obs, state):
        call_count[0] += 1
        return np.array([mpc.step(obs, state)])

    cache = str(tmp_path / "mpc.pkl")
    states1, rews1 = load_or_run_mpc_episode(cache, env, counting_action, params)
    assert os.path.exists(cache)
    first_calls = call_count[0]
    assert first_calls > 0

    # Second call must load from cache — counting_action is NOT called again
    states2, rews2 = load_or_run_mpc_episode(cache, env, counting_action, params)
    assert call_count[0] == first_calls, "MPC was re-run despite cache existing"
    assert len(states1) == len(states2)
    assert rews1 == rews2


# ---------------------------------------------------------------------------
# GradientMPC — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_cls,params_cls,make_fn,make_kwargs,action_dim",
    [
        (CSTR, CSTRParams, make_cstr_mpc, {"horizon": 3}, 1),
        (FirstOrderSystem, FirstOrderParams, make_first_order_mpc, {"horizon": 3}, 1),
        (FourTank, FourTankParams, make_four_tank_mpc, {"horizon": 3}, 2),
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
    """env.make_pid() returns a usable PID controller."""
    env = env_cls()
    pid = env.make_pid()
    assert pid is not None
    # StatefulPID / StatefulMIMOPID both expose a step() method
    assert callable(getattr(pid, "step", None))


def test_plane_env_make_pid_returns_controller():
    env = Airplane2D()
    pid = env.make_pid()
    assert pid is not None
    assert callable(getattr(pid, "step", None))


@pytest.mark.parametrize(
    "env_cls,params_cls,make_kwargs,expected_type",
    [
        (CSTR, CSTRParams, {"horizon": 3}, CasadiMPC),
        (FirstOrderSystem, FirstOrderParams, {"horizon": 3}, CasadiMPC),
        (FourTank, FourTankParams, {"horizon": 3}, CasadiMPC),
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
    assert isinstance(mpc, GradientMPC)
    assert mpc.action_dim == 2
