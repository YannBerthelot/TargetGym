import os
import shutil
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.pc_gym.cstr.env_jax import CSTR as JaxCSTR
from target_gym.pc_gym.cstr.env_jax import CSTRParams, CSTRState
from target_gym.plane.env_jax import Airplane2D as JaxPlane2D
from target_gym.plane.env_jax import PlaneParams, PlaneState
from target_gym.registry import REGISTRY


# -------------------------
# Utilities
# -------------------------
def are_images_similar(
    img1: np.ndarray, img2: np.ndarray, threshold: float = 0.95
) -> bool:
    if img1.shape != img2.shape:
        return False
    img1_norm = (img1 - img1.mean()) / img1.std()
    img2_norm = (img2 - img2.mean()) / img2.std()
    correlation = np.sum(img1_norm * img2_norm) / img1.size
    return correlation > threshold


# -------------------------
# Parametrization
# -------------------------

max_steps_for_video = 100
ENVIRONMENTS = [
    (
        JaxPlane2D,
        PlaneParams,
        PlaneState,
        lambda _: (0.8, 0.0),
    ),  # plane
    (
        JaxCSTR,
        CSTRParams,
        CSTRState,
        lambda _: 0.5,
    ),  # cstr
]


@pytest.mark.parametrize("jax_env_cls,EnvParamsCls,EnvStateCls,action_fn", ENVIRONMENTS)
def test_render_trajectory_param(jax_env_cls, EnvParamsCls, EnvStateCls, action_fn):
    jax_env = jax_env_cls()

    # Initialize environment
    key = jax.random.PRNGKey(0)
    obs, state = jax_env.reset(key)
    env_params = EnvParamsCls(
        max_steps_in_episode=max_steps_for_video
    )  # default parameters
    frames_list = []
    screen = None
    clock = None

    for _ in range(10):
        action = action_fn(None)
        obs, state, reward, terminated, truncated, info = jax_env.step(
            key, state, action, env_params
        )

        # Render
        rendered_frames, screen, clock = jax_env.render(
            screen, state, env_params, frames_list, clock
        )
        frames_list.extend(rendered_frames)


@pytest.mark.parametrize("jax_env_cls,EnvParamsCls,EnvStateCls,action_fn", ENVIRONMENTS)
def test_save_renders_param(jax_env_cls, EnvParamsCls, EnvStateCls, action_fn):
    temp_dir = tempfile.mkdtemp()
    try:
        jax_env = jax_env_cls()
        env_params = EnvParamsCls(max_steps_in_episode=100)

        file = jax_env.save_video(
            lambda _: action_fn(None),
            seed=0,
            folder=temp_dir,
            episode_index=0,
            params=env_params,
        )
        assert os.path.exists(file)

    finally:
        shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# Every registered environment renders
#
# The two hand-listed environments above predate the registry, and left
# sixteen renderers with no test at all -- they are user-facing (the README
# gallery, save_video), so a crash in one of them would ship. Rendering a few
# frames costs well under a second per environment.
# ---------------------------------------------------------------------------

RENDER_STEPS = 12


@pytest.mark.parametrize("name", list(REGISTRY), ids=list(REGISTRY))
def test_every_environment_renders(name):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 30})

    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    action_shape = env.action_space(params).shape or (1,)

    frames, screen, clock = [], None, None
    for _ in range(RENDER_STEPS):
        _, state, _, terminated, _ = step(key, state, jnp.zeros(action_shape), params)
        frames, screen, clock = env.render(screen, state, params, frames, clock)
        if bool(terminated):
            break

    assert frames, f"{name}: rendered no frames in {RENDER_STEPS} steps"

    first = np.asarray(frames[0])
    assert (
        first.ndim == 3 and first.shape[2] == 3
    ), f"{name}: frame is {first.shape}, expected (height, width, 3)"
    # A GIF cannot be assembled from frames of differing size, so a renderer
    # whose layout depends on the data is broken even though it draws.
    assert all(
        np.asarray(f).shape == first.shape for f in frames
    ), f"{name}: frame sizes differ across the episode"
    values = first.astype(np.int64)
    assert values.min() >= 0 and values.max() <= 255, f"{name}: pixels out of range"
    # A renderer that silently draws nothing still returns a frame, and every
    # other assertion here passes for a flat rectangle.
    assert values.std() > 1.0, f"{name}: frame is uniform -- the renderer drew nothing"
