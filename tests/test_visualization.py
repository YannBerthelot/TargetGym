import os
import shutil
import tempfile

import jax
import numpy as np
import pytest

# from gymnasium.utils.save_video import save_video
from target_gym.bicycle.env_jax import BikeParams, BikeState
from target_gym.bicycle.env_jax import RandlovBicycle as JaxBike2D
from target_gym.car.env_jax import Car2D as JaxCar2D
from target_gym.car.env_jax import CarParams, CarState
from target_gym.pc_gym.cstr.env_jax import CSTR as JaxCSTR
from target_gym.pc_gym.cstr.env_jax import CSTRParams, CSTRState
from target_gym.plane.env_jax import Airplane2D as JaxPlane2D
from target_gym.plane.env_jax import PlaneParams, PlaneState


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
        JaxCar2D,
        CarParams,
        CarState,
        lambda _: 0.5,
    ),  # car
    (
        JaxBike2D,
        BikeParams,
        BikeState,
        lambda _: (0.6, 0.0),
    ),  # bicycle
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
        obs, state, reward, terminated, info = jax_env.step(
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
        key = jax.random.PRNGKey(0)
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
