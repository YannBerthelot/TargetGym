import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import CSTR, Bike, Car, CarParams, CSTRParams, Plane, PlaneParams
from target_gym.interpolator import (
    ENV_IO_MAPPING,
    get_interpolator,
)
from target_gym.runners.utils import run_input_grid

STEPS = 10_000
RESOLUTION = 100  # small for test speed


@pytest.mark.parametrize(
    "env_class, env_params",
    [
        (Car, CarParams(max_steps_in_episode=STEPS)),
        (CSTR, CSTRParams(max_steps_in_episode=STEPS)),
        (Plane, PlaneParams(max_steps_in_episode=STEPS)),
        # (Bike, BikeParams(max_steps_in_episode=STEPS)),
    ],
)
def test_interpolator_round_trip(env_class, env_params):

    interpolator = get_interpolator(env_class, env_params, resolution=RESOLUTION)
    mapping = ENV_IO_MAPPING[env_class]
    input_names = mapping["input_names"]
    state_attr = mapping["state_attr"]

    # Automatically set input grids
    if len(input_names) == 2:

        if env_class == Plane:
            first_input = jnp.linspace(0, 1.0, RESOLUTION)
            second_input = jnp.zeros(1)
        else:
            first_input = jnp.linspace(-1.0, 1.0, RESOLUTION)
            second_input = jnp.linspace(-1.0, 1.0, RESOLUTION)
    else:
        env_instance = env_class()
        try:
            min_val = float(env_instance.action_space(env_params).low[0])
            max_val = float(env_instance.action_space(env_params).high[0])
            if env_class == Car:
                min_val = max(min_val, 0.0)
        except Exception:
            min_val, max_val = -1.0, 1.0
        first_input = jnp.linspace(min_val, max_val, RESOLUTION)
        second_input = None
    # Run the grid to get actual outputs
    final_values, df = run_input_grid(
        first_input,
        env_class(),
        env_params,
        steps=STEPS,
        input_name=input_names[0],
        second_input_levels=second_input,
        second_input_name=input_names[1] if second_input is not None else None,
        state_attr=state_attr,
    )

    predicted_inputs = interpolator(df["final_value"].to_numpy())
    # Mask NaNs
    mask = ~np.isnan(predicted_inputs)
    np.testing.assert_allclose(
        predicted_inputs[mask], df[input_names[0]].to_numpy()[mask], rtol=1e-2
    )
