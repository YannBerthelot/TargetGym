import os
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d

from target_gym import CSTR, Bike, Car, Plane
from target_gym.runners.utils import run_input_grid


def build_input_interpolator_from_df(
    df: pd.DataFrame,
    input_names: Union[str, List[str]] = "input",
    output_name: str = "final_value",
):
    """
    Build interpolator(s) for single- or two-input environments.

    Args:
        df: DataFrame with columns [input1, input2 (optional), final_value]
        input_names: string or list of input column names
        output_name: name of the column containing the output (state attribute)

    Returns:
        interp1d function or dict of interp1d functions
    """
    if isinstance(input_names, str):
        # Single-input environment
        df_sorted = df.sort_values(output_name)
        inputs = df_sorted[input_names].to_numpy()
        outputs = df_sorted[output_name].to_numpy()

        # Check monotonicity
        if not (np.all(np.diff(outputs) >= 0) or np.all(np.diff(outputs) <= 0)):
            raise ValueError(
                f"Output not monotonic for {input_names}, interpolation ambiguous."
            )

        interpolator = interp1d(
            outputs, inputs, bounds_error=False, fill_value=np.nan, kind="linear"
        )
        return interpolator

    elif isinstance(input_names, list) and len(input_names) == 2:
        # Two-input environment
        input1, input2 = input_names
        interpolators: Dict[float, interp1d] = {}
        tol = 1e-6

        # Build an interpolator for each fixed value of the second input
        for val2 in np.unique(df[input2]):
            df_fixed = df[np.abs(df[input2] - val2) < tol].sort_values(output_name)
            if df_fixed.empty:
                continue
            outputs = df_fixed[output_name].to_numpy()
            inputs = df_fixed[input1].to_numpy()

            if not (np.all(np.diff(outputs) >= 0) or np.all(np.diff(outputs) <= 0)):
                raise ValueError(
                    f"Output not monotonic for {input1} at {input2}={val2}, ambiguous."
                )

            interpolators[val2] = interp1d(
                outputs, inputs, bounds_error=False, fill_value=np.nan, kind="linear"
            )

        return interpolators

    else:
        raise ValueError("input_names must be a string or a list of 2 strings.")


def get_interpolator_from_run(
    run_func: callable,
    run_kwargs: dict,
    input_names: Union[str, List[str]] = "input",
    output_name: str = "final_value",
):
    """
    Run the grid function and build interpolators.

    Args:
        run_func: function returning (final_values, df)
        run_kwargs: kwargs to pass to run_func
        input_names: single input column or list of two input columns
        output_name: column name of the output (state attribute)

    Returns:
        interp1d function (single-input) or dict of interp1d (two-input)
    """
    _, df = run_func(**run_kwargs)
    return build_input_interpolator_from_df(
        df, input_names=input_names, output_name=output_name
    )


# Map each env to its input(s) and output state attribute
ENV_IO_MAPPING = {
    Plane: {"input_names": ["power", "stick"], "state_attr": "z"},
    Bike: {"input_names": ["power", "stick"], "state_attr": "z"},
    Car: {"input_names": ["throttle"], "state_attr": "velocity"},
    CSTR: {"input_names": ["T_c"], "state_attr": "T"},
}


def build_env_interpolator(
    env_class, env_params, input_levels=None, second_input_levels=None, steps=10_000
):
    mapping = ENV_IO_MAPPING[env_class]
    input_names = mapping["input_names"]
    state_attr = mapping["state_attr"]

    env_instance = env_class()
    final_values, df = run_input_grid(
        input_levels,
        env_instance,
        env_params,
        steps=steps,
        input_name=input_names[0],
        second_input_levels=second_input_levels,
        second_input_name=input_names[1] if len(input_names) == 2 else None,
        state_attr=state_attr,
    )
    if env_class is Plane:
        df = df[df["final_value"] > 0]
    if len(input_names) == 1:
        # Single input: interp1d
        return interp1d(
            df["final_value"].to_numpy(),
            df[input_names[0]].to_numpy(),
            bounds_error=False,
            fill_value=np.nan,
            kind="linear",
        )
    else:
        # Two inputs: only keep second_input=0 for round-trip
        df0 = df[df[input_names[1]] == 0.0].sort_values("final_value")
        return interp1d(
            df0["final_value"].to_numpy(),
            df0[input_names[0]].to_numpy(),
            bounds_error=False,
            fill_value=np.nan,
            kind="nearest",
        )


def get_interpolator(env_class, env_params, resolution: int = 100, steps: int = 10_000):
    mapping = ENV_IO_MAPPING[env_class]
    input_names = mapping["input_names"]

    # Automatically set input grids
    if len(input_names) == 2:

        if env_class == Plane:
            first_input = jnp.linspace(0, 1.0, resolution)
            second_input = jnp.zeros(1)
        else:
            first_input = jnp.linspace(-1.0, 1.0, resolution)
            second_input = jnp.linspace(-1.0, 1.0, resolution)
    else:
        env_instance = env_class()
        try:
            min_val = float(env_instance.action_space(env_params).low[0])
            max_val = float(env_instance.action_space(env_params).high[0])
            if env_class == Car:
                min_val = max(min_val, 0.0)
        except Exception:
            min_val, max_val = -1.0, 1.0
        first_input = jnp.linspace(min_val, max_val, resolution)
        second_input = None

    # Build interpolator
    interp = build_env_interpolator(
        env_class,
        env_params,
        input_levels=first_input,
        second_input_levels=second_input,
        steps=steps,
    )
    return interp
