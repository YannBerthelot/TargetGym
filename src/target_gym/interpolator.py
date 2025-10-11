from typing import Dict, List, Union

import jax
import jax.numpy as jnp
import pandas as pd

from target_gym import CSTR, Bike, Car, Plane
from target_gym.runners.utils import run_input_grid


def jax_interp1d(xq, xp, fp, left=jnp.nan, right=jnp.nan):
    """
    Minimal JAX-compatible version of interp1d (linear only).
    Assumes xp is sorted and monotonic.
    """
    idx = jnp.searchsorted(xp, xq) - 1
    idx = jnp.clip(idx, 0, xp.shape[0] - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = fp[idx], fp[idx + 1]
    slope = (y1 - y0) / (x1 - x0)
    yq = y0 + slope * (xq - x0)

    # Handle out-of-bounds
    yq = jnp.where(xq < xp[0], left, yq)
    yq = jnp.where(xq > xp[-1], right, yq)
    return yq


def build_input_interpolator_from_df(
    df: pd.DataFrame,
    input_names: Union[str, List[str]] = "input",
    output_name: str = "final_value",
):
    """
    Build JAX-compatible interpolators for single- or two-input environments.
    """
    if isinstance(input_names, str):
        # Single-input environment
        df_sorted = df.sort_values(output_name)
        inputs = jnp.array(df_sorted[input_names].to_numpy())
        outputs = jnp.array(df_sorted[output_name].to_numpy())

        # Check monotonicity (outside JAX, since it's a one-time check)
        diffs = jnp.diff(outputs)
        if not (jnp.all(diffs >= 0) | jnp.all(diffs <= 0)):
            raise ValueError(
                f"Output not monotonic for {input_names}, interpolation ambiguous."
            )

        def interpolator(query_outputs):
            return jax_interp1d(query_outputs, outputs, inputs)

        return interpolator

    elif isinstance(input_names, list) and len(input_names) == 2:
        # Two-input environment
        input1, input2 = input_names
        interpolators: Dict[float, callable] = {}
        tol = 1e-6

        for val2 in jnp.unique(jnp.array(df[input2].to_numpy())):
            df_fixed = df[abs(df[input2] - float(val2)) < tol].sort_values(output_name)
            if df_fixed.empty:
                continue
            outputs = jnp.array(df_fixed[output_name].to_numpy())
            inputs = jnp.array(df_fixed[input1].to_numpy())

            diffs = jnp.diff(outputs)
            if not (jnp.all(diffs >= 0) | jnp.all(diffs <= 0)):
                raise ValueError(
                    f"Output not monotonic for {input1} at {input2}={val2}, ambiguous."
                )

            def make_interp(xp, fp):
                return lambda xq: jax_interp1d(xq, xp, fp)

            interpolators[float(val2)] = make_interp(outputs, inputs)

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
    Run the grid function and build interpolators (JAX version).
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
        return lambda q: jax_interp1d(
            q,
            jnp.array(df["final_value"].to_numpy()),
            jnp.array(df[input_names[0]].to_numpy()),
        )
    else:
        df0 = df[df[input_names[1]] == 0.0].sort_values("final_value")
        return lambda q: jax_interp1d(
            q,
            jnp.array(df0["final_value"].to_numpy()),
            jnp.array(df0[input_names[0]].to_numpy()),
        )


def get_interpolator(env_class, env_params, resolution: int = 100, steps: int = 10_000):
    mapping = ENV_IO_MAPPING[env_class]
    input_names = mapping["input_names"]

    if len(input_names) == 2:
        if env_class == Plane:
            first_input = jnp.linspace(-1.0, 1.0, resolution)
            second_input = jnp.zeros(1)
        else:
            first_input = jnp.linspace(-1.0, 1.0, resolution)
            second_input = jnp.linspace(-1.0, 1.0, resolution)
    else:
        env_instance = env_class()
        try:
            bounds = env_instance.action_space(env_params)
            min_val = float(bounds.low[0])
            max_val = float(bounds.high[0])
            if env_class == Car:
                min_val = max(min_val, 0.0)
        except Exception:
            min_val, max_val = -1.0, 1.0
        first_input = jnp.linspace(min_val, max_val, resolution)
        second_input = None

    interp = build_env_interpolator(
        env_class,
        env_params,
        input_levels=first_input,
        second_input_levels=second_input,
        steps=steps,
    )
    return interp
