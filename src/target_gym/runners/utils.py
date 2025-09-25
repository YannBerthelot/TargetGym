import os
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pandas as pd


def run_constant_policy_final_value(
    env,
    params,
    action: Union[float, Tuple[float, float]],
    state_attr: str,
    steps: int = 10_000,
    key_seed: int = 0,
):
    """
    Run a constant policy in a JAX environment and return the final value of a specified state attribute.
    Works safely with JAX traced arrays.
    """
    key = jax.random.PRNGKey(key_seed)
    init_obs, init_state = env.reset_env(key, params)

    def step_fn(carry, _):
        key, state, done = carry
        obs, new_state, reward, new_done, info = env.step_env(
            key, state, action, params
        )
        truncated = new_state.time >= params.max_steps_in_episode
        state = jax.lax.cond(done, lambda _: state, lambda _: new_state, operand=None)
        done = jnp.logical_or(done, truncated)
        value = getattr(new_state, state_attr)
        last_value = (
            getattr(info["last_state"], state_attr) if "last_state" in info else value
        )
        return (key, state, done), (value, last_value, done)

    (_, final_state, done), (value_hist, last_value_hist, done_hist) = jax.lax.scan(
        step_fn, (key, init_state, False), None, length=steps
    )

    # Get index of first True in done_hist
    done_idx = jnp.argmax(done_hist)

    # Safely handle first step case using lax.cond
    final_value = jax.lax.cond(
        done_idx > 0,
        lambda idx: last_value_hist[idx - 1],
        lambda _: last_value_hist[-1],
        operand=done_idx,
    )

    return final_value


def run_input_grid(
    input_levels: jnp.ndarray,
    env,
    params,
    steps: int = 10_000,
    input_name: str = "input",
    second_input_levels: Optional[jnp.ndarray] = None,
    second_input_name: Optional[str] = None,
    state_attr: str = "velocity",
) -> Tuple[jnp.ndarray, pd.DataFrame]:
    """
    Runs a grid of constant inputs on an environment.

    Supports:
        - Single-input environments: Car, CSTR
        - Two-input environments: Plane, Bike

    Args:
        input_levels: 1D array of first input (throttle, T_c, power)
        env: JAX environment
        params: EnvParams
        steps: timesteps to run
        input_name: name for CSV column of first input
        second_input_levels: 1D array of second input (stick), optional
        second_input_name: CSV column name for second input, optional
        state_attr: which state attribute to track ("velocity", "T", "z", etc.)

    Returns:
        final_values: jnp array of final state_attr values
        df: pandas DataFrame with inputs and final values
    """
    if second_input_levels is None:
        # Single-input env
        def run_one_input(u):
            return run_constant_policy_final_value(
                env, params, action=u, state_attr=state_attr, steps=steps, key_seed=0
            )

        final_values = jax.vmap(run_one_input)(input_levels)
        df = pd.DataFrame({input_name: input_levels, "final_value": final_values})

    else:
        # Two-input env
        def run_one_first_input(u):
            return jax.vmap(
                lambda v: run_constant_policy_final_value(
                    env,
                    params,
                    action=(u, v),
                    state_attr=state_attr,
                    steps=steps,
                    key_seed=0,
                )
            )(second_input_levels)

        final_values = jax.vmap(run_one_first_input)(input_levels)

        # Flatten arrays for DataFrame
        df = pd.DataFrame(
            {
                input_name: jnp.repeat(input_levels, len(second_input_levels)),
                second_input_name: jnp.tile(second_input_levels, len(input_levels)),
                "final_value": final_values.flatten(),
            }
        )
    return final_values, df
