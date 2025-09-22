import time

import jax
import jax.numpy as jnp


def benchmark_env(env, params, steps: int = 1_000_000, batch_size: int = 1024):
    """
    Benchmark how many steps per second an environment can run in parallel.

    Args:
        env: JAX environment (must implement reset_env, step)
        params: environment parameters
        steps: number of timesteps to simulate
        batch_size: number of parallel environments

    Returns:
        steps_per_second: float
    """
    key = jax.random.PRNGKey(0)

    # Reset batch of environments
    keys = jax.random.split(key, batch_size)
    obs, state = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)

    # Dummy action (zeros)
    action_dim = env.action_space(params).shape[0]
    # actions = (
    #     jnp.zeros((batch_size, action_dim))
    #     if action_dim > 1
    #     else jnp.zeros((batch_size,))
    # )
    actions = jnp.zeros((batch_size, action_dim))

    def step_fn(carry, _):
        state = carry
        key = jax.random.PRNGKey(0)  # reuse same rng (no stochasticity assumed)
        obs, new_state, reward, done, info = jax.vmap(
            env.step, in_axes=(None, 0, 0, None)
        )(key, state, actions, params)
        return new_state, None

    # JIT compile scan loop
    step_fn_jit = jax.jit(
        lambda s: jax.lax.fori_loop(0, steps, lambda _, st: step_fn(st, None)[0], s)
    )

    # Warmup
    state = step_fn_jit(state)
    state.t.block_until_ready()

    # Timing
    t0 = time.time()
    state = step_fn_jit(state)
    state.t.block_until_ready()
    t1 = time.time()

    total_steps = steps * batch_size
    steps_per_second = total_steps / (t1 - t0)

    return steps_per_second


if __name__ == "__main__":
    from target_gym import (
        CSTR,
        Bike,
        BikeParams,
        Car,
        CarParams,
        CSTRParams,
        Plane,
        PlaneParams,
    )

    N_steps = int(1e8)
    max_steps_in_episode = 10_000

    plane_env = Plane()
    plane_params = PlaneParams(max_steps_in_episode=max_steps_in_episode)
    car_env = Car()
    car_params = CarParams(max_steps_in_episode=max_steps_in_episode)

    bike_env = Bike()
    bike_params = BikeParams(max_steps_in_episode=max_steps_in_episode)

    cstr_env = CSTR()
    cstr_params = CSTRParams(max_steps_in_episode=max_steps_in_episode)

    print(
        "Plane M-steps/sec:",
        benchmark_env(plane_env, plane_params, steps=N_steps, batch_size=1) / int(1e6),
    )
    print(
        "Car M-steps/sec:",
        benchmark_env(car_env, car_params, steps=N_steps, batch_size=1) / int(1e6),
    )
    print(
        "Bike M-steps/sec:",
        benchmark_env(bike_env, bike_params, steps=N_steps, batch_size=1) / int(1e6),
    )
    print(
        "CSTR M-steps/sec:",
        benchmark_env(cstr_env, cstr_params, steps=N_steps, batch_size=1) / int(1e6),
    )
