import jax
import jax.numpy as jnp


def compute_norm_from_coordinates(coordinates: jnp.ndarray) -> float:
    """Compute the norm of a vector given its coordinates"""
    return jnp.linalg.norm(coordinates, axis=0)


def log_scaled_reward(error, floor, envelope, xp=jnp):
    """Tracking reward that keeps paying for precision.

    Returns 1.0 at zero error and 0.0 at ``envelope``.

    A reward normalised by the state-space envelope (``1 - e/E``, however
    sharply exponentiated) spends nearly all of its dynamic range on errors the
    controller has already eliminated: with ``E`` = 12 km, the gap between a
    10 m and a 1 m altitude error is under 0.001, so the reward cannot tell a
    good controller from a great one.  Scaling in ``log(error)`` instead pays a
    *constant* increment for every halving of the error, all the way down to
    ``floor`` -- the finest error the corresponding sensor can resolve, below
    which further "improvement" is measuring noise.

    Args:
        error: absolute tracking error, in the same units as the other two.
        floor: sensor resolution; the error scale below which reward saturates.
        envelope: the error at which the reward reaches 0.
    """
    return xp.clip(1.0 - xp.log1p(error / floor) / xp.log1p(envelope / floor), 0.0, 1.0)


def compute_episode_returns_vectorized(rewards: jnp.ndarray, dones: jnp.ndarray):
    """
    Compute per-episode cumulative reward, handling both terminated and truncated episodes.

    Args:
        rewards: (T,) float array of rewards per timestep
        dones:   (T,) int/boolean array, 1 if episode ends at that step

    Returns:
        episode_returns: (num_episodes,) array of total return per episode
    """
    # Cumulative rewards
    cumsum_rewards = jnp.cumsum(rewards)

    # Rewards at episode boundaries (where done=1)
    final_returns = cumsum_rewards[dones.astype(bool)]

    # Previous boundaries (prepend 0 for the very first episode)
    prev_final = jnp.concatenate([jnp.array([0.0]), final_returns[:-1]])

    # Proper per-episode returns
    episode_returns = final_returns - prev_final

    # --- Handle truncated last episode ---
    last_is_done = dones[-1] > 0
    if not last_is_done:
        # Add the return from last boundary (or start) up to end
        last_return = cumsum_rewards[-1] - (
            final_returns[-1] if final_returns.size > 0 else 0.0
        )
        episode_returns = jnp.concatenate([episode_returns, jnp.array([last_return])])

    return episode_returns


def run_n_steps(env, policy, params, n_steps=10_000, seed=0):
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)

    def step_fn(carry, _):
        obs, state, key = carry
        key, subkey = jax.random.split(key)

        action = policy(obs)
        obs, new_state, reward, done, _ = env.step_env(subkey, state, action, params)

        carry = (
            jax.lax.cond(
                done,
                lambda _: env.reset_env(key, params),
                lambda _: (obs, new_state),
                operand=None,
            )[
                0
            ],  # new obs
            jax.lax.cond(
                done,
                lambda _: env.reset_env(key, params),
                lambda _: (obs, new_state),
                operand=None,
            )[
                1
            ],  # new state
            key,
        )

        # If episode ended, mark this reward as last in episode
        ep_done = done.astype(jnp.float32)
        return carry, (reward, ep_done)

    # Scan for n_steps
    (_, _, _), (rewards, ep_dones) = jax.lax.scan(
        step_fn, (obs, state, key), None, n_steps
    )
    valid_returns = compute_episode_returns_vectorized(rewards, ep_dones)
    mean_return = jnp.mean(valid_returns)
    return mean_return


def convert_raw_action_to_range(raw_action, min_action, max_action):
    """
    Assuming the action is roughly in (-1,1), we rescale to it (min_action,max_action).
    """
    action = min_action + 0.5 * (jnp.clip(raw_action, -1, 1) + 1) * (
        max_action - min_action
    )
    return action
