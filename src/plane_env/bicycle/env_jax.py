# env_jax.py
from typing import Tuple

import chex
import jax
import jax.numpy as jnp

# If you use gymnax, import environment and spaces. If not installed,
# the class still uses the same API as your example.
try:
    from gymnax.environments import environment, spaces
except Exception:
    # Minimal fallback types for static checking if gymnax isn't present.
    from typing import Any

    class spaces:
        Box = object

    class environment:
        class Environment:
            pass


from plane_env.bicycle.env import (
    EnvParams,
    EnvState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    get_obs,
)


class RandlovBicycle(environment.Environment[EnvState, EnvParams]):
    """
    JAX-compatible Randløv bicycle environment implementing equations from the paper.
    Continuous actions: [-1,1]^2 -> [Torque, Displacement].
    Observations: [omega, omega_dot, omega_ddot, theta, theta_dot]
    """

    def __init__(self):
        self.obs_shape = (5,)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams = None,
    ):
        """
        Perform one env step (JAX friendly).
        Returns (obs, new_state, reward, done, info)
        """
        if params is None:
            params = self.default_params

        new_state = compute_next_state(action, state, params, xp=jnp)
        reward = compute_reward(new_state, params, xp=jnp)
        terminated, truncated = check_is_terminal(new_state, params, xp=jnp)
        done = terminated | truncated

        obs = self.get_obs(new_state, params=params)
        return obs, new_state, reward, done, {"last_state": new_state}

    def get_obs(self, state: EnvState, params: EnvParams = None):
        """
        Observation vector per paper: omega, omega_dot, omega_ddot, theta, theta_dot
        """
        if params is None:
            params = self.default_params
        return get_obs(state, params=params, xp=jnp)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        terminated, truncated = check_is_terminal(state, params, xp=jnp)
        return terminated | truncated

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams = None
    ) -> Tuple[jnp.ndarray, EnvState]:
        """
        Reset the environment using JAX random keys.
        Starts the bicycle in upright equilibrium at origin.
        """
        if params is None:
            params = self.default_params

        # initialize tyre contact points near origin (front/back separated by l)
        xf0 = 0.0
        yf0 = 0.0
        xb0 = -params.l
        yb0 = 0.0

        state = EnvState(
            omega=0.0,
            omega_dot=0.0,
            theta=0.0,
            theta_dot=0.0,
            x_f=xf0,
            y_f=yf0,
            x_b=xb0,
            y_b=yb0,
            t=0,
        )
        obs = self.get_obs(state, params=params)
        return obs, state

    def action_space(self, params: EnvParams | None = None):
        """Continuous torque and displacement in [-1, 1]^2."""
        return spaces.Box(
            low=jnp.array([-1.0, -1.0]),
            high=jnp.array([1.0, 1.0]),
            shape=(2,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams | None = None):
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: EnvParams | None = None):
        # Simple placeholder Box describing flattened state
        inf = jnp.finfo(jnp.float32).max
        # 9 fields in EnvState
        return spaces.Box(-inf, inf, (9,), dtype=jnp.float32)
