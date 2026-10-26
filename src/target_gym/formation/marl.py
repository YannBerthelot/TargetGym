"""
The multi-agent view of the formation, in the JaxMARL ``MultiAgentEnv`` style.

Agents are ``"lead"``, ``"wingman_0"``, ..., ``"wingman_{K-1}"``. ``reset`` and
``step`` take and return dicts keyed by agent; ``dones`` also carries
``"__all__"``. Every agent receives the team reward, and ``info["costs"]``
reports each aircraft's own tracking cost for credit assignment. The step is
:class:`~target_gym.formation.env_jax.Formation`'s, so a team scores the same
here as on the registered task.
"""

from typing import Dict

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces

from target_gym.formation.env import (
    WINGMAN_OBS,
    FormationParams,
    FormationState,
    aircraft_costs,
)
from target_gym.formation.env_jax import Formation

LEAD = "lead"


def wingman_name(i: int) -> str:
    return f"wingman_{i}"


class FormationMARL:
    """A lead and ``num_wingmen`` wingmen (1 to 4), one agent per aircraft.

    Each agent acts with ``[power, stick, aileron]`` in [-1, 1]. The lead
    observes ``16 + 3 * num_wingmen`` values, each wingman the patrol's 26
    (see :class:`Formation`). The joint environment is ``self.env``.
    """

    def __init__(self, num_wingmen: int = 2, integration_method: str = "rk4_2"):
        self.env = Formation(num_wingmen, integration_method)
        self.num_wingmen = num_wingmen
        self.agents = [LEAD] + [wingman_name(i) for i in range(num_wingmen)]
        self.num_agents = len(self.agents)

    @property
    def default_params(self) -> FormationParams:
        return self.env.default_params

    def get_obs(self, state: FormationState) -> Dict[str, jnp.ndarray]:
        return self._split(self.env.get_obs(state))

    def _split(self, joint: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        lead = self.env.lead_obs_dim
        obs = {LEAD: joint[:lead]}
        for i in range(self.num_wingmen):
            start = lead + WINGMAN_OBS * i
            obs[wingman_name(i)] = joint[start : start + WINGMAN_OBS]
        return obs

    def reset(self, key: chex.PRNGKey, params: FormationParams = None):
        obs, state = self.env.reset_env(key, params)
        return self._split(obs), state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FormationState,
        actions: Dict[str, jnp.ndarray],
        params: FormationParams = None,
    ):
        """One step without the automatic reset. ``dones`` is true at a trip
        with ``params.continuing = False`` or at the time limit."""
        if params is None:
            params = self.default_params
        joint = jnp.concatenate([jnp.asarray(actions[a]) for a in self.agents])
        obs, new_state, reward, terminated, info = self.env.step_env(
            key, state, joint, params
        )
        done = terminated | (new_state.time >= params.max_steps_in_episode)
        # Each aircraft's tracking cost this step. A trip charges the team
        # ``info["trip_cost"]`` instead, so under the version-2 reward the
        # costs plus the trip cost add up to minus the reward.
        costs = jax.tree.map(
            lambda c: jnp.where(info["tripped"], 0.0, c),
            aircraft_costs(new_state, params),
        )
        info = dict(
            info,
            costs={
                LEAD: costs["lead"],
                **{
                    wingman_name(i): costs["wingmen"][i]
                    for i in range(self.num_wingmen)
                },
            },
        )
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        return self._split(obs), new_state, rewards, dones, info

    def step(
        self,
        key: chex.PRNGKey,
        state: FormationState,
        actions: Dict[str, jnp.ndarray],
        params: FormationParams = None,
    ):
        """JaxMARL-style step: resets every agent when ``dones["__all__"]``."""
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, rewards, dones, info = self.step_env(
            key, state, actions, params
        )
        obs_re, state_re = self.reset(key_reset, params)
        state = jax.tree.map(
            lambda re, st: jnp.where(dones["__all__"], re, st), state_re, state_st
        )
        obs = {
            a: jnp.where(dones["__all__"], obs_re[a], obs_st[a]) for a in self.agents
        }
        return obs, state, rewards, dones, info

    # -- spaces ------------------------------------------------------------
    def action_space(self, agent: str = None) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32)

    def observation_space(self, agent: str) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        dim = self.env.lead_obs_dim if agent == LEAD else WINGMAN_OBS
        return spaces.Box(-inf, inf, (dim,), dtype=jnp.float32)

    @property
    def observation_spaces(self) -> Dict[str, spaces.Box]:
        return {a: self.observation_space(a) for a in self.agents}

    @property
    def action_spaces(self) -> Dict[str, spaces.Box]:
        return {a: self.action_space(a) for a in self.agents}
