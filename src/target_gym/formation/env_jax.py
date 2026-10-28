"""
JAX-compatible formation environment: a lead and its wingmen, all learners.

:class:`Formation` is the registered, single-agent view: one joint action, the
aircraft's observations concatenated, one team reward. It is the form every
TargetGym tool reads (the conformance suite, the baselines, the protocol).
``marl.FormationMARL`` is the multi-agent view of the same step. See
:mod:`.env` for the transition, slot geometry, reward and observations.
"""

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from target_gym.base import canonical_reset, finish_step
from target_gym.expert_access import RegistryExperts
from target_gym.formation.env import (
    LEAD_OBS_BASE,
    MAX_WINGMEN,
    WINGMAN_OBS,
    FormationParams,
    FormationState,
    check_is_terminal,
    compute_next_state_formation,
    compute_reward,
    compute_reward_terms,
    formation_slots,
    get_obs_formation,
)
from target_gym.formation.rendering import _render
from target_gym.patrol.env import N_LEAD_LEGS, make_cruise_state
from target_gym.video import SavesVideo


class Formation(
    RegistryExperts,
    SavesVideo,
    environment.Environment[FormationState, FormationParams],
):
    """A lead and ``num_wingmen`` wingmen (1 to 4; 2 is the scored task).

    Action (9,): [lead power, lead stick, lead aileron, wingman 0 power, wingman 0 stick, wingman 0 aileron, wingman 1 power, wingman 1 stick, wingman 1 aileron] each in [-1, 1]

    In general ``3 * (num_wingmen + 1)`` values: ``[power, stick, aileron]``
    for the lead, then for each wingman.

    Observation: the lead's (``16 + 3 * num_wingmen`` values, see
    :func:`target_gym.formation.env.get_obs_lead`), then each wingman's, the
    patrol's 26 (:func:`target_gym.patrol.env.get_obs_full`). The tracked
    values are the lead's altitude and each wingman's slot error.

    The shipped controllers (``make_pid``, ``expert_policy``, ``make_mpc``)
    are the scored formation's; for another size build them with
    ``target_gym.formation.experts``, passing ``num_wingmen``.
    """

    render_plane = classmethod(_render)
    screen_width = 600
    screen_height = 400
    task_type: str = "formation"
    registry_name: str = "formation"

    def __init__(self, num_wingmen: int = 2, integration_method: str = "rk4_2"):
        if not 1 <= num_wingmen <= MAX_WINGMEN:
            raise ValueError(
                f"num_wingmen must be in 1..{MAX_WINGMEN}, got {num_wingmen}"
            )
        self.num_wingmen = num_wingmen
        self.integration_method = integration_method
        self.positions_history_xz = []
        self.positions_history_xy = []
        self.lead_obs_dim = LEAD_OBS_BASE + 3 * num_wingmen
        self.obs_shape = (self.lead_obs_dim + WINGMAN_OBS * num_wingmen,)
        seats = [self.lead_obs_dim + WINGMAN_OBS * i for i in range(num_wingmen)]
        # Lead altitude (2) against its target (10); each slot error (24 of
        # its seat) against the constant zero beside it (25).
        self.obs_value_index = (2, *(s + 24 for s in seats))
        self.obs_target_index = (10, *(s + 25 for s in seats))
        self.tracked_names = ("lead altitude (m)",) + tuple(
            f"wingman {i} slot error (m)" for i in range(num_wingmen)
        )

    @property
    def default_params(self) -> FormationParams:
        # The scored configuration: 200 steps, the patrol's episode, three
        # turn onsets of the route.
        return FormationParams(max_steps_in_episode=200)

    # -- core --------------------------------------------------------------
    def get_obs(self, state: FormationState, params=None) -> jnp.ndarray:
        return get_obs_formation(state)

    def compute_reward(self, state, params):
        return compute_reward(state, params)

    def reward_terms(self, state, params):
        return compute_reward_terms(state, params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: FormationState,
        action: jnp.ndarray,
        params: FormationParams = None,
    ):
        if params is None:
            params = self.default_params
        new_state, metrics = compute_next_state_formation(
            action, state, params, self.integration_method, key=key
        )
        # A crash, a collision or a lost wingman trips the whole formation:
        # the step is charged the trip cost and the formation restarts, or is
        # lost (``terminated``) with ``params.continuing = False``.
        new_state, reward, terminated, info = finish_step(
            self, key, state, new_state, params, metrics=metrics
        )
        return self.get_obs(new_state), new_state, reward, terminated, info

    def is_terminated(self, state: FormationState, params: FormationParams):
        """Natural termination only; the time limit is gymnax's ``is_truncated``."""
        terminated, _ = check_is_terminal(state, params)
        return terminated

    @canonical_reset
    def reset_env(self, key: chex.PRNGKey, params: FormationParams = None):
        if params is None:
            params = self.default_params
        k = self.num_wingmen
        alt_key, hdg_key, turn_key, back_key, lat_key, up_key, noise_key = (
            jax.random.split(key, 7)
        )
        lead_alt = jax.random.uniform(
            alt_key,
            minval=params.target_altitude_range[0],
            maxval=params.target_altitude_range[1],
        )
        lead_heading = jax.random.uniform(
            hdg_key,
            minval=params.target_heading_range[0],
            maxval=params.target_heading_range[1],
        )
        # The patrol's route: one turn rate per leg, drawn once and cycled.
        schedule = jax.random.uniform(
            turn_key,
            shape=(N_LEAD_LEGS,),
            minval=params.lead_turn_rate_range[0],
            maxval=params.lead_turn_rate_range[1],
        )

        def draw(key, bounds):
            return jax.random.uniform(key, minval=bounds[0], maxval=bounds[1])

        back, right, up = formation_slots(
            draw(back_key, params.slot_back_range),
            draw(lat_key, params.slot_lateral_range),
            draw(up_key, params.slot_up_range),
            k,
        )

        speed = params.initial_x_dot
        lead = make_cruise_state(
            x=0.0,
            y=0.0,
            z=lead_alt,
            psi=lead_heading,
            speed=speed,
            params=params,
            target_altitude=lead_alt,
            target_heading=lead_heading,
        )
        # Each wingman in its slot plus isotropic noise, at the lead's
        # velocity and heading: solvable, not trimmed.
        forward = jnp.array([jnp.cos(lead_heading), jnp.sin(lead_heading)])
        starboard = jnp.array([jnp.sin(lead_heading), -jnp.cos(lead_heading)])
        noise = params.follower_spawn_noise * jax.random.normal(noise_key, (k, 3))

        def spawn(b, r, u, n):
            xy = -b * forward + r * starboard
            return make_cruise_state(
                x=xy[0] + n[0],
                y=xy[1] + n[1],
                z=lead_alt + u + n[2],
                psi=lead_heading,
                speed=speed,
                params=params,
                target_altitude=lead_alt,
                target_heading=lead_heading,
            )

        wingmen = jax.vmap(spawn)(back, right, up, noise)
        state = FormationState(
            lead=lead,
            wingmen=wingmen,
            slot_back=back,
            slot_right=right,
            slot_up=up,
            lead_turn_rate=schedule[0],
            lead_turn_schedule=schedule,
            time=0,
        )
        return self.get_obs(state), state

    # -- rendering / video -------------------------------------------------
    def render(self, screen, state: FormationState, params, frames, clock):
        return self.render_plane(screen, state, params, frames, clock)

    # -- spaces ------------------------------------------------------------
    def action_space(self, params: FormationParams | None = None) -> spaces.Box:
        n = 3 * (self.num_wingmen + 1)
        return spaces.Box(
            low=-jnp.ones(n), high=jnp.ones(n), shape=(n,), dtype=jnp.float32
        )

    def observation_space(self, params: FormationParams | None = None) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, self.obs_shape, dtype=jnp.float32)

    def state_space(self, params: FormationParams | None = None) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        return spaces.Box(-inf, inf, (1,), dtype=jnp.float32)
