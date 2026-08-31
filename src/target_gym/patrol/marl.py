"""
Multi-agent close patrol / **formation** (Stage 2): a lead plus K wingmen, all
learners.

Generalises the two-plane cooperative task to an arbitrary formation of
``1 + num_wingmen`` aircraft (up to 5 total).  The lead learns to fly its
patrol pattern (track a target altitude and heading); each wingman learns to
hold its slot.  Slots are generated automatically, **evenly spread across both
sides** of the lead in a symmetric V (echelon) — wingman 0 to the right,
wingman 1 to the left, wingman 2 further right, and so on.

Everyone receives the **same team reward** (the lead's path-tracking plus the
mean of the wingmen's slot-keeping × heading-alignment), so success needs the
lead to fly a trackable pattern *and* every wingman to keep formation.  A
collision between **any pair** of aircraft is a shared, heavily-penalised
terminal state.

The interface follows the JaxMARL ``MultiAgentEnv`` convention: agents are
``"lead"``, ``"wingman_0"``, …, ``"wingman_{K-1}"``; ``reset`` / ``step`` return
dicts keyed by agent (plus a ``"__all__"`` done flag).  Reuses the exact 3D
physics and the patrol renderer.
"""

from typing import Dict

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from target_gym.base import EnvState
from target_gym.experts.pid import make_plane3d_heading_pid
from target_gym.patrol.env import (
    PatrolParams,
    PatrolState,
    decode_action,
    get_obs_full,
    heading_alignment,
    make_cruise_state,
    slot_error,
)
from target_gym.patrol.rendering import _render_scene
from target_gym.plane.dynamics import advance_gust
from target_gym.plane3d.env import (
    PlaneState3D,
    compute_next_state_3d,
    get_obs_heading,
    wrap_angle,
)
from target_gym.utils import log_scaled_reward

LEAD = "lead"
MAX_WINGMEN = 4


def wingman_name(i: int) -> str:
    return f"wingman_{i}"


@struct.dataclass
class FormationState(EnvState):
    """Lead + K wingmen (a tuple of PlaneState3D) and the per-wingman slots."""

    lead: PlaneState3D
    wingmen: tuple  # length-K tuple of PlaneState3D
    slot_back: jnp.ndarray  # (K,)
    slot_right: jnp.ndarray  # (K,)
    slot_up: jnp.ndarray  # (K,)
    # Shared formation turbulence gust (m/s) applied to every aircraft.
    gust_x: float = 0.0
    gust_y: float = 0.0
    gust_z: float = 0.0


@struct.dataclass
class PatrolMARLParams(PatrolParams):
    """Patrol params + cooperative reward weights and formation spacing.

    ``w_track`` weights the wingmen's slot-keeping, ``w_lead`` the lead's
    altitude+heading tracking (they should sum to 1).  ``lateral_spacing`` and
    ``back_spacing`` set the per-rank offset of the auto-generated slots.
    """

    w_track: float = 0.6
    w_lead: float = 0.4
    lateral_spacing: float = 130.0  # metres of lateral offset per rank
    back_spacing: float = 180.0  # metres of trailing offset per rank


def formation_slots(num_wingmen: int, lateral_spacing: float, back_spacing: float):
    """Return (back, right, up) arrays for a symmetric V, evenly spread both sides.

    Wingman i alternates side (even = right, odd = left) with rank i//2 + 1, so
    e.g. 4 wingmen sit at right/left/right/left of increasing offset — two per
    side — all co-altitude.
    """
    backs, rights, ups = [], [], []
    for i in range(num_wingmen):
        side = 1.0 if i % 2 == 0 else -1.0
        rank = i // 2 + 1
        rights.append(side * rank * lateral_spacing)
        backs.append(rank * back_spacing)
        ups.append(0.0)
    return (
        jnp.asarray(backs, dtype=jnp.float32),
        jnp.asarray(rights, dtype=jnp.float32),
        jnp.asarray(ups, dtype=jnp.float32),
    )


class PlanePatrolMARL:
    """Cooperative N-plane formation patrol (JaxMARL-style interface).

    ``num_wingmen`` in 1..4 gives 2..5 aircraft.  Each agent emits a 3-vector
    ``[power, stick, aileron]`` in [-1, 1]; all share the team reward.
    """

    def __init__(self, num_wingmen: int = 1, integration_method: str = "rk4_1"):
        if not (1 <= num_wingmen <= MAX_WINGMEN):
            raise ValueError(
                f"num_wingmen must be in 1..{MAX_WINGMEN} (2..{MAX_WINGMEN + 1} planes)"
            )
        self.num_wingmen = num_wingmen
        self.agents = [LEAD] + [wingman_name(i) for i in range(num_wingmen)]
        self.num_agents = len(self.agents)
        self.integration_method = integration_method
        _, self._pid_zero = make_plane3d_heading_pid()
        # Rendering state (mutated by the renderer across a rollout).
        self.screen_width = 600
        self.screen_height = 400

    @property
    def default_params(self) -> PatrolMARLParams:
        return PatrolMARLParams()

    # -- helpers -----------------------------------------------------------
    def _wingman_view(self, wingman, lead, i, state):
        """A lightweight PatrolState so the 1v1 obs/geometry helpers apply."""
        return PatrolState(
            follower=wingman,
            lead=lead,
            lead_pid=self._pid_zero,
            slot_back=state.slot_back[i],
            slot_right=state.slot_right[i],
            slot_up=state.slot_up[i],
            lead_turn_rate=0.0,
            time=0,
        )

    # -- observations ------------------------------------------------------
    def get_obs(self, state: FormationState) -> Dict[str, jnp.ndarray]:
        lead = state.lead
        # Lead sees its heading obs + the wingmen centroid offset in its frame.
        wx = jnp.mean(jnp.stack([w.x for w in state.wingmen]))
        wy = jnp.mean(jnp.stack([w.y for w in state.wingmen]))
        wz = jnp.mean(jnp.stack([w.z for w in state.wingmen]))
        psi = lead.psi
        fwd = jnp.array([jnp.cos(psi), jnp.sin(psi)])
        rgt = jnp.array([jnp.sin(psi), -jnp.cos(psi)])
        d = jnp.array([wx - lead.x, wy - lead.y])
        lead_obs = jnp.concatenate(
            [
                get_obs_heading(lead),
                jnp.stack([jnp.dot(d, fwd), jnp.dot(d, rgt), wz - lead.z]),
            ]
        )
        obs = {LEAD: lead_obs}
        for i, w in enumerate(state.wingmen):
            obs[wingman_name(i)] = get_obs_full(self._wingman_view(w, lead, i, state))
        return obs

    # -- reward & termination ---------------------------------------------
    def _reward_and_terminal(self, state: FormationState, params: PatrolMARLParams):
        lead = state.lead
        # Log-scaled in both errors, matching the 2D/3D plane tasks: an
        # envelope-normalised reward spends its whole dynamic range on errors
        # the lead has already eliminated (see ``log_scaled_reward``).
        alt_r = log_scaled_reward(
            jnp.abs(lead.target_altitude - lead.z),
            params.precision_floor,
            params.max_alt - params.min_alt,
        )
        hdg_r = log_scaled_reward(
            jnp.abs(wrap_angle(lead.psi - lead.target_heading)),
            params.heading_precision_floor,
            jnp.pi,
        )
        lead_r = alt_r * hdg_r

        track_terms, errs = [], []
        for i, w in enumerate(state.wingmen):
            view = self._wingman_view(w, lead, i, state)
            err = slot_error(view)
            errs.append(err)
            track_terms.append(
                jnp.exp(-0.5 * (err / params.slot_tolerance) ** 2)
                * heading_alignment(view, params)
            )
        track = jnp.mean(jnp.stack(track_terms))
        team = params.w_track * track + params.w_lead * lead_r

        planes = [lead] + list(state.wingmen)
        crash = jnp.zeros((), dtype=bool)
        for p in planes:
            crash = crash | (p.z <= params.min_alt) | (p.z >= params.max_alt)
        lost = jnp.max(jnp.stack(errs)) >= params.max_slot_error
        # Minimum pairwise separation.
        min_sep = jnp.array(jnp.inf)
        for a in range(len(planes)):
            for b in range(a + 1, len(planes)):
                pa, pb = planes[a], planes[b]
                d = jnp.sqrt(
                    (pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2 + 1e-8
                )
                min_sep = jnp.minimum(min_sep, d)
        collision = min_sep <= params.min_separation

        terminated = crash | lost | collision
        reward = jnp.where(terminated, -1.0 * params.max_steps_in_episode, team)
        return reward, terminated

    # -- reset -------------------------------------------------------------
    def reset(self, key: chex.PRNGKey, params: PatrolMARLParams = None):
        if params is None:
            params = self.default_params
        key, alt_key, hdg_key, noise_key = jax.random.split(key, 4)
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
        speed = params.initial_x_dot
        back, right, up = formation_slots(
            self.num_wingmen, params.lateral_spacing, params.back_spacing
        )

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
        cpsi, spsi = jnp.cos(lead_heading), jnp.sin(lead_heading)
        wingmen = []
        noise_keys = jax.random.split(noise_key, self.num_wingmen)
        for i in range(self.num_wingmen):
            # World slot position for wingman i.
            fwd = jnp.array([cpsi, spsi])
            rgt = jnp.array([spsi, -cpsi])
            xy = jnp.array([0.0, 0.0]) - back[i] * fwd + right[i] * rgt
            noise = params.follower_spawn_noise * jax.random.normal(noise_keys[i], (3,))
            wingmen.append(
                make_cruise_state(
                    x=xy[0] + noise[0],
                    y=xy[1] + noise[1],
                    z=lead_alt + up[i] + noise[2],
                    psi=lead_heading,
                    speed=speed,
                    params=params,
                    target_altitude=lead_alt,
                    target_heading=lead_heading,
                )
            )
        state = FormationState(
            lead=lead,
            wingmen=tuple(wingmen),
            slot_back=back,
            slot_right=right,
            slot_up=up,
            time=0,
        )
        return self.get_obs(state), state

    # -- step --------------------------------------------------------------
    def step_env(
        self,
        key: chex.PRNGKey,
        state: FormationState,
        actions: Dict[str, jnp.ndarray],
        params: PatrolMARLParams = None,
    ):
        if params is None:
            params = self.default_params
        method = self.integration_method

        # One shared turbulence gust for the whole formation (same air mass),
        # advanced with the step key and folded into eff_params.wind; each
        # aircraft still gets its own altitude-dependent shear in the engine.
        gust = advance_gust(
            jnp.array([state.gust_x, state.gust_y, state.gust_z]),
            params.turbulence_theta,
            params.turbulence_sigma,
            params.delta_t,
            key,
        )
        eff_params = params.replace(
            wind_x=params.wind_x + gust[0],
            wind_y=params.wind_y + gust[1],
            wind_z=params.wind_z + gust[2],
        )

        lp, ls, la = decode_action(actions[LEAD])
        new_lead, _ = compute_next_state_3d(
            lp, ls, la, state.lead, eff_params, integration_method=method
        )
        new_wingmen = []
        for i, w in enumerate(state.wingmen):
            wp, ws, wa = decode_action(actions[wingman_name(i)])
            nw, _ = compute_next_state_3d(
                wp, ws, wa, w, eff_params, integration_method=method
            )
            new_wingmen.append(nw)
        new_state = state.replace(
            lead=new_lead,
            wingmen=tuple(new_wingmen),
            time=state.time + 1,
            gust_x=gust[0],
            gust_y=gust[1],
            gust_z=gust[2],
        )

        reward, terminated = self._reward_and_terminal(new_state, params)
        truncated = new_state.time >= params.max_steps_in_episode
        done = terminated | truncated
        rewards = {a: reward for a in self.agents}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        info = {"terminated": terminated}
        return self.get_obs(new_state), new_state, rewards, dones, info

    def step(
        self,
        key: chex.PRNGKey,
        state: FormationState,
        actions: Dict[str, jnp.ndarray],
        params: PatrolMARLParams = None,
    ):
        """JaxMARL-style auto-resetting step."""
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, rewards, dones, info = self.step_env(
            key, state, actions, params
        )
        obs_re, state_re = self.reset(key_reset, params)
        state = jax.tree.map(
            lambda re, st: jax.lax.select(dones["__all__"], re, st), state_re, state_st
        )
        obs = {
            a: jax.lax.select(dones["__all__"], obs_re[a], obs_st[a])
            for a in self.agents
        }
        return obs, state, rewards, dones, info

    # -- spaces ------------------------------------------------------------
    def action_space(self, agent: str = None) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32)

    def observation_space(self, agent: str) -> spaces.Box:
        inf = jnp.finfo(jnp.float32).max
        dim = 18 if agent == LEAD else 26
        return spaces.Box(-inf, inf, (dim,), dtype=jnp.float32)

    @property
    def observation_spaces(self) -> Dict[str, spaces.Box]:
        return {a: self.observation_space(a) for a in self.agents}

    @property
    def action_spaces(self) -> Dict[str, spaces.Box]:
        return {a: self.action_space(a) for a in self.agents}

    # -- rendering ---------------------------------------------------------
    def render(self, screen, state: FormationState, params, frames, clock):
        if params is None:
            params = self.default_params
        if state is None:
            return _render_scene(self, screen, params, frames, clock, None, [], [], 0.0)
        reward, _ = self._reward_and_terminal(state, params)
        slots = [
            (
                float(state.slot_back[i]),
                float(state.slot_right[i]),
                float(state.slot_up[i]),
            )
            for i in range(self.num_wingmen)
        ]
        return _render_scene(
            self,
            screen,
            params,
            frames,
            clock,
            state.lead,
            list(state.wingmen),
            slots,
            float(reward),
        )
