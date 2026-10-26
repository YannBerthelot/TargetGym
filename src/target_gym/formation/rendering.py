"""Rendering for the formation: the patrol's two-panel scene, one ID ring per
aircraft and a slot ring per wingman."""

import jax

from target_gym.patrol.rendering import _render_scene


def _render(cls, screen, state, params, frames, clock):
    """Adapter from :class:`FormationState` to the patrol's scene."""
    from target_gym.formation.env import compute_reward

    if state is None:
        return _render_scene(cls, screen, params, frames, clock, None, [], [], 0.0)
    k = state.slot_back.shape[0]
    wingmen = [jax.tree.map(lambda x, i=i: x[i], state.wingmen) for i in range(k)]
    slots = [
        (float(state.slot_back[i]), float(state.slot_right[i]), float(state.slot_up[i]))
        for i in range(k)
    ]
    reward = float(compute_reward(state, params))
    return _render_scene(
        cls, screen, params, frames, clock, state.lead, wingmen, slots, reward
    )
