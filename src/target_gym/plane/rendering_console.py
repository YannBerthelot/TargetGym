"""
Rendering for the 2D airplane environments.

Two panels, both of them straight out of :mod:`target_gym.render_aircraft`:

  - **Left  - side elevation**, the same :func:`render_aircraft.side_scene` the
    3D tasks draw as their left panel, showing the aircraft against its
    commanded altitude on an adaptive scale.
  - **Right - the episode's trace**, altitude against its command over every
    step. The 3D tasks put a map here; a 2D aircraft has no lateral motion, and
    what the task actually is -- hold a level, follow a ramp, chase a sinusoid
    -- only shows over the whole episode.

This module used to draw its own console in matplotlib, with its own palette,
its own projection of the A320 mesh, its own parallax clouds and its own gauge
layout. That was a second implementation of the same picture, and the two drifted:
the 2D tasks kept a night-time sky and metres while the 3D ones moved to daylight
and feet. Everything shared now lives in the kit, and what is left here is the
composition.
"""

import numpy as np

from target_gym import render_aircraft as ra
from target_gym.render_kit import frame_stride, render_extra

pygame = render_extra("pygame")

HISTORY_KEYS = ("t", "altitude", "target", "speed", "pitch", "power", "reward")


def _render(cls, screen, state, params, frames, clock):
    """Two-panel renderer: side elevation (left) + episode trace (right)."""
    panel_w = cls.screen_width
    panel_h = cls.screen_height
    total_w = panel_w * 2

    if screen is None:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((total_w, panel_h))
        cls.screen = screen
        cls.positions_history_xz = []
        cls.history = {k: [] for k in HISTORY_KEYS}

    if clock is None:
        clock = pygame.time.Clock()
        cls.clock = clock

    if state is None:
        return frames, screen, clock

    # A new episode. An empty frame list marks it whatever the environment
    # counts in ``state.time``, which is what the shared render hook keys on.
    if not frames or state.time <= 1:
        cls.positions_history_xz = []
        cls.history = {k: [] for k in HISTORY_KEYS}

    max_alt_diff = float(params.max_alt) - float(params.min_alt)
    done_alt = (state.z <= params.min_alt) or (state.z >= params.max_alt)
    reward = (
        -1.0 * float(params.max_steps_in_episode)
        if done_alt
        else (
            (max_alt_diff - abs(float(state.target_altitude) - float(state.z)))
            / max_alt_diff
        )
        ** 2
    )

    # Histories advance every step, so both traces stay dense even though the
    # frames themselves are sampled.
    cls.positions_history_xz.append((float(state.x), float(state.z)))
    cls.history["t"].append(float(state.time))
    cls.history["altitude"].append(float(state.z))
    cls.history["target"].append(float(state.target_altitude))
    cls.history["speed"].append(float(state.x_dot))
    cls.history["pitch"].append(float(state.theta))
    cls.history["power"].append(float(state.power))
    cls.history["reward"].append(reward)

    stride = frame_stride(params)
    if not (state.time % stride == 0 or state.time <= 1 or not frames):
        return frames, screen, clock

    side_surf = ra.side_scene(panel_w, panel_h, state, params, cls.positions_history_xz)
    trace_surf = ra.trace_scene(
        panel_w, panel_h, state, params, cls.history, reward=reward
    )
    pygame.draw.line(trace_surf, ra.FRAME, (0, 0), (0, panel_h), 2)

    combined = pygame.Surface((total_w, panel_h))
    combined.blit(side_surf, (0, 0))
    combined.blit(trace_surf, (panel_w, 0))

    screen.blit(combined, (0, 0))
    pygame.display.flip()
    frames.append(
        np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    )
    cls.frames = frames
    return frames, screen, clock
