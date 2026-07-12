"""
Rendering for the close-patrol / formation environments.

Reuses the **Plane 3D render engine** (the projected 3D aircraft models) in a
two-panel layout, and draws an arbitrary number of aircraft (1 lead + K
wingmen):

  - **Left — rear / chase view.**  The camera sits behind the formation and
    looks *along the lead's heading*, so we see every aircraft from behind.
    Horizontal = lateral offset (right of the lead), vertical = altitude, and
    each aircraft is drawn at its heading/bank relative to the lead — so the
    lateral spacing, vertical spacing and orientation differences read directly
    and it never disagrees with the top-down panel.
  - **Right — top-down view** (world x-y).

A coloured ID ring under each aircraft distinguishes the lead (blue) from the
wingmen (a fixed palette); green rings mark the slots.  The HUD reports the
mean slot error, the minimum pairwise separation (red on collision), and the
instantaneous + cumulative reward.
"""

import numpy as np
import pygame
from pygame import gfxdraw

from target_gym.plane.rendering import draw_cloud, draw_dashed_line
from target_gym.plane3d.rendering import (
    _draw_hud_box,
    _draw_plane_topdown,
    _render_3d_plane,
    _rotation_matrix,
)

_LEAD = (40, 90, 210)
_SLOT = (30, 170, 90)
# Distinct wingman ID-ring colours (cycled if there are more wingmen).
_WING_COLORS = [
    (240, 130, 30),  # orange
    (150, 70, 200),  # purple
    (30, 165, 165),  # teal
    (210, 55, 140),  # magenta
]


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _draw_plane_rearview(surf, cx, cy, theta, phi, psi_rel, scale_px=1.0):
    """Draw the 3D plane as seen from *behind*, along its forward (+x) axis."""
    R = _rotation_matrix(theta, phi, psi_rel)
    camera_dir = np.array([-1, 0, 0])  # behind the aircraft, looking forward

    def project(pt, s):
        return (int(cx + pt[1] * s), int(cy - pt[2] * s))  # y=right, z=up

    _render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project)


def _id_ring(surf, sx, sy, color, r=17):
    gfxdraw.aacircle(surf, int(sx), int(sy), r, color)
    gfxdraw.aacircle(surf, int(sx), int(sy), r - 1, color)


def _slot_world(lead, back, right, up):
    """World (x, y, z) of a slot given (back, right, up) in the lead frame."""
    psi = float(lead.psi)
    fwd = np.array([np.cos(psi), np.sin(psi)])
    rgt = np.array([np.sin(psi), -np.cos(psi)])
    xy = np.array([float(lead.x), float(lead.y)]) - back * fwd + right * rgt
    return xy[0], xy[1], float(lead.z) + up


def _render_scene(cls, screen, params, frames, clock, lead, wingmen, slots, reward):
    """Draw the lead + K wingmen.  ``wingmen`` and ``slots`` are equal-length
    lists; each slot is a (back, right, up) tuple in the lead body frame."""
    panel_w, panel_h = cls.screen_width, cls.screen_height
    total_w = panel_w * 2

    if screen is None:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((total_w, panel_h))
        cls.trails = None
        cls.cum_reward = 0.0
        rng = np.random.default_rng(42)
        cls.cloud_positions = [
            (
                rng.integers(0, panel_w),
                rng.integers(40, panel_h // 2),
                rng.uniform(0.5, 1.4),
            )
            for _ in range(7)
        ]
        cls.font = pygame.font.SysFont("arial", 14)
    if clock is None:
        clock = pygame.time.Clock()
    if lead is None:
        return frames, screen, clock

    cls.cum_reward += reward
    n = len(wingmen)
    planes = [lead] + list(wingmen)  # index 0 = lead
    colors = [_LEAD] + [_WING_COLORS[i % len(_WING_COLORS)] for i in range(n)]
    pos = np.array([[float(p.x), float(p.y), float(p.z)] for p in planes])
    slot_world = [_slot_world(lead, b, r, u) for (b, r, u) in slots]
    if cls.trails is None:
        cls.trails = [[] for _ in planes]
    for i, p in enumerate(planes):
        cls.trails[i].append((float(p.x), float(p.y), float(p.z)))
    plane_px = max(0.8, min(1.6, panel_w * 0.0022))

    l_psi = float(lead.psi)
    cpsi, spsi = np.cos(l_psi), np.sin(l_psi)

    def lateral(px, py):
        return px * spsi - py * cpsi

    def _trail(surf, tr, w2s, color, project):
        stride = max(1, len(tr) // 300)
        for wp in tr[::stride]:
            px, py = w2s(*project(wp))
            if 0 <= px < panel_w and 0 <= py < panel_h:
                gfxdraw.filled_circle(surf, int(px), int(py), 1, color)

    # ── Left: rear / chase view (lateral vs altitude, from behind) ───────
    side = pygame.Surface((panel_w, panel_h))
    side.fill((135, 206, 235))
    lats = [lateral(p[0], p[1]) for p in pos]
    alts = [p[2] for p in pos]
    sc = panel_h * 0.42 / 360.0
    cu = 0.5 * (min(lats) + max(lats))
    cv = 0.5 * (min(alts) + max(alts))

    def side_w2s(wu, wz):
        return (panel_w / 2 + (wu - cu) * sc, panel_h / 2 - (wz - cv) * sc)

    for ccx, ccy, cscale in cls.cloud_positions:
        draw_cloud(
            side,
            ccx,
            ccy,
            scale=cscale * 0.7,
            seed=int(ccx),
            color=(200, 220, 240),
            outline_color=(160, 180, 200),
            outline_thickness=2,
        )
    _, gy = side_w2s(0, 0)
    if gy < panel_h:
        pygame.draw.rect(side, (100, 160, 80), (0, int(gy), panel_w, panel_h - int(gy)))
    for i, p in enumerate(planes):
        shade = tuple(int(c * 0.4) for c in colors[i])
        _trail(
            side, cls.trails[i], side_w2s, shade, lambda q: (lateral(q[0], q[1]), q[2])
        )
    for b, r, u in slots:
        wx, wy, wz = _slot_world(lead, b, r, u)
        px, py = side_w2s(lateral(wx, wy), wz)
        pygame.draw.circle(side, _SLOT, (int(px), int(py)), 7, 2)
    for i, p in enumerate(planes):
        px, py = side_w2s(lateral(p.x, p.y), float(p.z))
        _id_ring(side, px, py, colors[i])
        _draw_plane_rearview(
            side,
            px,
            py,
            float(p.theta),
            float(p.phi),
            _wrap(float(p.psi) - l_psi),
            scale_px=plane_px,
        )
    side.blit(
        cls.font.render("behind the lead →", True, (60, 60, 90)), (10, panel_h - 22)
    )

    # ── Right: top-down view (x-y) ───────────────────────────────────────
    top = pygame.Surface((panel_w, panel_h))
    top.fill((120, 170, 90))
    for i in range(0, panel_h, 40):
        shade = 120 + (i % 80 == 0) * 10
        gfxdraw.hline(top, 0, panel_w, i, (shade - 10, shade + 30, shade - 30))
    sc = panel_h * 0.42 / (360.0 + 90.0 * n)  # zoom out a little as planes are added
    cu = 0.5 * (pos[:, 0].min() + pos[:, 0].max())
    cv = 0.5 * (pos[:, 1].min() + pos[:, 1].max())

    def top_w2s(wx, wy):
        return (panel_w / 2 + (wx - cu) * sc, panel_h / 2 - (wy - cv) * sc)

    for i, p in enumerate(planes):
        shade = tuple(int(c * 0.4) for c in colors[i])
        _trail(top, cls.trails[i], top_w2s, shade, lambda q: (q[0], q[1]))
    for wi, (b, r, u) in enumerate(slots):
        wx, wy, wz = slot_world[wi]
        tsx, tsy = top_w2s(wx, wy)
        fpx, fpy = top_w2s(float(wingmen[wi].x), float(wingmen[wi].y))
        draw_dashed_line(top, _SLOT, (int(fpx), int(fpy)), (int(tsx), int(tsy)), 8, 6)
        pygame.draw.circle(top, _SLOT, (int(tsx), int(tsy)), 7, 2)
    for i, p in enumerate(planes):
        px, py = top_w2s(float(p.x), float(p.y))
        _id_ring(top, px, py, colors[i])
        _draw_plane_topdown(
            top, px, py, float(p.theta), float(p.phi), float(p.psi), scale_px=plane_px
        )
    lpx, lpy = top_w2s(float(lead.x), float(lead.y))
    top.blit(cls.font.render("Lead", True, _LEAD), (int(lpx) + 18, int(lpy) + 16))
    pygame.draw.line(top, (0, 0, 0), (0, 0), (0, panel_h), 2)

    # ── Metrics ──────────────────────────────────────────────────────────
    slot_errs = [
        float(np.linalg.norm(np.array([float(w.x), float(w.y), float(w.z)]) - sw))
        for w, sw in zip(wingmen, slot_world)
    ]
    mean_err = float(np.mean(slot_errs)) if slot_errs else 0.0
    # Minimum pairwise separation across all aircraft.
    min_sep = np.inf
    for a in range(len(pos)):
        for b in range(a + 1, len(pos)):
            min_sep = min(min_sep, float(np.linalg.norm(pos[a] - pos[b])))
    collision = min_sep <= float(params.min_separation)

    # ── Compose + HUD ────────────────────────────────────────────────────
    combined = pygame.Surface((total_w, panel_h))
    combined.blit(side, (0, 0))
    combined.blit(top, (panel_w, 0))
    left = [
        cls.font.render(f"Mean slot err: {mean_err:6.1f} m", True, _WING_COLORS[0]),
        cls.font.render(
            f"Min separation: {min_sep:6.1f} m",
            True,
            (200, 40, 40) if collision else (0, 0, 0),
        ),
        cls.font.render(f"Alt (lead): {int(lead.z):,} m", True, (0, 0, 0)),
    ]
    mid = [
        cls.font.render(
            f"Planes: {len(planes)}  (1 lead + {n} wingmen)", True, (0, 0, 0)
        ),
        cls.font.render(
            "COLLISION!" if collision else f"Time: {int(lead.time)}",
            True,
            (200, 40, 40) if collision else (0, 0, 0),
        ),
    ]
    right = [
        cls.font.render(
            f"Reward: {reward:+.3f}",
            True,
            (200, 40, 40) if reward < 0 else (20, 130, 40),
        ),
        cls.font.render(f"Cum. reward: {cls.cum_reward:+.1f}", True, (0, 0, 0)),
    ]
    _draw_hud_box(combined, [left, mid, right], total_w)

    screen.blit(combined, (0, 0))
    pygame.display.flip()
    frame = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    frames.append(frame)
    return frames, screen, clock


def _render(cls, screen, state, params, frames, clock):
    """Adapter for the single-agent :class:`PatrolState` (1 lead + 1 follower)."""
    from target_gym.patrol.env import compute_reward_patrol

    if state is None:
        return _render_scene(cls, screen, params, frames, clock, None, [], [], 0.0)
    reward = float(compute_reward_patrol(state, params))
    slots = [(float(state.slot_back), float(state.slot_right), float(state.slot_up))]
    return _render_scene(
        cls, screen, params, frames, clock, state.lead, [state.follower], slots, reward
    )
