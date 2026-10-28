"""
Rendering for the 3D airplane environment.

Two-panel layout:
  Left  - Side view (x-z plane), centered on aircraft, adaptive scale,
          3D aircraft model projected onto x-z plane
  Right - Top-down view (x-y plane) with task-specific overlay,
          green ground background

Both panels share the dotted trail and the console chrome. The palette, the
gauge stack, the A320 mesh and its projections all live in
:mod:`target_gym.render_aircraft`, which ``patrol`` draws with as well; this
module is only the two scenes and the layout that composes them.
"""

import numpy as np

from target_gym import render_aircraft as ra
from target_gym.render_kit import frame_stride, render_extra

pygame = render_extra("pygame")
gfxdraw = render_extra("pygame.gfxdraw")

# ─────────────────────────────────────────────────────────
#  Side-view scene (x-z, centered, adaptive scale)
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
#  Top-down scene (task-aware overlays)
# ─────────────────────────────────────────────────────────


def render_topdown_scene(
    panel_w,
    panel_h,
    state,
    params,
    positions_history_xy,
    cloud_positions,
    max_steps,
    task_type="heading",
):
    """Top-down view (x-y) with task-specific overlay."""
    surf = pygame.Surface((panel_w, panel_h))
    # Ground color for top-down view (looking down at terrain)
    surf.fill(ra.GROUND)  # top-down terrain; was (120, 170, 90) grass green

    scene_top = ra.HEADER_H
    scene_bot = panel_h - ra.gauge_height(4)
    scene_h = scene_bot - scene_top
    cx, cy = panel_w // 2, (scene_top + scene_bot) // 2

    # No terrain banding. It was a green hatch left from when this panel drew
    # grass, and against the console palette it read as interference rather
    # than ground. The scale bar carries the distance information instead.
    font_sm = pygame.font.SysFont(ra.MONO, 10)

    # Compute scale from trail history
    scale = 0.5  # default
    cur_x, cur_y = float(state.x), float(state.y)
    if len(positions_history_xy) > 1:
        xs = [p[0] for p in positions_history_xy]
        ys = [p[1] for p in positions_history_xy]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        # For circle/figure-8, also consider target radius
        if task_type in ("circle", "figure8") and float(state.target_radius) > 0:
            span = max(span, float(state.target_radius) * 2.5)
        # The hold is 2 * r * (_RACETRACK_LEG + 1) across its long axis, which
        # for the shipped leg length is 6r -- the circle's 2.5r allowance would
        # crop both legs off the panel and leave a picture of the caps alone.
        elif task_type == "racetrack" and float(state.target_radius) > 0:
            from target_gym.plane3d.env import _RACETRACK_LEG

            span = max(span, float(state.target_radius) * (_RACETRACK_LEG + 1) * 2.4)
        # 0.62 of the panel, not 0.4: the pattern is the subject of this view
        # and it was being drawn barely larger than the aircraft on it.
        scale = min(panel_w, scene_h) * 0.62 / span
        scale = min(scale, 0.5)

    def world_to_screen(wx, wy):
        sx = int(cx + (wx - cur_x) * scale)
        sy = int(cy - (wy - cur_y) * scale)  # y-flip
        return sx, sy

    ra.world_grid(
        surf, panel_w, scene_top, scene_bot, scale, cur_x, cur_y, world_to_screen
    )

    # -- trail --
    if len(positions_history_xy) > 1:
        stride = max(1, len(positions_history_xy) // 300)
        pts = positions_history_xy[::stride]
        for wx, wy in pts:
            sx, sy = world_to_screen(wx, wy)
            if 0 <= sx < panel_w and scene_top <= sy < scene_bot:
                gfxdraw.circle(surf, sx, sy, 2, ra.SCENE_DIM)
                gfxdraw.circle(surf, sx, sy, 1, ra.ACCENT)

    # -- task-specific overlay --
    if task_type == "heading":
        # Target heading (dashed red line)
        heading_len = min(panel_w, scene_h) * 0.4
        ra.draw_heading_dashed(
            surf,
            cx,
            cy,
            state.target_heading,
            heading_len,
            color=ra.TARGET,
            dash=10,
            gap=8,
        )
        # Current heading indicator (solid blue)
        ind_len = heading_len * 0.55
        sa = -state.psi
        ex = int(cx + ind_len * np.cos(sa))
        ey = int(cy + ind_len * np.sin(sa))
        pygame.draw.line(surf, ra.BLUE, (cx, cy), (ex, ey), 2)

    elif task_type == "circle":
        # Draw the target circle
        center_sx, center_sy = world_to_screen(
            float(state.target_x), float(state.target_y)
        )
        radius_px = int(float(state.target_radius) * scale)
        if radius_px > 2:
            pygame.draw.circle(
                surf,
                ra.TARGET,
                (center_sx, center_sy),
                radius_px,
                2,
            )
        # Small cross at center
        sz = 6
        pygame.draw.line(
            surf,
            ra.TARGET,
            (center_sx - sz, center_sy),
            (center_sx + sz, center_sy),
            1,
        )
        pygame.draw.line(
            surf,
            ra.TARGET,
            (center_sx, center_sy - sz),
            (center_sx, center_sy + sz),
            1,
        )

    elif task_type == "racetrack":
        # Two straight legs joined by two 180-degree caps, built in the pattern
        # frame and rotated out -- the same construction distance_to_racetrack
        # scores, so the drawn shape and the rewarded shape cannot disagree.
        from target_gym.plane3d.env import _RACETRACK_LEG

        r = float(state.target_radius)
        half_leg = r * _RACETRACK_LEG
        rcx, rcy = float(state.target_x), float(state.target_y)
        cos_o, sin_o = (
            np.cos(float(state.target_heading)),
            np.sin(float(state.target_heading)),
        )

        # Round the loop once: top leg, right cap, bottom leg, left cap. The
        # caps sweep through 0 and through pi respectively, so the four pieces
        # join end to end and the polyline closes without a seam.
        right_cap = np.linspace(np.pi / 2.0, -np.pi / 2.0, 60)
        left_cap = np.linspace(-np.pi / 2.0, -3.0 * np.pi / 2.0, 60)
        us = np.concatenate(
            [
                np.array([-half_leg, half_leg]),
                half_leg + r * np.cos(right_cap),
                np.array([half_leg, -half_leg]),
                -half_leg + r * np.cos(left_cap),
            ]
        )
        vs = np.concatenate(
            [
                np.array([r, r]),
                r * np.sin(right_cap),
                np.array([-r, -r]),
                r * np.sin(left_cap),
            ]
        )
        track_x = rcx + us * cos_o - vs * sin_o
        track_y = rcy + us * sin_o + vs * cos_o
        for i in range(len(us)):
            j = (i + 1) % len(us)
            x1, y1 = world_to_screen(track_x[i], track_y[i])
            x2, y2 = world_to_screen(track_x[j], track_y[j])
            pygame.draw.line(surf, ra.TARGET, (x1, y1), (x2, y2), 2)

        # Cross at the pattern centre, as the circle and figure-8 both carry.
        csx, csy = world_to_screen(rcx, rcy)
        sz = 6
        pygame.draw.line(surf, ra.TARGET, (csx - sz, csy), (csx + sz, csy), 1)
        pygame.draw.line(surf, ra.TARGET, (csx, csy - sz), (csx, csy + sz), 1)

    elif task_type == "figure8":
        # Draw the rotated lemniscate (Bernoulli parametrization)
        a = float(state.target_radius)
        fcx, fcy = float(state.target_x), float(state.target_y)
        orientation = float(state.target_heading)
        cos_o, sin_o = np.cos(orientation), np.sin(orientation)
        tau = np.linspace(0, 2.0 * np.pi, 400)
        denom = 1.0 + np.sin(tau) ** 2
        base_x = a * np.cos(tau) / denom
        base_y = a * np.sin(tau) * np.cos(tau) / denom
        curve_x = fcx + base_x * cos_o - base_y * sin_o
        curve_y = fcy + base_x * sin_o + base_y * cos_o
        for i in range(len(tau) - 1):
            x1, y1 = world_to_screen(curve_x[i], curve_y[i])
            x2, y2 = world_to_screen(curve_x[i + 1], curve_y[i + 1])
            pygame.draw.line(surf, ra.TARGET, (x1, y1), (x2, y2), 2)
        # Small cross at center
        csx, csy = world_to_screen(fcx, fcy)
        sz = 6
        pygame.draw.line(surf, ra.TARGET, (csx - sz, csy), (csx + sz, csy), 1)
        pygame.draw.line(surf, ra.TARGET, (csx, csy - sz), (csx, csy + sz), 1)

        # Nearest-point marker (orange dot) — shows where the aircraft
        # should be on the curve.
        from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

        ndx, ndy, _, _, _ = nearest_point_on_twisted_lemniscate(state, params)
        npx = float(state.x) + float(ndx)
        npy = float(state.y) + float(ndy)
        nsx, nsy = world_to_screen(npx, npy)
        pygame.draw.circle(surf, ra.ACCENT, (nsx, nsy), 6)
        pygame.draw.circle(surf, ra.SCENE_INK, (nsx, nsy), 6, 1)

    # -- aircraft --
    # Smaller than in the side panel. Here the glyph shares the frame with the
    # commanded pattern, and at the side panel's size a 37 m aircraft drew as
    # long as an 8.4 km lobe, which left the figure-8 looking like a doodle
    # beside it.
    plane_px_scale = max(0.8, min(1.6, panel_w * 0.0018))
    ra.draw_plane_topdown(
        surf,
        cx,
        cy,
        float(state.theta),
        float(state.phi),
        float(state.psi),
        scale_px=plane_px_scale,
    )

    # -- compass labels --
    # Pinned to the edges of the scene band rather than to a radius about the
    # centre, which put N underneath the HUD at this aspect ratio.
    font_sm = pygame.font.SysFont(ra.MONO, 13)
    m = 14
    for label, (tx, ty) in (
        ("N", (cx, scene_top + m)),
        ("S", (cx, scene_bot - m)),
        ("E", (panel_w - m, cy)),
        ("W", (m, cy)),
    ):
        # Was (40, 40, 40): chosen when this panel had a light green
        # ground, and left unreadable when the palette moved to the
        # shared dark console scheme.
        txt = font_sm.render(label, True, ra.SCENE_DIM)
        surf.blit(txt, (tx - txt.get_width() // 2, ty - txt.get_height() // 2))

    # -- HUD --
    heading_deg = ra.compass_deg(state.psi)
    bank_deg = float(np.rad2deg(state.phi))
    ail_deg = float(np.rad2deg(state.aileron))

    err_label, err_value, err_frac = "TRACK ERR", "--", 0.0
    if task_type == "heading":
        d = abs(
            (ra.compass_deg(state.target_heading) - heading_deg + 180.0) % 360.0 - 180.0
        )
        err_label, err_value, err_frac = "HDG ERR", f"{d:.0f}\u00b0", d / 180.0
    elif task_type in ("circle", "racetrack", "figure8"):
        if task_type == "circle":
            from target_gym.plane3d.env import distance_to_circle

            d = abs(float(distance_to_circle(state)))
        elif task_type == "racetrack":
            from target_gym.plane3d.env import distance_to_racetrack

            d = abs(float(distance_to_racetrack(state)))
        else:
            from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

            _, _, _, d, _ = nearest_point_on_twisted_lemniscate(state, params)
            d = abs(float(d))
        err_label, err_value = "TRACK ERR", f"{int(d):,} m"
        err_frac = min(1.0, d / max(float(state.target_radius), 1.0))

    ra.gauge_stack(
        surf,
        panel_w,
        panel_h,
        [
            (
                "HEADING",
                f"{heading_deg:.0f}\u00b0",
                heading_deg / 360.0,
                ra.ACCENT,
                None,
            ),
            (
                "BANK",
                f"{bank_deg:+.1f}\u00b0",
                (bank_deg + 45.0) / 90.0,
                ra.BLUE,
                0.5,
                0.5,
            ),
            (
                "AILERON",
                f"{ail_deg:+.0f}\u00b0",
                (ail_deg + 30.0) / 60.0,
                ra.TEAL,
                0.5,
                0.5,
            ),
            (
                "TRACK ERR" if err_label == "TRACK ERR" else err_label,
                err_value,
                err_frac,
                ra.ALERT if err_frac > 0.25 else ra.GOOD,
                None,
            ),
        ],
    )

    ra.gauge_stack(
        surf,
        panel_w,
        panel_h,
        [
            (
                "HEADING",
                f"{heading_deg:.0f}\u00b0",
                heading_deg / 360.0,
                ra.ACCENT,
                None,
            ),
            (
                "BANK",
                f"{bank_deg:+.1f}\u00b0",
                (bank_deg + 45.0) / 90.0,
                ra.BLUE,
                0.5,
                0.5,
            ),
            (
                "AILERON",
                f"{ail_deg:+.0f}\u00b0",
                (ail_deg + 30.0) / 60.0,
                ra.TEAL,
                0.5,
                0.5,
            ),
            (
                "TRACK ERR" if err_label == "TRACK ERR" else err_label,
                err_value,
                err_frac,
                ra.ALERT if err_frac > 0.25 else ra.GOOD,
                None,
            ),
        ],
    )

    # -- legend (bottom) --
    legend_y = scene_top + 14
    font_sm2 = pygame.font.SysFont(ra.MONO, 14)
    if task_type == "heading":
        pygame.draw.line(surf, ra.TARGET, (8, legend_y), (28, legend_y), 2)
        surf.blit(font_sm2.render("Target hdg", True, ra.SCENE_INK), (32, legend_y - 7))
    elif task_type == "circle":
        pygame.draw.circle(surf, ra.TARGET, (18, legend_y), 8, 2)
        surf.blit(
            font_sm2.render("Target circle", True, ra.SCENE_INK), (32, legend_y - 7)
        )
    elif task_type == "racetrack":
        pygame.draw.line(surf, ra.TARGET, (8, legend_y), (28, legend_y), 2)
        surf.blit(
            font_sm2.render("Holding pattern", True, ra.SCENE_INK), (32, legend_y - 7)
        )
    elif task_type == "figure8":
        pygame.draw.line(surf, ra.TARGET, (8, legend_y), (28, legend_y), 2)
        surf.blit(
            font_sm2.render("Target figure-8", True, ra.SCENE_INK), (32, legend_y - 7)
        )

    ra.draw_scale_bar(surf, scale, panel_w, scene_bot)
    return surf


# ─────────────────────────────────────────────────────────
#  Combined renderer (called by env_jax via classmethod)
# ─────────────────────────────────────────────────────────


def _render(cls, screen, state, params, frames, clock):
    """Two-panel renderer: side view (left) + top-down view (right)."""
    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    panel_w = cls.screen_width
    panel_h = cls.screen_height
    total_w = panel_w * 2

    if screen is None:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((total_w, panel_h))
        cls.positions_history_xz = []
        cls.positions_history_xy = []

        rng = np.random.default_rng(42)
        cloud_positions = []
        for _ in range(8):
            _cx = rng.integers(0, panel_w)
            _cy = rng.integers(50, panel_h // 2)
            _scale = rng.uniform(0.5, 1.5)
            _shape = rng.integers(0, params.max_steps_in_episode)
            cloud_positions.append((_cx, _cy, _scale, _shape))
        cls.cloud_positions = cloud_positions
        cls.screen = screen

    if clock is None:
        clock = pygame.time.Clock()
        cls.clock = clock

    if state is None:
        return None

    # Determine task type from the env class
    task_type = getattr(cls, "task_type", "heading")

    # Trail history advances every step, so the dotted track stays dense even
    # though frames are sampled.
    cls.positions_history_xz.append((float(state.x), float(state.z)))
    cls.positions_history_xy.append((float(state.x), float(state.y)))

    # Sample frames instead of emitting one per step. This renderer appended
    # every single step, so a 900-step hold produced 450 frames and a 7 MB gif
    # against the 1-2.5 MB the rest of the gallery runs at, and drawing two
    # pygame panels per step dominated the render cost.
    stride = frame_stride(params)
    if not (state.time % stride == 0 or state.time <= 1 or not frames):
        return frames, screen, clock

    # Left panel: side view
    side_surf = ra.side_scene(panel_w, panel_h, state, params, cls.positions_history_xz)

    # Right panel: top-down view
    top_surf = render_topdown_scene(
        panel_w,
        panel_h,
        state,
        params,
        cls.positions_history_xy,
        cls.cloud_positions,
        max_steps=params.max_steps_in_episode,
        task_type=task_type,
    )

    # Thin divider between panels
    pygame.draw.line(top_surf, ra.FRAME, (0, 0), (0, panel_h), 2)

    # Compose
    combined = pygame.Surface((total_w, panel_h))
    combined.blit(side_surf, (0, 0))
    combined.blit(top_surf, (panel_w, 0))

    # Display and capture
    screen.blit(combined, (0, 0))
    pygame.display.flip()
    frame = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    frames.append(frame)
    cls.frames = frames

    return frames, screen, clock
