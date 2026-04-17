"""
Rendering for the 3D airplane environment.

Two-panel layout:
  Left  - Side view (x-z plane), centered on aircraft, adaptive scale,
          3D aircraft model projected onto x-z plane
  Right - Top-down view (x-y plane) with task-specific overlay,
          green ground background

Both panels share dotted trail and HUD box.
"""

import time

import numpy as np
import pygame
from pygame import gfxdraw

from target_gym.plane.rendering import draw_dashed_line

# ─────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────


def _rotate_point(x, y, angle):
    """Rotate (x, y) by *angle* radians (positive = CCW)."""
    c, s = np.cos(angle), np.sin(angle)
    return x * c - y * s, x * s + y * c


def _rotate_and_translate(points, angle, cx, cy):
    return [
        (_rotate_point(x, y, angle)[0] + cx, _rotate_point(x, y, angle)[1] + cy)
        for x, y in points
    ]


def _draw_hud_box(surf, columns, screen_width, padding=8, line_height=22, hud_y=10):
    """Draw a rounded HUD box with text columns at the top of *surf*."""

    def rounded_width(col_texts, bucket=20):
        max_w = max(t.get_width() for t in col_texts)
        return ((max_w + bucket - 1) // bucket) * bucket

    col_widths = [rounded_width(col) + padding * 2 for col in columns]
    n_lines = max(len(c) for c in columns)
    hud_w = sum(col_widths)
    hud_h = n_lines * line_height + 2 * padding
    hud_x = (screen_width - hud_w) // 2

    rect = pygame.Rect(hud_x, hud_y, hud_w, hud_h)
    pygame.draw.rect(surf, (0, 0, 0), rect, border_radius=6)
    inner = rect.inflate(-6, -6)
    pygame.draw.rect(surf, (255, 255, 255), inner, border_radius=6)

    col_x = hud_x + padding
    for ci, col in enumerate(columns):
        for ri, txt in enumerate(col):
            surf.blit(txt, (col_x, hud_y + padding + ri * line_height))
        col_x += col_widths[ci]


# ─────────────────────────────────────────────────────────
#  Shared 3D plane geometry
# ─────────────────────────────────────────────────────────


def _build_plane_faces():
    """
    Build the 3D aircraft geometry in body frame (x=forward, y=right, z=up).
    Returns a list of (vertices_Nx3, base_color_rgb, name).
    Shared by both side-view and top-down projections.
    """
    L = 37.57  # A320 length
    hl = L / 2
    fw = 2.0  # fuselage half-width (y)
    fh = 2.0  # fuselage half-height (z)

    faces = []

    # Fuselage - left side (y = -fw)
    faces.append(
        (
            np.array(
                [
                    [-hl, -fw, -fh],
                    [hl * 0.65, -fw, -fh],
                    [hl * 0.8, -fw, 0],
                    [hl * 0.65, -fw, fh * 0.6],
                    [-hl, -fw, fh * 0.6],
                ]
            ),
            (220, 220, 220),
            "fuse_left",
        )
    )

    # Fuselage - right side (y = fw)
    faces.append(
        (
            np.array(
                [
                    [-hl, fw, -fh],
                    [hl * 0.65, fw, -fh],
                    [hl * 0.8, fw, 0],
                    [hl * 0.65, fw, fh * 0.6],
                    [-hl, fw, fh * 0.6],
                ]
            ),
            (220, 220, 220),
            "fuse_right",
        )
    )

    # Fuselage - top (z = fh*0.6)
    faces.append(
        (
            np.array(
                [
                    [-hl, -fw, fh * 0.6],
                    [hl * 0.65, -fw, fh * 0.6],
                    [hl * 0.8, 0, fh * 0.6],
                    [hl * 0.65, fw, fh * 0.6],
                    [-hl, fw, fh * 0.6],
                ]
            ),
            (245, 245, 245),
            "fuse_top",
        )
    )

    # Fuselage - bottom (z = -fh)
    faces.append(
        (
            np.array(
                [
                    [-hl, -fw, -fh],
                    [hl * 0.65, -fw, -fh],
                    [hl * 0.65, fw, -fh],
                    [-hl, fw, -fh],
                ]
            ),
            (180, 180, 180),
            "fuse_bot",
        )
    )

    # Nose cone
    faces.append(
        (
            np.array(
                [
                    [hl * 0.65, -fw, -fh],
                    [hl * 0.65, fw, -fh],
                    [hl * 0.65, fw, fh * 0.6],
                    [hl * 0.8, 0, 0],
                    [hl * 0.65, -fw, fh * 0.6],
                ]
            ),
            (200, 200, 200),
            "nose",
        )
    )

    # Wings
    ws = 17.9  # half wingspan
    wc_root = 6.0
    wc_tip = 2.5
    faces.append(
        (
            np.array(
                [
                    [-wc_root * 0.3, -fw, -fh * 0.3],
                    [wc_root * 0.7, -fw, -fh * 0.3],
                    [wc_tip * 0.7, -ws, -fh * 0.3],
                    [-wc_tip * 0.3, -ws, -fh * 0.3],
                ]
            ),
            (190, 190, 190),
            "wing_left",
        )
    )

    faces.append(
        (
            np.array(
                [
                    [-wc_root * 0.3, fw, -fh * 0.3],
                    [wc_root * 0.7, fw, -fh * 0.3],
                    [wc_tip * 0.7, ws, -fh * 0.3],
                    [-wc_tip * 0.3, ws, -fh * 0.3],
                ]
            ),
            (190, 190, 190),
            "wing_right",
        )
    )

    # Vertical stabilizer
    fin_h = 5.5
    fin_base = 6.0
    faces.append(
        (
            np.array(
                [
                    [-hl, 0, fh * 0.6],
                    [-hl + fin_base, 0, fh * 0.6],
                    [-hl + fin_base * 0.4, 0, fh * 0.6 + fin_h],
                    [-hl, 0, fh * 0.6 + fin_h * 0.8],
                ]
            ),
            (200, 200, 210),
            "vert_stab",
        )
    )

    # Horizontal stabilizers
    hs_span = 6.3
    hs_chord_root = 3.5
    hs_chord_tip = 1.5
    faces.append(
        (
            np.array(
                [
                    [-hl + 1, 0, fh * 0.6],
                    [-hl + 1 + hs_chord_root, 0, fh * 0.6],
                    [-hl + 1 + hs_chord_tip, -hs_span, fh * 0.6],
                    [-hl + 1, -hs_span * 0.8, fh * 0.6],
                ]
            ),
            (195, 195, 195),
            "hstab_left",
        )
    )

    faces.append(
        (
            np.array(
                [
                    [-hl + 1, 0, fh * 0.6],
                    [-hl + 1 + hs_chord_root, 0, fh * 0.6],
                    [-hl + 1 + hs_chord_tip, hs_span, fh * 0.6],
                    [-hl + 1, hs_span * 0.8, fh * 0.6],
                ]
            ),
            (195, 195, 195),
            "hstab_right",
        )
    )

    # Engines
    eng_len = 4.0
    eng_r = 1.0
    eng_x = wc_root * 0.2
    for eng_y, prefix in [(-ws * 0.35, "eng_left"), (ws * 0.35, "eng_right")]:
        faces.append(
            (
                np.array(
                    [
                        [eng_x - eng_len / 2, eng_y - eng_r, -fh * 0.3 - eng_r],
                        [eng_x + eng_len / 2, eng_y - eng_r, -fh * 0.3 - eng_r],
                        [eng_x + eng_len / 2, eng_y + eng_r, -fh * 0.3 - eng_r],
                        [eng_x - eng_len / 2, eng_y + eng_r, -fh * 0.3 - eng_r],
                    ]
                ),
                (160, 160, 160),
                f"{prefix}_bot",
            )
        )
        faces.append(
            (
                np.array(
                    [
                        [eng_x - eng_len / 2, eng_y - eng_r, -fh * 0.3 + eng_r * 0.3],
                        [eng_x + eng_len / 2, eng_y - eng_r, -fh * 0.3 + eng_r * 0.3],
                        [eng_x + eng_len / 2, eng_y + eng_r, -fh * 0.3 + eng_r * 0.3],
                        [eng_x - eng_len / 2, eng_y + eng_r, -fh * 0.3 + eng_r * 0.3],
                    ]
                ),
                (180, 180, 180),
                f"{prefix}_top",
            )
        )

    # Navigation lights (single 3D points, drawn last so always visible).
    # Convention: red on left wing (port), green on right wing (starboard),
    # white strobe at top of vertical stabilizer, white landing light on nose.
    nav_lights = [
        (
            np.array([-wc_tip * 0.3 * 0.5, -ws + 0.2, -fh * 0.3]),
            (255, 50, 50),
            "nav_port",
        ),  # red, left wing tip
        (
            np.array([-wc_tip * 0.3 * 0.5, ws - 0.2, -fh * 0.3]),
            (50, 255, 50),
            "nav_stbd",
        ),  # green, right wing tip
        (
            np.array([-hl + fin_base * 0.4, 0, fh * 0.6 + fin_h]),
            (255, 255, 255),
            "nav_tail",
        ),  # white, top of fin
        (np.array([hl * 0.8, 0, 0]), (255, 240, 200), "nav_nose"),  # warm white, nose
    ]

    return faces, L, fw, fh, nav_lights


def _render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project_fn):
    """
    Render the 3D plane model onto a surface.

    R:          3x3 rotation matrix
    scale_px:   pixels per body-frame meter
    camera_dir: unit vector pointing toward camera (for shading)
    project_fn: callable(rotated_pt) -> (screen_x, screen_y)
                maps a rotated 3D point to 2D screen coords
    """
    faces, L, fw, fh, nav_lights = _build_plane_faces()
    s = scale_px

    # Depth axis index: the axis aligned with camera_dir
    # For side view (camera along -y): depth = y
    # For top-down (camera along -z): depth = z
    depth_axis = int(np.argmax(np.abs(camera_dir)))

    transformed = []
    for verts, base_color, name in faces:
        rotated = (R @ verts.T).T
        center = rotated.mean(axis=0)
        depth = center[depth_axis]

        # Face normal for shading
        if len(rotated) >= 3:
            e1 = rotated[1] - rotated[0]
            e2 = rotated[2] - rotated[0]
            normal = np.cross(e1, e2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-8:
                normal = normal / norm_len
            else:
                normal = np.array([0, 0, 1])
        else:
            normal = np.array([0, 0, 1])

        shade = abs(np.dot(normal, camera_dir))
        shade = 0.4 + 0.6 * shade
        color = tuple(int(c * shade) for c in base_color)

        screen_pts = [project_fn(pt, s) for pt in rotated]
        transformed.append((depth, screen_pts, color, name))

    # Painter's algorithm — sort by camera_dir sign
    cam_sign = np.sign(camera_dir[depth_axis])
    transformed.sort(key=lambda t: -cam_sign * t[0])

    for _, screen_pts, color, name in transformed:
        if len(screen_pts) >= 3:
            gfxdraw.filled_polygon(surf, screen_pts, color)
            outline = tuple(max(0, c - 40) for c in color)
            gfxdraw.aapolygon(surf, screen_pts, outline)

    # Passenger windows
    hl = L / 2
    n_win = 14
    win_spacing = L * 0.55 / n_win
    for side_sign in [-1, 1]:
        for i in range(n_win):
            wx = -hl * 0.3 + i * win_spacing
            wy = side_sign * fw
            wz = fh * 0.15
            pt = R @ np.array([wx, wy, wz])
            sx, sy = project_fn(pt, s)
            # Only draw if facing camera
            if camera_dir[depth_axis] * pt[depth_axis] < 0:
                pygame.draw.circle(
                    surf, (100, 100, 120), (sx, sy), max(1, int(s * 0.3))
                )

    # Navigation lights — drawn last so they sit on top regardless of depth.
    # Each light gets a soft halo + a bright core so it remains visible when
    # the aircraft is small or end-on.
    light_r_core = max(2, int(s * 0.55))
    light_r_halo = max(3, int(s * 1.0))
    for pt_body, color, _name in nav_lights:
        pt = R @ pt_body
        sx, sy = project_fn(pt, s)
        # Halo (semi-transparent)
        halo_surf = pygame.Surface(
            (light_r_halo * 2 + 2, light_r_halo * 2 + 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            halo_surf,
            (color[0], color[1], color[2], 110),
            (light_r_halo + 1, light_r_halo + 1),
            light_r_halo,
        )
        surf.blit(halo_surf, (sx - light_r_halo - 1, sy - light_r_halo - 1))
        # Bright core
        pygame.draw.circle(surf, color, (sx, sy), light_r_core)


# ─────────────────────────────────────────────────────────
#  3D side-view plane drawing
# ─────────────────────────────────────────────────────────


def _rotation_matrix(theta, phi, psi):
    """
    Build rotation matrix R = Rz(psi) @ Ry(-theta) @ Rx(-phi).

    Body frame: x=forward, y=right, z=up.
    theta = pitch (nose up = positive)
    phi   = bank  (right wing down = positive)
    psi   = heading (CW from north when viewed from above)

    The side-view camera looks along the -y axis, so we project
    the rotated points onto the x-z plane.
    """
    ct, st = np.cos(-theta), np.sin(-theta)
    cp, sp = np.cos(-phi), np.sin(-phi)
    ch, sh = np.cos(psi), np.sin(psi)

    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def _draw_plane_sideview(surf, cx, cy, theta, phi, psi, scale_px=1.0):
    """Draw the 3D plane projected onto x-z plane (side view, camera along -y)."""
    R = _rotation_matrix(theta, phi, psi)
    camera_dir = np.array([0, -1, 0])

    def project(pt, s):
        return (int(cx + pt[0] * s), int(cy - pt[2] * s))

    _render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project)


def _draw_plane_topdown(surf, cx, cy, theta, phi, psi, scale_px=1.0):
    """Draw the 3D plane projected onto x-y plane (top-down view, camera along -z)."""
    R = _rotation_matrix(theta, phi, psi)
    camera_dir = np.array([0, 0, -1])

    def project(pt, s):
        return (int(cx + pt[0] * s), int(cy - pt[1] * s))

    _render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project)


def _draw_heading_dashed(surf, cx, cy, heading, length, color, dash=10, gap=8):
    """Draw a dashed line from (cx, cy) in the direction of *heading*."""
    sa = -heading  # world CCW -> screen CW
    dx, dy = np.cos(sa), np.sin(sa)
    drawn = 0.0
    while drawn < length:
        s = drawn
        e = min(drawn + dash, length)
        pygame.draw.line(
            surf,
            color,
            (int(cx + dx * s), int(cy + dy * s)),
            (int(cx + dx * e), int(cy + dy * e)),
            2,
        )
        drawn += dash + gap


# ─────────────────────────────────────────────────────────
#  Side-view scene (x-z, centered, adaptive scale)
# ─────────────────────────────────────────────────────────


def _pick_tick_interval(span_m):
    """Pick a nice altitude tick interval (in meters) for the given span."""
    # Target ~5-8 ticks on screen
    raw = span_m / 6.0
    # Round to a nice number
    for nice in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if nice >= raw:
            return nice
    return 10000


def render_side_scene(
    panel_w,
    panel_h,
    state,
    params,
    positions_history_xz,
    cloud_positions,
    max_steps,
):
    """Side view (x-z), centered on the aircraft, with adaptive scale."""
    surf = pygame.Surface((panel_w, panel_h))
    surf.fill((135, 206, 235))  # sky blue

    cx, cy = panel_w // 2, panel_h // 2
    cur_x, cur_z = float(state.x), float(state.z)

    # Adaptive scale based on ALTITUDE range only (not horizontal distance)
    min_span = 200.0  # show at least 200 m vertical to see detail
    tgt_alt = float(state.target_altitude)

    z_span = min_span
    if len(positions_history_xz) > 1:
        zs = [p[1] for p in positions_history_xz]
        z_trail_span = max(zs) - min(zs)
        z_span = max(z_trail_span * 1.5, min_span)

    # Include target altitude difference
    alt_diff = abs(tgt_alt - cur_z)
    if alt_diff > 0:
        z_span = max(z_span, alt_diff * 3.0)

    scale = panel_h * 0.4 / z_span
    scale = min(scale, 0.5)

    # Margins for altitude scale on the right
    alt_margin = 60

    def world_to_screen(wx, wz):
        sx = int(cx + (wx - cur_x) * scale)
        sy = int(cy - (wz - cur_z) * scale)  # z-up -> y-down
        return sx, sy

    # Ground — only draw if visible on screen
    _, ground_sy = world_to_screen(0, 0)
    if ground_sy < panel_h:
        pygame.draw.rect(
            surf, (100, 160, 80), (0, ground_sy, panel_w, panel_h - ground_sy)
        )
        gfxdraw.hline(surf, 0, panel_w, min(ground_sy, panel_h - 1), (60, 100, 40))

    # Subtle clouds behind (less visible)
    from target_gym.plane.rendering import draw_cloud

    for cloud_cx, cloud_cy, cscale, shape in cloud_positions:
        draw_cloud(
            surf,
            cloud_cx,
            cloud_cy,
            scale=cscale * 0.7,
            seed=shape,
            color=(200, 220, 240),
            outline_color=(160, 180, 200),
            outline_thickness=2,
        )

    # ── Altitude scale (right edge) ──
    font_tick = pygame.font.SysFont("arial", 11)
    tick_interval = _pick_tick_interval(z_span)
    # Visible altitude range
    alt_top = cur_z + (panel_h / 2) / scale
    alt_bot = cur_z - (panel_h / 2) / scale
    first_tick = int(alt_bot / tick_interval) * tick_interval
    if first_tick < alt_bot:
        first_tick += tick_interval

    tick_x = panel_w - alt_margin
    # Use an SRCALPHA overlay so ticks/grid/labels are properly transparent
    tick_overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    tick_color = (80, 80, 80, 90)
    grid_color = (80, 80, 80, 35)
    label_color = (60, 60, 60, 140)
    axis_color = (80, 80, 80, 90)

    alt_val = first_tick
    while alt_val <= alt_top:
        _, ty = world_to_screen(0, alt_val)
        if 0 <= ty < panel_h:
            # Tick mark
            pygame.draw.line(
                tick_overlay, tick_color, (tick_x, ty), (tick_x + 6, ty), 1
            )
            # Light horizontal grid line
            pygame.draw.line(tick_overlay, grid_color, (0, ty), (tick_x, ty), 1)
            # Label (in feet)
            alt_ft = int(alt_val * 3.281)
            label = font_tick.render(f"{alt_ft:,} ft", True, label_color)
            tick_overlay.blit(label, (tick_x + 8, ty - label.get_height() // 2))
        alt_val += tick_interval

    # Vertical line for the scale axis
    pygame.draw.line(tick_overlay, axis_color, (tick_x, 0), (tick_x, panel_h), 1)
    surf.blit(tick_overlay, (0, 0))

    # Aircraft — 3D model projected onto side view
    theta = float(state.theta)
    psi = float(state.psi)
    phi = float(state.phi)
    plane_px_scale = max(0.8, min(2.5, panel_w * 0.003))
    _draw_plane_sideview(surf, cx, cy, theta, phi, psi, scale_px=plane_px_scale)

    # Target altitude dashed line
    _, tgt_sy = world_to_screen(0, tgt_alt)
    if 0 <= tgt_sy < panel_h:
        draw_dashed_line(
            surf,
            (220, 40, 40),
            (0, tgt_sy),
            (tick_x, tgt_sy),
            dash_length=10,
            space_length=10,
        )
        font_sm = pygame.font.SysFont("arial", 14)
        txt_tgt = font_sm.render(
            f"Target: {int(tgt_alt * 3.281):,} ft",
            True,
            (220, 40, 40),
        )
        surf.blit(txt_tgt, (10, max(tgt_sy - 20, 0)))

    # Trail
    if len(positions_history_xz) > 1:
        stride = max(1, len(positions_history_xz) // 300)
        pts = positions_history_xz[::stride]
        for wx, wz in pts:
            sx, sy = world_to_screen(wx, wz)
            if 0 <= sx < panel_w and 0 <= sy < panel_h:
                gfxdraw.circle(surf, sx, sy, 2, (0, 0, 0))
                gfxdraw.circle(surf, sx, sy, 1, (255, 255, 255))

    # HUD
    font = pygame.font.SysFont("arial", 16)
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(state.time))
    max_alt_diff = params.max_alt - params.min_alt
    done_alt = (state.z <= params.min_alt) or (state.z >= params.max_alt)
    if done_alt:
        reward = -1.0 * params.max_steps_in_episode
    else:
        reward = (
            (max_alt_diff - abs(state.target_altitude - state.z)) / max_alt_diff
        ) ** 2

    ground_speed = float(np.sqrt(float(state.x_dot) ** 2 + float(state.y_dot) ** 2))
    left = [
        font.render(
            f"Altitude: {int(state.z * 3.281):,} ft - {int(state.z):,} m",
            True,
            (0, 0, 255),
        ),
        font.render(
            f"Distance: {int(state.x * 0.539957 / 1000)} nm - {int(state.x / 1000)} km",
            True,
            (0, 0, 255),
        ),
        font.render(
            f"Speed: {int(ground_speed * 1.944)} kt - {int(ground_speed * 3.6)} km/h",
            True,
            (0, 0, 255),
        ),
    ]
    mid = [
        font.render(f"Pitch: {np.rad2deg(state.theta):.1f}\u00b0", True, (0, 0, 255)),
        font.render(f"Bank: {np.rad2deg(state.phi):.1f}\u00b0", True, (0, 0, 255)),
        font.render(f"Power: {state.power * 100:.0f}%", True, (0, 0, 255)),
    ]
    right = [
        font.render(f"Time: {time_elapsed}", True, (0, 0, 255)),
        font.render(f"Reward: {reward:.2f}", True, (0, 0, 255)),
    ]
    _draw_hud_box(surf, [left, mid, right], panel_w)

    return surf


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
    surf.fill((120, 170, 90))

    cx, cy = panel_w // 2, panel_h // 2

    # Subtle terrain texture lines
    font_sm = pygame.font.SysFont("arial", 10)
    for i in range(0, panel_h, 40):
        shade = 120 + (i % 80 == 0) * 10
        gfxdraw.hline(surf, 0, panel_w, i, (shade - 10, shade + 30, shade - 30))

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
        scale = min(panel_w, panel_h) * 0.4 / span
        scale = min(scale, 0.5)

    def world_to_screen(wx, wy):
        sx = int(cx + (wx - cur_x) * scale)
        sy = int(cy - (wy - cur_y) * scale)  # y-flip
        return sx, sy

    # -- trail --
    if len(positions_history_xy) > 1:
        stride = max(1, len(positions_history_xy) // 300)
        pts = positions_history_xy[::stride]
        for wx, wy in pts:
            sx, sy = world_to_screen(wx, wy)
            if 0 <= sx < panel_w and 0 <= sy < panel_h:
                gfxdraw.circle(surf, sx, sy, 2, (60, 60, 60))
                gfxdraw.circle(surf, sx, sy, 1, (255, 255, 255))

    # -- task-specific overlay --
    if task_type == "heading":
        # Target heading (dashed red line)
        heading_len = min(panel_w, panel_h) * 0.4
        _draw_heading_dashed(
            surf,
            cx,
            cy,
            state.target_heading,
            heading_len,
            color=(220, 40, 40),
            dash=10,
            gap=8,
        )
        # Current heading indicator (solid blue)
        ind_len = heading_len * 0.55
        sa = -state.psi
        ex = int(cx + ind_len * np.cos(sa))
        ey = int(cy + ind_len * np.sin(sa))
        pygame.draw.line(surf, (30, 80, 200), (cx, cy), (ex, ey), 2)

    elif task_type == "circle":
        # Draw the target circle
        center_sx, center_sy = world_to_screen(
            float(state.target_x), float(state.target_y)
        )
        radius_px = int(float(state.target_radius) * scale)
        if radius_px > 2:
            pygame.draw.circle(
                surf,
                (220, 40, 40),
                (center_sx, center_sy),
                radius_px,
                2,
            )
        # Small cross at center
        sz = 6
        pygame.draw.line(
            surf,
            (220, 40, 40),
            (center_sx - sz, center_sy),
            (center_sx + sz, center_sy),
            1,
        )
        pygame.draw.line(
            surf,
            (220, 40, 40),
            (center_sx, center_sy - sz),
            (center_sx, center_sy + sz),
            1,
        )

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
            pygame.draw.line(surf, (220, 40, 40), (x1, y1), (x2, y2), 2)
        # Small cross at center
        csx, csy = world_to_screen(fcx, fcy)
        sz = 6
        pygame.draw.line(surf, (220, 40, 40), (csx - sz, csy), (csx + sz, csy), 1)
        pygame.draw.line(surf, (220, 40, 40), (csx, csy - sz), (csx, csy + sz), 1)

        # Nearest-point marker (orange dot) — shows where the aircraft
        # should be on the curve.
        from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

        ndx, ndy, _, _, _ = nearest_point_on_twisted_lemniscate(state, params)
        npx = float(state.x) + float(ndx)
        npy = float(state.y) + float(ndy)
        nsx, nsy = world_to_screen(npx, npy)
        pygame.draw.circle(surf, (255, 140, 0), (nsx, nsy), 6)
        pygame.draw.circle(surf, (80, 40, 0), (nsx, nsy), 6, 1)

    # -- aircraft --
    plane_px_scale = max(0.8, min(2.5, panel_w * 0.003))
    _draw_plane_topdown(
        surf,
        cx,
        cy,
        float(state.theta),
        float(state.phi),
        float(state.psi),
        scale_px=plane_px_scale,
    )

    # -- compass labels --
    compass_r = min(panel_w, panel_h) * 0.45
    font_sm = pygame.font.SysFont("arial", 13)
    for label, angle_deg in [("N", 90), ("E", 0), ("S", -90), ("W", 180)]:
        rad = np.deg2rad(angle_deg)
        tx = int(cx + compass_r * np.cos(-rad))
        ty = int(cy + compass_r * np.sin(-rad))
        txt = font_sm.render(label, True, (40, 40, 40))
        surf.blit(txt, (tx - txt.get_width() // 2, ty - txt.get_height() // 2))

    # -- HUD --
    font = pygame.font.SysFont("arial", 16)
    heading_deg = np.rad2deg(state.psi) % 360

    if task_type == "heading":
        target_deg = np.rad2deg(state.target_heading) % 360
        left = [
            font.render(f"Heading: {heading_deg:.0f}\u00b0", True, (0, 0, 255)),
            font.render(f"Target: {target_deg:.0f}\u00b0", True, (200, 30, 30)),
        ]
    elif task_type == "circle":
        from target_gym.plane3d.env import distance_to_circle

        d = float(distance_to_circle(state))
        left = [
            font.render(f"Heading: {heading_deg:.0f}\u00b0", True, (0, 0, 255)),
            font.render(f"Dist to circle: {int(d)} m", True, (200, 30, 30)),
        ]
    elif task_type == "figure8":
        from target_gym.plane3d.env import nearest_point_on_twisted_lemniscate

        _, _, _, d, _ = nearest_point_on_twisted_lemniscate(state, params)
        left = [
            font.render(f"Heading: {heading_deg:.0f}\u00b0", True, (0, 0, 255)),
            font.render(f"Dist to curve: {int(d)} m", True, (200, 30, 30)),
        ]
    else:
        left = [
            font.render(f"Heading: {heading_deg:.0f}\u00b0", True, (0, 0, 255)),
        ]

    right = [
        font.render(f"Bank: {np.rad2deg(state.phi):.1f}\u00b0", True, (0, 0, 255)),
        font.render(
            f"Aileron: {np.rad2deg(state.aileron):.0f}\u00b0", True, (0, 0, 255)
        ),
    ]
    _draw_hud_box(surf, [left, right], panel_w)

    # -- legend (bottom) --
    legend_y = panel_h - 26
    font_sm2 = pygame.font.SysFont("arial", 14)
    if task_type == "heading":
        pygame.draw.line(surf, (220, 40, 40), (8, legend_y), (28, legend_y), 2)
        surf.blit(font_sm2.render("Target hdg", True, (40, 40, 40)), (32, legend_y - 7))
    elif task_type == "circle":
        pygame.draw.circle(surf, (220, 40, 40), (18, legend_y), 8, 2)
        surf.blit(
            font_sm2.render("Target circle", True, (40, 40, 40)), (32, legend_y - 7)
        )
    elif task_type == "figure8":
        pygame.draw.line(surf, (220, 40, 40), (8, legend_y), (28, legend_y), 2)
        surf.blit(
            font_sm2.render("Target figure-8", True, (40, 40, 40)), (32, legend_y - 7)
        )

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

    # Left panel: side view
    side_surf = render_side_scene(
        panel_w,
        panel_h,
        state,
        params,
        cls.positions_history_xz,
        cls.cloud_positions,
        max_steps=params.max_steps_in_episode,
    )

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
    pygame.draw.line(top_surf, (0, 0, 0), (0, 0), (0, panel_h), 2)

    # Compose
    combined = pygame.Surface((total_w, panel_h))
    combined.blit(side_surf, (0, 0))
    combined.blit(top_surf, (panel_w, 0))

    # Update histories with world coordinates
    cls.positions_history_xz.append((float(state.x), float(state.z)))
    cls.positions_history_xy.append((float(state.x), float(state.y)))

    # Display and capture
    screen.blit(combined, (0, 0))
    pygame.display.flip()
    frame = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    frames.append(frame)
    cls.frames = frames

    return frames, screen, clock
