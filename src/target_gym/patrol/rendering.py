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
wingmen (a fixed palette); green rings mark the slots.  Both panels carry the
same console chrome as the rest of the suite: a step/status header along the
top and a stack of labelled bars along the bottom, rather than the floating
text box and blue-sky-and-grass scene this used to draw.
"""

import numpy as np

from target_gym import render_aircraft as ra
from target_gym.render_kit import render_extra

pygame = render_extra("pygame")
gfxdraw = render_extra("pygame.gfxdraw")

_LEAD = ra.BLUE
_SLOT = ra.GOOD
# Distinct wingman ID-ring colours (cycled if there are more wingmen), drawn
# from the shared console palette rather than the saturated primaries this
# used to carry -- they were picked against a sky-blue panel that no longer
# exists.
_WING_COLORS = [ra.TARGET, ra.TEAL, ra.ACCENT, (206, 147, 216)]


def _wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _draw_plane_rearview(surf, cx, cy, theta, phi, psi_rel, scale_px=1.0):
    """Draw the 3D plane as seen from *behind*, along its forward (+x) axis."""
    R = ra.rotation_matrix(theta, phi, psi_rel)
    camera_dir = np.array([-1, 0, 0])  # behind the aircraft, looking forward

    def project(pt, s):
        return (int(cx + pt[1] * s), int(cy - pt[2] * s))  # y=right, z=up

    ra.render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project)


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
        cls.font = pygame.font.SysFont(ra.MONO, 13)
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
    plane_px = max(1.0, min(2.2, panel_w * 0.0030))

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
    # Both panels keep their scene inside the band left by the header strip and
    # the gauge stack, so nothing important is drawn under the chrome.
    scene_top = ra.HEADER_H
    side_bot = panel_h - ra.gauge_height(4)
    side_h = side_bot - scene_top
    side_cy = (scene_top + side_bot) // 2

    side = pygame.Surface((panel_w, panel_h))
    side.fill(ra.SKY)
    # Frame the formation and its slots rather than a fixed 360 m window. The
    # fixed window was sized for the widest formation the environment can ask
    # for, so a two-ship in close trail drew as a pair of specks in an empty
    # panel. The floor keeps it from zooming in on a converged formation until
    # the glyphs overlap, and from breathing as the slot error decays.
    slot_lat = [lateral(w[0], w[1]) for w in slot_world]
    slot_alt = [w[2] for w in slot_world]
    lats = [lateral(p[0], p[1]) for p in pos] + slot_lat
    alts = [p[2] for p in pos] + slot_alt
    span = max(max(lats) - min(lats), max(alts) - min(alts), 300.0) * 1.6
    sc = min(panel_w, side_h) * 0.42 / span
    cu = 0.5 * (min(lats) + max(lats))
    cv = 0.5 * (min(alts) + max(alts))

    def side_w2s(wu, wz):
        return (panel_w / 2 + (wu - cu) * sc, side_cy - (wz - cv) * sc)

    ra.world_grid(side, panel_w, scene_top, side_bot, sc, cu, cv, side_w2s)
    _, gy = side_w2s(0, 0)
    if gy < side_bot:
        pygame.draw.rect(side, ra.TERRAIN, (0, int(gy), panel_w, side_bot - int(gy)))
        gfxdraw.hline(side, 0, panel_w, int(min(gy, side_bot - 1)), ra.SCENE_LINE)
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
        cls.font.render("behind the lead →", True, ra.SCENE_DIM), (12, side_bot - 20)
    )

    # ── Right: top-down view (x-y) ───────────────────────────────────────
    top_bot = panel_h - ra.gauge_height(3)
    top_h = top_bot - scene_top
    top_cy = (scene_top + top_bot) // 2

    top = pygame.Surface((panel_w, panel_h))
    top.fill(ra.GROUND)
    xs = list(pos[:, 0]) + [w[0] for w in slot_world]
    ys = list(pos[:, 1]) + [w[1] for w in slot_world]
    top_span = max(max(xs) - min(xs), max(ys) - min(ys), 400.0) * 1.6
    sc = min(panel_w, top_h) * 0.42 / top_span
    cu = 0.5 * (min(xs) + max(xs))
    cv = 0.5 * (min(ys) + max(ys))

    def top_w2s(wx, wy):
        return (panel_w / 2 + (wx - cu) * sc, top_cy - (wy - cv) * sc)

    ra.world_grid(top, panel_w, scene_top, top_bot, sc, cu, cv, top_w2s)

    for i, p in enumerate(planes):
        shade = tuple(int(c * 0.4) for c in colors[i])
        _trail(top, cls.trails[i], top_w2s, shade, lambda q: (q[0], q[1]))
    for wi, (b, r, u) in enumerate(slots):
        wx, wy, wz = slot_world[wi]
        tsx, tsy = top_w2s(wx, wy)
        fpx, fpy = top_w2s(float(wingmen[wi].x), float(wingmen[wi].y))
        ra.draw_dashed_line(
            top, _SLOT, (int(fpx), int(fpy)), (int(tsx), int(tsy)), 8, 6
        )
        pygame.draw.circle(top, _SLOT, (int(tsx), int(tsy)), 7, 2)
    for i, p in enumerate(planes):
        px, py = top_w2s(float(p.x), float(p.y))
        _id_ring(top, px, py, colors[i])
        ra.draw_plane_topdown(
            top, px, py, float(p.theta), float(p.phi), float(p.psi), scale_px=plane_px
        )
    lpx, lpy = top_w2s(float(lead.x), float(lead.y))
    top.blit(cls.font.render("Lead", True, _LEAD), (int(lpx) + 18, int(lpy) + 16))
    pygame.draw.line(top, ra.FRAME, (0, 0), (0, panel_h), 2)

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

    # ── Compose + gauges ─────────────────────────────────────────────────
    sep_ref = float(params.min_separation)
    status = ("COLLISION", ra.ALERT) if collision else ("NOMINAL", ra.GOOD)
    ra.gauge_stack(
        side,
        panel_w,
        panel_h,
        [
            (
                "SLOT ERR",
                f"{mean_err:,.0f} m",
                min(1.0, mean_err / max(3.0 * sep_ref, 1.0)),
                ra.TARGET if mean_err > sep_ref else ra.GOOD,
                None,
            ),
            (
                "MIN SEP",
                "--" if not np.isfinite(min_sep) else f"{min_sep:,.0f} m",
                (
                    0.0
                    if not np.isfinite(min_sep)
                    else min(1.0, min_sep / max(4.0 * sep_ref, 1.0))
                ),
                ra.ALERT if collision else ra.ACCENT,
                0.25,
            ),
            (
                "ALTITUDE",
                f"{int(float(lead.z) * 3.281):,} ft",
                float(lead.z) / 12000.0,
                ra.BLUE,
                None,
            ),
            ("REWARD", f"{reward:+.3f}", max(0.0, float(reward)), ra.TEAL, None),
        ],
    )
    ra.header(side, panel_w, lead.time, status)

    lead_hdg = ra.compass_deg(lead.psi)
    lead_bank = float(np.rad2deg(float(lead.phi)))
    ra.gauge_stack(
        top,
        panel_w,
        panel_h,
        [
            ("LEAD HDG", f"{lead_hdg:.0f}\u00b0", lead_hdg / 360.0, ra.ACCENT, None),
            (
                "LEAD BANK",
                f"{lead_bank:+.1f}\u00b0",
                (lead_bank + 45.0) / 90.0,
                ra.BLUE,
                0.5,
                0.5,
            ),
            (
                "FORMATION",
                f"{len(planes)} ac",
                min(1.0, len(planes) / 5.0),
                ra.TEAL,
                None,
            ),
        ],
    )

    combined = pygame.Surface((total_w, panel_h))
    combined.blit(side, (0, 0))
    combined.blit(top, (panel_w, 0))

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
