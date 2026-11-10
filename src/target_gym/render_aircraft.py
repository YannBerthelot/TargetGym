"""
The pygame half of the render kit: everything the aircraft panels share.

:mod:`target_gym.render_kit` is matplotlib, which is what the industrial plants
draw with. The aircraft draw with pygame instead, because they project a solid
model every frame and matplotlib is too slow for that. So they need their own
kit, and this is it: the palette, the console chrome (header strip, gauge
stack, world grid), the aircraft models (the A320 and the F/A-18 Hornet) and
the three projections that draw them.

It lives here rather than in ``plane3d/rendering.py`` because four modules draw
aircraft -- ``plane3d`` (side and top-down), ``patrol`` (rear and top-down),
``plane/rendering_console.py`` (matplotlib, but flying this same mesh) and
``plane/rendering.py`` -- and until this module existed they reached into each
other for it. ``patrol`` imported eleven private names out of ``plane3d``,
which meant a palette change in one aircraft environment silently restyled
another and nothing said so.

The palette mirrors :mod:`target_gym.render_kit`'s hex constants, lifted a
little: the plants are diagrams on a near-black ground, while these panels are
scenes that need a solid pale aircraft to read against the sky.

Nothing here is fingerprinted. ``provenance._env_sources`` skips ``rendering*``
modules inside an environment package, and this module is not in one, so moving
code between the aircraft renderers never marks a recorded baseline stale.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import NamedTuple

import numpy as np

from target_gym.plane3d import hornet
from target_gym.render_kit import format_clock, render_extra

pygame = render_extra("pygame")
gfxdraw = render_extra("pygame.gfxdraw")


def draw_dashed_line(surface, color, start_pos, end_pos, dash_length=5, space_length=5):
    """A dashed segment, used for target altitudes, headings and slot links."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    dx, dy = x2 - x1, y2 - y1
    distance = max(1e-6, (dx**2 + dy**2) ** 0.5)
    dashes = int(distance / (dash_length + space_length))
    for i in range(max(dashes, 0)):
        s = i * (dash_length + space_length) / distance
        e = min(1.0, (i * (dash_length + space_length) + dash_length) / distance)
        pygame.draw.line(
            surface,
            color,
            (x1 + dx * s, y1 + dy * s),
            (x1 + dx * e, y1 + dy * e),
        )


def scene_band(panel_h, n_gauge_rows):
    """The vertical strip a panel's scene may use, between header and gauges.

    Returns ``(top, bottom, height, centre_y)``. Every aircraft panel centres
    its view on this rather than on the raw surface, so nothing lands under the
    chrome.
    """
    top = HEADER_H
    bot = panel_h - gauge_height(n_gauge_rows)
    return top, bot, bot - top, (top + bot) // 2


def world_grid(surf, panel_w, top, bot, scale, cu, cv, w2s):
    """A world-locked grid across a panel's scene band.

    Locked to world coordinates rather than to the panel, so it slides under
    the aircraft as it flies and the view reads as movement over ground instead
    of a pattern spinning in place. The spacing walks a 1-2-5 ladder until a
    cell is wide enough to read; below that it draws nothing rather than a
    moire.
    """
    if not np.isfinite(scale) or scale <= 0:
        return
    want_px = min(panel_w, bot - top) / 5.0
    step = 1.0
    while step * scale < want_px:
        step *= 2.0 if f"{step:e}"[0] in "15" else 2.5
    if step * scale < 24.0:
        return
    grid = pygame.Surface((panel_w, surf.get_height()), pygame.SRCALPHA)
    half_u = panel_w / 2 / scale
    u = np.floor((cu - half_u) / step) * step
    while u < cu + half_u:
        sx, _ = w2s(u, cv)
        pygame.draw.line(grid, (*SCENE_LINE, 170), (sx, top), (sx, bot), 1)
        u += step
    half_v = (bot - top) / 2 / scale
    v = np.floor((cv - half_v) / step) * step
    while v < cv + half_v:
        _, sy = w2s(cu, v)
        pygame.draw.line(grid, (*SCENE_LINE, 170), (0, sy), (panel_w, sy), 1)
        v += step
    surf.blit(grid, (0, 0))


# ── Palette ───────────────────────────────────────────────────────────────
# Shared with target_gym.render_kit, which the industrial plants render with,
# and with plane/rendering.py. The 3D tasks drew a sky-blue field over bright
# green terrain; beside a furnace or a reactor in the same gallery that read as
# a different project rather than a different environment.
SKY = (150, 172, 196)  # daylight, not dusk -- see SCENE_INK below
PANEL = (44, 60, 80)  # HUD panel fill
GROUND = (128, 148, 170)  # top-down map ground, a shade below the sky
#: Terrain in the *side* view, which has to separate from the sky at a glance.
#: GROUND is only a shade below SKY, which is right for the map -- it is the
#: whole panel there -- and useless on an elevation, where the horizon then
#: disappeared and an aircraft 12 000 ft low looked level.
TERRAIN = (86, 104, 124)
ALERT = (239, 83, 80)  # render_kit RED    #ef5350
FRAME = (66, 90, 118)  # render_kit FRAME  #1e3248
INK = (196, 216, 236)  # render_kit TEXT   #c4d8ec
DIM = (128, 154, 176)  # render_kit DIM    #4a6678
ACCENT = (0, 188, 212)  # render_kit CYAN   #00bcd4
TARGET = (255, 202, 40)  # render_kit AMBER  #ffca28
GOOD = (102, 187, 106)  # render_kit GREEN  #66bb6a
WELL = (26, 38, 54)  # gauge trough
BLUE = (66, 165, 245)  # render_kit BLUE  #42a5f5
TEAL = (38, 166, 154)  # render_kit TEAL  #26a69a
#: Monospaced stack, matching ``render_kit.MONO``. Proportional type is
#: most of what made these panels read as a game rather than a console.
#: Ink for things drawn *on a scene* rather than on the chrome. The panels
#: are two different grounds: the gauge stack and header are dark console
#: furniture, while the sky and the map are lit daylight. INK and DIM are pale,
#: which is right on the dark panel and invisible on the light one, so scene
#: text, hairlines and graticules use these instead.
SCENE_INK = (26, 38, 54)
SCENE_DIM = (72, 92, 114)
SCENE_LINE = (96, 118, 142)

MONO = "menlo,dejavusansmono,couriernew,monospace"


#: How fast the graticule scrolls relative to the ground, for parallax.
GRATICULE_PARALLAX = 0.35

#: Facets around the fuselage, and spanwise panels per wing. The model was
#: fourteen flat slabs, which lit in hard steps and made the painter's sort pop
#: the engine nacelles through the wing whenever their centroids crossed.
#: Smaller, more numerous faces fix both: the shading gradient becomes smooth
#: and the depth ordering has far less to get wrong.
N_RING, SPAN_PANELS = 12, 5

#: Height of the header strip, and of one row of the gauge stack, in pixels.
HEADER_H = 28
GAUGE_ROW = 26


def gauge_height(n_rows: int) -> int:
    """Height the gauge stack claims at the foot of a panel."""
    return 10 + GAUGE_ROW * int(n_rows)


def gauge_stack(surf, panel_w, panel_h, rows, height=None):
    """A render_kit-style gauge stack across the bottom of a panel.

    The 3D tasks showed their KPIs as a floating box of text, which is the
    single biggest reason they did not look like the rest of the suite: every
    other environment reads its values off horizontal bars with limit ticks and
    a target marker. Bars also show *where in its range* a value sits, which a
    number cannot.
    """
    height = gauge_height(len(rows)) if height is None else height
    top = panel_h - height
    panel = pygame.Surface((panel_w, height), pygame.SRCALPHA)
    panel.fill((*PANEL, 236))
    surf.blit(panel, (0, top))
    pygame.draw.line(surf, FRAME, (0, top), (panel_w, top), 1)

    font_l = pygame.font.SysFont(MONO, 11)
    font_v = pygame.font.SysFont(MONO, 12)
    pad, lab_w, val_w = 14, 92, 104
    bar_x = pad + lab_w
    bar_w = max(40, panel_w - bar_x - val_w - pad)
    row_h = height // max(len(rows), 1)
    for i, row in enumerate(rows):
        label, value, frac, color, marker = row[:5]
        # A sixth field gives the bar an origin other than the left edge, so a
        # signed quantity (bank, pitch, aileron) grows either way from centre
        # instead of reading as a magnitude that is never negative.
        origin = float(row[5]) if len(row) > 5 else 0.0
        y = top + i * row_h + row_h // 2
        surf.blit(font_l.render(label, True, DIM), (pad, y - 6))
        pygame.draw.rect(surf, WELL, (bar_x, y - 5, bar_w, 11), border_radius=2)
        f = min(1.0, max(0.0, float(frac)))
        x0, x1 = sorted((bar_x + int(bar_w * origin), bar_x + int(bar_w * f)))
        if x1 - x0 > 1:
            pygame.draw.rect(surf, color, (x0, y - 5, x1 - x0, 11), border_radius=2)
        # A marker sitting on the bar's own origin says nothing, so skip it.
        if marker is not None and abs(float(marker) - origin) > 1e-6:
            mx = bar_x + int(bar_w * min(1.0, max(0.0, float(marker))))
            pygame.draw.line(surf, TARGET, (mx, y - 9), (mx, y + 9), 2)
        v = font_v.render(value, True, INK)
        surf.blit(v, (panel_w - pad - v.get_width(), y - 7))


def header(surf, panel_w, step, status=None, clock=None):
    """Step counter left, status pill right -- as ``render_kit.frame`` draws."""
    # Its own strip, matching the gauge stack's. Without one the header sat
    # directly on the sky, and once the scene palette went to daylight its pale
    # console text stopped being readable at all.
    strip = pygame.Surface((panel_w, HEADER_H), pygame.SRCALPHA)
    strip.fill((*PANEL, 236))
    surf.blit(strip, (0, 0))
    pygame.draw.line(surf, FRAME, (0, HEADER_H), (panel_w, HEADER_H), 1)

    font = pygame.font.SysFont(MONO, 12)
    left = f"STEP {int(step):>5d}"
    if clock is not None:
        left = f"{left}   {clock}"
    surf.blit(font.render(left, True, DIM), (14, 10))
    if status is not None:
        text, color = status
        lab = font.render(text, True, color)
        x = panel_w - 14 - lab.get_width()
        pygame.draw.circle(surf, color, (x - 12, 16), 4)
        surf.blit(lab, (x, 10))


# ── Aircraft models ───────────────────────────────────────────────────────
# A model is a set of flat faces in body axes (x forward, y left, z up, metres)
# plus its nav lights. Each is put together from the three part builders below
# (a body of revolution, a lifting surface, a flat plate), and every projection
# takes the model it draws as an argument. A new airframe is therefore one new
# builder in ``MODEL_BUILDERS``; ``model_for`` then picks it for any params
# whose wingspan is its own.


class Face(NamedTuple):
    """One flat facet of a model, in body axes."""

    verts: np.ndarray
    color: tuple
    name: str
    #: A single polygon standing in for a thin surface (wing, tail, fin). It
    #: has no interior, so it is lit from either side.
    plate: bool = False


def body_faces(stations, color, name, *, y=0.0, n=N_RING, apex=None, cap=None):
    """A fuselage, nacelle or canopy: elliptic rings joined by quads.

    ``stations`` are ``(x, half_width, half_height, centre_z)``, tail first.
    ``color`` is an RGB, or ``color(angle)`` for a tone that varies round the
    ring. ``apex`` closes the last ring to a point (a nose cone); ``cap``
    closes the first ring with a flat face of that colour (a nozzle, a tail).
    """
    shade = color if callable(color) else (lambda _a: color)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rings = [
        np.stack([np.full(n, x), y + ry * np.cos(ang), cz + rz * np.sin(ang)], 1)
        for x, ry, rz, cz in stations
    ]
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(n):
            j = (i + 1) % n
            faces.append(
                Face(np.array([a[i], a[j], b[j], b[i]]), shade(ang[i]), f"{name}_{i}")
            )
    if apex is not None:
        last, tip = rings[-1], np.asarray(apex, dtype=float)
        for i in range(n):
            j = (i + 1) % n
            faces.append(
                Face(np.array([last[i], last[j], tip]), shade(ang[i]), f"{name}_nose")
            )
    if cap is not None:
        faces.append(Face(rings[0][::-1].copy(), cap, f"{name}_cap"))
    return faces


def surface_faces(root, tip, color, name, panels=SPAN_PANELS):
    """A wing, tailplane or fin: a flat plate split spanwise into *panels*.

    ``root`` and ``tip`` are each the two ends of a chord, as 3D points, so
    sweep, taper, dihedral and cant all come from where they are placed.
    """
    (ra_, rb), (ta, tb) = np.asarray(root, float), np.asarray(tip, float)
    faces = []
    for i in range(panels):
        t0, t1 = i / panels, (i + 1) / panels
        quad = [ra_ + t0 * (ta - ra_), rb + t0 * (tb - rb)]
        quad += [rb + t1 * (tb - rb), ra_ + t1 * (ta - ra_)]
        faces.append(Face(np.array(quad), color, f"{name}_{i}", plate=True))
    return faces


def plate_face(points, color, name):
    """One flat polygon of a thin part: a fin, a strake."""
    return Face(np.asarray(points, dtype=float), color, name, plate=True)


def outward(faces):
    """Wind every face to point away from the model's centre.

    Lighting is signed, so a face wound inward would light on the wrong side.
    The first hand-written model had eight of fourteen faces wound inward,
    which left the two sides of the fuselage with the *same* normal: nothing in
    the picture said which side you were looking at.
    """
    centroid = np.vstack([f.verts for f in faces]).mean(axis=0)

    def points_in(f):
        v = f.verts
        normal = np.cross(v[1] - v[0], v[2] - v[0])
        return np.dot(normal, v.mean(axis=0) - centroid) < 0

    return tuple(f._replace(verts=f.verts[::-1]) if points_in(f) else f for f in faces)


@dataclass(frozen=True, eq=False)
class AircraftModel:
    """A low-poly aircraft, as every projection in this module draws it.

    ``nav_lights`` are ``(point, rgb, name)`` in body axes. ``windows`` are body
    points drawn as small dark dots on the side facing the camera.
    ``light_radius`` is a nav light's core in metres; its halo is twice that.
    """

    name: str
    faces: tuple
    nav_lights: tuple
    windows: tuple = ()
    light_radius: float = 0.95

    @cached_property
    def _verts(self):
        return np.vstack([f.verts for f in self.faces])

    @property
    def length(self) -> float:
        """Nose to tail, metres."""
        return float(np.ptp(self._verts[:, 0]))

    @property
    def span(self) -> float:
        """Wingtip to wingtip, metres."""
        return float(np.ptp(self._verts[:, 1]))

    @cached_property
    def reach(self) -> float:
        """Farthest any face or light lies from the origin, halo included."""
        lights = [p for p, _c, _n in self.nav_lights]
        r = np.linalg.norm(np.vstack([self._verts, *lights]), axis=1).max()
        return float(r) + 2.0 * self.light_radius


def build_a320():
    """The A320 of ``plane``, the ``plane3d`` tasks and ``wake_surf``."""
    hl, fw, fh = 37.57 / 2, 2.0, 2.0
    cz = -0.2 * fh

    def livery(a):
        # Lighter over the crown, the way a white livery reads.
        tone = int(196 + 42 * (0.5 + 0.5 * np.sin(a)))
        return (tone, tone, min(255, tone + 3))

    # A tapered prism rather than four flat sides. The nose closes to a single
    # apex: a last ring at 0.12 of radius collapsed the cone, and the nose read
    # as a blunt wedge at shallow angles.
    stations = [(-hl, 0.55), (-hl * 0.72, 0.92), (-hl * 0.3, 1.0), (hl * 0.2, 1.0)]
    stations += [(hl * 0.5, 0.95), (hl * 0.68, 0.80), (hl * 0.78, 0.46)]
    faces = body_faces(
        [(x, fw * k, 0.8 * fh * k, cz) for x, k in stations],
        livery,
        "fuse",
        apex=(hl * 0.86, 0.0, cz * 0.35),
    )

    ws, wcr, wct = 17.9, 6.0, 2.5
    hs, hcr, hct = 6.3, 3.5, 1.5
    wz, hz = -fh * 0.3, fh * 0.6
    for sgn, side in ((1, "port"), (-1, "stbd")):
        y_root = sgn * fw
        faces += surface_faces(
            ((-wcr * 0.3, y_root, wz), (wcr * 0.7, y_root, wz)),
            ((-wct * 0.3, sgn * ws, wz + 0.9), (wct * 0.7, sgn * ws, wz + 0.9)),
            (198, 200, 204),
            f"wing_{side}",
        )
        faces += surface_faces(
            ((-hl + 1, 0, hz), (-hl + 1 + hcr, 0, hz)),
            (
                (-hl + 1, sgn * hs * 0.8, hz + 0.25),
                (-hl + 1 + hct, sgn * hs, hz + 0.25),
            ),
            (196, 198, 205),
            f"hstab_{side}",
            panels=3,
        )
    fin_h, fin_b = 5.5, 6.0
    faces.append(
        plate_face(
            [
                [-hl, 0, hz],
                [-hl + fin_b, 0, hz],
                [-hl + fin_b * 0.4, 0, hz + fin_h],
                [-hl, 0, hz + fin_h * 0.8],
            ],
            (204, 206, 214),
            "vert_stab",
        )
    )
    # Engines: short nacelles slung under and ahead of the wing.
    nacelle = [(-4.6, 1.25), (-1.0, 1.45), (2.4, 1.15)]
    for sgn in (1, -1):
        faces += body_faces(
            [(x, r, r, -fh * 0.85) for x, r in nacelle],
            (150, 154, 160),
            f"eng_{sgn}",
            y=sgn * 7.4,
            n=8,
        )

    # Port is +y: with x forward and z up, a right-handed frame puts y left.
    nav_lights = (
        (np.array([-wct * 0.15, ws - 0.2, -fh * 0.3]), (255, 50, 50), "nav_port"),
        (np.array([-wct * 0.15, -ws + 0.2, -fh * 0.3]), GOOD, "nav_stbd"),
        (np.array([-hl + fin_b * 0.4, 0, hz + fin_h]), (255, 255, 255), "nav_tail"),
        (np.array([hl * 0.8, 0, 0]), (255, 240, 200), "nav_nose"),
    )
    n_win, length = 14, 2 * hl
    windows = tuple(
        np.array([-hl * 0.3 + i * length * 0.55 / n_win, sgn * fw, fh * 0.15])
        for sgn in (-1, 1)
        for i in range(n_win)
    )
    return AircraftModel("a320", outward(faces), nav_lights, windows)


def build_hornet():
    """The F/A-18 Hornet of ``patrol``, ``patrol_range_bearing`` and ``formation``.

    True scale, our own low-poly silhouette. The wing is hornet.py's: its span
    and area [HARV], 26.7 deg of leading-edge sweep and a taper of 0.35, which
    give a 4.82 m root chord and HARV's 3.51 m MAC. The rest is placed by eye to
    public F/A-18C figures: 56 ft nose to tail (AFF's own limit), 4.7 m tall on
    its gear, leading-edge extensions from the cockpit to the wing, all-moving
    stabilators 6.6 m across, and twin fins canted 20 deg outboard.
    """
    nose, tail = 0.5 * 56.0 * hornet.FT, -0.5 * 56.0 * hornet.FT  # +-8.53 m

    def grey(a):
        # Navy light grey, a shade lighter over the spine.
        tone = int(184 + 34 * (0.5 + 0.5 * np.sin(a)))
        return (tone, tone + 2, min(255, tone + 6))

    # Fuselage: widest over the side-by-side engines, narrowing to the cockpit
    # and the radome, closed at the back by the nozzle face.
    faces = body_faces(
        [
            (-7.9, 0.95, 0.55, 0.05),
            (-6.0, 1.25, 0.70, 0.10),
            (-3.0, 1.45, 0.80, 0.10),
            (0.0, 1.40, 0.85, 0.10),
            (2.5, 1.05, 0.85, 0.05),
            (4.3, 0.70, 0.80, 0.10),
            (6.2, 0.52, 0.58, 0.00),
            (7.6, 0.30, 0.32, -0.05),
        ],
        grey,
        "fuse",
        apex=(nose, 0.0, -0.08),
        cap=(96, 98, 104),
    )
    faces += body_faces(
        [(3.2, 0.30, 0.25, 0.80), (4.4, 0.42, 0.40, 0.88), (5.7, 0.26, 0.24, 0.78)],
        (64, 78, 98),
        "canopy",
        n=8,
        apex=(6.2, 0.0, 0.72),
    )

    half_span = 0.5 * hornet.SPAN
    taper, tan_le = 0.35, np.tan(np.radians(26.7))
    c_root = 2.0 * hornet.WING_AREA / (hornet.SPAN * (1.0 + taper))  # 4.82 m
    le0 = 0.6  # leading edge on the centreline (ours, by eye)

    def chord(y, z):
        """Leading and trailing edge of the wing's chord at span station y."""
        le = le0 - abs(y) * tan_le
        return (le, y, z), (
            le - c_root * (1.0 - (1.0 - taper) * abs(y) / half_span),
            y,
            z,
        )

    wing, wz = (186, 190, 198), 0.3
    for sgn, side in ((1, "port"), (-1, "stbd")):
        faces += surface_faces(
            chord(sgn * 1.3, wz), chord(sgn * half_span, wz), wing, f"wing_{side}"
        )
        # Leading-edge extension: from beside the cockpit out to the wing's
        # leading edge, which it meets 2.1 m from the centreline.
        lex_le, _ = chord(sgn * 2.1, wz)
        root_le, _ = chord(sgn * 1.3, wz)
        faces.append(
            plate_face(
                [
                    (5.4, sgn * 0.55, wz),
                    (2.6, sgn * 1.15, wz),
                    (0.8, sgn * 1.65, wz),
                    lex_le,
                    root_le,
                ],
                wing,
                f"lex_{side}",
            )
        )
        # All-moving stabilators, 6.58 m tip to tip, a little anhedral.
        faces += surface_faces(
            ((-5.6, sgn * 1.0, 0.1), (-8.05, sgn * 1.0, 0.1)),
            ((-7.66, sgn * 3.29, 0.0), (tail, sgn * 3.29, 0.0)),
            wing,
            f"stab_{side}",
            panels=2,
        )
        # Twin fins, canted 20 deg outboard, 2.3 m from root to tip.
        cant = np.radians(20.0)
        ty, tz = sgn * (0.95 + 2.3 * np.sin(cant)), 0.75 + 2.3 * np.cos(cant)
        faces += surface_faces(
            ((-3.0, sgn * 0.95, 0.75), (-6.3, sgn * 0.95, 0.75)),
            ((-4.6, ty, tz), (-5.8, ty, tz)),
            (192, 196, 204),
            f"fin_{side}",
            panels=2,
        )
        # Engine intakes under the strakes, listed front first so the cap
        # closes the mouth and the apex fairs the duct into the fuselage.
        faces += body_faces(
            [(2.0, 0.50, 0.55, -0.35), (-1.0, 0.48, 0.52, -0.35)],
            (176, 180, 188),
            f"intake_{side}",
            y=sgn * 1.15,
            n=8,
            apex=(-3.0, sgn * 0.9, -0.2),
            cap=(48, 52, 60),
        )
        faces += body_faces(
            [(-8.35, 0.40, 0.40, 0.05), (-7.7, 0.46, 0.46, 0.05)],
            (112, 108, 104),
            f"nozzle_{side}",
            y=sgn * 0.52,
            n=8,
            cap=(40, 40, 44),
        )

    tip_le, _ = chord(half_span, wz)
    nav_lights = (
        (np.array([tip_le[0] - 0.8, half_span - 0.1, wz]), (255, 50, 50), "nav_port"),
        (np.array([tip_le[0] - 0.8, -half_span + 0.1, wz]), GOOD, "nav_stbd"),
    )
    return AircraftModel("hornet", outward(faces), nav_lights, light_radius=0.45)


#: Model builders by name. A new airframe is one more entry here.
MODEL_BUILDERS = {"a320": build_a320, "hornet": build_hornet}


@lru_cache(maxsize=None)
def aircraft_model(name: str) -> AircraftModel:
    """The registered model *name*, built once."""
    return MODEL_BUILDERS[name]()


def register_model(name: str, builder) -> None:
    """Add or replace the model built by ``builder()`` under *name*."""
    MODEL_BUILDERS[name] = builder
    aircraft_model.cache_clear()


def model_for(params, default: str = "a320") -> AircraftModel:
    """The model of the airframe *params* fly: the registered one nearest its
    wingspan, or *default* for params that carry none. Read off the span rather
    than a params field of its own, because params values are fingerprinted."""
    span = getattr(params, "wingspan", None)
    if span is None:
        return aircraft_model(default)
    span = float(span)
    models = [aircraft_model(name) for name in MODEL_BUILDERS]
    return min(models, key=lambda m: abs(np.log(m.span / span)))


def _as_model(model) -> AircraftModel:
    return aircraft_model(model) if isinstance(model, str) else model


#: Direction *toward* the light: high, and on the camera's side so the near
#: face of the fuselage is the lit one. Fixed in world space on purpose.
LIGHT = np.array([-0.35, -0.55, 0.76]) / np.linalg.norm([-0.35, -0.55, 0.76])

#: Draw the aircraft this many times oversize, then smoothscale down.
SUPERSAMPLE = 3


def render_3d_plane(target, cx, cy, R, scale_px, camera_dir, project_fn, model="a320"):
    """
    Render an aircraft model onto a surface.

    R:          3x3 rotation matrix
    scale_px:   pixels per body-frame meter
    camera_dir: the direction the camera looks along (sets the depth sort)
    project_fn: callable(rotated_pt) -> (screen_x, screen_y)
                maps a rotated 3D point to 2D screen coords
    model:      the :class:`AircraftModel` to draw, or its registered name
    """
    model = _as_model(model)
    s = scale_px

    # Draw into a supersampled scratch surface, then shrink it. pygame's
    # ``filled_polygon`` has no antialiasing, so at this glyph size every facet
    # edge was a hard staircase, and the finer model made that worse by having
    # more edges. Drawing at SUPERSAMPLE times the size and smoothscaling down
    # is the cheapest fix and needs no change to the callers.
    #
    # ``project_fn`` is affine, so its basis can simply be probed: the large
    # scale factor keeps the integer rounding inside it from distorting the
    # result.
    origin = np.array(project_fn(np.zeros(3), 1000.0), dtype=float)
    basis = (
        np.array(
            [np.array(project_fn(e, 1000.0), dtype=float) - origin for e in np.eye(3)]
        )
        / 1000.0
    )
    half = int(max(28.0, scale_px * model.reach) * SUPERSAMPLE)
    canvas = pygame.Surface((2 * half, 2 * half), pygame.SRCALPHA)
    surf, project_fn = canvas, (
        lambda pt, sc: (
            int(
                half
                + sc * float(np.asarray(pt, dtype=float) @ basis[:, 0]) * SUPERSAMPLE
            ),
            int(
                half
                + sc * float(np.asarray(pt, dtype=float) @ basis[:, 1]) * SUPERSAMPLE
            ),
        )
    )

    # Depth axis index: the axis aligned with camera_dir
    # For side view (camera along -y): depth = y
    # For top-down (camera along -z): depth = z
    depth_axis = int(np.argmax(np.abs(camera_dir)))

    transformed = []
    for verts, base_color, name, plate in model.faces:
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

        # Lambert against a *fixed world* light, not the camera. A headlight
        # is invariant to which way the aircraft faces, so it cannot convey a
        # turn; a fixed light makes the lit side change as the aircraft comes
        # round, which is what the side panel is for. Signed, not ``abs()``:
        # the windings are normalised outward (``outward``), so a face turned
        # away from the light is genuinely dark. Plates are lit from either face.
        lambert = float(np.dot(normal, LIGHT))
        if plate:
            lambert = abs(lambert)
        shade = 0.30 + 0.70 * max(lambert, 0.0)
        color = tuple(int(c * shade) for c in base_color)

        screen_pts = [project_fn(pt, s) for pt in rotated]
        transformed.append((depth, screen_pts, color, plate))

    # Painter's algorithm — sort by camera_dir sign
    cam_sign = np.sign(camera_dir[depth_axis])
    transformed.sort(key=lambda t: -cam_sign * t[0])

    for _, screen_pts, color, plate in transformed:
        if len(screen_pts) >= 3:
            gfxdraw.filled_polygon(surf, screen_pts, color)
            # Antialias the edge in the face's own colour. It used to be
            # drawn 40 counts darker, which was a reasonable seam on fourteen
            # big slabs and became a black wireframe once the model had a
            # hundred small ones. The supersampled downscale does the real
            # smoothing; this only fills the edge pixels.
            gfxdraw.aapolygon(surf, screen_pts, color)
            if plate:
                # A plate seen edge-on, as every wing and fin is from dead
                # astern, has no area, and its one-pixel outline in the
                # supersampled canvas shrank to a third of a pixel: a Hornet
                # from behind was two nozzles and nothing else. One screen
                # pixel keeps the surface a line, as it looks.
                pygame.draw.polygon(surf, color, screen_pts, SUPERSAMPLE)

    # Passenger windows, on the side facing the camera.
    for wpt in model.windows:
        pt = R @ wpt
        if camera_dir[depth_axis] * pt[depth_axis] < 0:
            pygame.draw.circle(
                surf, (100, 100, 120), project_fn(pt, s), max(1, int(s * 0.3))
            )

    # Navigation lights — drawn last so they sit on top regardless of depth.
    # Each light gets a soft halo + a bright core so it remains visible when
    # the aircraft is small or end-on.
    # Sized in canvas units like every other primitive here: these are drawn
    # into the supersampled scratch surface, so they need the same factor or
    # they come out SUPERSAMPLE times too small once it is shrunk.
    # These are drawn into the supersampled canvas, so the radius that reaches
    # the screen is this divided by SUPERSAMPLE. Sized to land near 3 px of
    # core and 6 px of halo at the current glyph scale: the port and starboard
    # lights are the clearest cue for whether the aircraft is coming or going,
    # and at the previous factors they shrank to about a pixel and a half.
    light_r_core = max(2 * SUPERSAMPLE, int(s * model.light_radius * SUPERSAMPLE))
    light_r_halo = max(4 * SUPERSAMPLE, int(s * 2 * model.light_radius * SUPERSAMPLE))
    for pt_body, color, _name in model.nav_lights:
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

    shrunk = pygame.transform.smoothscale(
        canvas, (2 * half // SUPERSAMPLE, 2 * half // SUPERSAMPLE)
    )
    target.blit(
        shrunk,
        (int(origin[0]) - half // SUPERSAMPLE, int(origin[1]) - half // SUPERSAMPLE),
    )


def rotation_matrix(theta, phi, psi):
    """
    Build rotation matrix R = Rz(psi) @ Ry(-theta) @ Rx(-phi).

    Body frame: x=forward, y=LEFT, z=up -- right-handed, which is what the
    right-handed Rz/Ry/Rx below and the right-handed world (+x East,
    +y North, +z up) require. It read "y=right" here, which is left-handed,
    and drew the aircraft mirrored about its fuselage.
    theta = pitch (nose up = positive)
    phi   = bank  (positive = LEFT wing down, matching the dynamics)
    psi   = heading, CCW from +x/East (``arctan2(y_dot, x_dot)``), not a
            compass bearing -- see ``compass_deg`` for the display form

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


def draw_plane_sideview(surf, cx, cy, theta, phi, psi, scale_px=1.0, model="a320"):
    """Draw the 3D plane projected onto x-z plane, viewed from the south.

    The camera direction is ``+y``, not ``-y``, and the two have to agree with
    the projection below or the aircraft shows its far side.

    ``project`` maps world ``+x`` (East) to the screen right. Screen-right for
    a camera looking along ``d`` with world up ``z`` is ``d x z``, and that
    equals ``+x`` only for ``d = +y``: the camera stands to the *south* and
    looks north. The painter's sort here is the only cue for which way the
    aircraft faces, since shading uses ``abs(dot(...))`` and there is no
    back-face culling, so with ``-y`` the sort drew the far side on top.
    Measured: heading north, away from the camera, put the *nose* on top, and
    heading south put the fin on top, inverted on every heading. It shows up
    on a figure-8 as the aircraft failing to turn to face you as it comes round
    a lobe, flipping instead at the apex where ``sin(psi)`` changes sign.
    """
    R = rotation_matrix(theta, phi, psi)
    camera_dir = np.array([0, 1, 0])

    def project(pt, s):
        return (int(cx + pt[0] * s), int(cy - pt[2] * s))

    render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project, model)


def draw_plane_topdown(surf, cx, cy, theta, phi, psi, scale_px=1.0, model="a320"):
    """Draw the 3D plane projected onto x-y plane (top-down view, camera along -z)."""
    R = rotation_matrix(theta, phi, psi)
    camera_dir = np.array([0, 0, -1])

    def project(pt, s):
        return (int(cx + pt[0] * s), int(cy - pt[1] * s))

    render_3d_plane(surf, cx, cy, R, scale_px, camera_dir, project, model)


def draw_heading_dashed(surf, cx, cy, heading, length, color, dash=10, gap=8):
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


def pick_tick_interval(span_m):
    """Pick a nice altitude tick interval (in meters) for the given span."""
    # Target ~5-8 ticks on screen
    raw = span_m / 6.0
    # Round to a nice number
    for nice in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        if nice >= raw:
            return nice
    return 10000


def compass_deg(psi_rad):
    """State heading -> the compass bearing this panel's rose implies.

    ``state.psi`` is ``arctan2(y_dot, x_dot)``, a mathematical angle measured
    counter-clockwise from +x. The top-down panel maps +x to screen right and
    +y to screen up, and labels those E and N, so +x is East and +y is North.
    Printing psi raw and calling it "Heading" therefore disagreed with the
    compass rose drawn beside it in two ways at once: offset by 90 degrees and
    running the opposite way round. An aircraft tracking south-west read as
    218 where a pilot would say 234, and 0 pointed screen-right rather than
    screen-up.

    Only the display converts. ``psi`` keeps its convention in the state, the
    observation and the reward, where the maths angle is the right thing.
    """
    return (90.0 - np.rad2deg(float(psi_rad))) % 360.0


def draw_scale_bar(surf, scale, panel_w, panel_h):
    """A distance ruler for the top-down panel, plus an honesty label.

    The aircraft glyph is sized from the panel width, not from the map scale,
    so it is drawn some hundreds of times oversize: on a figure-8 the view
    spans about 21 km across 400 px, roughly 130 m per pixel, while the model
    is drawn at ~1.8 px per body metre. Without a ruler there is nothing in
    frame to calibrate against, and the aircraft appears to cross the whole
    pattern in a few of its own lengths, which reads as an aerobatic display
    rather than an airliner on 8.4 km lobes.

    Drawing the aircraft to true scale would make it a quarter of a pixel and
    lose the attitude display the panel exists for, so the glyph stays and the
    picture says so instead.
    """
    # A round number of kilometres spanning roughly a quarter of the panel.
    target_px = panel_w * 0.25
    for km in (1, 2, 5, 10, 20, 50, 100, 200):
        bar_px = km * 1000.0 * scale
        if bar_px >= target_px:
            break
    if not np.isfinite(bar_px) or bar_px <= 0 or bar_px > panel_w * 0.8:
        return

    bar_px = int(bar_px)
    font = pygame.font.SysFont(MONO, 13)
    label = font.render(f"{km} km", True, SCENE_INK)
    note = pygame.font.SysFont(MONO, 11).render(
        "aircraft not to scale", True, SCENE_DIM
    )

    # Bottom right. The task legend already owns the bottom left corner, and
    # drawing here collided with it.
    right_margin = 12
    x0 = panel_w - right_margin - label.get_width() - 8 - bar_px
    y0 = panel_h - 18
    pygame.draw.line(surf, SCENE_INK, (x0, y0), (x0 + bar_px, y0), 2)
    for x in (x0, x0 + bar_px):
        pygame.draw.line(surf, SCENE_INK, (x, y0 - 4), (x, y0 + 4), 2)
    surf.blit(label, (x0 + bar_px + 8, y0 - 9))
    surf.blit(note, (panel_w - right_margin - note.get_width(), y0 - 24))


#: How much of the side panel's scene band must separate the aircraft from the
#: ground line. The aircraft is drawn at the band's centre and a banked wingtip
#: reaches about an eighth of the band below it, so a third leaves clear air
#: under the turn and keeps the ground to a strip rather than a third of the
#: view.
_GROUND_CLEARANCE = 0.35


def side_scene(panel_w, panel_h, state, params, positions_history_xz):
    """Side elevation (x-z), centred on the aircraft, with an adaptive scale.

    Every aircraft environment draws its altitude through this one function:
    the 3D tasks as their left panel, the 2D tasks as theirs. It needs
    ``x``, ``z``, ``theta``, ``target_altitude``, ``power`` and ``time`` on the
    state, and reads ``phi``, ``psi`` and the velocity components if they are
    there.
    """
    surf = pygame.Surface((panel_w, panel_h))
    surf.fill(SKY)

    # The panel now carries a header strip and a gauge stack, so the scene is
    # centred in what is left rather than in the raw surface -- otherwise the
    # aircraft sits low and the ground line renders under the gauges.
    scene_top = HEADER_H
    scene_bot = panel_h - gauge_height(5)
    scene_h = scene_bot - scene_top
    cx, cy = panel_w // 2, (scene_top + scene_bot) // 2
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

    # Then cap it so the ground stays out of the way. The span above is driven
    # by the distance to the target, which on a fresh episode is several
    # kilometres, and zooming that far out drags the horizon up into the middle
    # of the panel: at 3 700 m with a target at 7 500 m the ground took 37% of
    # the view and the wingtips of a banked aircraft crossed it. Since the
    # scene is centred on the aircraft, the ground sits ``cur_z * scale`` below
    # centre, so requiring that to be at least GROUND_CLEARANCE of the band is
    # a cap on the span. The target line may then fall off the top, which the
    # dashed-line draw already handles, and the altitude ruler still says where
    # it is.
    z_span = min(z_span, max(min_span, cur_z * 0.4 / _GROUND_CLEARANCE))

    scale = scene_h * 0.4 / z_span
    scale = min(scale, 0.5)

    # Margins for altitude scale on the right
    alt_margin = 78

    def world_to_screen(wx, wz):
        sx = int(cx + (wx - cur_x) * scale)
        sy = int(cy - (wz - cur_z) * scale)  # z-up -> y-down
        return sx, sy

    # Ground — only draw if visible on screen
    _, ground_sy = world_to_screen(0, 0)
    if ground_sy < panel_h:
        pygame.draw.rect(surf, TERRAIN, (0, ground_sy, panel_w, panel_h - ground_sy))
        gfxdraw.hline(surf, 0, panel_w, min(ground_sy, panel_h - 1), SCENE_LINE)

    # A scrolling coordinate graticule instead of clouds. The clouds were the
    # largest objects in the panel and carried no information, which made the
    # view read as weather rather than as a simulation. Vertical rules moving
    # past at the same parallax give the same sense of speed while saying
    # "distance axis", and they cost nothing to read past.
    grat = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    spacing = max(48, int(panel_w / 9))
    offset = int(float(state.x) * scale * GRATICULE_PARALLAX) % spacing
    for gx in range(-spacing, panel_w + spacing, spacing):
        pygame.draw.line(
            grat, (*SCENE_LINE, 150), (gx - offset, 0), (gx - offset, panel_h), 1
        )
    surf.blit(grat, (0, 0))

    # ── Altitude scale (right edge) ──
    font_tick = pygame.font.SysFont(MONO, 11)
    tick_interval = pick_tick_interval(z_span)
    # Visible altitude range
    alt_top = cur_z + (scene_h / 2) / scale
    alt_bot = cur_z - (scene_h / 2) / scale
    first_tick = int(alt_bot / tick_interval) * tick_interval
    if first_tick < alt_bot:
        first_tick += tick_interval

    tick_x = panel_w - alt_margin
    # Use an SRCALPHA overlay so ticks/grid/labels are properly transparent
    tick_overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    tick_color = (*SCENE_DIM, 190)
    grid_color = (*SCENE_LINE, 120)
    label_color = (*SCENE_DIM, 220)
    axis_color = (*SCENE_DIM, 190)

    # Never below ground: the adaptive span reaches 11 km when the aircraft
    # starts far from its target, and the ticks then ran to -32 810 ft.
    alt_val = max(first_tick, 0.0)
    while alt_val <= alt_top:
        _, ty = world_to_screen(0, alt_val)
        if scene_top <= ty < scene_bot:
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
    pygame.draw.line(
        tick_overlay, axis_color, (tick_x, scene_top), (tick_x, scene_bot), 1
    )
    surf.blit(tick_overlay, (0, 0))

    # Aircraft — 3D model projected onto side view
    theta = float(state.theta)
    # The 2D aircraft has no bank or heading, so this reads them defensively and
    # draws the same mesh at wings level. One elevation panel for both, rather
    # than a second hand-rolled one that drifts.
    psi = float(getattr(state, "psi", 0.0))
    phi = float(getattr(state, "phi", 0.0))
    # Smaller than it was. At 0.0055 the glyph spanned a third of the panel,
    # which left no room between a banked wingtip and anything else in frame.
    plane_px_scale = max(1.2, min(3.0, panel_w * 0.0038))
    draw_plane_sideview(surf, cx, cy, theta, phi, psi, scale_px=plane_px_scale)

    # Target altitude dashed line
    _, tgt_sy = world_to_screen(0, tgt_alt)
    if scene_top <= tgt_sy < scene_bot:
        draw_dashed_line(
            surf,
            TARGET,
            (0, tgt_sy),
            (tick_x, tgt_sy),
            dash_length=10,
            space_length=10,
        )
        font_sm = pygame.font.SysFont(MONO, 14)
        txt_tgt = font_sm.render(
            f"Target: {int(tgt_alt * 3.281):,} ft",
            True,
            TARGET,
        )
        surf.blit(txt_tgt, (10, max(tgt_sy - 20, scene_top + 2)))

    # Trail
    if len(positions_history_xz) > 1:
        stride = max(1, len(positions_history_xz) // 300)
        pts = positions_history_xz[::stride]
        for wx, wz in pts:
            sx, sy = world_to_screen(wx, wz)
            if 0 <= sx < panel_w and scene_top <= sy < scene_bot:
                gfxdraw.circle(surf, sx, sy, 2, SCENE_DIM)
                gfxdraw.circle(surf, sx, sy, 1, ACCENT)

    # HUD
    # Simulated flight time, not playback time, and labelled as such. It used
    # to render as a bare "00:13:18" from ``time.gmtime(state.time)``, which
    # was wrong twice over. It read ``state.time`` as seconds, silently
    # assuming ``delta_t == 1``; and with a clip subsampled to 400 frames at
    # 30 fps, 800 s of flight plays in 13 s, so this clock's *minutes* digit
    # advanced once per second of playback and every viewer read it as a
    # seconds counter. A 4-minute airliner circuit then looks like a 4-second
    # aerobatic one.
    time_elapsed = format_clock(float(state.time) * float(params.delta_t))
    max_alt_diff = params.max_alt - params.min_alt
    done_alt = (state.z <= params.min_alt) or (state.z >= params.max_alt)
    if done_alt:
        reward = -1.0 * params.max_steps_in_episode
    else:
        reward = (
            (max_alt_diff - abs(state.target_altitude - state.z)) / max_alt_diff
        ) ** 2

    # True airspeed over whichever velocity components the state carries: the
    # 3D aircraft has all three, the 2D one has x and z.
    speed = float(
        np.sqrt(
            sum(float(getattr(state, f, 0.0)) ** 2 for f in ("x_dot", "y_dot", "z_dot"))
        )
    )
    alt_frac = (float(state.z) - params.min_alt) / max(max_alt_diff, 1.0)
    tgt_frac = (float(state.target_altitude) - params.min_alt) / max(max_alt_diff, 1.0)
    pitch_deg = float(np.rad2deg(state.theta))
    gauge_stack(
        surf,
        panel_w,
        panel_h,
        [
            (
                "ALTITUDE",
                f"{int(float(state.z) * 3.281):,} ft",
                alt_frac,
                ACCENT,
                tgt_frac,
            ),
            (
                "AIRSPEED",
                f"{speed * 1.944:.0f} kt",
                speed / 300.0,
                BLUE,
                None,
            ),
            (
                "PITCH",
                f"{pitch_deg:+.1f}\u00b0",
                (pitch_deg + 20.0) / 40.0,
                TEAL,
                0.5,
                0.5,
            ),
            ("POWER", f"{state.power * 100:.0f} %", float(state.power), GOOD, None),
            ("REWARD", f"{reward:.2f}", float(reward), TARGET, None),
        ],
    )

    err = abs(float(state.target_altitude) - float(state.z))
    band = err / max(max_alt_diff, 1.0)
    status = (
        ("NOMINAL", GOOD)
        if band < 0.02
        else ("WATCH", TARGET) if band < 0.08 else ("ALARM", ALERT)
    )
    header(surf, panel_w, state.time, status, time_elapsed)

    return surf


def trace_scene(panel_w, panel_h, state, params, history, *, reward=0.0):
    """Altitude against its command over the whole episode.

    The 2D tasks' counterpart to the 3D tasks' map panel. There is no lateral
    motion to draw, and what the task actually is -- hold a level, follow a
    ramp, chase a sinusoid -- only shows over the episode, which the elevation
    beside it cannot say because it is centred on the aircraft.

    ``history`` holds the ``altitude`` and ``target`` series in metres.
    """
    surf = pygame.Surface((panel_w, panel_h))
    surf.fill(SKY)
    top, bot, band_h, _ = scene_band(panel_h, 3)

    alt = list(history.get("altitude", ()))
    tgt = list(history.get("target", ()))
    n = min(len(alt), len(tgt))
    pad_l, pad_r = 14, 74
    plot_w = max(1, panel_w - pad_l - pad_r)

    lo = float(params.min_alt)
    hi = float(params.max_alt)
    if n > 1:
        seen = alt[:n] + tgt[:n]
        margin = 0.15 * max(max(seen) - min(seen), 1.0)
        lo = max(lo, min(seen) - margin)
        hi = min(hi, max(seen) + margin)
    span = max(hi - lo, 1.0)
    steps = max(int(params.max_steps_in_episode), 2)

    def to_xy(i, a):
        return (
            pad_l + int(plot_w * i / (steps - 1)),
            bot - int((band_h - 8) * (float(a) - lo) / span) - 4,
        )

    # Altitude ruler down the right, matching the elevation panel's.
    font_tick = pygame.font.SysFont(MONO, 11)
    interval = pick_tick_interval(span)
    tick_x = panel_w - pad_r
    v = max(int(lo / interval) * interval, 0.0)
    overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    while v <= hi:
        _, ty = to_xy(0, v)
        if top <= ty < bot:
            pygame.draw.line(overlay, (*SCENE_LINE, 120), (pad_l, ty), (tick_x, ty), 1)
            pygame.draw.line(
                overlay, (*SCENE_DIM, 190), (tick_x, ty), (tick_x + 6, ty), 1
            )
            lab = font_tick.render(f"{int(v * 3.281):,} ft", True, (*SCENE_DIM, 220))
            overlay.blit(lab, (tick_x + 8, ty - lab.get_height() // 2))
        v += interval
    pygame.draw.line(overlay, (*SCENE_DIM, 190), (tick_x, top), (tick_x, bot), 1)
    surf.blit(overlay, (0, 0))

    if n > 1:
        stride = max(1, n // plot_w)
        idx = list(range(0, n, stride))
        # Both traces solid. Dashing the command by segment drew nothing: at a
        # step every few pixels each segment is shorter than one dash period,
        # so ``draw_dashed_line`` emitted zero dashes and the commanded
        # altitude was simply absent from the panel.
        pygame.draw.lines(surf, TARGET, False, [to_xy(i, tgt[i]) for i in idx], 2)
        pygame.draw.lines(surf, ACCENT, False, [to_xy(i, alt[i]) for i in idx], 2)
        pygame.draw.circle(surf, ACCENT, to_xy(n - 1, alt[n - 1]), 4)

    # Axis furniture, above the gauge stack rather than under it.
    font_sm = pygame.font.SysFont(MONO, 11)
    axis_y = bot - 13
    surf.blit(font_sm.render("0", True, SCENE_DIM), (pad_l, axis_y))
    lab = font_sm.render(f"{steps} steps", True, SCENE_DIM)
    surf.blit(lab, (tick_x - lab.get_width(), axis_y))
    for i, (text, color) in enumerate((("commanded", TARGET), ("flown", ACCENT))):
        y = top + 8 + i * 13
        pygame.draw.line(surf, color, (pad_l, y), (pad_l + 14, y), 2)
        surf.blit(font_sm.render(text, True, SCENE_INK), (pad_l + 19, y - 6))

    err = abs(float(state.target_altitude) - float(state.z))
    fuel = float(getattr(state, "fuel", 0.0))
    fuel0 = float(getattr(params, "initial_fuel", 0.0)) or max(fuel, 1.0)
    cum = float(sum(history.get("reward", ())))
    gauge_stack(
        surf,
        panel_w,
        panel_h,
        [
            (
                "ALT ERROR",
                f"{err * 3.281:,.0f} ft",
                min(1.0, err / max(params.max_alt - params.min_alt, 1.0) * 8.0),
                ALERT if err * 3.281 > 500 else GOOD,
                None,
            ),
            ("FUEL", f"{fuel:,.0f} kg", min(1.0, fuel / fuel0), BLUE, None),
            ("RETURN", f"{cum:,.0f}", min(1.0, cum / max(steps, 1)), TARGET, None),
        ],
    )
    return surf
