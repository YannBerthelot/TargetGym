"""Composite environment clips into mosaics, one per registry group.

A row of separate GIFs looks ragged for two reasons that no amount of HTML
fixes: their aspect ratios run from 3.0:1 to 1.87:1, so nothing lines up, and
their frame counts differ (30 to 450), so they drift out of phase and the grid
shimmers. One pre-rendered mosaic solves both -- uniform tiles, a single loop,
and one request instead of twelve.

    uv run python scripts/make_gallery_mosaic.py --set flagship
    uv run python scripts/make_gallery_mosaic.py --set all

``flagship`` is the curated four the homepage carries, one per registry group,
laid out two across so each tile renders at roughly twice the width the old
four-column grid gave it. The group sets are the full gallery, and they are
built *from the registry* rather than from a hand-written list, so an
environment cannot be added without appearing in its group's mosaic.

Writes ``videos/mosaic_<set>.webp``, which is what the README and the
documentation embed, plus a palette-quantised ``.gif`` beside it.

Note that the ``.gif`` is a local convenience only: ``.gitignore`` excludes
``videos/**/*.gif`` apart from ``*_short.gif``, so it is never committed and
therefore never served. It cannot act as a fallback for a reader whose browser
lacks animated WebP -- if that fallback is ever wanted, the ignore rule has to
change first.
"""

from __future__ import annotations

import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.registry import GROUPS, REGISTRY, display_name  # noqa: E402

# The four the homepage carries: one per registry group, so the curation says
# something about the library's spread rather than being four clips someone
# liked. All four share the console's 1.87:1 frame, which the 3D aircraft clips
# (3.0:1) do not -- mixing those in letterboxes every tile down to the shortest,
# and at two columns that waste is very visible.
# Captions name the difficulty in plain terms. "non-minimum phase" was here and
# told a reader nothing; the four-tank's actual problem is that the obvious
# pairing of valves to tanks is unstable, which is both concrete and the reason
# the shipped PID crosses its loops.
FLAGSHIP = (
    ("plane_energy", "Aircraft - hold altitude and airspeed at once"),
    ("four_tank", "Process - the obvious valve pairing is unstable"),
    ("glass_furnace", "Industrial - 6 of 9 states hidden"),
    ("wind_turbine", "Energy - turbulent inflow, unmeasured"),
)

# Columns per set. Two for the flagship is the whole point of it: the tiles are
# the same pixels as before, but at half the columns they render twice as wide.
COLS = {"flagship": 2, "aircraft": 4, "process": 3, "industrial": 3, "energy": 2}

# Tiles are the sources' own size. Upscaling a 646-wide console to fill a larger
# tile only softens it; making the *grid* narrower is what makes a tile read
# bigger on a page that renders at width:100%.
TILE_W, TILE_H = 646, 346
PAD = 8
LABEL_H = 44

# The console palette, not white. A white surround around a near-black console
# frames every tile in glare and is the first thing the eye lands on; these are
# render_kit's BG, PANEL, TEXT and FRAME, so the mosaic reads as one dark
# instrument panel rather than twelve pictures pinned to a wall.
BG = (8, 12, 20)  # render_kit.BG     #080c14
LABEL_BG = (14, 24, 36)  # render_kit.PANEL  #0e1824
LABEL_FG = (196, 216, 236)  # render_kit.TEXT   #c4d8ec
RULE = (30, 50, 72)  # render_kit.FRAME  #1e3248

FONT_SIZE = 23
N_FRAMES = 40  # enough to read the motion; the file is a hero image, not a demo
DURATION_MS = 100
COLORS = 64  # the palette scripts/shorten_gifs.py already uses for the gallery
QUALITY = 72

# Environments whose clip is shared with another: the two patrol variants render
# the same formation.
SPECIAL_VIDEOS = {
    "patrol": "videos/patrol/pid_formation_short.gif",
    "patrol_range_bearing": "videos/patrol/pid_formation_short.gif",
}
VIDEO_CANDIDATES = (
    "videos/{name}/pid_output_short.gif",
    "videos/{name}/pid_output.gif",
)


def _font(size: int):
    """A real typeface at a readable size, not PIL's 11px bitmap default.

    matplotlib is already a dependency and ships DejaVu Sans, so this needs no
    new package and resolves the same way on every platform the suite runs on.
    """
    try:
        import matplotlib.font_manager as fm

        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def _video(name: str) -> str | None:
    for candidate in (
        SPECIAL_VIDEOS.get(name),
        *(c.format(name=name) for c in VIDEO_CANDIDATES),
    ):
        if candidate and (ROOT / candidate).exists():
            return candidate
    return None


def _title(name: str) -> str:
    return display_name(name)


def sets() -> dict[str, list[tuple[str, str]]]:
    """``{set name: [(clip path, caption)]}`` for the flagship and every group."""
    out: dict[str, list[tuple[str, str]]] = {}
    flagship = [
        (path, caption)
        for name, caption in FLAGSHIP
        if (path := _video(name)) is not None
    ]
    out["flagship"] = flagship

    for group in GROUPS:
        tiles = [
            (path, _title(name))
            for name, spec in REGISTRY.items()
            if spec.group == group and (path := _video(name)) is not None
        ]
        if tiles:
            out[group] = tiles
    return out


def _frames(path: pathlib.Path, n: int) -> list[Image.Image]:
    """``n`` frames from a clip, letterboxed onto a uniform tile.

    Sources are cycled rather than stretched: a clip with 450 frames is sampled
    ``i % 450`` so every tile advances at the same rate and the mosaic loops as
    one animation.
    """
    src = Image.open(path)
    raw = [f.convert("RGB") for f in ImageSequence.Iterator(src)]
    out = []
    for i in range(n):
        frame = raw[i % len(raw)]
        scale = min(TILE_W / frame.width, (TILE_H - LABEL_H) / frame.height)
        size = (max(int(frame.width * scale), 1), max(int(frame.height * scale), 1))
        tile = Image.new("RGB", (TILE_W, TILE_H), BG)
        tile.paste(
            frame.resize(size, Image.LANCZOS),
            ((TILE_W - size[0]) // 2, (TILE_H - LABEL_H - size[1]) // 2),
        )
        out.append(tile)
    return out


def build(set_name: str, tiles: list[tuple[str, str]]) -> int:
    cols = COLS.get(set_name, 3)
    rows = -(-len(tiles) // cols)
    out_webp = ROOT / "videos" / f"mosaic_{set_name}.webp"
    out_gif = ROOT / "videos" / f"mosaic_{set_name}.gif"

    if not tiles:
        print(f"  {set_name}: no clips found, skipped")
        return 1
    clips = [_frames(ROOT / rel, N_FRAMES) for rel, _ in tiles]
    labels = [label for _, label in tiles]

    font = _font(FONT_SIZE)
    width = cols * TILE_W + (cols + 1) * PAD
    height = rows * TILE_H + (rows + 1) * PAD
    frames = []
    for i in range(N_FRAMES):
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)
        for k, (clip, label) in enumerate(zip(clips, labels)):
            col, row = k % cols, k // cols
            x = PAD + col * (TILE_W + PAD)
            y = PAD + row * (TILE_H + PAD)
            canvas.paste(clip[i], (x, y))
            top = y + TILE_H - LABEL_H
            draw.rectangle([x, top, x + TILE_W, y + TILE_H], fill=LABEL_BG)
            draw.line([x, top, x + TILE_W, top], fill=RULE, width=1)
            draw.text(
                (x + 14, top + (LABEL_H - FONT_SIZE) // 2 - 2),
                label,
                fill=LABEL_FG,
                font=font,
            )
        frames.append(canvas)

    out_webp.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_webp,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=DURATION_MS,
        loop=0,
        quality=QUALITY,
        method=6,
    )
    print(
        f"  wrote {out_webp.relative_to(ROOT)}  {width}x{height}  "
        f"{len(tiles)} tiles  {out_webp.stat().st_size / 1e6:.2f} MB"
    )

    small = [f.resize((width // 2, height // 2), Image.LANCZOS) for f in frames[::2]]
    palette = small[0].quantize(colors=COLORS, method=Image.MEDIANCUT)
    quantised = [
        f.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for f in small
    ]
    quantised[0].save(
        out_gif,
        save_all=True,
        append_images=quantised[1:],
        duration=DURATION_MS * 2,
        loop=0,
        optimize=True,
    )
    return 0


def main() -> int:
    import argparse

    available = sets()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=[*sorted(available), "all"], default="flagship")
    args = ap.parse_args()

    names = sorted(available) if args.set == "all" else [args.set]
    return max(build(n, available[n]) for n in names)


if __name__ == "__main__":
    raise SystemExit(main())
