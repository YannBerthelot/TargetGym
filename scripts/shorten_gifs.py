#!/usr/bin/env python3
"""Create lightweight ``_short.gif`` copies of every GIF under ``videos/``.

Strategy: **trim** the GIF to the first N frames that fit under the target
size, keeping the original frame rate untouched.  This preserves smooth
playback — the video is simply shorter, not choppier.

If the average bytes-per-frame is very high (large resolution), a binary
search finds the maximum number of frames that fit.

Usage
-----
    python scripts/shorten_gifs.py              # default 10 MB target
    python scripts/shorten_gifs.py --target 5   # 5 MB target
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _get_gif_info(path: Path) -> tuple[list[Image.Image], list[int]]:
    """Return (frames, durations_ms) for a GIF file."""
    img = Image.open(path)
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        while True:
            frames.append(img.copy())
            durations.append(img.info.get("duration", 33))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames, durations


def _save_gif(frames: list[Image.Image], durations: list[int], out: Path) -> int:
    """Save frames as a GIF, return file size in bytes."""
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return out.stat().st_size


def shorten_gif(
    src: Path,
    dst: Path,
    target_bytes: int,
) -> tuple[int, int, int]:
    """Create a trimmed GIF.  Returns (original_bytes, new_bytes, n_frames)."""
    orig_bytes = src.stat().st_size
    frames, durations = _get_gif_info(src)
    n_orig = len(frames)

    if n_orig == 0:
        return orig_bytes, 0, 0

    # Already fits — just copy (re-save with optimize).
    if orig_bytes <= target_bytes:
        size = _save_gif(frames, durations, dst)
        return orig_bytes, size, n_orig

    # Estimate how many frames will fit (linear approximation).
    avg_bytes_per_frame = orig_bytes / n_orig
    estimate = max(1, int(target_bytes / avg_bytes_per_frame))

    # Binary search for the maximum number of frames that fits.
    lo, hi = 1, min(estimate + estimate // 2 + 10, n_orig)

    # Verify hi is actually too large (if not, the estimate was conservative).
    size_hi = _save_gif(frames[:hi], durations[:hi], dst)
    if size_hi <= target_bytes:
        # Try expanding — maybe we can fit more.
        while hi < n_orig:
            hi = min(hi * 2, n_orig)
            size_hi = _save_gif(frames[:hi], durations[:hi], dst)
            if size_hi > target_bytes:
                break
        if size_hi <= target_bytes:
            return orig_bytes, size_hi, hi

    best_n, best_size = 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        size = _save_gif(frames[:mid], durations[:mid], dst)
        if size <= target_bytes:
            best_n, best_size = mid, size
            lo = mid + 1
        else:
            hi = mid - 1

    # Re-save the best result.
    if best_n != hi + 1:
        best_size = _save_gif(frames[:best_n], durations[:best_n], dst)

    return orig_bytes, best_size, best_n


def main():
    parser = argparse.ArgumentParser(description="Shorten GIFs under videos/")
    parser.add_argument(
        "--target", type=float, default=10.0, help="Target size in MB (default: 10)"
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="videos",
        help="Root directory to scan (default: videos)",
    )
    args = parser.parse_args()

    target_bytes = int(args.target * 1024 * 1024)
    videos_dir = Path(args.videos_dir)

    if not videos_dir.is_dir():
        print(f"Directory not found: {videos_dir}", file=sys.stderr)
        sys.exit(1)

    gifs = sorted(videos_dir.rglob("*.gif"))
    # Skip existing _short.gif files.
    gifs = [g for g in gifs if not g.stem.endswith("_short")]

    if not gifs:
        print("No GIFs found.")
        return

    print(f"Processing {len(gifs)} GIF(s), target {args.target:.0f} MB ...\n")

    for gif in gifs:
        dst = gif.with_name(gif.stem + "_short.gif")
        orig_mb = gif.stat().st_size / (1024 * 1024)
        orig_bytes, new_bytes, n_frames = shorten_gif(gif, dst, target_bytes)
        n_orig = len(_get_gif_info(gif)[0])
        new_mb = new_bytes / (1024 * 1024)
        wall_time = ""
        if n_orig > 0:
            # Compute approximate wall-clock duration of the short version.
            _, durs = _get_gif_info(gif)
            total_ms = sum(durs[:n_frames])
            wall_time = f", ~{total_ms / 1000:.0f}s"
        tag = " (kept all)" if n_frames == n_orig else ""
        print(
            f"  {gif.relative_to(videos_dir)}: "
            f"{orig_mb:.1f} MB -> {new_mb:.1f} MB "
            f"({n_frames}/{n_orig} frames{wall_time}){tag}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
