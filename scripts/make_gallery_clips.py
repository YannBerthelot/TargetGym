#!/usr/bin/env python3
"""Regenerate the README gallery clips for the control-room environments.

Every non-pygame renderer was rebuilt on ``target_gym.render_kit``, so the
clips shipped in ``videos/`` no longer show what the environments look like.
This regenerates them from the registered PID baseline, downscales and
palette-quantises each one, and writes ``videos/<env>/pid_output_short.gif``.

The aircraft clips are pygame scene renders and are deliberately untouched.

Usage
-----
    python scripts/make_gallery_clips.py                 # every console env
    python scripts/make_gallery_clips.py --envs hvac battery
    python scripts/make_gallery_clips.py --width 700 --max-mb 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.registry import REGISTRY  # noqa: E402

# Per environment: how many control steps to run, and how many of the rendered
# frames to keep. Step counts are chosen so the clip shows the behaviour the
# environment is *about* -- a full setback recovery for HVAC, several transport
# delays for the kiln, a swell for the boiler drum.
CLIPS = {
    "first_order": dict(steps=200, frames=60),
    "cstr": dict(steps=300, frames=80),
    "four_tank": dict(steps=500, frames=90),
    "ph_neutralization": dict(steps=500, frames=90),
    "distillation": dict(steps=500, frames=90),
    "glass_furnace": dict(steps=900, frames=90),
    "reactor": dict(steps=6000, frames=100),
    "hvac": dict(steps=288, frames=72),
    "cement_kiln": dict(steps=360, frames=90),
    "boiler_drum": dict(steps=900, frames=110),
    "wind_turbine": dict(steps=600, frames=90),
    "battery": dict(steps=540, frames=100),
}


def _force_stride(env, stride: int):
    """Point the environment's render hook at a chosen frame interval.

    Each renderer carries its own stride, tuned for full-length videos, so a
    clip of a few hundred steps can come back with a couple of dozen frames.
    The hooks all take ``stride`` as a trailing keyword, so the interval can be
    driven from the frame count the clip actually wants.
    """
    cls = type(env)
    patched = []
    for attr in dir(cls):
        if not attr.startswith("render_"):
            continue
        bound = getattr(cls, attr)
        if not callable(bound):
            continue

        def wrapper(screen, state, params, frames, clock, _b=bound, _s=stride):
            return _b(screen, state, params, frames, clock, stride=_s)

        # staticmethod, or attribute access re-binds ``self`` into the first
        # positional slot and shifts every argument along by one.
        setattr(cls, attr, staticmethod(wrapper))
        patched.append((cls, attr, bound))
    return patched


def _restore(patched):
    for cls, attr, original in patched:
        setattr(cls, attr, original)


def render_frames(name: str, steps: int, want: int) -> list:
    """Run the PID baseline and collect roughly *want* rendered frames."""
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.params_cls().replace(max_steps_in_episode=steps + 5)
    policy = spec.make_pid() if spec.has_pid else env.expert_policy
    if hasattr(policy, "reset"):
        policy.reset()

    patched = _force_stride(env, max(1, steps // max(want, 1)))
    try:
        key = jax.random.PRNGKey(0)
        obs, state = env.reset_env(key, params)
        step = jax.jit(env.step_env)
        frames, screen, clock = [], None, None
        for _ in range(steps):
            try:
                action = policy(obs)
            except TypeError:
                action = policy.step(obs)
            obs, state, _, terminated, _ = step(
                key, state, jnp.atleast_1d(jnp.asarray(action)), params
            )
            frames, screen, clock = env.render(screen, state, params, frames, clock)
            if bool(terminated):
                break
    finally:
        _restore(patched)
    return frames


def subsample(frames: list, count: int) -> list:
    """Evenly thin a frame list down to *count* frames, keeping the last one."""
    if len(frames) <= count:
        return frames
    idx = np.unique(np.linspace(0, len(frames) - 1, count).round().astype(int))
    return [frames[i] for i in idx]


def write_gif(frames: list, out: Path, width: int, fps: int, colors: int) -> int:
    """Downscale, quantise and write a GIF. Returns its size in bytes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for f in frames:
        im = Image.fromarray(np.asarray(f).astype(np.uint8))
        h = round(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
        images.append(im.quantize(colors=colors, method=Image.MEDIANCUT))
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=sorted(CLIPS))
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--max-mb", type=float, default=2.6)
    args = ap.parse_args()

    for name in args.envs:
        if name not in CLIPS:
            print(f"  {name:20s} SKIPPED (not a console environment)")
            continue
        cfg = CLIPS[name]
        frames = render_frames(name, cfg["steps"], cfg["frames"])
        kept = subsample(frames, cfg["frames"])
        out = ROOT / "videos" / name / "pid_output_short.gif"

        width, colors = args.width, args.colors
        for _ in range(4):
            size = write_gif(kept, out, width, args.fps, colors)
            if size <= args.max_mb * 1e6:
                break
            # Too big: shed resolution first, down to a floor that still gives
            # 2x the width the README displays. Palette depth is the thing to
            # protect -- at 48 colours the instrument bars lose their hues, and
            # cyan and blue collapse into green.
            if width > 520:
                width = max(520, int(width * 0.85))
            else:
                colors = max(64, colors - 16)
        print(
            f"  {name:20s} {len(frames):4d} rendered -> {len(kept):3d} kept  "
            f"{width}px  {colors} colours  {size / 1e6:5.2f} MB"
        )


if __name__ == "__main__":
    main()
