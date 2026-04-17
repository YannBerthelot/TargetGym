import argparse
import glob
import os

from tqdm import tqdm

from target_gym.runners.cstr_runner import run_all_modes as run_cstr
from target_gym.runners.cstr_runner import run_figures as run_cstr_figures
from target_gym.runners.cstr_runner import run_videos as run_cstr_videos
from target_gym.runners.first_order_runner import run_all_modes as run_first_order
from target_gym.runners.first_order_runner import run_figures as run_first_order_figures
from target_gym.runners.first_order_runner import run_videos as run_first_order_videos
from target_gym.runners.four_tank_runner import run_all_modes as run_four_tank
from target_gym.runners.four_tank_runner import run_figures as run_four_tank_figures
from target_gym.runners.four_tank_runner import run_videos as run_four_tank_videos
from target_gym.runners.plane3d_runner import run_all_modes as run_plane3d
from target_gym.runners.plane3d_runner import run_figures as run_plane3d_figures
from target_gym.runners.plane3d_runner import run_videos as run_plane3d_videos
from target_gym.runners.plane_runner import run_all_modes as run_plane
from target_gym.runners.plane_runner import run_figures as run_plane_figures
from target_gym.runners.plane_runner import run_videos as run_plane_videos
from target_gym.runners.reactor_runner import run_all_modes as run_reactor
from target_gym.runners.reactor_runner import run_figures as run_reactor_figures
from target_gym.runners.reactor_runner import run_videos as run_reactor_videos

ALL_RUNNERS = {
    "plane": run_plane,
    "plane3d": run_plane3d,
    "cstr": run_cstr,
    "first_order": run_first_order,
    "four_tank": run_four_tank,
    "reactor": run_reactor,
}

VIDEO_RUNNERS = {
    "plane": run_plane_videos,
    "plane3d": run_plane3d_videos,
    "cstr": run_cstr_videos,
    "first_order": run_first_order_videos,
    "four_tank": run_four_tank_videos,
    "reactor": run_reactor_videos,
}

FIGURE_RUNNERS = {
    "plane": run_plane_figures,
    "plane3d": run_plane3d_figures,
    "cstr": run_cstr_figures,
    "first_order": run_first_order_figures,
    "four_tank": run_four_tank_figures,
    "reactor": run_reactor_figures,
}


def _has_outputs(directory: str, ext: str) -> bool:
    """True if *directory* contains at least one file with the given extension."""
    return bool(glob.glob(os.path.join(directory, f"*.{ext}")))


def _run_selected(runners, envs, output_dir=None, output_ext=None):
    """Run selected runners, optionally skipping envs whose outputs exist.

    When *envs* is ``None`` (i.e. the user did **not** pass ``--env``),
    existing outputs in ``<output_dir>/<name>/`` cause that env to be
    skipped.  When *envs* is explicitly provided the runner always runs
    (the user asked for it specifically, e.g. ``make videos-reactor``).
    """
    selected = {k: v for k, v in runners.items() if envs is None or k in envs}
    skip_existing = envs is None and output_dir is not None and output_ext is not None
    for name, run_fn in tqdm(selected.items(), desc="Environments"):
        if skip_existing and _has_outputs(f"{output_dir}/{name}", output_ext):
            tqdm.write(
                f"\n── {name} ── (skipped, outputs exist in {output_dir}/{name}/)"
            )
            continue
        tqdm.write(f"\n── {name} ──")
        run_fn()


def run_all(envs: list[str] | None = None):
    """Run figures and video generation for the given environments (default: all)."""
    # When running everything, skip envs that already have BOTH figures and videos.
    if envs is None:
        selected = {}
        for name, run_fn in ALL_RUNNERS.items():
            has_figs = _has_outputs(f"figures/{name}", "png")
            has_vids = _has_outputs(f"videos/{name}", "gif")
            if has_figs and has_vids:
                tqdm.write(f"── {name} ── (skipped, figures & videos exist)")
            else:
                selected[name] = run_fn
        for name, run_fn in tqdm(selected.items(), desc="Environments"):
            tqdm.write(f"\n── {name} ──")
            run_fn()
    else:
        _run_selected(ALL_RUNNERS, envs)


def run_videos(envs: list[str] | None = None):
    """Run only video generation (fast, single-seed)."""
    _run_selected(VIDEO_RUNNERS, envs, output_dir="videos", output_ext="gif")


def run_figures(envs: list[str] | None = None):
    """Run only figure generation (includes multi-seed comparisons)."""
    _run_selected(FIGURE_RUNNERS, envs, output_dir="figures", output_ext="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate figures and videos for target-gym environments."
    )
    parser.add_argument(
        "--env",
        nargs="*",
        choices=list(ALL_RUNNERS.keys()),
        default=None,
        metavar="ENV",
        help=f"Environments to run (default: all). Choices: {', '.join(ALL_RUNNERS)}",
    )
    parser.add_argument(
        "--only",
        choices=["videos", "figures"],
        default=None,
        help="Run only videos or only figures (default: both)",
    )
    args = parser.parse_args()

    if args.only == "videos":
        run_videos(args.env)
    elif args.only == "figures":
        run_figures(args.env)
    else:
        run_all(args.env)
