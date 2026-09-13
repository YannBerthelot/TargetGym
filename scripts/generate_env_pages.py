"""Generate one documentation page per environment, from the registry.

Modelled on the Gymnasium environment pages, which are the convention readers
already know: a picture, then action space, observation space, rewards, starting
state, episode end, arguments, in that order.

Generated rather than written, for the same reason ``docs/environments.md`` is:
eighteen hand-maintained pages drift the moment a parameter changes, and a
documentation page that quietly disagrees with the code is worse than none. Run

    uv run python scripts/generate_env_pages.py            # rewrite
    uv run python scripts/generate_env_pages.py --check    # CI: fail if stale

Everything on a page is read from the environment itself: spaces from
``observation_space``/``action_space``, tracked variables from
``obs_value_index``, rewards and termination from the docstrings of
``compute_reward`` and ``check_is_terminal``, arguments from the params
dataclass, baseline scores from ``data/baseline_returns.json``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

import jax  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.registry import REGISTRY, display_name  # noqa: E402

OUT_DIR = ROOT / "docs" / "environments"
BASELINES = ROOT / "data" / "baseline_returns.json"

# Where each environment's gif lives, relative to the repository root. Only
# environments with a rendered clip get a picture; the rest simply omit it.
VIDEO_CANDIDATES = ("videos/{name}/pid_output.gif",)
# One candidate, deliberately: whatever ``target_gym.runners.runners`` writes.
#
# This used to prefer a hand-placed ``*_short.gif`` and fall back to the
# generated clip, and the comment here already recorded that going wrong once --
# "the gallery showed pre-re-skin aircraft for a week". It went wrong the same
# way a second time. Thirteen pages were still showing shorts that no generator
# refreshes, so after the aircraft renderers were unified the gallery was split
# between two visual styles, and the glass furnace's short played for 1.8 s
# against the 9.8 s of every regenerated clip.
#
# A fallback chain whose first entry nothing maintains is a trap, so there is no
# chain now. If a page has no picture, that is because no clip was rendered for
# it, which is a fact worth seeing rather than papering over.
SPECIAL_VIDEOS: dict[str, str] = {}


def _video(name: str) -> str | None:
    """The clip path for *name*, without checking whether the file is there.

    It deliberately does not test for existence. The clips are a build artifact:
    ``.gitignore`` excludes ``videos/**/*.gif`` and the docs-deploy workflow
    renders them before mkdocs runs, so they are present on the published site
    and absent from a clean checkout. Branching on presence made these pages
    depend on whether the generator happened to run on a machine that had them,
    which is exactly what CI caught: pages committed from a working tree with
    clips disagreed with pages generated in CI without them, and all twenty-one
    came back stale.
    """
    for candidate in (
        SPECIAL_VIDEOS.get(name),
        *(c.format(name=name) for c in VIDEO_CANDIDATES),
    ):
        if candidate:
            return candidate
    return None


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    out: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip() and out:
            break
        out.append(line.strip())
    return " ".join(out)


def _env_module(spec):
    package = type(spec.make_env()).__module__.rsplit(".", 1)[0]
    try:
        return importlib.import_module(package + ".env")
    except ModuleNotFoundError:
        return None


def _action_labels(env, n: int) -> list[str]:
    """Action meanings, where the environment's own docstring states them.

    Several class docstrings carry a line like
    ``Action  (2): [fuel, feedwater], raw in [-1, 1]``. That is the only place
    the meanings are written down, so it is the only honest source for them --
    inventing labels here would put words in the environment's mouth.
    """
    doc = type(env).__doc__ or ""
    for line in doc.splitlines():
        if "ction" not in line or "):" not in line:
            continue
        # Everything after the arity marker. Two spellings are in use and both
        # are read here: the bracketed list the aircraft carry,
        #   Action (3,): [power, stick, aileron] each in [-1, 1]
        # and the prose form the process plants carry,
        #   Action (1,): base flow, raw in [-1, 1] -> [q3_min, q3_max]
        # The earlier version took the first "[" on the line, which on the
        # second spelling is the *range* -- so it read "-1, 1" as two labels,
        # matched no single-action environment, and left eleven of twenty-two
        # pages with a blank meaning column.
        text = line.split("):", 1)[1].strip()
        for cut in (", raw in", " raw in", " each in", "->"):
            at = text.find(cut)
            if at >= 0:
                text = text[:at]
        text = text.strip().rstrip(",").strip()
        if text.startswith("["):
            stop = text.find("]")
            if stop < 0:
                continue
            text = text[1:stop]
        parts = [x.strip() for x in text.split(",") if x.strip()]
        if len(parts) == n:
            return parts
    return []


def _space_table(space, labels: list[str]) -> str:
    low = np.broadcast_to(np.asarray(space.low, float), space.shape or (1,))
    high = np.broadcast_to(np.asarray(space.high, float), space.shape or (1,))
    rows = ["| # | meaning | min | max |", "|---|---|---|---|"]
    for i, (lo, hi) in enumerate(zip(low, high)):
        rows.append(
            f"| {i} | {labels[i] if i < len(labels) else ''} | {lo:g} | {hi:g} |"
        )
    return "\n".join(rows)


def _params_table(params, limit: int = 14) -> str:
    fields = [
        (k, v)
        for k, v in vars(params).items()
        if isinstance(v, (int, float, bool)) and not isinstance(v, bool)
    ]
    rows = ["| parameter | default |", "|---|---|"]
    for k, v in fields[:limit]:
        rows.append(
            f"| `{k}` | {v:g} |" if isinstance(v, float) else f"| `{k}` | {v} |"
        )
    if len(fields) > limit:
        rows.append(f"| … | {len(fields) - limit} more, see the params dataclass |")
    return "\n".join(rows)


def _baseline_section(name: str, spec) -> str:
    recorded = {}
    if BASELINES.exists():
        recorded = json.loads(BASELINES.read_text()).get(name, {})
    lines = []
    if not spec.has_pid:
        return f"No baseline ships for this environment. {spec.baselines_note or ''}".strip()
    if recorded.get("pid_returns"):
        pid = np.mean(recorded["pid_returns"])
        steps = recorded["steps"]
        lines.append(
            f"Measured over {recorded['seeds']} seeds on a {steps}-step episode "
            f"(see [Baselines](../baselines.md)):\n"
        )
        lines.append("| controller | return | per step |")
        lines.append("|---|---|---|")
        lines.append(f"| PID | {pid:.1f} | {pid / steps:.3f} |")
        if recorded.get("mpc_returns"):
            mpc = np.mean(recorded["mpc_returns"])
            lines.append(f"| MPC | {mpc:.1f} | {mpc / steps:.3f} |")
    else:
        lines.append("A tuned PID ships with this environment.")
    # Both degradation notes are surfaced, not just the MPC's. The homepage
    # promises that "where a baseline is weak, the docs say how weak", and a
    # weak *expert* is the case that misleads hardest: a reader who beats it
    # concludes they beat a tuned controller. Rendered in full rather than
    # truncated at the first period -- these notes carry the measured numbers
    # that make the gap checkable, and the first sentence alone drops them.
    if spec.expert_degraded:
        lines.append(
            f'\n!!! warning "The shipped expert is weak here"\n'
            f"    {spec.expert_degraded}"
        )
    if spec.mpc_degraded:
        lines.append(
            f'\n!!! warning "The MPC is not an upper bound here"\n'
            f"    {spec.mpc_degraded.split('.')[0]}."
        )
    return "\n".join(lines)


def page(name: str, spec) -> str:
    env = spec.make_env()
    params = spec.make_test_params()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    module = _env_module(spec)

    obs_space = env.observation_space(params)
    act_space = env.action_space(params)
    n_obs = int(np.prod(obs_space.shape or (1,)))
    tracked = env.obs_value_index
    tracked = tracked if isinstance(tracked, tuple) else (tracked,)

    from target_gym.runners.runners import _tracked_labels

    try:
        labels = _tracked_labels(env, len(tracked))
    except Exception:
        labels = []

    title = display_name(name)
    video = _video(name)
    from target_gym.registry import control_step_seconds

    dt = control_step_seconds(env, params)
    episode = int(params.max_steps_in_episode)

    out = [f"# {title}", ""]
    if video:
        out += [
            # ``../../``, not ``../``.
            #
            # mkdocs serves these with directory URLs, so this page is
            # ``/environments/<name>/`` and a relative source resolves against
            # that directory, not against ``/environments/``. With one level the
            # browser asked for ``/environments/videos/<name>/pid_output.gif``
            # and got a 404 on every environment page, while the homepage
            # worked because it sits at the site root. mkdocs rewrites relative
            # links in Markdown but not inside raw HTML, which is why this has
            # to be right here.
            f'<p align="center"><img src="../../{video}" width="480px"/></p>',
            "",
        ]
    if module and module.__doc__:
        out += [_first_paragraph(module.__doc__), ""]

    out += [
        "| | |",
        "|---|---|",
        f"| Action space | `Box({act_space.shape or (1,)})`, all actions in [-1, 1] |",
        f"| Observation space | `Box({obs_space.shape or (1,)})` |",
        f"| Tracked variable(s) | {', '.join(labels) if labels else 'see below'} |",
        f"| Episode length | {episode} steps ({episode * dt:g} s at {dt:g} s per step) |",
        f"| Import | `from target_gym import {type(env).__name__}, "
        f"{type(params).__name__}` |",
        f"| Cite as | `{spec.versioned_name}` |",
        "",
        "## Action space",
        "",
        "Actions are normalised to `[-1, 1]` and mapped onto the plant's real",
        "actuator range inside the environment.",
        "",
        _space_table(
            act_space, _action_labels(env, act_space.shape[0] if act_space.shape else 1)
        ),
        "",
        "## Observation space",
        "",
        f"{n_obs} values. Indices {tracked} carry the tracked variable(s) that the",
        "reward scores.",
        "",
        "## Rewards",
        "",
    ]
    reward_doc = _first_paragraph(
        getattr(module, "compute_reward", None).__doc__
        if module and hasattr(module, "compute_reward")
        else None
    )
    out += [
        reward_doc or "See the environment's `compute_reward`.",
        "",
        "Every environment in this suite scores on one contract: the reward is",
        "`(tracking terms, multiplied) x (1 - weighted costs)`, bounded in",
        "`[0, 1]`, and reaches 1 only while the target is held exactly. See",
        "[Reward shaping](../reward-shaping.md).",
        "",
        "## Starting state",
        "",
        f"`reset` samples the initial condition and the target; state has "
        f"{len(state.__dataclass_fields__)} fields.",
        "",
        "## Episode end",
        "",
    ]
    term_doc = _first_paragraph(
        getattr(module, "check_is_terminal", None).__doc__
        if module and hasattr(module, "check_is_terminal")
        else None
    )
    out += [
        f"**Termination.** {term_doc or 'See `check_is_terminal`.'}",
        "",
        f"**Truncation.** After {episode} steps.",
        "",
        "## Baselines",
        "",
        _baseline_section(name, spec),
        "",
        "## Arguments",
        "",
        _params_table(params),
        "",
    ]
    return "\n".join(out) + "\n"


def build() -> dict[str, str]:
    return {name: page(name, spec) for name, spec in REGISTRY.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if any page is stale")
    args = ap.parse_args()

    pages = build()
    if args.check:
        stale = []
        for name, content in pages.items():
            path = OUT_DIR / f"{name}.md"
            if not path.exists() or path.read_text() != content:
                stale.append(name)
        if stale:
            print(f"stale environment pages: {stale}; run {__file__}")
            return 1
        print(f"{len(pages)} environment pages are up to date")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in pages.items():
        (OUT_DIR / f"{name}.md").write_text(content)
    print(f"wrote {len(pages)} pages to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
