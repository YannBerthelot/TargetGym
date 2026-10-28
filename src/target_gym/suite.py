"""The capability suite: where a policy stands on each kind of difficulty.

As in bsuite (Osband et al., 2020), every task is tagged with the capabilities
it probes, and a policy's scores are averaged per capability into a profile,
drawn as two radars:

- **Hold**: the holding score ``S`` of ``target_gym.scoring``, 0 at the expert
  and 1 at the ceiling.
- **Reach**: ``log(transient_ratio) / log(best known)``, 0 at the expert and 1
  at the oracle's recorded move ratio. Moves have no ceiling.

Each score is drawn at ``radius = 0.5 + 0.5 * score``, clipped to [0, 1]: the
expert is the 0.5 ring, a policy worse than the expert falls inside it, and 0
is as far below the expert as the ceiling (or the best known) is above. A task
with no room (its expert already within the ceiling, or no best known move
recorded) has no point, and a capability is the mean of its tasks' points.

Profile a policy::

    from target_gym.suite import plot_profile, references, run_suite

    out = run_suite(make_policy, seeds=10)  # make_policy(name) -> a policy
    fig = plot_profile({**references(), "mine": out["profile"]})

docs/capability-profile.md says how to read it and why each task has its tags.
"""

from __future__ import annotations

import math

import numpy as np

from target_gym import scoring

__all__ = ["AXES", "CAPABILITIES", "LABELS", "plot_profile", "profile"]
__all__ += ["radius", "references", "run_suite"]

#: The capabilities, in their order around the radar.
AXES = (
    "basic",
    "instability",
    "nonlinearity",
    "delay",
    "coupling",
    "inverse_response",
    "disturbance",
    "moving_target",
    "economic",
    "partial_observability",
    "multi_agent",
    "safety",
)

LABELS = {
    "basic": "basic",
    "instability": "instability",
    "nonlinearity": "nonlinearity",
    "delay": "delay and slow",
    "coupling": "coupling",
    "inverse_response": "inverse response",
    "disturbance": "disturbance",
    "moving_target": "moving target",
    "economic": "economic",
    "partial_observability": "partial observability",
    "multi_agent": "multi-agent",
    "safety": "safety",
}

#: The capabilities each task probes, from its PHYSICS.md. The evidence for
#: every tag is the table in docs/capability-profile.md, which a test keeps in
#: step with this one.
CAPABILITIES = {
    "first_order": ("basic",),
    "cstr": ("basic", "nonlinearity"),
    "unstable_cstr": ("instability", "disturbance", "safety"),
    "ph_neutralization": ("nonlinearity", "disturbance"),
    "four_tank": ("coupling", "inverse_response"),
    "distillation": ("coupling", "delay"),
    "boiler_drum": ("inverse_response", "coupling", "safety", "instability"),
    "cement_kiln": ("delay", "partial_observability", "nonlinearity"),
    "glass_furnace": ("delay", "partial_observability"),
    "hvac": ("delay", "economic", "partial_observability"),
    "reactor": ("delay", "partial_observability", "economic", "moving_target"),
    "compressor_surge": ("safety", "nonlinearity", "disturbance"),
    "wind_turbine": ("disturbance", "economic", "safety"),
    "battery": ("economic", "moving_target", "safety"),
    "plane": ("disturbance",),
    "plane_energy": ("moving_target", "coupling"),
    "plane_sine": ("moving_target",),
    "plane3d_heading": ("coupling",),
    "plane3d_circle": ("moving_target",),
    "plane3d_racetrack": ("moving_target",),
    "plane3d_figure8": ("moving_target",),
    "patrol": ("moving_target",),
    "patrol_bearing_only": ("moving_target", "partial_observability"),
    "formation": ("multi_agent", "moving_target"),
}

_NAN = float("nan")


def radius(s: float) -> float:
    """Where a score sits on the radar: ``0.5 + 0.5 * s`` in [0, 1]."""
    if math.isnan(s):
        return _NAN
    return min(max(0.5 + 0.5 * s, 0.0), 1.0)


def _reach_score(m: dict) -> float:
    """``log(transient_ratio) / log(best known)``; NaN without a best known
    above the expert."""
    best = m.get("oracle_transient_ratio", _NAN)
    if not best > 1.0:
        return _NAN
    return math.log(max(m["transient_ratio"], 1e-300)) / math.log(best)


def profile(results: dict, tags: dict | None = None) -> dict:
    """``{"hold": {capability: radius}, "reach": {...}}``: per capability, the
    mean radius of its tasks in ``results`` (task -> ``score_policy``'s
    numbers) that have one; NaN where none has."""
    tags = CAPABILITIES if tags is None else tags
    radii = {
        name: {"hold": radius(m["score"]), "reach": radius(_reach_score(m))}
        for name, m in results.items()
    }
    axes = dict.fromkeys(t for ts in tags.values() for t in ts)
    out = {}
    for which in ("hold", "reach"):
        out[which] = {}
        for axis in axes:
            vals = [
                radii[name][which]
                for name, ts in tags.items()
                if axis in ts and name in radii and not math.isnan(radii[name][which])
            ]
            out[which][axis] = float(np.mean(vals)) if vals else _NAN
    return out


def references(rows: dict | None = None, tags: dict | None = None) -> dict:
    """The expert's and the oracle's profiles from the recorded protocol rows
    (``scoring.recorded()``): the expert is the 0.5 ring wherever a task has
    room, and the oracle sits where its recorded hold and moves put it."""
    rows = scoring.recorded() if rows is None else rows
    expert, oracle = {}, {}
    for name, row in rows.items():
        b, o = row["pid"]["hold_tracking"], row["mpc"]["hold_tracking"]
        best = scoring.best_known(row)["oracle_transient_ratio"]
        known = {"oracle_transient_ratio": best}
        expert[name] = {"score": scoring.score(b, b, row["ceiling"])}
        expert[name] |= {"transient_ratio": 1.0, **known}
        oracle[name] = {"score": scoring.score(o, b, row["ceiling"])}
        oracle[name] |= {"transient_ratio": best, **known}
    return {"expert": profile(expert, tags), "oracle": profile(oracle, tags)}


def run_suite(make_policy, seeds: int = 10, tasks=None) -> dict:
    """Score ``make_policy(name)`` on every task, or on ``tasks``, with
    ``scoring.score_policy``, and fold the results into a profile.

    ``make_policy(name)`` returns a fresh ``policy(obs)`` or
    ``policy(obs, state)`` for task ``name``; it is called once per episode.
    Returns ``{"tasks": {name: numbers}, "profile": profile(numbers)}``.
    """
    names = list(CAPABILITIES) if tasks is None else list(tasks)
    results = {
        name: scoring.score_policy(name, lambda name=name: make_policy(name), seeds)
        for name in names
    }
    return {"tasks": results, "profile": profile(results)}


# Light-surface categorical slots, in order (the expert is the gray ring).
_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
_EXPERT_GRAY = "#898781"


def plot_profile(profiles: dict, axes=AXES):
    """Two radars side by side, hold and reach, with one polygon per profile
    in ``profiles`` (name -> ``profile``). ``"expert"`` is drawn as a dashed
    gray ring; the others take the categorical colours in order. A capability
    with no point in a polygon is skipped by it, and its label is grayed when
    no profile has one. Returns the matplotlib figure."""
    import matplotlib.pyplot as plt

    angles = np.linspace(0.0, 2.0 * np.pi, len(axes), endpoint=False)
    fig, pair = plt.subplots(
        1, 2, figsize=(13, 6.5), subplot_kw={"projection": "polar"}
    )
    tops = {"hold": "ceiling", "reach": "best known"}
    for ax, which in zip(pair, ("hold", "reach")):
        colors = iter(_COLORS)
        anywhere = np.zeros(len(axes), bool)
        for name, prof in profiles.items():
            r = np.array([prof[which].get(a, _NAN) for a in axes], float)
            ok = ~np.isnan(r)
            anywhere |= ok
            th, rr = np.r_[angles[ok], angles[ok][:1]], np.r_[r[ok], r[ok][:1]]
            if name == "expert":
                ax.plot(th, rr, "--", color=_EXPERT_GRAY, lw=1.5, label=name)
                continue
            color = next(colors)
            ax.plot(th, rr, "-o", color=color, lw=2, ms=5, label=name)
            ax.fill(th, rr, color=color, alpha=0.12)
        ax.set_xticks(angles, [LABELS[a] for a in axes], fontsize=9)
        ax.tick_params(axis="x", pad=10)
        for label, ok in zip(ax.get_xticklabels(), anywhere):
            label.set_color("#262625" if ok else "#b5b3ad")
        ax.set_ylim(0.0, 1.0)
        ticks = ["", "expert", "", tops[which]]
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], ticks, fontsize=8, color="#6b6a65")
        ax.grid(color="#e0dfda", lw=0.8)
        ax.set_title(which.capitalize(), pad=22)
        if not anywhere.any():
            ax.text(
                0.5,
                0.5,
                "nothing recorded yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#6b6a65",
            )
    handles, names = pair[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False)
    fig.subplots_adjust(bottom=0.12, wspace=0.45)
    return fig
