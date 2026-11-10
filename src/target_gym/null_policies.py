"""The null policies: what a task must beat to be a control problem.

A task is a control problem only if something that does no real control cannot
pass for the expert (docs/proposals/industrial-fixes.md, check 1). The rule,
fixed before any result was seen: a null policy fails the task if it wins 6 or
more of the 10 held-out seeds with a ratio above 1.

The null policies know no physics. Each is a static function of the target,
with at most one integrator:

- a constant action at the middle of the box, and at each bound;
- the target echoed as the action, where the two share units;
- a quadratic table of the target, fitted to where the expert settles on the
  training seeds;
- that table with an I-only trim on each tracked output, tuned by SIMC where the
  recorded bump test is valid.

A policy a review found to beat the expert stays as a witness on that task
(``WITNESSES``). It needs no tuning rule, because a win proves the defect however
the gain was chosen, and it can only make the gate stricter.

Each is a ``benchmark.Policy``, so it runs as a reference row beside the expert::

    from target_gym.benchmark import run_policy, task_info
    from target_gym.null_policies import null_policies

    for policy in null_policies("cstr"):
        rollout = run_policy(policy, env, params, task_info(...), seeds=seeds)

The tables and gains are read from ``data/null_policies/<task>.json``, since the
bump-test evidence is outside the wheel, and each task's results are in
``data/credibility/<task>.json``. ``scripts/record_credibility.py`` writes both.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from target_gym import eval as E
from target_gym import records, scoring
from target_gym.benchmark import episodes_of, run_policy, shipped_policy, task_info
from target_gym.registry import REGISTRY

__all__ = ["EXCEPTIONS", "EXPERT_FIGURES", "HELD_OUT", "NO_FIGURE", "Null", "Table"]
__all__ += ["WINS_TO_FAIL", "credibility_row", "gate", "null_policies", "run_on_seeds"]
__all__ += ["wins_like_the_expert"]

HELD_OUT = tuple(range(200, 210))
WINS_TO_FAIL = 6

#: The action echoes the target only where it commands the tracked quantity.
ECHO = ("battery",)

_RVSM = {
    "kind": "requirement",
    "value": 65.0 * 0.3048,
    "unit": "m",
    "what": "automatic altitude control band, +-65 ft, straight and level in still "
    "air; a requirement, not a measured hold",
    "source": "14 CFR 91 Appendix G, section 2(c)(2)(i)",
    "url": "https://www.law.cornell.edu/cfr/text/14/appendix-G_to_part_91",
    "status": "read, primary",
}

#: What a credibility row says of a task with no sourced figure. Only the owner
#: can grant an exception to check 2 (docs/proposals/industrial-fixes.md).
NO_FIGURE = {"status": "no sourced figure"}

#: Per task, the sourced real-world figure its expert's hold answers to
#: (check 2), with what it is and where it comes from. A task not listed has
#: none.
_SLOT_FLOWN = {
    "kind": "measured",
    "value": 1.52,
    "unit": "m",
    "what": "F/A-18 trail aircraft, relative position within 2 ft in straight and "
    "level flight, plus up to 3 ft of GPS error; light turbulence at most, no "
    "manoeuvring, and not scaled by span. The design requirement was 10 ft",
    "source": "NASA Tech Briefs, Autonomous Formation Flight, February 2004",
    "url": "https://www.techbriefs.com/component/content/article/29246-drc0146",
    "status": "read, secondary",
}

EXPERT_FIGURES = {
    "plane": _RVSM,
    "plane_energy": _RVSM,
    "plane_sine": _RVSM,
    "plane3d_heading": _RVSM,
    "plane3d_circle": _RVSM,
    "plane3d_racetrack": _RVSM,
    "glass_furnace": {
        "kind": "band",
        "value": 10.0,
        "unit": "K",
        "what": "melting-section band of a float furnace, 1550 C +-10 C; not the "
        "crown and not a measured hold",
        "source": "a temperature-control vendor's blog, naming no original",
        "url": "https://lorentzzi.com/es/glass-production-precise-temperature-"
        "control-solutions-from-lorentzzi",
        "status": "read, secondary",
    },
    "battery": {
        "kind": "measured",
        "value": 94.0,
        "unit": "PJM performance score, percent",
        "what": "average RegD score of PJM's battery class, Nov 2015; a fleet average "
        "of Manual 12's score, which scripts/pjm_regulation_score.py computes on the "
        "expert. A 1 MW battery averaged 59 to 63 over a month (Watson et al., 2018), "
        "and PJM certifies at 75",
        "source": "PJM, Performance, Mileage and the Mileage Ratio, 2015, slide 10",
        "url": "https://www.pjm.com/-/media/DotCom/committees-groups/task-forces/rmistf/20151111/20151111-item-05-performance-based-regulation-concepts.ashx",
        "status": "read, primary",
    },
    "boiler_drum": {
        "kind": "measured",
        "value": 25.4,
        "unit": "mm",
        "what": "level variation of +-1 inch during load upsets after a "
        "feedforward-feedback upgrade, from +-4 inches, on an 80,000 pph industrial "
        "boiler, far smaller than the task's. Whether it is field or simulated is "
        "not stated. Pressure has no figure",
        "source": "Swayne, MS thesis, University of Tennessee, 1992, abstract only",
        "url": "https://trace.tennessee.edu/utk_gradthes/12293",
        "status": "read, abstract only",
    },
    "formation": _SLOT_FLOWN,
    "patrol": _SLOT_FLOWN,
    "patrol_range_bearing": _SLOT_FLOWN,
    "wake_surf": NO_FIGURE,  # no public station-keeping figure for fello'fly
    "plane3d_figure8": {
        "kind": "requirement",
        "value": 926.0,
        "unit": "m",
        "what": "flight technical error budget for RNP 1 with autopilot or flight "
        "director, 0.5 NM; a budget, not a hold",
        "source": "FAA AC 90-105A, Appendix C, C.3.1, page C-3",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/"
        "AC_90-105A.pdf",
        "status": "read, primary",
    },
    "unstable_cstr": {
        "kind": "measured",
        "value": 3.0,
        "unit": "K",
        "what": "+2 and -3 C deviations under PID at the unstable region of a "
        "laboratory CSTR, sodium thiosulfate and hydrogen peroxide; a rig, not a plant",
        "source": "Go, Gobaco and Ha, BS thesis, De La Salle University, 1995, "
        "abstract only",
        "url": "https://animorepository.dlsu.edu.ph/etd_bachelors/9284",
        "status": "read, abstract only",
    },
    "wind_turbine": {
        "kind": "requirement",
        "value": 2.0,
        "unit": "percent of rated power",
        "what": "accuracy of a completed active power control, over 1 minute, at a "
        "plant's point of connection, not one turbine; an expired revision",
        "source": "Energinet, Technical regulation 3.2.5, revision 4, section 5.1",
        "url": "https://en.energinet.dk/media/5pbflmly/technical-regulation-325-"
        "wind-power-plants-above-11-kw-rev-4_expired.pdf",
        "status": "read, primary",
    },
    "cement_kiln": {
        "kind": "target",
        "value": 0.3,
        "unit": "percent free lime",
        "what": "target on free lime variability, sigma under 0.3 %; a target, not "
        "a measured hold",
        "source": "a maintenance-software vendor's page, naming no original",
        "url": "https://oxmaint.com/industries/cement-plant/cement-plant-process-"
        "control-apc-expert-systems",
        "status": "read, secondary",
    },
}

#: The class exception to check 2 (docs/proposals/industrial-fixes.md).
_NO_FIGURE_FOUND = (
    "check 2: no published figure could be found or opened, after two searches "
    "with a re-fetch check (accepted 2026-11-06)"
)

#: Per task, the written exception the owner accepted.
EXCEPTIONS = {
    "compressor_surge": _NO_FIGURE_FOUND,
    "cstr": _NO_FIGURE_FOUND,
    "distillation": _NO_FIGURE_FOUND,
    "first_order": _NO_FIGURE_FOUND,
    "four_tank": _NO_FIGURE_FOUND,
    "hvac": _NO_FIGURE_FOUND,
    "ph_neutralization": _NO_FIGURE_FOUND,
    "reactor": f"{_NO_FIGURE_FOUND}; check 1: a plain integrator on Tavg beats the "
    "rod program at the $1 rod step, ours, and no source settles a better price "
    "(accepted 2026-11-07)",
    "unstable_cstr": "check 2: tau_j of 6 s is synthetic, and PHYSICS.md says so "
    "(accepted 2026-11-06)",
}


def wins_like_the_expert(ratio: float, win_rate: float) -> bool:
    """The gate's rule, on 10 seeds."""
    return ratio > 1.0 and win_rate >= WINS_TO_FAIL / len(HELD_OUT)


class Table(NamedTuple):
    """``table(targets) -> actions``: a quadratic in the targets, each scaled by
    ``mean`` and ``scale``, with coefficients ``coef``."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def features(self, targets) -> np.ndarray:
        z = (np.asarray(targets, float) - self.mean) / self.scale
        i, j = np.triu_indices(z.shape[1])
        return np.hstack([np.ones((len(z), 1)), z, z[:, i] * z[:, j]])

    def __call__(self, targets) -> np.ndarray:
        return self.features(targets) @ self.coef


def _rod_integrator(obs):
    """The review's rod integrator: 0.0163 per step on the reactivity, whose
    span is 0.015, moving the rod from where it is read."""
    rod = obs[:, [2]] + 2.0 / 0.015 * 0.0163 * (obs[:, [3]] - obs[:, [0]])
    return np.hstack([rod, np.zeros_like(rod)])


def _tavg_integrator(obs):
    """A plain integrator on what the rods control, moving the rod from where it is
    read. SIMC's gain on a hand bump test of the rebuilt plant: 19 K per unit of
    action, a lag of one 30 s step and a step of delay, so 1 / (2 * 19 * 2)."""
    rod = obs[:, [2]] + 0.0132 * (obs[:, [5]] - obs[:, [4]])
    return np.hstack([rod, np.zeros_like(rod)])


#: Per task, the policies a review found to beat the expert. The rebuild's
#: secondary side ended the first of these: power follows the turbine, and
#: holding it says nothing about Tavg.
WITNESSES = {
    "reactor": (
        ("rod integrator", _rod_integrator),
        ("Tavg integrator", _tavg_integrator),
    )
}


class Null:
    """``feedforward(obs) -> action``, with ``gains`` trimming each tracked
    error by an integrator that stops at the box."""

    oracle = False

    def __init__(self, name: str, feedforward, gains=None):
        self.name, self.feedforward, self.gains = name, feedforward, gains

    def reset(self, key, obs, task):
        self.task = task
        return np.zeros((task.n_episodes, task.n_act))

    def act(self, carry, obs, t):
        task, obs = self.task, np.asarray(obs, float)
        ff = self.feedforward(obs)
        if self.gains is not None:
            target = obs[:, list(task.target_index)]
            error = target - obs[:, list(task.value_index)]
            carry = carry + error @ self.gains.T
            carry = np.clip(carry, task.action_low - ff, task.action_high - ff)
        return ff + carry, carry


def run_on_seeds(policy, name: str, seeds) -> dict:
    """``run_policy`` on the test configuration of ``name``, one episode a seed."""
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    task = task_info(name, env, params, len(seeds))
    return run_policy(policy, env, params, task, seeds=seeds)


def null_policies(name: str) -> list[Null]:
    """Every null policy of ``name``, from its stored table and gains."""
    row = records.load_row("null_policies", name)
    if not row:
        raise ValueError(f"no stored null policies for {name!r}")
    spec = REGISTRY[name]
    info = task_info(name, spec.make_env(), spec.make_test_params(), 1)
    low, high = info.action_low, info.action_high
    value = [(f, low + f * (high - low)) for f in (0.0, 0.5, 1.0)]
    out = [
        Null(f"constant {f}", lambda o, u=u: np.tile(u, (len(o), 1))) for f, u in value
    ]
    targets = list(info.target_index)
    if name in ECHO:
        out.append(Null("echo", lambda o: o[:, targets]))
    table = Table(*(np.asarray(row["table"][k], float) for k in Table._fields))
    out.append(Null("table", lambda o: table(o[:, targets])))
    if row["i_gains"] is not None:
        gains = np.asarray(row["i_gains"], float)
        out.append(Null("I-only", lambda o: table(o[:, targets]), gains))
    out.extend(Null(*witness) for witness in WITNESSES.get(name, ()))
    return out


def gate(name: str) -> dict:
    """``{policy: (ratio, win_rate)}`` on the held-out seeds."""
    params = REGISTRY[name].make_test_params()
    expert = episodes_of(run_on_seeds(shipped_policy("pid"), name, HELD_OUT))
    out = {}
    for policy in null_policies(name):
        m = scoring.compare(
            episodes_of(run_on_seeds(policy, name, HELD_OUT)),
            expert,
            scoring.ceiling(name, params),
            E.scored_burn_in(name, params),
            E.fixed_settle(name),
        )
        out[policy.name] = (m["ratio"], m["win_rate"])
    return out


def credibility_row(name: str, results: dict) -> dict:
    """``name``'s credibility record from ``gate``'s results: each null's result,
    the ones that beat the expert, its sourced figure and accepted exception."""
    return {
        "nulls": {k: {"ratio": r, "win_rate": w} for k, (r, w) in results.items()},
        "null_wins": sorted(k for k, v in results.items() if wins_like_the_expert(*v)),
        "expert_figure": EXPERT_FIGURES.get(name, NO_FIGURE),
        "exception": EXCEPTIONS.get(name),
    }
