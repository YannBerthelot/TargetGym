"""The null-policy gate (docs/proposals/industrial-fixes.md, check 1).

A task is a control problem only if something that does no real control cannot
pass for the expert. The rule, fixed before any result was seen: a null policy
fails the task if it wins 6 or more of the 10 held-out seeds with a ratio above
1. A task that fails is a strict xfail naming the finding, and the fix removes
its own line, so a fix that works goes red until it does.

The null policies know no physics. Each is a static function of the target,
with at most one integrator:

- a constant action at the middle of the box, and at each bound;
- the target echoed as the action, where the two share units;
- a quadratic table of the target, fitted to where the expert settles on the
  training seeds;
- that table with an I-only trim on each tracked output, tuned by SIMC where the
  recorded bump test is valid.

A policy a review found to beat the expert stays here as a witness on that task
(``WITNESSES``). It needs no tuning rule, because a win proves the defect however
the gain was chosen, and it can only make the gate stricter.
"""

from __future__ import annotations

import json
import pathlib
import statistics

import numpy as np
import pytest

from target_gym import eval as E
from target_gym import scoring
from target_gym.benchmark import episodes_of, run_policy, shipped_policy, task_info
from target_gym.identification import TRAIN_SEEDS
from target_gym.registry import REGISTRY

HELD_OUT = tuple(range(200, 210))
WINS_TO_FAIL = 6
EVIDENCE = pathlib.Path(__file__).parent.parent / "evidence" / "oracle_horizons"

#: The action echoes the target only where it commands the tracked quantity.
ECHO = ("battery",)

#: Per task, the recorded pair (action index, name, tracked name) that holds each
#: tracked output, as the horizon protocol pairs them (H18). A task with no
#: recorded bump test on a raw action has no I-only null: the aircraft have none
#: yet, and ``unstable_cstr``'s goes through the expert's inner loop.
PAIRS = {
    "first_order": ((0, "u", "x"),),
    "cstr": ((0, "coolant", "C_a"),),
    "ph_neutralization": ((0, "base", "pH"),),
    "glass_furnace": ((0, "fuel", "T_crown"),),
    "reactor": ((0, "rod", "n"),),
    "hvac": ((0, "heating", "T_air"),),
    "battery": ((0, "power", "P"),),
    "four_tank": ((0, "pump1", "h1"), (1, "pump2", "h2")),
    "distillation": ((0, "L", "yD"), (1, "V", "xB")),
    "boiler_drum": ((1, "feedwater", "level"), (0, "fuel", "pressure")),
    "cement_kiln": ((0, "fuel", "lime"),),
    "compressor_surge": ((0, "speed", "dp"),),
    "wind_turbine": ((1, "torque", "P"),),
}


def wins_like_the_expert(ratio: float, win_rate: float) -> bool:
    """The gate's rule, on 10 seeds."""
    return ratio > 1.0 and win_rate >= WINS_TO_FAIL / len(HELD_OUT)


def fit_table(targets, actions):
    """``table(targets) -> actions``: a least-squares quadratic in the targets,
    scaled per output. A target that never moves leaves the mean action."""
    mean, scale = targets.mean(axis=0), targets.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)

    def features(t):
        z = (np.asarray(t, float) - mean) / scale
        i, j = np.triu_indices(z.shape[1])
        return np.hstack([np.ones((len(z), 1)), z, z[:, i] * z[:, j]])

    coef = np.linalg.lstsq(features(targets), actions, rcond=1e-6)[0]
    return lambda t: features(t) @ coef


def simc_i_gain(k: float, tau: float, theta: float) -> float:
    """The I-only gain per step of a first-order lag with delay, both in steps.
    SIMC's rule for a pure delay is ``1 / (k (tau_c + theta))``. The lag counts as
    delay, the half rule's cautious limit, and ``tau_c`` equals that delay."""
    return 1.0 / (2.0 * k * (tau + theta))


def i_gains(name: str, n_act: int, n_out: int, step: float) -> np.ndarray | None:
    """``(n_act, n_out)`` gains per step from the recorded bump tests, or None
    with no usable test. A pair needs a valid first-order fit on most seeds: an
    invalid fit reads something else (``reactor``'s reads the slow thermal tail,
    and tunes a loop 300 times too slow), and a pure integrator around an
    integrator cannot settle."""
    if name not in PAIRS:
        return None
    rows = json.loads((EVIDENCE / f"{name}.json").read_text())
    rows = next(v for k, v in rows.items() if k in ("analyses", "pairs", "tests"))
    gains = np.zeros((n_act, n_out))
    for out, (act, mv, cv) in enumerate(PAIRS[name]):
        fits = [
            r["fit"]
            for r in rows
            if (r["mv"], r["cv"], r["label"]) == (mv, cv, "valid")
            and r["fit"]["model"] == "fopdt"
        ]
        if len(fits) * 2 <= len(TRAIN_SEEDS):
            continue
        delay = [0.0 if isinstance(f["theta"], str) else f["theta"] for f in fits]
        gains[act, out] = simc_i_gain(
            statistics.median(f["k"] for f in fits),
            statistics.median(f["tau"] for f in fits) / step,
            statistics.median(delay) / step,
        )
    return gains if gains.any() else None


def _rod_integrator(obs):
    """The review's rod integrator: 0.0163 per step on the reactivity, whose
    span is 0.015, moving the rod from where it is read."""
    return obs[:, [2]] + 2.0 / 0.015 * 0.0163 * (obs[:, [3]] - obs[:, [0]])


def _tavg_integrator(obs):
    """A plain integrator on what the rods control, moving the rod from where it is
    read. SIMC's gain on a hand bump test of the rebuilt plant: 19 K per unit of
    action, a lag of one 30 s step and a step of delay, so 1 / (2 * 19 * 2)."""
    return obs[:, [2]] + 0.0132 * (obs[:, [5]] - obs[:, [4]])


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


def _rollout(policy, name: str, seeds) -> dict:
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    task = task_info(name, env, params, len(seeds))
    return run_policy(policy, env, params, task, seeds=seeds)


def settled_points(rollout: dict) -> tuple[np.ndarray, np.ndarray]:
    """(targets, actions): the mean of each over the last quarter of every
    target cycle, where the controller has settled if it ever does."""
    targets, actions = [], []
    for i in range(len(rollout["length"])):
        cycle = np.flatnonzero(rollout["target_change"][i])
        cuts = [0, *cycle, int(rollout["length"][i])]
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:  # a target set on the first step
                continue
            tail = slice(b - max((b - a) // 4, 1), b)
            targets.append(rollout["target"][i, tail].mean(axis=0))
            actions.append(rollout["action"][i, tail].mean(axis=0))
    return np.array(targets), np.array(actions)


def null_policies(name: str) -> list[Null]:
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    info = task_info(name, env, params, 1)
    low, high = info.action_low, info.action_high
    value = [(f, low + f * (high - low)) for f in (0.0, 0.5, 1.0)]
    out = [
        Null(f"constant {f}", lambda o, u=u: np.tile(u, (len(o), 1))) for f, u in value
    ]
    targets = list(info.target_index)
    if name in ECHO:
        out.append(Null("echo", lambda o: o[:, targets]))
    table = fit_table(
        *settled_points(_rollout(shipped_policy("pid"), name, TRAIN_SEEDS))
    )
    out.append(Null("table", lambda o: table(o[:, targets])))
    step = float(params.delta_t)
    gains = i_gains(name, info.n_act, len(targets), step)
    if gains is not None:
        out.append(Null("I-only", lambda o: table(o[:, targets]), gains))
    out.extend(Null(*witness) for witness in WITNESSES.get(name, ()))
    return out


def gate(name: str) -> dict:
    """``{policy: (ratio, win_rate)}`` on the held-out seeds."""
    params = REGISTRY[name].make_test_params()
    expert = episodes_of(_rollout(shipped_policy("pid"), name, HELD_OUT))
    out = {}
    for policy in null_policies(name):
        m = scoring.compare(
            episodes_of(_rollout(policy, name, HELD_OUT)),
            expert,
            scoring.ceiling(name, params),
            E.scored_burn_in(name, params),
            E.fixed_settle(name),
        )
        out[policy.name] = (m["ratio"], m["win_rate"])
    return out


#: What an aircraft expert loses to, until PR 9. TECS's ``time_const`` of 2 pulls
#: past the limit load on a height capture, and each trip costs far more than a
#: null policy's error. With the protocol's start of 5 all seven tasks pass.
_STALE_TECS = "PR 9: TECS's time_const 2 trips the new envelope; at 5 this passes"

#: The tasks a null policy beats today, each a finding of the industrial review.
#: The PR that fixes one removes its line.
KNOWN = {
    "cstr": "the expert was tuned on an instant analyser; PR 9 retunes it",
    "distillation": "the expert was tuned on instant analysers; PR 9 retunes it",
    "ph_neutralization": "the expert's gains predate the probe, so it cycles; PR 9 "
    "retunes them",
    "plane": _STALE_TECS,
    "plane3d_circle": _STALE_TECS,
    "plane3d_figure8": _STALE_TECS,
    "plane3d_heading": _STALE_TECS,
    "plane3d_racetrack": _STALE_TECS,
    "plane_energy": _STALE_TECS,
    "plane_sine": _STALE_TECS,
    "reactor": "PR 9: an integrator on Tavg beats the rod program's dead band",
    "unstable_cstr": "the expert's gains predate the thermowell and the valve, so it "
    "trips; PR 9 retunes them",
}


@pytest.mark.parametrize(
    "ratio, win_rate, fails",
    [
        (1.2, 0.6, True),
        (1.2, 0.5, False),  # 5 of 10 wins is not enough
        (1.0, 0.9, False),  # a tie on the ratio
        (0.9, 0.8, False),  # most seeds, but dearer overall
    ],
)
def test_a_null_policy_fails_a_task_with_six_wins_and_a_ratio_above_1(
    ratio, win_rate, fails
):
    assert wins_like_the_expert(ratio, win_rate) is fails


def test_the_table_recovers_a_quadratic_map_of_one_target():
    targets = np.linspace(-1.0, 3.0, 9)[:, None]
    actions = 1.0 + 2.0 * targets + targets**2  # at t = 2: 1 + 4 + 4 = 9
    np.testing.assert_allclose(fit_table(targets, actions)(np.array([[2.0]])), [[9.0]])


def test_a_target_that_never_moves_gets_the_mean_action():
    targets, actions = np.full((4, 1), 5.0), np.array([[1.0], [2.0], [3.0], [6.0]])
    np.testing.assert_allclose(fit_table(targets, actions)(np.array([[5.0]])), [[3.0]])


def test_the_i_gain_is_simcs_with_the_lag_counted_as_delay():
    """k 2, lag 8 steps and delay 2: 1 / (2 * 2 * (8 + 2)) = 0.025 per step."""
    assert simc_i_gain(2.0, 8.0, 2.0) == pytest.approx(0.025)


def test_an_invalid_bump_test_tunes_no_loop():
    """All 10 of ``reactor``'s fits are invalid, ``cstr``'s are all valid. Those
    fits read ``n``, from before the rebuild; PR 9 measures Tavg's."""
    assert i_gains("reactor", 1, 1, 1.0) is None
    assert i_gains("cstr", 1, 1, 0.25)[0, 0] != 0.0


def test_a_task_with_no_bump_test_has_no_i_only_null():
    assert i_gains("plane", 2, 1, 1.0) is None


def test_the_trim_removes_the_error_a_table_leaves():
    """``first_order`` with a table that is wrong by a constant: the integrator
    reads the error and takes it out, and it stops at the box."""
    name = "first_order"
    params = REGISTRY[name].make_test_params()
    gains = i_gains(name, 1, 1, float(params.delta_t))
    wrong = Null("off", lambda o: np.zeros((len(o), 1)), gains)
    out = _rollout(wrong, name, (0,))
    assert abs(out["value"][0, -1, 0] - out["target"][0, -1, 0]) < 0.05
    assert np.abs(out["action"]).max() <= 1.0 + 1e-6


@pytest.mark.large
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(strict=True, reason=KNOWN[n]) if n in KNOWN else (),
        )
        for n in sorted(REGISTRY)
    ],
)
def test_no_null_policy_passes_for_the_expert(name):
    fails = {k: v for k, v in gate(name).items() if wins_like_the_expert(*v)}
    assert not fails, f"{name}: (ratio, win rate) {fails}"
