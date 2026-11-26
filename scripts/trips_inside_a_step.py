"""Count limits crossed and recovered inside one control step. A measurement only.

A plant tests its trip once per control step, on the state the step ends on
(``base.failure_kernel``). A plant that integrates the step in N sub-steps could
leave its envelope after sub-step k and be back inside by the end: a trip the
plant itself would have called, had the step been shorter, and that nobody sees.

This counts them. For each control step of a scored run it asks the plant's own
trip test about the proposal the step would have made after k = 1 .. N - 1 of its
N sub-steps, everything else as in the full step, and reports the steps where
one of those trips and the full step does not.

Runs are the null-policy gate's: the task's test configuration, the held-out
seeds (``null_policies.HELD_OUT``), the shipped expert and every null policy.
Nothing here changes a plant: the cut-short proposals are computed beside the
run, which goes on from the full step.

    uv run python scripts/trips_inside_a_step.py OUT.json [task ...]

Covered: every plant whose sub-steps go through ``integrate_dynamics``, and
unstable_cstr, whose own sub-step loop is replayed by hand below. Not covered,
and said so in the output: plants that already read inside the step (the
compressor's soft minimum over its sub-steps, the reactor's trip at every
physics sub-step, the aircraft's worst load of the inner loop), and plants
that take one sub-step or integrate nothing.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from target_gym import integration  # noqa: E402
from target_gym.base import actuate  # noqa: E402
from target_gym.benchmark import run_policy, shipped_policy, task_info  # noqa: E402
from target_gym.null_policies import HELD_OUT, null_policies  # noqa: E402
from target_gym.registry import REGISTRY, all_specs  # noqa: E402

#: Reads inside its step already, or has nothing to read, with the reason.
NOT_COUNTED = {
    "compressor_surge": "its trip variable is a soft minimum over its sub-steps, "
    "and a flow reversal in any of them is recorded",
    "reactor": "its trip is tested at every physics sub-step of a control step",
    "battery": "stepped in closed form, no sub-steps",
    "glass_furnace": "its own stepping, one Euler step per plant step",
}
#: Times the cut-short stepper was traced: a plant that never reaches it does
#: not integrate through ``integrate_dynamics``.
reached: list = []


def _cut_short(family, k, n):
    """A stepper taking the first ``k`` of ``n`` sub-steps of ``family``."""

    def step(field, y, h, drift):
        reached.append(1)
        out, aux = integration.integrate_dynamics(
            y, h * k / n, f"{family}_{k}", compute_velocity=field
        )
        return out, jax.tree_util.tree_map(lambda m: m[-1], aux)

    return step


def watch(env, inside):
    """The plant, with ``crossed_inside`` beside its reward terms: 1 where a
    proposal cut short inside the step trips. Everything else is the plant's,
    so the copy is of a subclass of its class."""

    class Watched(type(env)):
        def step_env(self, key, state, action, params=None):
            obs, new, reward, done, info = super().step_env(key, state, action, params)
            crossed = inside(key, state, action, params)
            terms = dict(info.get("reward_terms", {}))
            terms["crossed_inside"] = crossed.astype(jnp.float32)
            return obs, new, reward, done, {**info, "reward_terms": terms}

    copy = Watched.__new__(Watched)
    copy.__dict__.update(env.__dict__)
    return copy


def through_the_integrator(env, family, n):
    """``inside(key, state, action, params)`` for a plant whose sub-steps are
    the shared integrator's."""
    short = []
    for k in range(1, n):
        integration.STEPPERS[f"first{k}of{n}"] = _cut_short(family, k, n)
        short.append(type(env)(integration_method=f"first{k}of{n}_1"))

    def inside(key, state, action, params):
        crossed = jnp.zeros((), bool)
        for env_k in short:
            crossed |= env_k.step_env(key, state, action, params)[4]["tripped"]
        return crossed

    return inside


def unstable_cstr_inside(env, n):
    """The same for unstable_cstr, whose ``advance`` loops over its own
    sub-steps: the inputs of ``compute_next_state``, then ``advance`` over the
    first k sub-steps, then the trip's own test on the temperature."""
    from target_gym.pc_gym.unstable_cstr import env as plant
    from target_gym.utils import convert_raw_action_to_range

    def inside(key, state, action, params):
        p = params
        raw, held = actuate(env, state, jnp.reshape(action, (1,)), p)
        feed = convert_raw_action_to_range(raw.reshape(()), p.I_f_min, p.I_f_max)
        second = held.time * p.delta_t
        _, T_f, f, scale = plant.events(held, p, second, plant.wander_second(held, p))
        x = jnp.stack([held.I, held.M, held.T])
        crossed = jnp.zeros((), bool)
        for k in range(1, n):
            part = p.replace(delta_t=p.delta_t * k / n)
            I, M, T = plant.advance(x, feed, part, (T_f, f, scale), f"rk4_{k}")
            proposal = held.replace(I=I, M=M, T=T, time=held.time + 1)
            crossed |= env.is_terminated(proposal, p)
        return crossed

    return inside


def measure(name):
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    family, n = re.fullmatch(r"(.+)_(\d+)", env.integration_method).groups()
    n = int(n)
    if name in NOT_COUNTED:
        return {"sub_steps": n, "not_counted": NOT_COUNTED[name]}
    if n == 1:
        return {"sub_steps": 1, "not_counted": "one sub-step: nothing inside"}
    if name == "unstable_cstr":
        inside = unstable_cstr_inside(env, n)
    else:
        reached.clear()
        try:
            inside = through_the_integrator(env, family, n)
        except ValueError as refusal:
            return {"sub_steps": n, "not_counted": f"refuses the probe: {refusal}"}
    watched = watch(env, inside)
    task = task_info(name, env, params, len(HELD_OUT))
    rows = {}
    for policy in [shipped_policy("pid"), *null_policies(name)]:
        label = getattr(policy, "name", "expert")
        run = run_policy(policy, watched, params, task, seeds=HELD_OUT)
        crossed = run["terms"]["crossed_inside"] > 0.5
        tripped = np.asarray(run["tripped"], bool)
        rows[label] = {
            "steps": int(crossed.size),
            "trips_at_step_end": int(tripped.sum()),
            "crossed_inside_and_tripped": int((crossed & tripped).sum()),
            "crossed_inside_and_recovered": int((crossed & ~tripped).sum()),
        }
    if name != "unstable_cstr" and not reached:
        return {"sub_steps": n, "not_counted": "never reaches integrate_dynamics"}
    return {"sub_steps": n, "policies": rows}


if __name__ == "__main__":
    out, names = sys.argv[1], sys.argv[2:] or [s.name for s in all_specs()]
    table = {}
    for name in names:
        try:
            table[name] = measure(name)
        except Exception as error:  # noqa: BLE001 - report and go on
            table[name] = {"error": repr(error)[:300]}
        print(name, json.dumps(table[name]), flush=True)
        pathlib.Path(out).write_text(json.dumps(table, indent=1))
