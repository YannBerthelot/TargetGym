"""Every registered MPC is built and driven.

Before this, four of the sixteen MPC baselines were ever constructed by the
suite. The other twelve -- the reactor, the furnace, the kiln, the drum, the
battery and the rest -- could have stopped building, or started returning
actions the environment cannot accept, and nothing would have said so. An MPC
is one half of what a learned policy is measured against, so a silently broken
one quietly moves the bar.

These construct each MPC, drive it in closed loop for a few steps, and check
the contract the environment relies on: an action of the right shape, inside
the action space, finite, and leaving the plant in a finite state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.provenance import baseline_fingerprint, load_recorded_baselines
from target_gym.registry import REGISTRY

MPC_ENVS = [name for name, spec in REGISTRY.items() if spec.has_mpc]
CLOSED_LOOP_STEPS = 3


def _bounds(space, shape):
    low = np.broadcast_to(np.asarray(space.low, dtype=float), shape)
    high = np.broadcast_to(np.asarray(space.high, dtype=float), shape)
    return low, high


def _cheap_mpc(spec, env, params):
    """Build the MPC with the smallest horizon its factory will accept.

    What this test checks -- action shape, bounds, finiteness, and a finite
    plant afterwards -- is independent of how far the controller plans. The
    shipped horizon is not: it sets the size of the traced rollout, and for the
    four aircraft that made three closed-loop steps cost 82 s of the fast CI
    job, more than half of it. A gradient MPC compiles a rollout ``horizon``
    deep and then runs ``n_iter`` optimiser passes over it, per instance.

    Shrinking both keeps every assertion below meaningful and removes the
    compile. The shipped configuration is still exercised in full by the slow
    quality contract further down this file, which is where a bad horizon would
    show up as bad control rather than as a bad action shape.
    """
    # The registry wraps every factory as ``make(env, params, **kwargs)``, so the
    # signature says nothing about which knobs the underlying builder accepts --
    # the gradient controllers take all three, the CasADi ones only ``horizon``.
    # Try the most reduced form first and fall back until one is accepted.
    for kwargs in (
        {"horizon": 5, "n_iter": 2, "n_tail": 0},
        {"horizon": 5, "n_iter": 2},
        {"horizon": 5},
        {},
    ):
        try:
            return spec.make_mpc(env, params, **kwargs)
        except (TypeError, ValueError):
            continue
    return spec.make_mpc(env, params)


@pytest.mark.parametrize("name", MPC_ENVS)
def test_registered_mpc_builds_and_controls(name):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 20})

    mpc = _cheap_mpc(spec, env, params)
    mpc.reset()

    space = env.action_space(params)
    expected = space.shape or (1,)

    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)

    for i in range(CLOSED_LOOP_STEPS):
        obs = env.get_obs(state, params)
        action = np.atleast_1d(np.asarray(mpc.step(obs, state), dtype=float))

        assert (
            action.shape == expected
        ), f"{name}: MPC returned {action.shape}, action space is {expected}"
        assert np.all(np.isfinite(action)), f"{name}: non-finite action at step {i}"
        low, high = _bounds(space, action.shape)
        assert np.all(action >= low - 1e-6) and np.all(
            action <= high + 1e-6
        ), f"{name}: action {action} outside [{low}, {high}] at step {i}"

        _, state, reward, terminated, _ = step(key, state, jnp.asarray(action), params)
        assert np.isfinite(float(reward)), f"{name}: non-finite reward at step {i}"
        if bool(terminated):
            break

    for leaf in jax.tree_util.tree_leaves(state):
        assert np.all(
            np.isfinite(np.asarray(leaf))
        ), f"{name}: MPC drove the plant to a non-finite state"


def test_every_environment_without_an_mpc_says_why():
    """A missing baseline must be documented, not silent."""
    undocumented = [
        name
        for name, spec in REGISTRY.items()
        if not spec.has_mpc and not spec.baselines_note
    ]
    assert (
        not undocumented
    ), f"environments with no MPC and no baselines_note: {undocumented}"


# ─── The quality contract ───────────────────────────────
#
# Everything above checks that an MPC *runs*. Nothing checked that it controls,
# and for a long time several did not: the wind turbine returned -0.02 against
# the PID's 393, the battery -61, the glass furnace -23, and the 2D aircraft flew
# into the ground. All of them passed the tests above, because emitting finite,
# in-bounds actions is exactly what a controller that has given up does.
#
# That gap mattered more than an ordinary baseline bug. The MPC is what a learned
# policy is measured against, so "RL beat MPC" on any of those environments was a
# statement about a broken baseline, not about RL.
#
# What this catches, measured by reverting each fix and re-running:
#
#   wind turbine surrogate removed  -> caught (terminates at step 20 of 400)
#   battery surrogate removed       -> caught (-27.5% of the PID)
#   glass furnace scale reverted    -> NOT caught
#
# The furnace case is the honest limit. That bug costs -17.6% of the PID over ten
# seeds but only -4.4% over the five this can afford, against +3.7% when fixed --
# an eight-point gap that no sane tolerance separates. Tightening the threshold
# until it caught this one bug would be fitting the test to a known answer and
# would make it flaky. A subtle objective error is below this contract's
# resolution; the ten-seed table in docs/baselines.md is what finds those.
#
# The tolerance is set from the defects rather than chosen. As a fraction of the
# PID's return the gross failures cost -100% (turbine), -88% (four tank) and
# -27% (battery), while every environment that legitimately sits level with its
# PID is within 4% once fixed. 10% separates those two populations with room on
# both sides.
#
# Ten seeds, not two or three. Two seeds hides the battery entirely -- its MPC
# scored 277 on seed 0 and 65 on seed 1, so the average of the two looks healthy
# while the ten-seed truth is -61. That is the same trap that produced three
# wrong conclusions in the work this file came from, and it applies to the
# tripwire, not only to the ranking.
#
# Episodes are each environment's own, from EnvSpec.test_params, and are no
# longer capped. The cap was 250 steps and existed to bound CI cost back when
# this test rolled the plants out itself; it truncated the reactor from 1200
# steps, the four-tank from 500 and the boiler drum from 400, which measures
# something other than the episode the environment defines. Since the
# episode-length audit those episodes satisfy
# N >= max(10 * tau_actuator, 3 * T_period) -- long enough that holding the
# target, rather than reaching it, is what gets scored.
#
# None of that is paid here any more. This file used to be the slow job:
# profiled with --durations, [plane] alone took 836 s of a 19:47 run, and since
# xdist parallelises across tests rather than within one, that single test set
# roughly 70% of the wall-clock floor and came close to the job's 30-minute
# timeout on CI's four slower cores. The aircraft are expensive structurally --
# their gradient MPC plans 30 steps and, for the 2D plane, holds the last action
# for 60 more, so choosing one action optimises a 90-step rollout fifty times
# over.
#
# The measurement now happens in scripts/record_baselines.py, by hand, and this
# file reads what it wrote. That also made it affordable to stop compromising on
# episode length and seed count at the same time, which is why both moved up
# rather than down.

# Seeds and episode length now live in scripts/record_baselines.py, which is
# what actually rolls the plant out. Only the tolerance is asserted here.
PID_SHORTFALL_TOLERANCE = 0.10


@pytest.mark.parametrize("name", MPC_ENVS)
def test_recorded_baseline_still_describes_this_tree(name):
    """A recorded measurement must not be believed after the code moved.

    This is the whole safety of recording rather than re-measuring. The record
    carries a fingerprint of the environment's modules, the shared controller and
    integration code, the tuned gains and the parameter values it was taken at;
    if any of those moved, the numbers below describe code that no longer exists
    and the right answer is to refuse them, not to average them.

    Comments and docstrings are excluded from the fingerprint, so editing prose
    does not send anyone off to spend forty minutes of CPU.
    """
    spec = REGISTRY[name]
    if not spec.has_pid:
        pytest.skip(f"{name}: {spec.baselines_note}")

    recorded = load_recorded_baselines()
    assert name in recorded, (
        f"{name}: no recorded baseline. Run "
        f"`uv run python scripts/record_baselines.py --envs {name}` and commit "
        f"src/target_gym/data/baseline_returns.json."
    )
    current = baseline_fingerprint(spec)
    assert recorded[name]["fingerprint"] == current, (
        f"{name}: the recorded baseline was taken against different code "
        f"(recorded {recorded[name]['fingerprint']}, current {current}). Its "
        f"physics, controllers, gains or parameters have changed since, so its "
        f"numbers no longer say anything about this tree. Re-measure with "
        f"`uv run python scripts/record_baselines.py --envs {name}` and commit "
        f"the result with the change that invalidated it."
    )


@pytest.mark.parametrize("name", MPC_ENVS)
def test_mpc_controls_at_least_as_well_as_the_pid(name):
    """The MPC is presented as an upper bound; hold it to that.

    Asserted from the recorded measurement rather than by reproducing it. The
    rollouts cost about forty minutes of CPU, and one aircraft parametrisation
    alone was 836 s of a 19:47 slow job -- which, since xdist parallelises across
    tests and not within one, set roughly 70% of that job's wall-clock floor and
    came close to its 30-minute timeout on CI's slower cores.

    Reading the number instead of producing it makes this *stronger*, not
    weaker. It now runs in the fast job on every push and across the whole
    Python matrix, where before it ran once per merge to main on one
    interpreter. What guards it is the fingerprint test above; what produces it
    is scripts/record_baselines.py.

    Termination is checked first because it is the sharper signal. A controller
    that trips the turbine or flies the aircraft into the ground can still
    average acceptably across seeds, and averaging is exactly what hid it.
    """
    spec = REGISTRY[name]
    if not spec.has_pid:
        pytest.skip(f"{name}: {spec.baselines_note}")
    if spec.mpc_degraded:
        pytest.xfail(f"{name}: {spec.mpc_degraded}")

    recorded = load_recorded_baselines().get(name)
    if recorded is None:
        pytest.fail(f"{name}: no recorded baseline -- run scripts/record_baselines.py.")

    assert recorded["mpc_trips"] == 0, (
        f"{name}: the MPC tripped the plant {recorded['mpc_trips']} time(s) over "
        f"{recorded['seeds']} windows -- it left the operating envelope. A trip is "
        f"charged at the failure cost for the downtime (``base.failure_kernel``); "
        f"a planner that trips is not an upper bound. Terminal conditions are "
        f"booleans, so a penalty behind ``where(tripped, ...)`` gives the planner "
        f"the cost of a trip but no gradient away from the boundary; a "
        f"differentiable barrier on the approach is what works (see "
        f"make_wind_turbine_mpc)."
    )

    pid = np.array(recorded["pid_returns"])
    mpc = np.array(recorded["mpc_returns"])
    # Scale the allowance by the PID's own magnitude, so this reads the same way
    # for a return of 99 and one of 1100.
    allowance = PID_SHORTFALL_TOLERANCE * abs(pid.mean())
    assert mpc.mean() >= pid.mean() - allowance, (
        f"{name}: MPC {mpc.mean():.2f} vs PID {pid.mean():.2f} "
        f"({(mpc.mean() - pid.mean()) / abs(pid.mean()):+.1%}, allowance "
        f"{PID_SHORTFALL_TOLERANCE:.0%}). Per seed: MPC {np.round(mpc, 1).tolist()}, "
        f"PID {np.round(pid, 1).tolist()}. An MPC that has stopped controlling "
        f"still emits finite in-bounds actions, so check its objective before its "
        f"gains: every instance of this so far was an objective the planner could "
        f"not descend (a clipped plateau) or one that did not share the reward's "
        f"minimiser (a mis-scaled or non-monotone term), never a tuning problem."
    )
