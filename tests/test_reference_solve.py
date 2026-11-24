"""A plant's integrator lands where an independent adaptive solver lands.

Contract (owner, 2026-11-24, "Reference solve in the tests"): with its noise off,
in float64, on the square wave of ``tests/test_plant_numerics.py``, a plant costs
what it costs when every integration it asks of ``integrate_dynamics`` is done by
SciPy's LSODA at a tight tolerance, on the plant's own vector field with its input
held, to within 0.1 %.

The step-halving check of ``test_plant_numerics`` compares an integrator with
itself, which a method that converges slowly, or to the wrong place, passes. This
one compares it with code that shares nothing with it: ODEPACK's LSODA, adaptive in
step and order and switching to a stiff method by itself. The tolerance is the same
0.1 % of the cost, for the same reason (a tenth of the smallest room reported).

What the reference replaces is each call of the shared integrator, so what a plant
holds over a call stays held: the compressor's actuators over a sub-step, the
kiln's calcination rate, an aircraft's surfaces over an inner step. Those holds are
the plant's model, and the step-halving check is what covers them.

The solve is a host callback inside the plant's compiled step, so each control
step calls back into Python and the run is short: ``STEPS`` steps, three
half-periods of the square wave. In float64 on both sides, so rounding (the
precision check's subject) stays out of it.

A task left out, or known to fail, is named in a table below with its reason. A
plant that never reaches the reference fails here, not passes: a comparison of a
plant with itself is no check.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from target_gym import integration
from tests.test_plant_numerics import (
    HELD_BY_ITS_EXPERT,
    _cases,
    _compare,
    _key,
    _quiet,
    _tripped,
)

#: Three half-periods of the square wave: the start, a step down and a step up.
STEPS = 60
#: LSODA's tolerances: seven orders inside the 0.1 % the costs are held to.
RTOL, ATOL = 1e-10, 1e-12

#: Does not integrate through ``integrate_dynamics``, with the reason.
STEPS_BY_ITSELF = {
    "glass_furnace": "its own stepping takes euler_1 only (glass_furnace/env_jax.py)",
    "battery": "stepped in closed form over each period, nothing is integrated",
    "unstable_cstr": "its own sub-step: the initiator in closed form, then RK4 by "
    "hand (unstable_cstr/env.py, advance)",
}
#: Fails this contract today: a finding, strict so that a fix shows.
OFF_THE_REFERENCE: dict = {}


#: A plant's field compiled once, by the function and the settings it carries.
_compiled: dict = {}
#: One entry per solve, emptied by the fixture.
calls: list = []


def _is_array(leaf):
    return isinstance(leaf, (jax.Array, np.ndarray))


def reference_step(field, y, h, drift):
    """``y`` after ``h`` under ``field``, by LSODA.

    Traced with the plant's step: the solve is a host callback, handed the
    state and the arrays the field holds. A plant passes its field as a
    ``Partial``, a function that stays and arrays that change from call to
    call, so the field is compiled once, apart, with those arrays as arguments,
    and the thousands of evaluations LSODA asks for cost microseconds each.
    """
    leaves, tree = jax.tree_util.tree_flatten(field)
    arrays = [leaf for leaf in leaves if _is_array(leaf)]
    fixed = tuple(None if _is_array(leaf) else leaf for leaf in leaves)

    def rate(arrays, x):
        held = iter(arrays)
        rebuilt = tree.unflatten([next(held) if f is None else f for f in fixed])
        return rebuilt(x)[0]

    def solve(start, length, *arrays):
        calls.append(1)
        if (tree, fixed) not in _compiled:
            _compiled[(tree, fixed)] = (
                jax.jit(rate),
                jax.jit(jax.jacfwd(rate, argnums=1)),
            )
        fast, jacobian = _compiled[(tree, fixed)]
        start = np.asarray(start, np.float64)
        solved = solve_ivp(
            lambda t, x: np.asarray(fast(arrays, x)),
            (0.0, float(length)),
            start,
            method="LSODA",
            jac=lambda t, x: np.asarray(jacobian(arrays, x)),
            rtol=RTOL,
            atol=ATOL * (1.0 + np.abs(start)),
        )
        assert solved.success, solved.message
        return solved.y[:, -1].astype(start.dtype)

    # One solve per plant where a task steps several under ``vmap``.
    end = jax.pure_callback(
        solve,
        jax.ShapeDtypeStruct(y.shape, y.dtype),
        y,
        h,
        *arrays,
        vmap_method="sequential",
    )
    return end, field(end)[1]


@pytest.fixture
def reference():
    """``"reference_N"`` as an integration setting, in double precision, for the
    test's length. Double precision is set for the process, not a scope: the
    solve is called back on a thread of the runtime's, which a scope set on the
    test's thread does not reach."""
    calls.clear()
    integration.STEPPERS["reference"] = reference_step
    wide = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", wide)
    del integration.STEPPERS["reference"]


def _drive(spec):
    """``(reset, act)``: the square wave, or the expert where the plant needs it."""
    if spec.name not in HELD_BY_ITS_EXPERT:
        return None, lambda memory, obs, t, shape: (
            jnp.full(shape, jnp.where((t // 20) % 2 == 0, 0.5, -0.5)),
            memory,
        )
    import target_gym as tg

    expert = tg.make_expert(spec.name)
    return expert.reset, lambda memory, obs, t, shape: expert.step(memory, obs)


def _run(spec, env, step, params, start):
    """Cost and trip per step under ``step``, one step at a time in Python."""
    reset, act = _drive(spec)
    shape = env.action_space(params).shape
    state, obs = start, env.get_obs(start, params)
    memory = reset() if reset else None
    costs, trips = [], []
    for t in range(STEPS):
        action, memory = act(memory, obs, t, shape)
        obs, state, reward, _, info = step(_key(t), state, action, params)
        costs.append(-float(reward))
        trips.append(bool(_tripped(info)))
    return np.asarray(costs), np.asarray(trips)


@pytest.mark.medium
@pytest.mark.parametrize("spec", _cases(STEPS_BY_ITSELF, OFF_THE_REFERENCE))
def test_the_plant_costs_what_it_costs_under_a_reference_solver(spec, reference):
    env, params = spec.make_env(), _quiet(spec)
    count = re.fullmatch(r".+_(\d+)", env.integration_method).group(1)
    exact = type(env)(integration_method=f"reference_{count}")
    _, start = env.reset_env(jax.random.PRNGKey(0), params)
    # The same start for both, float32 values held in float64.
    start = jax.tree_util.tree_map(
        lambda leaf: (
            leaf.astype(jnp.float64)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            else leaf
        ),
        start,
    )
    shipped = _run(spec, env, jax.jit(env.step_env), params, start)
    solved = _run(spec, exact, jax.jit(exact.step_env), params, start)
    assert len(calls) >= STEPS, (
        f"{spec.name} reached the reference {len(calls)} times in {STEPS} steps: "
        "it does not integrate through integrate_dynamics, so name it in "
        "STEPS_BY_ITSELF"
    )
    _compare(
        spec.name,
        f"{env.integration_method} against LSODA at {RTOL:g}",
        *shipped,
        *solved,
    )
