"""A plant's arithmetic does not decide its cost: not its precision, not its step.

Contract (owner, 2026-11-23, after the reactor's boron mixing was found to round small
flows in single precision: 36 kg of dilution over 10 h removed nothing and was paid for):

- **Precision.** With its noise off, a plant run in float32 costs what it costs in
  float64, from the same start under the same inputs, to within 0.1 %.
- **Step.** With its noise off, driven by a square wave of half its action range, a
  plant costs the same at its integration step and at half of it, to within 0.1 %. A
  square wave, because a plant held at rest in calm air costs next to nothing, which
  no percentage can be read against, and because it reaches the fast modes. A plant
  that is unstable open loop is driven by its expert: left alone, two runs of it part
  whatever the integrator.

0.1 % is a tenth of the smallest room the benchmark reports on a task (about 1 %, the
nearly saturated tasks, docs/proposals/charter-amendment-6-draft.md), so neither
rounding nor the integrator can pass for room or hide it.

Both compare the steps before the first trip of either run, since a restart draws
afresh, and need 30 of them. The inputs of the precision check are held constant and
small on purpose: a trickle held for hours is what single precision loses.

A task left out, or known to fail, is named in a table below with its reason, so a hole
shows at collection and not inside a test.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import registry

try:
    from jax import enable_x64
except ImportError:  # JAX 0.6; 0.10 removed the experimental name
    from jax.experimental import enable_x64

TOLERANCE = 1e-3
MIN_STEPS = 30
#: Raw actions held for the whole run: at rest, three trickles and one plain move.
LEVELS = (0.0, 0.005, 0.02, -0.02, 0.1)
#: Steps of the precision run: a trickle needs hours to show.
LONG, SHORT = 3000, 1000

#: Process noise the registry's ``noise_fields`` does not list, switched off here.
ALSO_QUIET = {"glass_furnace": {"walk_sd": (0.0, 0.0, 0.0, 0.0, 0.0)}}

#: Not comparable across precisions, with the reason.
DRAWS_INSIDE_A_STEP = {
    "plane3d_heading": "draws its clearance heading inside a step, and a draw differs "
    "between float32 and float64; its dynamics are plane3d_circle's, which is checked",
}
#: Fails the precision contract today: a finding, strict so that a fix shows.
ROUNDS_IN_FLOAT32 = {
    "reactor": "the boron mixing rounds small flows in float32 (found 2026-11-23; the "
    "fix is on plant/reactor-baseload-losses)",
}
#: Takes one integration setting only, with the reason.
ONE_STEP_ONLY = {
    "glass_furnace": "its own stepping takes euler_1 only (glass_furnace/env_jax.py)",
}
#: Stepped under its expert, not on a square wave, with the reason.
HELD_BY_ITS_EXPERT = {
    "unstable_cstr": "unstable open loop by design",
    "patrol": "an aircraft on station: a square wave flies it off and trips in 23 steps",
    "patrol_range_bearing": "as patrol",
    "wake_surf": "as patrol, in 27 steps",
}
#: Fails the step contract today: a finding, strict so that a fix shows.
STEP_TOO_COARSE = {
    "unstable_cstr": "at rk4_4 under its expert the cost moves 0.13 % when the step is "
    "halved (2.8704 against 2.8667 over 1000 steps, 2026-11-23)",
}


def _cases(skipped, failing):
    out = []
    for spec in registry.all_specs():
        marks = []
        if spec.name in skipped:
            marks.append(pytest.mark.skip(reason=skipped[spec.name]))
        if spec.name in failing:
            marks.append(pytest.mark.xfail(strict=True, reason=failing[spec.name]))
        out.append(pytest.param(spec, id=spec.name, marks=marks))
    return out


def _quiet(spec):
    off = {"sensor_noise_scale": 0.0, **{f: 0.0 for f in spec.noise_fields}}
    return spec.make_test_params(**off, **ALSO_QUIET.get(spec.name, {}))


def _tripped(info):
    return jnp.asarray(info.get("tripped", False))


def _key(t):
    return jax.random.fold_in(jax.random.PRNGKey(0), t)


def _compare(name, what, costs_a, trips_a, costs_b, trips_b):
    """``costs`` and ``trips`` per step; the two runs agree before either trips."""
    either = np.asarray(trips_a) | np.asarray(trips_b)
    steps = int(np.argmax(either)) if either.any() else len(either)
    assert steps >= MIN_STEPS, f"{name}, {what}: a trip after {steps} steps"
    a, b = float(np.sum(costs_a[:steps])), float(np.sum(costs_b[:steps]))
    assert a == pytest.approx(b, rel=TOLERANCE, abs=1e-12), (
        f"{name}, {what}: {a:.8g} against {b:.8g} over {steps} steps, "
        f"{abs(a - b) / max(abs(b), 1e-300):.1e} apart"
    )


def _open_loop(env, params, start, dtype, steps):
    """Cost and trip per step, ``(len(LEVELS), steps)``, each level held throughout."""
    shape = env.action_space(params).shape

    def run(level):
        action = jnp.full(shape, level, dtype)

        def body(state, t):
            _, after, reward, _, info = env.step_env(_key(t), state, action, params)
            # A plant may hand a field back in float32 whatever it was given
            # (the 3D aircraft's mean wind, the battery's availability cost):
            # held in the wider type, its value is unchanged.
            after = jax.tree_util.tree_map(
                lambda new, old: new.astype(old.dtype), after, state
            )
            return after, (-reward, _tripped(info))

        return jax.lax.scan(body, start, jnp.arange(steps))[1]

    cost, tripped = jax.jit(jax.vmap(run))(jnp.asarray(LEVELS, dtype))
    return np.asarray(cost, np.float64), np.asarray(tripped)


@pytest.mark.medium
@pytest.mark.parametrize("spec", _cases(DRAWS_INSIDE_A_STEP, ROUNDS_IN_FLOAT32))
def test_single_precision_costs_what_double_precision_costs(spec):
    env, params = spec.make_env(), _quiet(spec)
    steps = min(int(params.max_steps_in_episode), LONG)
    _, start = env.reset_env(jax.random.PRNGKey(0), params)
    single = _open_loop(env, params, start, jnp.float32, steps)
    with enable_x64():
        # The same start, to the bit: float32 values held in float64.
        wide = jax.tree_util.tree_map(
            lambda leaf: (
                leaf.astype(jnp.float64)
                if jnp.issubdtype(leaf.dtype, jnp.floating)
                else leaf
            ),
            start,
        )
        double = _open_loop(env, params, wide, jnp.float64, steps)
    for i, level in enumerate(LEVELS):
        _compare(
            spec.name,
            f"raw action {level:+g} held, float32 against float64",
            single[0][i],
            single[1][i],
            double[0][i],
            double[1][i],
        )


def _square_wave(env, params, start, steps):
    """Cost and trip per step, the raw action going from +0.5 to -0.5 every 20 steps."""
    shape = env.action_space(params).shape

    def body(state, t):
        action = jnp.full(shape, jnp.where((t // 20) % 2 == 0, 0.5, -0.5))
        _, state, reward, _, info = env.step_env(_key(t), state, action, params)
        return state, (-reward, _tripped(info))

    cost, tripped = jax.jit(lambda s: jax.lax.scan(body, s, jnp.arange(steps))[1])(
        start
    )
    return np.asarray(cost, np.float64), np.asarray(tripped)


def _under_its_expert(name):
    """Cost and trip per step under the shipped expert, which reads observations."""
    expert = tg.make_expert(name)

    def run(env, params, start, steps):
        def body(carry, t):
            state, obs, memory = carry
            action, memory = expert.step(memory, obs)
            obs, state, reward, _, info = env.step_env(_key(t), state, action, params)
            return (state, obs, memory), (-reward, _tripped(info))

        carry = (start, env.get_obs(start, params), expert.reset())
        cost, tripped = jax.jit(lambda c: jax.lax.scan(body, c, jnp.arange(steps))[1])(
            carry
        )
        return np.asarray(cost, np.float64), np.asarray(tripped)

    return run


def _half_step(env):
    """The same plant integrated with twice the sub-steps: ``rk4_2`` for ``rk4_1``."""
    method, count = re.fullmatch(r"(.+)_(\d+)", env.integration_method).groups()
    return type(env)(integration_method=f"{method}_{2 * int(count)}")


@pytest.mark.medium
@pytest.mark.parametrize("spec", _cases(ONE_STEP_ONLY, STEP_TOO_COARSE))
def test_half_the_integration_step_costs_the_same(spec):
    env, params = spec.make_env(), _quiet(spec)
    steps = min(int(params.max_steps_in_episode), SHORT)
    _, start = env.reset_env(jax.random.PRNGKey(0), params)
    held = spec.name in HELD_BY_ITS_EXPERT
    run = _under_its_expert(spec.name) if held else _square_wave
    _compare(
        spec.name,
        f"{env.integration_method} against half its step, "
        + ("under its expert" if held else "on a square wave"),
        *run(env, params, start, steps),
        *run(_half_step(env), params, start, steps),
    )
