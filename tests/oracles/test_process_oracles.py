"""The five process oracles on the shared solver (``oracles.controller``):
cstr, first_order, four_tank, unstable_cstr and compressor_surge.

Contract, from docs/proposals/oracle-smoothing-protocol.md (sections 2, 3)
and the owner decisions in oracle-audit.md:

- exactness: on a plan that meets the constraints, the planner's model with
  its exact rewrites (T1 to T3) and the env agree, so K8's replay of the
  planned moves on ``step_env`` matches the plan to float64 rounding;
- K5: a smoothed site's width keeps its one-step effect on the tracked
  output within half that output's resolution at the expert's states;
- T3: every planned state keeps inside a hard limit the objective pushes
  against, when the trip cost exceeds what a violation could gain;
- a failed solve applies the previous plan's next move;
- no oracle reads expert code (its imports, followed);
- T1r (Amendment 1): a command past an actuator's travel over the step moves
  it as a command at that travel does, so bounding moves there is exact;
- G10 (solver spec): the tie-break's worst case over an episode, every move
  of every solve at full size, is the 0.25 reward units half a resolution
  costs in one step.
"""

import ast
import functools
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import records
from target_gym.benchmark import episode_keys
from target_gym.experts.mpc import plan_params
from target_gym.oracles.controller import enable_x64, float64
from target_gym.registry import REGISTRY
from target_gym.runners.runners import pid_policy

FIVE = ("cstr", "first_order", "four_tank", "unstable_cstr", "compressor_surge")
SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "target_gym"


def _oracle(name, **overrides):
    """The env, the planner's params and the registered oracle on them."""
    spec = REGISTRY[name]
    params = plan_params(spec, spec.make_test_params(**overrides))
    return spec.make_env(), params, spec.make_mpc(spec.make_env(), params)


@pytest.mark.medium
@pytest.mark.parametrize("name", FIVE)
def test_the_exact_rewrites_replay_on_the_env(name):
    """Ten stages from training seed 100's reset, under the moves that hold
    the actuators (raw 0 where the reset holds none in the box) plus a
    0.05 sine. The two runs differ only in the order of a few float64
    operations a step (2.2e-16 each), so the relative reward gap and the
    output gap, in resolutions, sit many decades under 1e-10 and 1e-6 unless
    a rewrite is not exact. The compressor's sites run exact (width 0)."""
    env, params, oracle = _oracle(name)
    with enable_x64():
        _, state = env.reset_env(jax.random.PRNGKey(100), params)
        st, m = float64(state), oracle.model
        hold = jnp.where(jnp.abs(m.hold(st)) < 1.0, m.hold(st), 0.0)
        wave = 0.05 * jnp.sin(0.7 * jnp.arange(oracle.P))[:, None]
        moves = jnp.clip(hold + wave, -1.0, 1.0)
        d = {"task": m.data(st), "end": 10 * oracle.tc}
        d["widths"] = jnp.zeros_like(jnp.asarray(oracle.widths))
        cert = oracle._certify(d, st, m.state_x(st), moves, 10)
    assert not bool(cert["replay_trip"]), "the plan must meet the constraints"
    assert float(cert["replay_gap"]) <= 1e-10
    assert float(cert["output_gap"]) <= 1e-6


def _expert_probes(name, seed=100, every=10):
    """A subset of K5's probes: the expert's states on one training seed."""
    spec = REGISTRY[name]
    env, params, policy = spec.make_env(), spec.make_test_params(), pid_policy(spec)
    T = int(params.max_steps_in_episode)
    k_reset, k_steps = episode_keys(jax.random.PRNGKey(seed), 1, T)
    obs, state = env.reset_env(k_reset[0], params)
    step, out = jax.jit(env.step_env), []
    for t in range(T):
        u = np.clip(np.atleast_1d(np.asarray(policy(np.asarray(obs)), float)), -1, 1)
        if t % every == 0:
            out.append((state, u))
        obs, state, *_ = step(k_steps[t, 0], state, jnp.asarray(u), params)
    return out


@pytest.mark.medium
def test_the_compressor_widths_meet_the_k5_bound():
    """Each site's width is a rung s / 10^k of its scale and meets K5's
    bound at the expert's states on training seed 100, a subset of its
    probes: w ln 2 (CM's softplus gap) times the sum over the step's ten
    evaluations of |d dp / d delta| is at most half the transmitter's
    resolution (GUM), in kPa. The slopes are forward-mode derivatives of the
    exact step, independent of the width tool's code."""
    from target_gym.compressor_surge.oracle import compressor_step

    env, _, oracle = _oracle("compressor_surge")
    sites = records.load_row("oracle_widths", "compressor_surge")["sites"]
    p, m = oracle.params, oracle.model
    visited = _expert_probes("compressor_surge")  # the expert runs as shipped
    with enable_x64():
        probes = [(float64(s), jnp.asarray(u, jnp.float64)) for s, u in visited]

        @functools.partial(jax.jit, static_argnums=2)
        def slope(state, u, i):
            def dp(delta):
                deltas = jnp.zeros((2, delta.size)).at[i].set(delta)
                x1 = compressor_step(
                    env, p, m.state_x(state), u, 0, m.data(state), (0.0, 0.0), deltas
                )[0]
                return x1[1] / 1e3  # kPa

            return jnp.sum(jnp.abs(jax.jacfwd(dp)(jnp.zeros(10))))

        for i, site in enumerate(sorted(sites)):
            w, s = sites[site]["width"], sites[site]["scale"]
            assert math.log10(s / w) == pytest.approx(round(math.log10(s / w)))
            worst = max(float(slope(st, u, i)) for st, u in probes)
            assert w * math.log(2.0) * worst <= p.precision_floor / 2, site


@pytest.mark.medium
def test_a_plan_keeps_every_state_inside_a_hard_limit_it_pushes_against():
    """first_order with its trip at x_max = 1.0 and targets above it, ten
    steps. A unit of x past the limit saves at most 2 e / e_floor^2, about
    2 x 0.5 / 3.6e-5 = 2.8e4 a step, 2.8e5 over the horizon, under the trip
    cost of 2e8 a unit: no slack is worth taking. So every planned state is
    at most x_max (plus IPOPT's bound relaxation, 1e-8 relative), and the
    plan rides the limit (within 1e-3)."""
    env, params, oracle = _oracle(
        "first_order", x_max=1.0, target_x_range=(1.4, 1.5), max_steps_in_episode=10
    )
    _, state = env.reset_env(jax.random.PRNGKey(100), params)
    oracle.step(None, state)
    planned = np.asarray(oracle.shooting.states(oracle._plan))[:10, 0]
    assert not oracle.certificates[-1]["failed"]
    assert planned.max() <= 1.0 + 1e-7
    assert planned.max() >= 1.0 - 1e-3


@pytest.mark.medium
def test_a_failed_solve_applies_the_previous_plans_next_move():
    """A solve from a state with no finite value fails; the move applied is
    the one the last plan had planned next, never anything else."""
    env, params, oracle = _oracle("first_order")
    _, state = env.reset_env(jax.random.PRNGKey(100), params)
    oracle.step(None, state)
    planned = np.asarray(oracle.shooting.moves(oracle._plan))[:, 0]
    broken = state.replace(x=jnp.nan, time=state.time + 1)
    applied = oracle.step(None, broken)
    assert oracle.certificates[-1]["failed"]
    assert applied == float(np.clip(planned[1], -1.0, 1.0))


def _imports(path: pathlib.Path) -> set[pathlib.Path]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        for name in names:
            rel = (
                pathlib.Path(*name.split(".")[1:])
                if name.startswith("target_gym.")
                else None
            )
            for f in (
                ((SRC / rel).with_suffix(".py"), SRC / rel / "__init__.py")
                if rel
                else ()
            ):
                if f.is_file():
                    found.add(f)
    return found


@pytest.mark.parametrize("name", FIVE)
def test_the_oracle_reads_no_expert_code(name):
    """Every module the oracle imports, followed to the end, is outside
    experts/ and no experts*.py, and the spec declares no expert reads."""
    spec = REGISTRY[name]
    todo = [
        SRC
        / pathlib.Path(*type(spec.make_env()).__module__.split(".")[1:-1])
        / "oracle.py"
    ]
    seen: set[pathlib.Path] = set()
    while todo:
        path = todo.pop()
        if path not in seen:
            seen.add(path)
            todo.extend(_imports(path))
    expert = sorted(str(p.relative_to(SRC)) for p in seen if p.parent.name == "experts" or p.name.startswith("experts"))  # fmt: skip
    assert expert == []
    assert spec.oracle_reads == () and spec.oracle_gains_keys == ()


@pytest.mark.parametrize("actuator, sign", [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)])
def test_a_command_past_the_compressors_reach_moves_it_as_one_at_the_reach(
    actuator, sign
):
    """The drive (0) or the valve (1), up or down, from the reset: over one
    step the env moves it by at most its rate times the step, so a command
    at the opposite end of its range lands on the same next state as one at
    the reach. Float32 env: equal to 1e-6 relative."""
    spec = REGISTRY["compressor_surge"]
    env, p = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    lo, hi, pos, up, down = (
        (p.N_min, p.N_max, float(state.N_ramp), p.drive_rate, p.drive_rate)
        if actuator == 0
        else (0.0, 1.0, float(state.x), p.valve_open_rate, p.valve_close_rate)
    )
    at = pos + sign * p.delta_t * (up if sign > 0 else down)

    def raw(command):
        hold = [float(state.N_ramp), float(state.x)]
        hold[actuator] = command
        ranges = [(p.N_min, p.N_max), (0.0, 1.0)]
        return jnp.asarray(
            [2.0 * (c - a) / (b - a) - 1.0 for c, (a, b) in zip(hold, ranges)]
        )

    reached = env.step_env(key, state, raw(at), p)[1]
    past = env.step_env(key, state, raw(hi if sign > 0 else lo), p)[1]
    for field in ("N_ramp", "x", "N", "dp", "m_c", "phi_min"):
        np.testing.assert_allclose(
            getattr(past, field), getattr(reached, field), rtol=1e-6, err_msg=field
        )


@pytest.mark.parametrize("name", FIVE)
def test_the_tie_break_costs_half_a_resolution_step_over_an_episode(name):
    _, _, mpc = _oracle(name)
    assert mpc.shooting.model.tie * mpc.model.nu * mpc.n_ep == pytest.approx(0.25)
