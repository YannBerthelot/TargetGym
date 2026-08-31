"""Shared conformance contract, run against every registered environment.

Motivation
----------
TargetGym's per-environment tests were thorough about *shapes* and *signs* but
each was written in isolation, so a defect in a shared convention could hide in
one environment while the others were fine. The glass furnace's pull-rate
disturbance is the worked example: it read its randomness from the PRNG key
handed to ``step_env``, but every rollout helper in this repository
(``run_episode_headless``, ``save_video``, the ``lax.scan`` bodies in the
runners) passes *the same key at every step*. The "AR(1) noise" was therefore a
deterministic monotone ramp, and 334 passing tests said nothing about it.

Anything asserted here is a claim about *all* environments, so a new
environment inherits the whole battery by adding one line to
``target_gym.registry``.

Grouped by concern:
    1. Reset contract
    2. Step contract (incl. the gymnax >= 1.0 six-value API)
    3. Determinism and PRNG hygiene
    4. Numerical health over a full episode
    5. JAX transform compatibility (jit / vmap / scan)
    6. Baseline coverage
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry

ALL_SPECS = list(registry.all_specs())
SPEC_IDS = [s.name for s in ALL_SPECS]


@pytest.fixture(params=ALL_SPECS, ids=SPEC_IDS)
def spec(request):
    return request.param


def _zero_action(env, params):
    """A valid mid-range action for this environment."""
    space = env.action_space(params)
    return jnp.zeros(space.shape, dtype=jnp.float32)


def _leaves(state):
    """Flatten a state pytree to finite-checkable arrays."""
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(state)]


# ---------------------------------------------------------------------------
# 1. Reset contract
# ---------------------------------------------------------------------------


def test_reset_returns_obs_matching_observation_space(spec):
    env, params = spec.make_env(), spec.make_test_params()
    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
    assert obs.shape == env.observation_space(params).shape
    assert np.all(np.isfinite(np.asarray(obs)))


def test_reset_starts_the_clock_at_zero(spec):
    env, params = spec.make_env(), spec.make_test_params()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert int(state.time) == 0


def test_reset_is_deterministic_given_a_seed(spec):
    """Same key must reproduce the same initial condition exactly."""
    env, params = spec.make_env(), spec.make_test_params()
    obs_a, _ = env.reset_env(jax.random.PRNGKey(7), params)
    obs_b, _ = env.reset_env(jax.random.PRNGKey(7), params)
    assert jnp.allclose(obs_a, obs_b)


def test_reset_varies_with_seed(spec):
    """Different seeds must give different episodes, or the task is degenerate."""
    obs = [
        spec.make_env().reset_env(jax.random.PRNGKey(s), spec.make_test_params())[0]
        for s in range(8)
    ]
    stacked = jnp.stack(obs)
    assert not jnp.allclose(
        stacked, stacked[0]
    ), "every seed produced an identical observation"


# ---------------------------------------------------------------------------
# 2. Step contract
# ---------------------------------------------------------------------------


def test_step_env_returns_five_values_with_natural_termination_only(spec):
    """``step_env`` reports natural termination; gymnax owns truncation."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    out = env.step_env(key, state, _zero_action(env, params), params)
    assert len(out) == 5
    obs, new_state, reward, terminated, info = out
    assert obs.shape == env.observation_space(params).shape
    assert jnp.asarray(terminated).dtype == jnp.bool_
    assert np.isfinite(float(reward))
    assert isinstance(info, dict)


def test_gymnax_step_returns_six_values(spec):
    """gymnax >= 1.0: (obs, state, reward, terminated, truncated, info)."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    out = env.step(key, state, _zero_action(env, params), params)
    assert len(out) == 6, f"expected 6-value step, got {len(out)}"
    *_, terminated, truncated, info = out
    assert jnp.asarray(terminated).dtype == jnp.bool_
    assert jnp.asarray(truncated).dtype == jnp.bool_
    assert "terminated" in info and "truncated" in info


def test_time_advances_monotonically(spec):
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = _zero_action(env, params)
    previous = int(state.time)
    _step = jax.jit(env.step_env)
    for _ in range(5):
        _, state, _, terminated, _ = _step(key, state, action, params)
        assert int(state.time) > previous
        previous = int(state.time)
        if bool(terminated):
            break


def test_truncation_fires_at_the_step_limit(spec):
    """``is_truncated`` must trigger once the clock reaches the limit."""
    env, params = spec.make_env(), spec.make_test_params()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    at_limit = state.replace(time=params.max_steps_in_episode)
    assert bool(env.is_truncated(at_limit, params))
    if params.max_steps_in_episode > 1:
        before = state.replace(time=params.max_steps_in_episode - 1)
        assert not bool(env.is_truncated(before, params))


def test_action_space_sample_is_accepted(spec):
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = env.action_space(params).sample(jax.random.PRNGKey(1))
    _, new_state, reward, _, _ = env.step_env(key, state, action, params)
    assert np.isfinite(float(reward))
    assert all(np.all(np.isfinite(x)) for x in _leaves(new_state))


# ---------------------------------------------------------------------------
# 3. Determinism and PRNG hygiene
# ---------------------------------------------------------------------------


def test_step_is_deterministic_given_key_state_and_action(spec):
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(3)
    _, state = env.reset_env(key, params)
    action = _zero_action(env, params)
    obs_a, _, reward_a, _, _ = env.step_env(key, state, action, params)
    obs_b, _, reward_b, _, _ = env.step_env(key, state, action, params)
    assert jnp.allclose(obs_a, obs_b)
    assert float(reward_a) == float(reward_b)


def _terminal_disturbances(spec, seeds, split_key, steps):
    """Disturbance values after ``steps`` for each seed, one key regime.

    The environment, its parameters and the compiled step are built once and
    reused across every seed. ``jax.jit`` keys its cache on the function it
    wraps, and ``env.step_env`` is a bound method, so building a fresh
    environment per seed would compile the same step function again for each
    one -- which for this parametrised test is where nearly all of its time
    used to go.
    """
    env = spec.make_env()
    params = spec.make_test_params(**spec.disturbance_overrides)
    action = _zero_action(env, params)
    step = jax.jit(env.step_env)
    reset = jax.jit(env.reset_env)

    out = []
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        _, state = reset(key, params)
        rolling = key
        for _ in range(steps):
            if split_key:
                rolling, sub = jax.random.split(rolling)
            else:
                sub = key  # constant -- what every rollout helper here does
            _, state, _, terminated, _ = step(sub, state, action, params)
            if bool(terminated):
                break
        out.append([float(getattr(state, f)) for f in spec.disturbance_fields])
    return out


def test_disturbance_magnitude_is_independent_of_key_splitting(spec):
    """A disturbance must not depend on whether the *caller* splits the key.

    Every rollout helper in this repository -- ``run_episode_headless``,
    ``save_video``, the ``lax.scan`` bodies in the runners, and gymnax's own
    ``Environment.step`` when handed a constant key -- drives ``step_env`` with
    the same key at every step. An environment that draws its per-step noise
    directly from that key then redraws the *identical* innovation forever,
    collapsing a zero-mean process into a deterministic ramp toward
    ``innovation / (1 - rho)``. Randomness must instead be derived from the
    state, e.g. ``jax.random.fold_in(state.some_key, state.time)``.

    Comparing realised RMS under constant vs split keys is the general form of
    that check. A correct environment gives the same magnitude either way; a
    broken one inflates by roughly ``1 / (1 - rho)``.

    Note a monotonicity check is *not* sufficient: a fast-reverting process
    (the aircraft gusts, theta = 0.2) converges to a fixed point and plateaus,
    so its increments stop being monotone while the value is still wrong.
    """
    if not spec.disturbance_fields:
        pytest.skip(f"{spec.name} declares no stochastic disturbance")

    steps, seeds = 200, range(6)
    constant = np.array(_terminal_disturbances(spec, seeds, False, steps))
    split = np.array(_terminal_disturbances(spec, seeds, True, steps))
    rms_constant = float(np.sqrt((constant**2).mean()))
    rms_split = float(np.sqrt((split**2).mean()))

    if rms_split < 1e-9:
        pytest.skip(f"{spec.name} disturbance inactive under test params")

    assert rms_constant < 2.5 * rms_split, (
        f"{spec.name}{list(spec.disturbance_fields)}: RMS is {rms_constant:.3f} "
        f"with a constant key vs {rms_split:.3f} with split keys "
        f"({rms_constant / rms_split:.1f}x). The per-step noise is drawn from "
        "the passed key rather than derived from the state, so under the "
        "repo's own rollout helpers this disturbance is a deterministic offset "
        "rather than a zero-mean process."
    )


def test_distinct_keys_give_distinct_stochastic_trajectories(spec):
    """Where an env is stochastic, the key must actually influence the outcome."""
    if not spec.disturbance_fields:
        pytest.skip(f"{spec.name} declares no stochastic disturbance")
    env = spec.make_env()
    params = spec.make_test_params(**spec.disturbance_overrides)
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    action = _zero_action(env, params)

    finals = []
    for seed in (0, 1, 2):
        s, k = state, jax.random.PRNGKey(100 + seed)
        _step = jax.jit(env.step_env)
        for _ in range(20):
            k, sub = jax.random.split(k)
            _, s, _, terminated, _ = _step(sub, s, action, params)
            if bool(terminated):
                break
        finals.append(tuple(float(getattr(s, f)) for f in spec.disturbance_fields))
    assert len(set(finals)) > 1, "PRNG key had no effect on the disturbance"


# ---------------------------------------------------------------------------
# 4. Numerical health
# ---------------------------------------------------------------------------


def test_full_episode_stays_finite(spec):
    """No NaN or Inf anywhere in state, obs or reward over a whole episode."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    action = _zero_action(env, params)

    jitted = jax.jit(env.step_env)
    for step in range(400):
        key, sub = jax.random.split(key)
        obs, state, reward, terminated, _ = jitted(sub, state, action, params)
        assert np.all(np.isfinite(np.asarray(obs))), f"non-finite obs at step {step}"
        assert np.isfinite(float(reward)), f"non-finite reward at step {step}"
        for leaf in _leaves(state):
            if leaf.dtype.kind == "f":
                assert np.all(np.isfinite(leaf)), f"non-finite state at step {step}"
        if bool(terminated) or int(state.time) >= params.max_steps_in_episode:
            break


def test_reward_is_bounded_over_an_episode(spec):
    """Per-step reward must stay within a sane magnitude."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = _zero_action(env, params)
    rewards = []
    _step = jax.jit(env.step_env)
    for _ in range(120):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = _step(sub, state, action, params)
        rewards.append(float(reward))
        if bool(terminated):
            break
    limit = 10.0 * max(1.0, float(params.max_steps_in_episode))
    assert (
        np.max(np.abs(rewards)) <= limit
    ), f"reward magnitude {np.max(np.abs(rewards))}"


# ---------------------------------------------------------------------------
# 5. JAX transform compatibility
# ---------------------------------------------------------------------------


def test_step_env_is_jittable(spec):
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = _zero_action(env, params)

    jitted = jax.jit(lambda k, s, a: env.step_env(k, s, a, params))
    obs_j, _, reward_j, _, _ = jitted(key, state, action)
    obs_e, _, reward_e, _, _ = env.step_env(key, state, action, params)
    assert jnp.allclose(obs_j, obs_e, atol=1e-4)
    assert float(reward_j) == pytest.approx(float(reward_e), rel=1e-4, abs=1e-4)


def test_env_is_vmappable_over_seeds(spec):
    """Batched rollout is the point of a JAX env: reset+step must vmap."""
    env, params = spec.make_env(), spec.make_test_params()
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    obs, states = jax.vmap(lambda k: env.reset_env(k, params))(keys)
    assert obs.shape[0] == n

    action = jnp.broadcast_to(
        _zero_action(env, params), (n,) + env.action_space(params).shape
    )
    obs2, _, rewards, terminated, _ = jax.vmap(
        lambda k, s, a: env.step_env(k, s, a, params)
    )(keys, states, action)
    assert obs2.shape[0] == n
    assert rewards.shape == (n,)
    assert np.all(np.isfinite(np.asarray(rewards)))
    assert jnp.asarray(terminated).shape == (n,)


def test_env_runs_under_lax_scan(spec):
    """The env must compose with ``lax.scan``, which is how rollouts are done."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = _zero_action(env, params)

    def body(carry, _):
        s, k = carry
        k, sub = jax.random.split(k)
        _, s2, r, _, _ = env.step_env(sub, s, action, params)
        return (s2, k), r

    (_, _), rewards = jax.lax.scan(body, (state, key), None, length=10)
    assert rewards.shape == (10,)
    assert np.all(np.isfinite(np.asarray(rewards)))


# ---------------------------------------------------------------------------
# 6. Baseline coverage
# ---------------------------------------------------------------------------


def test_baselines_are_present_or_documented(spec):
    """Every env ships PID + MPC, or explains in the registry why it does not.

    The README promises both for every environment. Where that is not yet true
    the gap must be explicit, so it stays visible instead of being discovered
    by a user.
    """
    if spec.has_pid and spec.has_mpc:
        return
    assert spec.baselines_note, (
        f"{spec.name} is missing a "
        f"{'PID' if not spec.has_pid else ''}"
        f"{' and ' if not spec.has_pid and not spec.has_mpc else ''}"
        f"{'MPC' if not spec.has_mpc else ''} baseline "
        "and has no baselines_note explaining why"
    )


def test_pid_baseline_produces_valid_actions(spec):
    if not spec.has_pid:
        pytest.skip(f"{spec.name}: {spec.baselines_note}")
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    pid = spec.make_pid()
    pid.reset()
    space = env.action_space(params)
    _step = jax.jit(env.step_env)
    for _ in range(10):
        action = jnp.asarray(pid(obs))
        assert np.all(np.isfinite(np.asarray(action)))
        assert action.shape == space.shape or action.size == int(
            np.prod(space.shape)
        ), f"PID emitted shape {action.shape}, env expects {space.shape}"
        obs, state, reward, terminated, _ = _step(key, state, action, params)
        assert np.isfinite(float(reward))
        if bool(terminated):
            break


def test_registry_matches_group_vocabulary(spec):
    assert spec.group in registry.GROUPS


# ---------------------------------------------------------------------------
# 7. Controller effectiveness
#
# The gap this section closes: ``test_pid_baseline_produces_valid_actions``
# checks that a PID emits finite, correctly-shaped actions -- and it passed a
# glass-furnace PID that was tracking *fuel percentage* as its temperature
# setpoint, because the observation vector gained fields and the controller's
# hardcoded indices did not follow. A controller can be entirely well-formed
# and still not control. These tests assert that it does.
# ---------------------------------------------------------------------------


# Envs and params are cached per spec. Building them fresh per episode forces
# JIT recompilation every time -- ``get_obs`` takes ``params`` as a *static*
# argname, so each new params instance is a new compilation unit, and the
# recompiles dominated this contract's runtime.
_EFFECTIVENESS_CACHE: dict[str, tuple] = {}


def _effectiveness_env(spec):
    """Cached (env, params, jitted_step) for the effectiveness contract.

    ``step_env`` is jitted once per environment. Unjitted, every call
    re-traces -- for the reactor that means re-tracing a 10-substep stiff
    integrator scan on each control step, which dominated the runtime.
    """
    if spec.name not in _EFFECTIVENESS_CACHE:
        env = spec.make_env()
        params = spec.make_test_params(**spec.effectiveness_overrides)
        step = jax.jit(lambda k, st, a: env.step_env(k, st, a, params))
        _EFFECTIVENESS_CACHE[spec.name] = (env, params, step)
    return _EFFECTIVENESS_CACHE[spec.name]


def _episode_return(spec, policy, seed, params=None):
    env, p, step = _effectiveness_env(spec)
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, p)
    total = 0.0
    # Loop on the *clock*, not on a step count. ``max_steps_in_episode`` is in
    # physics steps, and the reactor advances ``control_period`` (10) of them
    # per ``step_env`` -- counting iterations ran its episodes 10x too long.
    while int(state.time) < int(p.max_steps_in_episode):
        key, sub = jax.random.split(key)
        obs, state, reward, terminated, _ = step(sub, state, jnp.asarray(policy(obs)))
        total += float(reward)
        if bool(terminated):
            break
    return total


# Kept deliberately small: this contract runs one full episode per (policy,
# seed) pair across every registered environment, so the product dominates the
# slow suite's runtime. Three constants bracketing the action range and two
# seeds is enough to catch a controller that is not controlling -- the failure
# it exists to detect (a mis-indexed setpoint) is not a marginal one.
CONSTANT_ACTIONS = (-0.5, 0.0, 0.5)
EFFECTIVENESS_SEEDS = 2
# No extra episode cap: ``EnvSpec.test_params`` already sizes each episode to
# that environment's own dynamics. A flat cap is meaningless across envs whose
# characteristic times differ by orders of magnitude -- 150 physics steps is
# 150 s for the reactor, whose xenon transient runs for hours, so the PID had
# no time to demonstrate anything and lost to a constant.


@pytest.mark.slow
def test_pid_outperforms_the_best_constant_action(spec):
    """The shipped PID must beat every open-loop constant action.

    A deliberately weak bar -- a constant action is the most trivial policy
    there is -- but it is exactly the bar a mis-wired controller fails, and it
    needs no per-environment threshold, so every environment inherits it.
    """
    if not spec.has_pid:
        pytest.skip(f"{spec.name}: {spec.baselines_note}")
    if spec.expert_degraded:
        pytest.xfail(f"{spec.name}: {spec.expert_degraded}")

    env, eff_params, _ = _effectiveness_env(spec)
    action_shape = env.action_space(eff_params).shape

    pid = spec.make_pid()
    pid_returns = []
    for seed in range(EFFECTIVENESS_SEEDS):
        pid.reset()
        pid_returns.append(_episode_return(spec, lambda o: pid(o), seed))
    pid_mean = float(np.mean(pid_returns))

    best_constant, best_value = -np.inf, None
    for c in CONSTANT_ACTIONS:
        mean = float(
            np.mean(
                [
                    _episode_return(
                        spec, lambda o, c=c: jnp.full(action_shape, c), seed
                    )
                    for seed in range(EFFECTIVENESS_SEEDS)
                ]
            )
        )
        if mean > best_constant:
            best_constant, best_value = mean, c

    assert pid_mean > best_constant, (
        f"{spec.name}: PID scores {pid_mean:.2f}, worse than holding a constant "
        f"action of {best_value:+.1f} ({best_constant:.2f}). The controller is "
        "well-formed but not controlling -- check that its observation indices "
        "still match the environment's get_obs layout."
    )


@pytest.mark.slow
def test_pid_is_deterministic_across_resets(spec):
    """Resetting the controller must reproduce the run exactly.

    A stateful controller that leaks integrator state between episodes makes
    every benchmark number order-dependent.
    """
    if not spec.has_pid:
        pytest.skip(f"{spec.name}: {spec.baselines_note}")
    pid = spec.make_pid()
    pid.reset()
    first = _episode_return(spec, lambda o: pid(o), 0)
    pid.reset()
    second = _episode_return(spec, lambda o: pid(o), 0)
    assert first == pytest.approx(second, rel=1e-5, abs=1e-5)


# ---------------------------------------------------------------------------
# The model must stay physical wherever an optimiser can drive it
# ---------------------------------------------------------------------------

# Attitude angles only, and only the ones that are *integrated*. "alpha" is
# deliberately absent: it is angle of attack on the aircraft but residual
# calcination extent on the cement kiln, where it is a fraction in [0, 1] and an
# array. Matching on names alone produced that false positive.
ANGLE_FIELDS = ("theta", "phi", "psi", "gamma")

# Environments whose attitude winds up without bound. Recorded rather than
# skipped: the contract is right and these are the gap it found.
UNDAMPED_ATTITUDE = {"plane", "plane3d_heading", "plane3d_circle", "plane3d_figure8"}
MAX_TURNS = 4.0  # full rotations before we call it a tumble rather than a manoeuvre


def test_angles_stay_bounded_under_extreme_actions(spec):
    """An angle that accumulates without bound means damping is missing.

    This is the contract the aircraft's zero-drag stall defect slipped past for
    so long. Every drag test probed attached flow, where ``CD = cd0 + k*CL**2``
    is self-consistent -- testing drag there tests the formula against itself,
    and cannot reveal that it has no separated-flow term at all. The two tests
    that did reach past the stall were *satisfied* by the defect: a lift
    collapse to zero is a maximal collapse, and a sweep asserting ``isfinite``
    and ``cd > 0`` is happy with a wing producing less drag than in cruise.

    A benchmark is where this matters most, because an optimiser goes looking
    for the region nobody modelled. So the contract is about reachability, not
    about the design point: drive the plant to its limits and the state must
    stay physical, or the episode must end.
    """
    env = spec.make_env()
    params = spec.make_test_params()
    probe = env.reset_env(jax.random.PRNGKey(0), params)[1]
    fields = [
        f
        for f in ANGLE_FIELDS
        if hasattr(probe, f) and np.ndim(np.asarray(getattr(probe, f))) == 0
    ]
    if not fields:
        pytest.skip(f"{spec.name} has no scalar attitude state")
    if spec.name in UNDAMPED_ATTITUDE:
        pytest.xfail(
            f"{spec.name}: attitude is integrated with no aerodynamic rate "
            "damping -- a real tail's incidence carries a q*l/V term that "
            "opposes pitch rate, and nothing here does. A departed aircraft "
            "therefore tumbles indefinitely. Distinct from the post-stall drag "
            "defect (fixed): that removed the forces, this removes the moment "
            "that would arrest the rotation."
        )

    space = env.action_space(params)
    shape = space.shape or (1,)
    low = np.broadcast_to(np.asarray(space.low, float), shape)
    high = np.broadcast_to(np.asarray(space.high, float), shape)
    step = jax.jit(env.step_env)

    worst = 0.0
    for frac in (0.0, 1.0):
        action = jnp.asarray(low + frac * (high - low))
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)
        for _ in range(int(params.max_steps_in_episode)):
            _, state, _, terminated, _ = step(key, state, action, params)
            for f in fields:
                worst = max(worst, abs(float(getattr(state, f))))
            if bool(terminated):
                break

    turns = worst / (2 * np.pi)
    assert turns < MAX_TURNS, (
        f"{spec.name}: an angle reached {turns:.1f} full rotations "
        f"({np.rad2deg(worst):.0f} deg) while the episode continued. Unbounded "
        "accumulation means the model has no damping for that axis."
    )
