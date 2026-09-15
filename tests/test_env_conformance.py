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


def test_env_declares_where_the_value_and_its_target_live(spec):
    """Both index attributes must exist and address finite observations.

    ``runners.rollout`` reads both to know what it is plotting and scoring, so
    an environment missing one cannot be rolled out at all. That is not a
    theoretical failure: ``Plane3DRacetrack`` shipped without
    ``obs_target_index``, and since the gain search scores candidates inside a
    ``try`` it reported every one of them as ``-inf`` and tuned nothing, while
    ``scripts/record_baselines.py`` could not have recorded the environment.

    The equivalents in ``tests/test_runners.py`` name three classes by hand,
    which is why a fourth got past them. This one sweeps the registry.
    """
    from target_gym.runners.runners import _as_tuple

    env, params = spec.make_env(), spec.make_test_params()
    for attr in ("obs_value_index", "obs_target_index"):
        assert hasattr(env, attr), (
            f"{spec.name}: no {attr}. Nothing can roll this environment out "
            f"until it says which observation carries its tracked value and "
            f"which carries the setpoint."
        )

    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)
    obs = np.asarray(obs)
    value_idx = _as_tuple(env.obs_value_index)
    target_idx = _as_tuple(env.obs_target_index)
    assert len(value_idx) == len(target_idx), (
        f"{spec.name}: {len(value_idx)} tracked value(s) against "
        f"{len(target_idx)} target(s). Every tracked value needs its setpoint."
    )
    for i in (*value_idx, *target_idx):
        assert 0 <= i < obs.shape[-1], (
            f"{spec.name}: index {i} is outside an observation of "
            f"{obs.shape[-1]} elements."
        )
        assert np.isfinite(obs[i]), f"{spec.name}: observation {i} is not finite."


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


def test_reward_is_finite_and_a_cost_over_an_episode(spec):
    """Version 2: every per-step reward is finite and non-positive (a cost);
    the failure charge is the only step that may exceed the envelope's cost."""
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    action = _zero_action(env, params)
    _step = jax.jit(env.step_env)
    for _ in range(120):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = _step(sub, state, action, params)
        assert np.isfinite(float(reward)) and float(reward) <= 0.0, float(reward)
        if bool(terminated):
            break


def test_reward_is_bounded_over_an_episode(spec):
    """Version 1: per-step reward must stay within a sane magnitude."""
    env, params = spec.make_env(), spec.make_test_params(reward_version=1)
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
    # Float32 under jit reorders sums; an observation formed as a difference
    # of kilometre-scale positions (the racetrack's cross-track) can move by
    # an ulp of those positions, so the tolerance follows the vector's scale.
    scale = float(jnp.max(jnp.abs(obs_e)))
    assert jnp.allclose(obs_j, obs_e, atol=1e-4 + 1e-6 * scale)
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
    # Loop on the *clock*, as ``runners.rollout`` does, so the episode ends
    # exactly when the environment says it does.
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
# characteristic times differ by orders of magnitude -- 150 steps is 25 min
# for the reactor, whose xenon transient runs for hours, so the PID had no time
# to demonstrate anything and lost to a constant.


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

# Attitude *rates*, not attitude. "alpha" is deliberately absent from the angle
# list: it is angle of attack on the aircraft but residual calcination extent on
# the cement kiln, where it is a fraction in [0, 1] and an array.
ANGLE_RATE_FIELDS = ("theta_dot", "phi_dot", "psi_dot")

# Half a rotation per second. No transport aircraft sustains this, and no
# process plant has an attitude at all, so exceeding it means the state is
# running away rather than manoeuvring.
MAX_RATE_DEG_S = 180.0


def test_attitude_rates_stay_bounded_under_extreme_actions(spec):
    """A state that runs away means the model has no damping for that axis.

    This began as a check on the *angle*, which was wrong: held at full nose-up
    elevator an aircraft loops, and a looping aircraft accumulates pitch angle
    forever with a perfectly bounded rate. The measurement showed it plainly --
    7 to 14 deg/s of pitch rate while altitude cycled between 3000 and 5400 m.
    That is a manoeuvre, not a divergence.

    The rate is the quantity that distinguishes them, and it is the one the
    original defect broke: with a separated wing producing no drag and no
    damping, pitch rate reached 270 deg/s and was still climbing.

    The contract is stated over the reachable state space rather than the design
    point, because a benchmark is where that distinction bites: an optimiser
    goes looking for the corner nobody modelled.
    """
    env = spec.make_env()
    params = spec.make_test_params()
    probe = env.reset_env(jax.random.PRNGKey(0), params)[1]
    fields = [
        f
        for f in ANGLE_RATE_FIELDS
        if hasattr(probe, f) and np.ndim(np.asarray(getattr(probe, f))) == 0
    ]
    if not fields:
        pytest.skip(f"{spec.name} has no scalar attitude rate")

    space = env.action_space(params)
    shape = space.shape or (1,)
    low = np.broadcast_to(np.asarray(space.low, float), shape)
    high = np.broadcast_to(np.asarray(space.high, float), shape)
    step = jax.jit(env.step_env)

    worst, where = 0.0, None
    for frac in (0.0, 1.0):
        action = jnp.asarray(low + frac * (high - low))
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)
        for _ in range(int(params.max_steps_in_episode)):
            _, state, _, terminated, _ = step(key, state, action, params)
            for f in fields:
                rate = abs(np.rad2deg(float(getattr(state, f))))
                if rate > worst:
                    worst, where = rate, f
            if bool(terminated):
                break

    assert worst < MAX_RATE_DEG_S, (
        f"{spec.name}: {where} reached {worst:.0f} deg/s under extreme actions. "
        "A rate that large is a state running away, not a manoeuvre -- the axis "
        "has no damping."
    )


# ---------------------------------------------------------------------------
# 7. Model review checks
#
# Promoted from docs/model-review-checklist.md, which was a page of prose and a
# set of scratch scripts. Each check below found something real once, and each
# is cheap enough to keep. The allowlists exist so that a check reports *new*
# defects rather than re-reporting the known-benign findings every run -- and
# so that adding an entry to one is a deliberate act with a reason attached.
# ---------------------------------------------------------------------------


# Check 3. Fields written into state and never read back. All four are records
# of a commanded or diagnostic quantity: the local variable of that name carries
# the value into the dynamics, and the state field only reports it.
KNOWN_WRITE_ONLY_STATE_FIELDS = {
    ("HVACState", "T_surface"),
    ("HVACState", "Q_command"),
    ("GlassFurnaceState", "T_stack"),
    ("WindTurbineState", "torque_cmd"),
}


def test_no_new_write_only_state_fields():
    """Check 3: a field written but never read hides a disagreement.

    ``state.m`` was once set to 92 588 kg, above the aircraft's maximum takeoff
    weight, while the dynamics integrated ``initial_mass`` directly. Nothing
    read the field, so a 20-tonne error sat there until fuel burn made mass
    load-bearing.
    """
    import ast
    import pathlib
    import re

    src = pathlib.Path("src/target_gym")
    texts = {f: f.read_text() for f in sorted(src.rglob("*.py"))}
    everything = "\n".join(texts.values())

    found = set()
    for path, text in texts.items():
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover - not expected
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name.endswith("State")):
                continue
            for item in node.body:
                if not (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                ):
                    continue
                name = item.target.id
                if not re.search(rf"\.{re.escape(name)}\b", everything):
                    found.add((node.name, name))

    new = found - KNOWN_WRITE_ONLY_STATE_FIELDS
    assert not new, (
        f"state fields written but never read: {sorted(new)}. Either the "
        "dynamics meant to use one and do not, or it is a diagnostic record -- "
        "read it before adding it to KNOWN_WRITE_ONLY_STATE_FIELDS."
    )


# Check 4. Tuple unpacks that discard a component. All three discard position
# coordinates the equations of motion genuinely do not depend on, or a scan
# carry.
KNOWN_DISCARDED_UNPACKS = {
    ("plane/dynamics.py", "_, z, theta = positions"),
    ("plane3d/dynamics.py", "_, _, z, theta, phi = positions"),
    ("utils.py", "_, rewards = result"),
}


def test_no_state_component_is_silently_discarded():
    """Check 4: ``x_dot, z_dot, _ = velocities`` threw away the pitch rate.

    That one line is why the aircraft had no pitch damping and a departed
    airframe tumbled indefinitely.
    """
    import pathlib
    import re

    src = pathlib.Path("src/target_gym")
    found = set()
    for path in sorted(src.rglob("*.py")):
        rel = str(path.relative_to(src))
        for line in path.read_text().splitlines():
            m = re.match(r"\s*([\w\s,]+)=\s*([\w\.\[\]]+)\s*$", line)
            if not m:
                continue
            lhs = [x.strip() for x in m.group(1).split(",")]
            if len(lhs) >= 2 and "_" in lhs:
                found.add((rel, line.strip()))

    new = found - KNOWN_DISCARDED_UNPACKS
    assert not new, (
        f"tuple unpacks discarding a component: {sorted(new)}. Confirm the "
        "physics genuinely does not depend on it before allowlisting."
    )


@pytest.mark.slow
def test_actuator_can_move_the_tracked_variable(spec):
    """Check 8: an actuator whose authority was written down, not derived.

    The aileron moment once applied the wing's lift-curve slope to the control
    deflection, rolling the aircraft at 84 deg/s against a transport's 25-30.
    The plant-agnostic form is weaker but catches the opposite failure: an
    actuator that cannot move its own tracked variable is decorative, and every
    score against it measures the initial condition.
    """
    from target_gym.runners.runners import _as_tuple

    env, params = spec.make_env(), spec.make_test_params()
    idx = list(_as_tuple(env.obs_value_index))
    space = env.action_space(params)
    shape = space.shape or (1,)
    low = np.broadcast_to(np.asarray(space.low, float), shape)
    high = np.broadcast_to(np.asarray(space.high, float), shape)
    step = jax.jit(env.step_env)

    excursion = 0.0
    for action in (jnp.asarray(low), jnp.asarray(high)):
        key = jax.random.PRNGKey(0)
        obs, state = env.reset_env(key, params)
        start = np.asarray(obs)[idx]
        for _ in range(min(int(params.max_steps_in_episode), 400)):
            key, sub = jax.random.split(key)
            obs, state, _, terminated, _ = step(sub, state, action, params)
            excursion = max(
                excursion, float(np.abs(np.asarray(obs)[idx] - start).max())
            )
            if bool(terminated):
                break

    assert excursion > 0.0, (
        f"{spec.name}: full actuator travel in both directions never moved the "
        "tracked variable. The actuator has no authority over the thing the "
        "environment scores."
    )


# Environments whose terminal guard cannot fire, and why that is known rather
# than suspected. A guard that can never trip is harmless as defence and
# actively misleading as documentation -- it reads as a failure mode the plant
# has, and it makes ``mpc_terminated_early`` a structural zero rather than an
# earned one. The four-tank's PHYSICS.md already records its dead ``h_max`` in
# this spirit; this pins the rest so none of them can silently become reachable
# after a parameter change without somebody noticing it had not been.
KNOWN_UNREACHABLE_TERMINALS = {
    "first_order": (
        "x is first-order toward K*u, which the action bounds cap at +/-2, "
        "from a start inside +/-0.5, so |x| never approaches the +/-3 trip"
    ),
}


def test_unreachable_terminals_are_still_unreachable(spec):
    """Check 8: a guard that cannot fire must be known not to fire.

    Only the listed environments are asserted unreachable. Everything else is
    left alone: most trips here are reachable and several are the point of the
    task.
    """
    reason = KNOWN_UNREACHABLE_TERMINALS.get(spec.name)
    if reason is None:
        pytest.skip(f"{spec.name} is not claimed unreachable")

    env = spec.make_env()
    params = spec.make_test_params()
    space = env.action_space(params)
    shape = space.shape or (1,)
    low = np.broadcast_to(np.asarray(space.low, float), shape)
    high = np.broadcast_to(np.asarray(space.high, float), shape)
    step = jax.jit(env.step_env)

    for frac in (0.0, 0.5, 1.0):
        action = jnp.asarray(low + frac * (high - low))
        key = jax.random.PRNGKey(0)
        _, state = env.reset_env(key, params)
        for _ in range(int(params.max_steps_in_episode)):
            _, state, _, terminated, _ = step(key, state, action, params)
            assert not bool(terminated), (
                f"{spec.name}: terminated under a constant action, but is "
                f"listed as unreachable because {reason}. Either the guard is "
                "now live and the entry should go, or something moved."
            )


@pytest.mark.slow
def test_plant_does_not_accelerate_without_input(spec):
    """Check 7: can the energy budget be bounded from outside?

    Energy may only enter through the actuator, so with the actuator at zero
    nothing may grow without bound. Growth alone proves nothing -- position
    grows because an aircraft flies forward, and cumulative fade grows because
    it counts -- so what is asserted is that growth does not *accelerate*: the
    mean increment over the last fifth of an unforced run against the first.
    An integrator holds near 1; an unstable mode runs away.
    """
    env = spec.make_env()
    params = spec.make_test_params()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    fields = [
        f
        for f in state.__dataclass_fields__
        if f != "time"
        and np.ndim(getattr(state, f, None)) == 0
        and np.issubdtype(np.asarray(getattr(state, f)).dtype, np.floating)
    ]
    if not fields:
        pytest.skip(f"{spec.name} has no scalar float state")

    step = jax.jit(env.step_env)
    action = _zero_action(env, params)
    traj = [np.array([float(getattr(state, f)) for f in fields])]
    for _ in range(min(int(params.max_steps_in_episode), 600)):
        key, sub = jax.random.split(key)
        _, state, _, terminated, _ = step(sub, state, action, params)
        traj.append(np.array([float(getattr(state, f)) for f in fields]))
        if bool(terminated):
            break

    tr = np.stack(traj)
    assert np.isfinite(tr).all(), f"{spec.name}: non-finite state under zero input"
    if len(tr) < 10:
        pytest.skip(f"{spec.name} terminates too early to measure a trend")

    inc = np.abs(np.diff(tr, axis=0))
    fifth = max(len(inc) // 5, 1)
    early = inc[:fifth].mean(axis=0)
    late = inc[-fifth:].mean(axis=0)

    # Only fields that actually move are measurable. The floor has to scale
    # with the field, because JAX computes in float32 and a stationary field
    # still jitters in its last bits: ``plane3d_figure8`` flies wings level
    # under zero input, so its heading is constant at 1.4296085 rad, and the
    # increments are 1e-9 early against 1.5e-8 late -- both far below the
    # 1.7e-7 that is one float32 ULP at that magnitude. An absolute 1e-12 floor
    # admitted that as a 15x acceleration and failed the check on round-off.
    noise = 32.0 * np.finfo(np.float32).eps * np.maximum(np.abs(tr).mean(axis=0), 1.0)
    measurable = early > np.maximum(noise, 1e-12)
    ratio = np.where(measurable, late / np.maximum(early, 1e-12), 0.0)
    j = int(np.argmax(ratio))

    assert ratio[j] < ACCELERATION_LIMIT, (
        f"{spec.name}: {fields[j]} grows {ratio[j]:.1f}x faster at the end of an "
        "unforced run than at the start. With the actuator at zero nothing is "
        "supplying it, so the plant is producing it."
    )


# Measured worst over all eighteen: the glass furnace's T_work at 3.25x over
# 3000 unforced steps. 8x leaves room for the shorter run used here without
# admitting a genuinely unstable mode.
ACCELERATION_LIMIT = 8.0


# Check 5. Environments with a known seam that survives refinement, and what it
# is. Everything else must join smoothly.
KNOWN_SEAMS = {
    "plane": "shock-stall model near 308 m/s, outside the reachable envelope",
    # The same aircraft, flying a moving setpoint: same plant, same seam.
    "plane_steps": "shock-stall model near 308 m/s, unreachable (same plant as plane)",
    "plane_sine": "shock-stall model near 308 m/s, unreachable (same plant as plane)",
    "plane_energy": "shock-stall model near 308 m/s, unreachable (same plant as plane)",
    "plane3d_racetrack": "shock-stall model near 308 m/s, unreachable (same airframe)",
    "plane3d_heading": "shock-stall model near 308 m/s, unreachable",
    "plane3d_circle": "shock-stall model near 308 m/s, unreachable",
    "plane3d_figure8": "shock-stall model near 308 m/s, unreachable",
    "battery": "state-of-charge-dependent power limit binding near soc 0.115",
    "reactor": "fuel-temperature feedback near 718 K",
    "glass_furnace": "m_batch = maximum(.., 0) -- the batch blanket running out",
}
SEAM_LIMIT = 0.5


@pytest.mark.slow
def test_regimes_join_smoothly(spec):
    """Check 5: do the regimes join?

    Past the stall the aircraft collapsed lift, and because drag was
    ``cd0 + k*CL**2`` it collapsed drag with it, so a separated wing had *less*
    drag than in cruise.

    Sampling cannot test this: comparing the largest step to the typical step
    flags any function whose derivative grows, which ranks Arrhenius above every
    real seam. These dynamics are differentiable, so compare the two one-sided
    Jacobians of a single step instead --

        jump = |J(x+eps) - J(x-eps)| / (|J(x+eps)| + |J(x-eps)|)

    -- which goes to zero for any smooth function however steep, and holds
    across a kink. Both sides must be active: where one Jacobian is exactly zero
    the ratio is 1 by construction, and that is a saturation, not two physical
    descriptions failing to join.
    """
    env, params = spec.make_env(), spec.make_test_params()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    fields = [
        f
        for f in state.__dataclass_fields__
        if f != "time"
        and np.ndim(getattr(state, f, None)) == 0
        and np.issubdtype(np.asarray(getattr(state, f)).dtype, np.floating)
    ]
    if not fields:
        pytest.skip(f"{spec.name} has no scalar float state")
    action = _zero_action(env, params)

    worst, where = 0.0, None
    for f in fields:
        base = float(getattr(state, f))
        span = abs(base) if abs(base) > 1e-6 else 1.0
        xs = jnp.linspace(base - 0.75 * span, base + 0.75 * span, 121)
        eps = 1e-3 * span

        def one(x, _f=f):
            s = state.replace(**{_f: x})
            _, s2, _, _, _ = env.step_env(key, s, action, params)
            return jnp.stack([getattr(s2, g) for g in fields])

        try:
            jac = jax.jit(jax.vmap(jax.jacfwd(one)))
            Jm, Jp = jac(xs - eps), jac(xs + eps)
        except Exception:  # pragma: no cover - a plant that will not linearise
            continue
        num, den = jnp.abs(Jp - Jm), jnp.abs(Jp) + jnp.abs(Jm)
        peak = jnp.max(den, axis=0, keepdims=True)
        both = jnp.minimum(jnp.abs(Jp), jnp.abs(Jm)) > 0.05 * jnp.maximum(peak, 1e-30)
        jump = np.nan_to_num(
            np.asarray(jnp.where(both & (den > 0), num / jnp.maximum(den, 1e-30), 0.0))
        )
        if jump.size and jump.max() > worst:
            worst = float(jump.max())
            k = int(np.argmax(jump.max(axis=0)))
            where = f"{f} -> {fields[k]}"

    if spec.name in KNOWN_SEAMS:
        pytest.skip(f"{spec.name}: known seam -- {KNOWN_SEAMS[spec.name]}")
    assert worst < SEAM_LIMIT, (
        f"{spec.name}: the dynamics have a seam at {where} (jump {worst:.2f}). "
        "Two descriptions of this plant meet there and do not join. A rapid "
        "transition is fine; a discontinuous derivative is a modelling error "
        "unless it is a physical limit -- document it in KNOWN_SEAMS if so."
    )
