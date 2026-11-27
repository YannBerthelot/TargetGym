"""The state estimator an observation-only MPC plans from (item 5's reading of
2026-11-28, docs/task-protocol.md).

Contract, from the brief and Kalman filtering's textbook results, not from the
code:

- it is an extended Kalman filter on the task's own model: the env's step with
  its process noise zeroed predicts, its readings (``env.readings``) say how
  noisy each observation is, and the env's own step noise and reset spread give
  the process noise and the prior;
- it is fed the observations and the actions only: ``reset(obs)`` and
  ``update(belief, action, obs)``, and ``belief.state`` is an env state the MPC
  reads in place of the true one;
- on a linear-Gaussian plant it is the Kalman filter, so its covariance
  converges to the steady state of the discrete Riccati equation, its mean is
  the textbook filter's, and its error is consistent with its stated covariance
  (the normalised estimation error squared, NEES, is chi-square);
- a hidden field the observations never see keeps its prior, whatever its
  true value;
- ``jit``, ``vmap`` and ``lax.scan`` work on it.

The plant here is a tank with a hidden inflow, built on the same base helpers
as every task (``canonical_reset``, ``measured_obs``, ``finish_step``), so its
readings, sensor history and noise behave as a task's do. Its model is linear,
so every expected value comes from the textbook Kalman filter.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import spaces
from scipy import linalg, stats

from target_gym.base import (
    EnvParams,
    EnvState,
    FieldReading,
    actuator_field,
    canonical_reset,
    finish_step,
    measured_obs,
)
from target_gym.estimation import Belief
from target_gym.estimation import Estimator as _Estimator

LEAK = 0.1  # the level loses a tenth of itself each step
LEVEL_SIGMA = 0.05  # the level gauge's scatter
INFLOW_SIGMA = 0.02  # the hidden inflow's random walk, per step


@struct.dataclass
class TankParams(EnvParams):
    delta_t: float = 1.0
    max_steps_in_episode: int = 400
    inflow_sigma: float = INFLOW_SIGMA
    level_std: float = 0.5  # the reset's spread of the level
    inflow_std: float = 0.2  # and of the hidden inflow
    reward_version: int = 2
    restart_steps: int = 1
    failure_cost: float = 0.0


@struct.dataclass
class TankState(EnvState):
    level: jax.Array
    inflow: jax.Array  # hidden
    spare: jax.Array  # hidden, and in neither the dynamics nor any reading
    root: jax.Array  # the cube root's stand-in, solved by bisection (_bisect)
    actuator_held: jax.Array = actuator_field(1)


def Estimator(env, params, fields, **kwargs):
    """The filter on the tank, which is not in the registry, so its noise's
    field is named here as a task's registry entry names it."""
    return _Estimator(env, params, fields, noise_fields=("inflow_sigma",), **kwargs)


def _bisect(level):
    """The ``y`` with ``y + y**3 / 10 = level``, by bisection as pH's solve is:
    exact to float32, with a JAX derivative of zero."""

    def body(bounds, _):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        below = mid + mid**3 / 10.0 < level
        return (jnp.where(below, mid, lo), jnp.where(below, hi, mid)), None

    (lo, hi), _ = jax.lax.scan(body, (level - 20.0, level + 20.0), None, length=40)
    return 0.5 * (lo + hi)


class Tank:
    """``level' = (1 - LEAK) level + u + inflow``; the inflow walks."""

    registry_name = None

    def __init__(self, readings=(FieldReading("level", sigma=LEVEL_SIGMA),)):
        self.readings = readings

    @property
    def default_params(self):
        return TankParams()

    def action_space(self, params=None):
        return spaces.Box(-jnp.ones(1), jnp.ones(1), (1,), jnp.float32)

    @canonical_reset
    def reset_env(self, key, params=None):
        p = self.default_params if params is None else params
        k1, k2, k3 = jax.random.split(key, 3)
        level = p.level_std * jax.random.normal(k1)
        state = TankState(
            time=0,
            level=level,
            inflow=p.inflow_std * jax.random.normal(k2),
            spare=jax.random.normal(k3),
            root=_bisect(level),
        )
        return self.get_obs(state, p), state

    def step_env(self, key, state, action, params=None):
        p = self.default_params if params is None else params
        noise = jax.random.normal(jax.random.fold_in(key, state.time))
        level = (1.0 - LEAK) * state.level + jnp.reshape(action, ()) + state.inflow
        new = state.replace(
            time=state.time + 1,
            level=level,
            inflow=state.inflow + p.inflow_sigma * noise,
            root=_bisect(level),
        )
        new, reward, done, info = finish_step(self, key, state, new, p)
        return self.get_obs(new, p), new, reward, done, info

    @measured_obs
    def get_obs(self, state, params=None):
        return jnp.stack([state.level, state.root])

    def is_terminated(self, state, params):
        return jnp.asarray(False)

    def reward_terms(self, state, params, tripped=None):
        return {"level": state.level**2}

    def compute_reward(self, state, params):
        return -(state.level**2)


# The textbook model of the tank on (level, inflow).
F = np.array([[1.0 - LEAK, 1.0], [0.0, 1.0]])
H = np.array([[1.0, 0.0]])


def _steady_posterior(q, r):
    """The Kalman filter's steady posterior covariance: the stabilising
    solution of the filter's Riccati equation (SciPy's DARE on the dual
    system), then one measurement update."""
    prior = linalg.solve_discrete_are(F.T, H.T, q, r)
    gain = prior @ H.T @ np.linalg.inv(H @ prior @ H.T + r)
    return prior - gain @ H @ prior


def _run(est, env, params, seed, steps, action=0.0):
    """One episode on a constant action: the true states and the beliefs."""
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    belief = est.reset(obs)
    states, beliefs = [state], [belief]
    u = jnp.array([action])
    for _ in range(steps):
        obs, state, *_ = env.step_env(key, state, u, params)
        belief = est.update(belief, u, obs)
        states.append(state)
        beliefs.append(belief)
    return states, beliefs


def _episode(est, env, params, steps, action=0.0):
    """``_run`` as one ``lax.scan``, a function of the seed for ``jit`` and
    ``vmap``: the observations, true states and beliefs at the reset and after
    each step, stacked."""
    u = jnp.array([action])

    def run(seed):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        belief = est.reset(obs)

        def step(carry, _):
            state, belief = carry
            obs, state, *_ = env.step_env(key, state, u, params)
            belief = est.update(belief, u, obs)
            return (state, belief), (obs, state, belief)

        _, after = jax.lax.scan(step, (state, belief), None, length=steps)
        first = jax.tree.map(lambda x: x[None], (obs, state, belief))
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b]), first, after)

    return run


@pytest.fixture(scope="module")
def walking():
    """The tank whose inflow walks, read by its level gauge alone."""
    env = Tank()
    params = env.default_params
    return env, params, Estimator(env, params, ("level", "inflow"), channels=(0,))


def test_the_covariance_converges_to_the_kalman_steady_state():
    """With the inflow's walk stated (``random_walk``) and none in the plant,
    the process noise is exactly diag(0, 0.02^2) and the reading's 0.05^2, so
    after 300 steps the filter's covariance is the Riccati equation's."""
    env = Tank()
    params = env.default_params.replace(inflow_sigma=0.0)
    est = Estimator(
        env,
        params,
        ("level", "inflow"),
        channels=(0,),
        random_walk={"inflow": INFLOW_SIGMA},
    )
    _, _, beliefs = jax.jit(_episode(est, env, params, 300))(0)
    expected = _steady_posterior(
        np.diag([0.0, INFLOW_SIGMA**2]), np.array([[LEVEL_SIGMA**2]])
    )
    # float32 arithmetic over a converged recursion: a few parts in 1e5.
    np.testing.assert_allclose(beliefs.cov[-1], expected, rtol=1e-3, atol=1e-8)


def test_the_mean_is_the_textbook_kalman_filters():
    """The same observations through a NumPy Kalman filter written from the
    textbook, started from the filter's own prior, give the same means."""
    env = Tank()
    params = env.default_params.replace(inflow_sigma=0.0)
    est = Estimator(
        env,
        params,
        ("level", "inflow"),
        channels=(0,),
        random_walk={"inflow": INFLOW_SIGMA},
    )
    obs, _, beliefs = jax.jit(_episode(est, env, params, 60, action=0.3))(3)
    q = np.diag([0.0, INFLOW_SIGMA**2])
    r = np.array([[LEVEL_SIGMA**2]])
    x = np.array([float(est.prior.state.level), float(est.prior.state.inflow)])
    p = np.asarray(est.prior.cov, float)
    means = []
    for t, y in enumerate(np.asarray(obs[:, :1], float)):
        if t:
            x = F @ x + np.array([0.3, 0.0])
            p = F @ p @ F.T + q
        gain = p @ H.T @ np.linalg.inv(H @ p @ H.T + r)
        x = x + gain @ (y - H @ x)
        p = (np.eye(2) - gain @ H) @ p
        means.append(x.copy())
    got = np.stack([beliefs.state.level, beliefs.state.inflow], axis=1)
    np.testing.assert_allclose(got, np.array(means), rtol=1e-4, atol=1e-5)


def test_the_process_noise_is_the_plants_own_step_noise(walking):
    """Unless told otherwise the filter takes the noise the env's step draws:
    the inflow's 0.02 a step and nothing on the level. Estimated from the
    draws, so within five standard errors of a variance from 1024 of them."""
    _, _, est = walking
    q = np.asarray(est.process_noise, float)
    tol = 5 * INFLOW_SIGMA**2 * np.sqrt(2 / 1023)
    assert abs(q[1, 1] - INFLOW_SIGMA**2) < tol
    assert abs(q[0, 0]) < 1e-12 and abs(q[0, 1]) < tol


def test_the_prior_is_the_resets_spread(walking):
    """Before any observation the filter believes what the reset draws: level
    and inflow at their mean 0 with spreads 0.5 and 0.2, within five standard
    errors of 1024 draws."""
    _, _, est = walking
    cov = np.asarray(est.prior.cov, float)
    for i, std in enumerate((0.5, 0.2)):
        assert abs(cov[i, i] - std**2) < 5 * std**2 * np.sqrt(2 / 1023)
    assert abs(float(est.prior.state.level)) < 5 * 0.5 / 32
    assert abs(float(est.prior.state.inflow)) < 5 * 0.2 / 32


@pytest.mark.parametrize(
    "sigma, step, scale, variance",
    [
        (0.05, 0.0, 1.0, 0.05**2),
        # A reading rounded to 0.1 adds the rounding's uniform error, 0.1^2 / 12.
        (0.05, 0.1, 1.0, 0.05**2 + 0.1**2 / 12),
        (0.05, 0.0, 2.0, 0.1**2),
    ],
)
def test_the_measurement_noise_is_the_readings(sigma, step, scale, variance):
    env = Tank(readings=(FieldReading("level", sigma=sigma, step=step),))
    params = env.default_params.replace(sensor_noise_scale=scale)
    est = Estimator(env, params, ("level", "inflow"), channels=(0,))
    np.testing.assert_allclose(est.measurement_noise, [[variance]], rtol=1e-5)


def _nees(est, env, params, fields, runs=200, steps=150):
    """The mean over ``runs`` episodes of the NEES of ``fields`` at the end."""
    _, states, beliefs = jax.jit(jax.vmap(_episode(est, env, params, steps)))(
        jnp.arange(runs)
    )
    order = [est.fields.index(f) for f in fields]
    true = jnp.stack([getattr(states, f)[:, -1] for f in fields], axis=-1)
    mean = jnp.stack([getattr(beliefs.state, f)[:, -1] for f in fields], axis=-1)
    cov = beliefs.cov[:, -1][:, order][:, :, order]
    err = np.asarray(true - mean, float)
    return float(np.mean(np.einsum("ri,rij,rj->r", err, np.linalg.inv(cov), err)))


def _chi2_band(dof, runs=200):
    """The mean of ``runs`` chi-square(dof) draws lies here 99.9 % of the time."""
    lo, hi = stats.chi2.ppf([0.0005, 0.9995], dof * runs) / runs
    return lo, hi


@pytest.mark.parametrize(
    "reading, fields",
    [
        (FieldReading("level", sigma=LEVEL_SIGMA), ("level", "inflow")),
        (
            FieldReading("level", sigma=LEVEL_SIGMA, step=0.1),
            ("level", "inflow"),
        ),
        # A gauge 2 steps late and lagging 1.5 steps: the filter carries what
        # the instrument still has to report, the state's sensor history.
        (
            FieldReading("level", sigma=LEVEL_SIGMA, dead_time=2.0, lag=1.5),
            ("level", "inflow", "sensor_history"),
        ),
    ],
    ids=["plain", "rounded", "late"],
)
def test_the_error_is_consistent_with_the_stated_covariance(reading, fields):
    env = Tank(readings=(reading,))
    params = env.default_params
    est = Estimator(env, params, fields, channels=(0,))
    lo, hi = _chi2_band(2)
    assert lo < _nees(est, env, params, ("level", "inflow")) < hi


def test_a_solve_with_no_jax_derivative_is_filtered_by_secants():
    """The second reading is of a field solved by bisection, whose JAX
    derivative is zero, as pH's is. Read alone, through secants across the
    belief's spread, it still gives a consistent filter."""
    env = Tank(readings=(FieldReading("root", sigma=LEVEL_SIGMA),))
    params = env.default_params
    est = Estimator(
        env, params, ("level", "inflow", "root"), channels=(1,), linearize="secant"
    )
    lo, hi = _chi2_band(2)
    assert lo < _nees(est, env, params, ("level", "inflow")) < hi


def test_secants_agree_with_jax_derivatives_on_a_linear_model(walking):
    """Central differences are exact on a linear model, so both linearisations
    give the same filter."""
    env, params, est = walking
    by_secant = Estimator(
        env, params, ("level", "inflow"), channels=(0,), linearize="secant"
    )
    _, _, a = jax.jit(_episode(est, env, params, 40, action=-0.2))(5)
    _, _, b = jax.jit(_episode(by_secant, env, params, 40, action=-0.2))(5)
    np.testing.assert_allclose(a.cov, b.cov, rtol=1e-3, atol=1e-8)
    np.testing.assert_allclose(a.state.inflow, b.state.inflow, atol=1e-5)


def test_a_field_no_observation_sees_keeps_its_prior():
    """The spare field enters neither the dynamics nor a reading, so the filter
    learns nothing of it: two episodes whose true spares differ end with the
    same estimate, the prior's mean, and the prior's spread."""
    env = Tank()
    params = env.default_params
    est = Estimator(env, params, ("level", "inflow", "spare"), channels=(0,))
    spare = est.fields.index("spare")
    _, states, beliefs = jax.jit(jax.vmap(_episode(est, env, params, 30)))(
        jnp.arange(2)
    )
    assert abs(float(states.spare[0, -1] - states.spare[1, -1])) > 0.1
    np.testing.assert_allclose(
        beliefs.state.spare[:, -1], float(est.prior.state.spare), atol=1e-6
    )
    np.testing.assert_allclose(
        beliefs.cov[:, -1, spare, spare], float(est.prior.cov[spare, spare]), rtol=1e-4
    )


def test_a_model_that_still_draws_noise_is_refused():
    """Unzeroed, the inflow's walk would be predicted along one fixed noise
    path, the same in every episode: a bias common to them all."""
    env = Tank()
    with pytest.raises(ValueError, match="noise_fields"):
        _Estimator(env, env.default_params, ("level", "inflow"), channels=(0,))


def test_it_is_fed_observations_and_actions_only():
    """By its interface: nothing takes a state, so nothing can read one."""
    assert list(inspect.signature(_Estimator.reset).parameters) == ["self", "obs"]
    assert list(inspect.signature(_Estimator.update).parameters) == [
        "self",
        "belief",
        "action",
        "obs",
    ]
    assert Belief._fields == ("state", "cov")


def test_a_scan_under_jit_is_the_python_loop(walking):
    env, params, est = walking
    states, beliefs = _run(est, env, params, seed=2, steps=25, action=0.1)
    _, state, belief = jax.jit(_episode(est, env, params, 25, action=0.1))(2)
    np.testing.assert_allclose(state.level[-1], states[-1].level, rtol=1e-6)
    np.testing.assert_allclose(
        belief.state.inflow[-1], beliefs[-1].state.inflow, rtol=1e-4
    )
    np.testing.assert_allclose(belief.cov[-1], beliefs[-1].cov, rtol=1e-4)


def test_a_vmapped_batch_is_each_episode_alone(walking):
    env, params, est = walking
    run = jax.jit(_episode(est, env, params, 25))
    _, _, batch = jax.jit(jax.vmap(_episode(est, env, params, 25)))(jnp.arange(3))
    for i in range(3):
        _, _, alone = run(i)
        np.testing.assert_allclose(
            batch.state.inflow[i], alone.state.inflow, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(batch.cov[i], alone.cov, rtol=1e-5)


@pytest.mark.medium
def test_on_ph_the_hidden_buffer_flow_converges_within_its_stated_spread():
    """The real pH plant with its acid and flow schedules held still: the buffer
    flow ``q2``, an OU process no reading shows, is then its one hidden
    disturbance. Read through both probes, late and lagged as they are, by
    secants, since the pH solve is a bisection: after an hour of a held valve,
    the error on ``q2`` over 32 episodes is consistent with the stated spread,
    and that spread is under 0.6 of the prior's (0.45 measured, flanders,
    2026-11-28)."""
    import target_gym as tg

    env, params = tg.make("ph_neutralization", acid_upset_size=0.0, flow_swing=0.0)
    fields = ("Wa", "Wb", "Wa_s1", "Wb_s1", "q2", "pH", "pH_s1", "sensor_history")
    est = _Estimator(env, params, fields, channels=(0, 4), linearize="secant")
    q2 = est.fields.index("q2")  # the scalars come first, so it is q2's place

    def run(seed):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset_env(key, params)
        hold = env.initial_action(state, params)  # the valves where the reset runs

        def step(carry, _):
            state, belief = carry
            obs, state, *_ = env.step_env(key, state, hold, params)
            return (state, est.update(belief, hold, obs)), None

        (state, belief), _ = jax.lax.scan(
            step, (state, est.reset(obs)), None, length=720
        )
        return state.q2 - belief.state.q2, belief.cov[q2, q2]

    err, var = jax.jit(jax.vmap(run))(jnp.arange(32))
    lo, hi = _chi2_band(1, runs=32)
    assert lo < float(np.mean(np.asarray(err) ** 2 / np.asarray(var))) < hi
    assert float(jnp.sqrt(var[0])) < 0.6 * float(jnp.sqrt(est.prior.cov[q2, q2]))
