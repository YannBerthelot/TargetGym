"""``NewtonMPC``, the oracles' planner, on toy plants whose optimum is known.

The contract, from the planner's docstring and the oracles it serves:

- It minimises the stage cost summed over the plan, the environment's own
  step reward in floor units by default, weighted per step of the episode
  (``weights``, read from the step the plan starts at). A zero weight past
  the episode's end makes the plan shrink to the episode.
- Actions stay in the box ``[-1, 1]``.
- It re-solves every ``resolve_every`` steps and follows its plan between.
- A new episode (the clock going back) starts cold, as a fresh planner does.
- Each solve starts from the better, under its own objective, of the
  previous plan and the guide's rollout, so a planner given a guide is never
  worse than the guide under its model.
- The solve compiles once: a second planner of the same configuration reuses
  the first one's compilation.

The toy is an integrator, ``x' = x + u / 2`` per step, started at 0 with a
target of 1: two full-scale steps reach the target exactly, so a window that
opens at step 3 costs nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from target_gym.experts.mpc import NewtonMPC


@struct.dataclass
class ToyParams:
    max_steps_in_episode: int = 6
    e_floor: float = 0.01
    a: float = 1.0  # x' = a x + gain u
    gain: float = 0.5
    noise: float = 0.0


@struct.dataclass
class ToyState:
    x: float
    target: float
    time: int


class Toy:
    def step_env(self, key, state, action, params):
        u = jnp.clip(jnp.reshape(action, ()), -1.0, 1.0)
        xi = jax.random.normal(jax.random.fold_in(key, state.time))
        x = params.a * state.x + params.gain * u + params.noise * xi
        s = state.replace(x=x, time=state.time + 1)
        reward = -(((s.target - s.x) / params.e_floor) ** 2)
        return self.get_obs(s, params), s, reward, jnp.zeros((), bool), {}

    def get_obs(self, state, params):
        return jnp.stack([state.x, state.target])


ENV = Toy()


def _state(x=0.0, target=1.0, time=0):
    return ToyState(x=jnp.float32(x), target=jnp.float32(target), time=jnp.int32(time))


def _window(params, window_start=3, pre_weight=1e-3, horizon=None):
    """Weights per step of the episode: ``pre_weight`` before the window, 1
    in it, 0 past the episode's end, long enough for any plan start."""
    steps = int(params.max_steps_in_episode)
    horizon = steps if horizon is None else horizon
    w = np.zeros(steps + horizon)
    w[:steps] = 1.0
    w[:window_start] = pre_weight
    return w


def _run(planner, params, state=None, steps=None, key=jax.random.PRNGKey(0)):
    state = _state() if state is None else state
    xs, us = [], []
    for _ in range(int(params.max_steps_in_episode) if steps is None else steps):
        u = planner.step(None, state)
        us.append(float(np.asarray(u).reshape(())))
        _, state, *_ = ENV.step_env(key, state, jnp.asarray(u), params)
        xs.append(float(state.x))
    return np.array(xs), np.array(us)


def _shrinking(params, **kw):
    return NewtonMPC(
        ENV,
        params,
        horizon=int(params.max_steps_in_episode),
        weights=_window(params),
        **kw,
    )


def test_it_reaches_the_target_in_two_full_scale_steps():
    params = ToyParams()
    xs, us = _run(_shrinking(params), params)
    np.testing.assert_allclose(us[:2], 1.0, atol=1e-6)
    np.testing.assert_allclose(xs[1:], 1.0, atol=1e-5)  # the window and step 2


def test_its_actions_stay_in_the_box():
    """A target out of reach: full scale every step, and never past it."""
    params = ToyParams()
    xs, us = _run(_shrinking(params), params, state=_state(target=10.0))
    assert np.all(us <= 1.0) and np.all(us >= -1.0)
    np.testing.assert_allclose(us, 1.0, atol=1e-6)
    np.testing.assert_allclose(xs[-1], 3.0, atol=1e-5)


def test_it_resolves_every_period_and_follows_the_plan_between():
    params = ToyParams()
    planner = _shrinking(params, resolve_every=4)
    xs, _ = _run(planner, params)
    np.testing.assert_allclose(xs[1:], 1.0, atol=1e-5)
    assert planner.solver_report()["solver_calls"] == 2  # steps 0 and 4


def test_a_new_episode_without_a_reset_plays_as_a_fresh_planner():
    """The clock going back to 0 drops the old plan: the second episode's
    actions are a fresh planner's, from the same state."""
    params = ToyParams()
    planner = _shrinking(params)
    _run(planner, params)
    start = _state(x=0.3, target=-0.4)
    _, again = _run(planner, params, state=start)
    _, fresh = _run(_shrinking(params), params, state=start)
    np.testing.assert_allclose(again, fresh, atol=1e-6)


def test_reset_forgets_the_plan():
    params = ToyParams()
    planner = _shrinking(params)
    _run(planner, params, steps=3)
    planner.reset()
    start = _state(x=-0.2, target=0.6, time=3)
    u_after_reset = planner.step(None, start)
    u_fresh = _shrinking(params).step(None, start)
    assert u_after_reset == pytest.approx(u_fresh, abs=1e-6)


class _Proportional:
    """A guide as the experts are, ``step(params, state, obs) -> (action,
    state)``: ``u = clip(gain (target - x))`` on ``obs = [x, target]``."""

    def __init__(self, gain):
        self.params = jnp.float32(gain)
        self.init_state = jnp.zeros(())

    @staticmethod
    def step(gain, state, obs, xp=jnp):
        return xp.reshape(xp.clip(gain * (obs[1] - obs[0]), -1.0, 1.0), (1,)), state


def test_a_guide_better_than_the_plan_is_where_the_solve_starts():
    """With no iterations the plan is where the solve started. From x = 0 the
    zero plan never moves, and costs 3 window steps at |e| = 1. Gain 2 is
    deadbeat on this plant (full scale twice, then hold at the target), so
    its rollout costs only the approach and is taken: its first action, 1."""
    params = ToyParams()
    guided = _shrinking(params, n_iter=0, n_iter_cold=0, guide=_Proportional(2.0))
    assert guided.step(None, _state()) == pytest.approx(1.0)


def test_a_guide_worse_than_the_plan_is_ignored():
    """A negative gain drives x away from the target, which costs more than
    staying at 0: the zero plan is kept, and its first action is 0."""
    params = ToyParams()
    guided = _shrinking(params, n_iter=0, n_iter_cold=0, guide=_Proportional(-2.0))
    assert guided.step(None, _state()) == pytest.approx(0.0)


def test_it_holds_an_unstable_plant_at_its_target():
    """x' = 1.3 x + u with process noise the planner does not see (its params
    are noise-free, as ``plan_params`` makes them). Deadbeat control leaves
    exactly the noise, mean |e| = sigma sqrt(2 / pi) = 0.8 sigma; 2 sigma
    allows for the 150-step sample and the bound."""
    sigma = 0.01
    plant = ToyParams(max_steps_in_episode=200, a=1.3, gain=1.0, noise=sigma)
    planner = NewtonMPC(ENV, plant.replace(noise=0.0), horizon=10)
    xs, _ = _run(planner, plant, state=_state(x=0.5, target=0.2))
    assert np.mean(np.abs(xs[50:] - 0.2)) < 2 * sigma


_TRACES = {"n": 0}


def _counting_stage(env, state, action, params):
    _TRACES["n"] += 1  # a Python side effect: runs only while tracing
    _, new, reward, _, _ = env.step_env(jax.random.PRNGKey(0), state, action, params)
    return new, -reward


def test_a_second_planner_reuses_the_first_ones_compilation():
    params = ToyParams()
    first = _shrinking(params, stage_fn=_counting_stage)
    first.step(None, _state())
    traced = _TRACES["n"]
    assert traced > 0
    second = _shrinking(params, stage_fn=_counting_stage)
    second.step(None, _state(x=0.2))
    assert _TRACES["n"] == traced
