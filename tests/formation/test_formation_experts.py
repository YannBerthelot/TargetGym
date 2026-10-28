"""The formation's baselines.

The PID team flies a steady formation lead in the lead seat and the patrol's
expert, the formation hold, in each wingman seat; how closely it holds is in
tests/patrol/test_formation_hold.py. The oracle
flies the patrol's lead autopilot in the lead seat and the patrol's
twin-autopilot law in each wingman seat, with one residual planner over the
whole formation. The laws are checked here cheaply; the planner and the
closed-loop runs are in the slow job.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.experts.pid import make_plane3d_heading_pid, plane3d_heading_pid_step
from target_gym.formation.env import slot_errors
from target_gym.formation.experts import make_formation_pid_controller
from target_gym.patrol.env import PatrolState
from target_gym.patrol.experts import (
    TWIN_GAINS,
    make_patrol_pid_controller,
    twin_law_init,
    twin_law_step,
)

#: Held over the scored second half on seed 1 (scripts/measure_hold.py): the
#: lead 0.80 m off its altitude, the wingmen 0.23 and 0.27 m off their slots.
#: The bounds are three altimeter floors and a third of the slot floor.
ALT_HOLD_BOUND = 3.0
SLOT_HOLD_BOUND = 1.0


def _reset(seed=0):
    spec = registry.get("formation")
    env = spec.make_env()
    params = spec.make_test_params()
    obs, state = env.reset_env(jax.random.PRNGKey(seed), params)
    return spec, env, params, obs, state


def _seat(env, obs, i):
    start = env.lead_obs_dim + 26 * i
    return obs[start : start + 26]


def test_every_wingman_seat_flies_the_patrol_expert():
    """Each wingman seat is the patrol's expert on that seat's observation,
    the patrol's full observation."""
    _, env, _, obs, _ = _reset()
    team = make_formation_pid_controller()
    action, _ = jax.jit(team.act)(team.init_state, obs)

    wing = make_patrol_pid_controller()
    for i in range(2):
        seat, _ = jax.jit(wing.act)(wing.init_state, _seat(env, obs, i))
        # Two compiled graphs order float32 sums differently; the pitch loop's
        # gains (Kp_theta 8 on a pitch error built from energy rates) carry
        # that rounding to about 1e-5 of the stick, so 1e-4 is float32
        # agreement, not a looser law.
        np.testing.assert_allclose(
            np.asarray(action[3 * (i + 1) : 3 * (i + 2)]), np.asarray(seat), atol=1e-4
        )


@pytest.mark.parametrize("k", [1, 4])
def test_the_pid_team_flies_any_formation(k):
    from target_gym import Formation

    env = Formation(num_wingmen=k)
    obs, _ = env.reset_env(jax.random.PRNGKey(0), env.default_params)
    team = make_formation_pid_controller(num_wingmen=k)
    action, _ = team.act(team.init_state, obs)
    assert action.shape == (3 * (k + 1),)


@pytest.mark.slow
def test_the_pid_team_holds_the_formation():
    """A full test episode on two seeds: no trip, and every wingman ends
    within patrol's 60 m formation tolerance of its slot."""
    spec, env, params, _, _ = _reset()
    team = spec.make_pid_controller()
    act = jax.jit(team.act)
    step = jax.jit(env.step_env)
    for seed in range(2):
        obs, state = env.reset_env(jax.random.PRNGKey(seed), params)
        carry = team.init_state
        for _ in range(int(params.max_steps_in_episode)):
            a, carry = act(carry, obs)
            obs, state, _, _, info = step(jax.random.PRNGKey(seed), state, a, params)
            assert not bool(info["tripped"])
        assert float(jnp.max(slot_errors(state))) < params.slot_tolerance


def _patrol_view(state, i, lead_pid):
    return PatrolState(
        follower=jax.tree.map(lambda x: x[i], state.wingmen),
        lead=state.lead,
        lead_pid=lead_pid,
        slot_back=state.slot_back[i],
        slot_right=state.slot_right[i],
        slot_up=state.slot_up[i],
        lead_turn_rate=state.lead_turn_rate,
        lead_turn_schedule=state.lead_turn_schedule,
        time=state.time,
        gust_x=state.gust_x,
        gust_y=state.gust_y,
        gust_z=state.gust_z,
    )


def test_without_a_plan_the_oracle_is_the_seat_laws():
    """No residual: the lead flies the patrol's lead autopilot on the heading
    the route commands, and each wingman the patrol's twin law on its patrol
    view of the state."""
    spec, env, params, obs, state = _reset(seed=2)
    oracle = spec.make_mpc(env, plan_params(spec, params), horizon=0)
    oracle.reset()
    action = np.asarray(oracle.step(obs, state))

    lead_params, lead_state = make_plane3d_heading_pid()
    lead, _ = plane3d_heading_pid_step(lead_params, lead_state, obs[:15])
    np.testing.assert_allclose(action[:3], np.asarray(lead), atol=1e-6)
    for i in range(2):
        view = _patrol_view(state, i, lead_state)
        seat, _ = twin_law_step(lead_params, TWIN_GAINS, twin_law_init(), view)
        np.testing.assert_allclose(
            action[3 * (i + 1) : 3 * (i + 2)], np.asarray(seat), atol=1e-5
        )


@pytest.mark.slow
def test_the_oracle_holds_the_whole_formation():
    """A full test episode: no trip, and over the scored second half the lead
    holds its altitude and every wingman its slot to within a few floors,
    where the PID team is tens of metres off. The autopilot alone left the
    lead about 140 m off its altitude; the planner is what removes that."""
    spec, env, params, _, _ = _reset()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    step = jax.jit(env.step_env)
    seed = 1  # the slower of the two hold seeds to recover from its climb
    obs, state = env.reset_env(jax.random.PRNGKey(seed), params)
    oracle.reset()
    alt, slots = [], []
    for _ in range(int(params.max_steps_in_episode)):
        a = oracle.step(obs, state)
        obs, state, _, _, info = step(jax.random.PRNGKey(seed), state, a, params)
        assert not bool(info["tripped"])
        alt.append(abs(float(state.lead.target_altitude - state.lead.z)))
        slots.append(np.asarray(slot_errors(state)))
    half = len(alt) // 2
    assert np.mean(alt[half:]) < ALT_HOLD_BOUND
    assert np.max(np.mean(slots[half:], axis=0)) < SLOT_HOLD_BOUND
