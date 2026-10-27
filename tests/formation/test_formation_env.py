"""The formation task: a lead and its wingmen, every aircraft a learner.

Each aircraft is the patrol's, so most of what is checked here is the
composition: the lead's route is exogenous, the slots form a V, the team cost
is the lead's cost plus each wingman's, any aircraft's trip restarts the whole
formation, and with the patrol's autopilot in the lead seat a formation step
is a patrol step for every wingman.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import Formation, FormationParams
from target_gym.experts.pid import make_plane3d_heading_pid, plane3d_heading_pid_step
from target_gym.formation.env import (
    MAX_WINGMEN,
    aircraft_costs,
    compute_reward,
    compute_reward_terms,
    compute_reward_v1,
    formation_slots,
    min_separation,
    slot_errors,
)
from target_gym.patrol.env import (
    N_LEAD_LEGS,
    PatrolState,
    compute_reward_terms_patrol,
    get_obs_full,
    lead_leg,
)
from target_gym.patrol.env_jax import PlanePatrol
from target_gym.plane3d.env import compute_reward_terms_heading, wrap_angle
from target_gym.reward import trip_cost

KEY = jax.random.PRNGKey(0)


def _reset(k=2, seed=0, **overrides):
    env = Formation(num_wingmen=k)
    params = env.default_params.replace(max_steps_in_episode=200, **overrides)
    obs, state = env.reset_env(jax.random.PRNGKey(seed), params)
    return env, params, obs, state


def _move_wingman(state, i, **fields):
    w = jax.tree.map(lambda x: x[i], state.wingmen).replace(**fields)
    wingmen = jax.tree.map(lambda all_, one: all_.at[i].set(one), state.wingmen, w)
    return state.replace(wingmen=wingmen)


def _wingman(state, i):
    return jax.tree.map(lambda x: x[i], state.wingmen)


# ─── Construction ───────────────────────────────────────


class TestConstruction:
    def test_default_is_a_lead_and_two_wingmen(self):
        env = Formation()
        assert env.num_wingmen == 2
        assert env.obs_shape == (74,)
        assert env.action_space().shape == (9,)

    @pytest.mark.parametrize("k", range(1, MAX_WINGMEN + 1))
    def test_layout_scales_with_the_wingmen(self, k):
        env = Formation(num_wingmen=k)
        assert env.lead_obs_dim == 16 + 3 * k
        assert env.obs_shape == (16 + 3 * k + 26 * k,)
        assert env.action_space().shape == (3 * (k + 1),)
        obs, _ = env.reset_env(KEY, env.default_params)
        assert obs.shape == env.obs_shape
        assert bool(jnp.all(jnp.isfinite(obs)))

    @pytest.mark.parametrize("k", [0, MAX_WINGMEN + 1])
    def test_rejects_a_formation_out_of_range(self, k):
        with pytest.raises(ValueError):
            Formation(num_wingmen=k)

    def test_tracked_values_are_the_lead_altitude_and_each_slot_error(self):
        env, _, obs, state = _reset(follower_spawn_noise=0.0)
        assert len(env.tracked_names) == 3
        value = np.asarray(obs)[list(env.obs_value_index)]
        target = np.asarray(obs)[list(env.obs_target_index)]
        assert value[0] == pytest.approx(float(state.lead.z))
        assert target[0] == pytest.approx(float(state.lead.target_altitude))
        np.testing.assert_allclose(value[1:], 0.0, atol=1e-2)
        np.testing.assert_allclose(target[1:], 0.0)


# ─── Slots ──────────────────────────────────────────────


class TestSlots:
    def test_slots_alternate_sides_and_grow_with_rank(self):
        back, right, up = (
            np.asarray(a) for a in formation_slots(200.0, 100.0, 20.0, 4)
        )
        assert list(np.sign(right)) == [1, -1, 1, -1]
        np.testing.assert_allclose(np.abs(right), [100, 100, 200, 200])
        np.testing.assert_allclose(back, [200, 200, 400, 400])
        np.testing.assert_allclose(up, [20, 20, 40, 40])

    def test_the_v_is_drawn_per_episode_within_the_ranges(self):
        p = FormationParams()
        draws = [_reset(seed=s)[3] for s in range(8)]
        backs = np.array([float(s.slot_back[0]) for s in draws])
        laterals = np.array([float(s.slot_right[0]) for s in draws])
        ups = np.array([float(s.slot_up[0]) for s in draws])
        assert np.all((backs >= p.slot_back_range[0]) & (backs <= p.slot_back_range[1]))
        assert np.all(
            (laterals >= p.slot_lateral_range[0])
            & (laterals <= p.slot_lateral_range[1])
        )
        assert np.all((ups >= p.slot_up_range[0]) & (ups <= p.slot_up_range[1]))
        assert np.ptp(backs) > 1.0 and np.ptp(laterals) > 1.0
        for s in draws:  # symmetric: the second wingman mirrors the first
            assert float(s.slot_right[1]) == pytest.approx(-float(s.slot_right[0]))

    def test_wingmen_spawn_in_their_slots(self):
        _, _, _, state = _reset(follower_spawn_noise=0.0)
        np.testing.assert_allclose(np.asarray(slot_errors(state)), 0.0, atol=1e-2)

    def test_the_formation_starts_clear_of_a_collision(self):
        env = Formation()
        keys = jax.random.split(KEY, 64)
        states = jax.vmap(lambda k: env.reset_env(k, env.default_params)[1])(keys)
        sep = jax.vmap(min_separation)(states)
        assert float(sep.min()) > FormationParams().min_separation


# ─── The lead's route ───────────────────────────────────


class TestRoute:
    def test_the_route_is_in_the_state(self):
        p = FormationParams()
        _, _, _, state = _reset()
        sched = np.asarray(state.lead_turn_schedule)
        assert sched.shape == (N_LEAD_LEGS,)
        assert np.all(np.abs(sched) <= p.lead_turn_rate_range[1])
        assert float(state.lead_turn_rate) == pytest.approx(float(sched[0]))

    @pytest.mark.parametrize("lead_action", [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    def test_the_reference_follows_the_route_whatever_the_lead_does(self, lead_action):
        env, params, _, state = _reset()
        action = jnp.concatenate([jnp.asarray(lead_action), jnp.zeros(6)])
        for _ in range(3):
            _, new, _, _, _ = env.step_env(KEY, state, action, params)
            expected = wrap_angle(state.lead.target_heading + state.lead_turn_rate)
            assert float(new.lead.target_heading) == pytest.approx(
                float(expected), abs=1e-6
            )
            leg = int(lead_leg(state.time + 1, params))
            assert float(new.lead_turn_rate) == pytest.approx(
                float(state.lead_turn_schedule[leg])
            )
            state = new


# ─── Patrol, seat for seat ──────────────────────────────


def _patrol_view(state, i, lead_pid):
    return PatrolState(
        follower=_wingman(state, i),
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


@pytest.mark.parametrize("k", [1, 2])
def test_with_the_autopilot_in_the_lead_seat_each_wingman_flies_a_patrol(k):
    """The lead flown by the patrol's own autopilot, on the heading it is
    commanded this step: every wingman then sees patrol's observation and
    moves exactly as patrol's follower does, so a policy trained on patrol
    can fly a wingman seat unchanged."""
    env, params, obs, state = _reset(k=k, turbulence_sigma=3.0)
    patrol = PlanePatrol()
    pid_params, pid = make_plane3d_heading_pid()
    actions = jax.random.uniform(jax.random.PRNGKey(7), (k, 3), minval=-1, maxval=1)
    for _ in range(3):
        lead_action, new_pid = plane3d_heading_pid_step(pid_params, pid, obs[:15])
        joint = jnp.concatenate([lead_action, actions.ravel()])
        obs2, new, _, _, _ = env.step_env(KEY, state, joint, params)
        for i in range(k):
            view = _patrol_view(state, i, pid)
            start = env.lead_obs_dim + 26 * i
            # Batched over the wingmen, float32 can round the last bit apart.
            np.testing.assert_allclose(
                np.asarray(obs[start : start + 26]),
                np.asarray(get_obs_full(view)),
                rtol=1e-5,
            )
            _, pnew, _, _, _ = patrol.step_env(KEY, view, actions[i], params)
            for a, b in zip(
                jax.tree.leaves((pnew.follower, pnew.lead)),
                jax.tree.leaves((_wingman(new, i), new.lead)),
            ):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)
        obs, state, pid = obs2, new, new_pid


# ─── Reward ─────────────────────────────────────────────


class TestReward:
    def test_the_team_cost_is_the_lead_cost_plus_each_wingman_cost(self):
        _, params, _, state = _reset()
        terms = compute_reward_terms(state, params)
        lead = compute_reward_terms_heading(state.lead, params)["tracking"]
        wingmen = sum(
            compute_reward_terms_patrol(_patrol_view(state, i, None), params)[
                "tracking"
            ]
            for i in range(2)
        )
        assert float(terms["tracking"]) == pytest.approx(
            float(lead + wingmen), rel=1e-6
        )
        assert float(terms["failure"]) == 0.0
        assert float(compute_reward(state, params)) == pytest.approx(
            -float(terms["tracking"]), rel=1e-6
        )

    def test_aircraft_costs_add_up_to_the_team_cost(self):
        _, params, _, state = _reset(seed=3)
        costs = aircraft_costs(state, params)
        assert np.asarray(costs["wingmen"]).shape == (2,)
        total = float(costs["lead"]) + float(np.sum(costs["wingmen"]))
        terms = compute_reward_terms(state, params)
        assert total == pytest.approx(float(terms["tracking"]), rel=1e-6)

    def test_the_floors_are_the_instruments(self):
        p = FormationParams()
        assert p.e_floor_altitude == 1.0  # altimeter
        assert p.e_floor_heading == pytest.approx(0.0087)  # AHRS
        assert p.e_floor_slot == 3.0  # relative GPS

    @pytest.mark.parametrize("k", range(1, MAX_WINGMEN + 1))
    def test_a_trip_costs_more_than_any_formation_can_track(self, k):
        p = FormationParams()
        lead = (p.max_alt / p.e_floor_altitude) ** 2 + (np.pi / p.e_floor_heading) ** 2
        wingman = (p.max_slot_error / p.e_floor_slot) ** 2 + (
            np.pi / p.e_floor_heading
        ) ** 2
        assert p.failure_cost > lead + k * wingman

    def test_version_1_is_the_old_team_reward(self):
        _, params, _, state = _reset(follower_spawn_noise=0.0)
        v1 = float(compute_reward_v1(state, params))
        assert 0.0 < v1 <= 1.0
        old = params.replace(reward_version=1)
        assert float(compute_reward(state, old)) == pytest.approx(v1)

    def test_version_1_charges_a_trip_its_downtime(self):
        """The formation restarts in its slots, so a 0 on the tripped step
        would make a trip nearly free."""
        _, params, _, state = _reset()
        lead = state.lead
        state = _move_wingman(state, 0, x=lead.x, y=lead.y, z=lead.z + 1.0)
        assert float(compute_reward_v1(state, params)) == -params.restart_steps


# ─── Trips ──────────────────────────────────────────────


class TestTrips:
    def test_wingmen_colliding_with_each_other_trip(self):
        env, params, _, state = _reset()
        w0 = _wingman(state, 0)
        state = _move_wingman(state, 1, x=w0.x, y=w0.y + 5.0, z=w0.z)
        assert bool(env.is_terminated(state, params))

    def test_a_wingman_colliding_with_the_lead_trips(self):
        env, params, _, state = _reset()
        lead = state.lead
        state = _move_wingman(state, 0, x=lead.x + 5.0, y=lead.y, z=lead.z)
        assert bool(env.is_terminated(state, params))

    def test_a_lost_wingman_trips(self):
        env, params, _, state = _reset()
        w1 = _wingman(state, 1)
        state = _move_wingman(state, 1, z=w1.z + 1600.0)
        assert bool(env.is_terminated(state, params))

    def test_any_aircraft_leaving_the_envelope_trips(self):
        env, params, _, state = _reset()
        assert not bool(env.is_terminated(state, params))
        assert bool(
            env.is_terminated(
                state.replace(lead=state.lead.replace(z=params.max_alt + 1.0)), params
            )
        )

    def test_a_trip_is_charged_once_and_the_whole_formation_restarts(self):
        env, params, _, state = _reset()
        lead = state.lead
        state = _move_wingman(state, 0, x=lead.x, y=lead.y, z=lead.z + 1.0)
        action = jnp.zeros(9)
        _, new, reward, terminated, info = env.step_env(KEY, state, action, params)
        assert bool(info["tripped"])
        assert not bool(terminated)
        assert float(reward) == pytest.approx(-float(trip_cost(params)), rel=1e-6)
        assert float(info["trip_cost"]) == pytest.approx(float(trip_cost(params)))
        assert int(new.time) == int(state.time) + 1
        assert float(min_separation(new)) > params.min_separation
        assert float(new.lead.x) == 0.0  # a fresh formation

    def test_episodic_use_ends_the_episode_instead(self):
        env, params, _, state = _reset(continuing=False)
        lead = state.lead
        state = _move_wingman(state, 0, x=lead.x, y=lead.y, z=lead.z + 1.0)
        _, _, _, terminated, info = env.step_env(KEY, state, jnp.zeros(9), params)
        assert bool(info["tripped"]) and bool(terminated)


# ─── Disturbance ────────────────────────────────────────


def test_one_gust_for_the_whole_formation_and_any_size():
    """The formation shares one air mass, the gust is drawn from the step
    clock (a constant rollout key does not replay one draw), and it reaches a
    formation of any size."""
    env = Formation(num_wingmen=MAX_WINGMEN)
    params = env.default_params.replace(turbulence_sigma=4.0)
    _, state = env.reset_env(KEY, params)
    zero = jnp.zeros(env.action_space().shape)

    def step(s, _):
        _, s, _, _, _ = env.step_env(KEY, s, zero, params)
        return s, s.gust_y

    _, g = jax.jit(lambda s: jax.lax.scan(step, s, None, length=150))(state)
    g = np.asarray(g)
    assert np.std(g) > 1.0
    assert len(np.unique(np.round(g, 6))) > 100


def test_every_aircraft_flies_through_one_batched_dynamics_call(monkeypatch):
    """The lead and its wingmen share one traced copy of the aircraft
    dynamics. A planner differentiates the step through every aircraft, and a
    second copy for the lead alone doubled the gradient program: the oracle's
    compile then took about 10 GB, enough to kill a CI runner."""
    import target_gym.formation.env as formation_env

    calls = []
    real = formation_env.compute_next_state_3d

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(formation_env, "compute_next_state_3d", spy)
    env = Formation()
    params = env.default_params
    _, state = env.reset_env(KEY, params)
    jax.make_jaxpr(lambda a: env.step_env(KEY, state, a, params))(
        jnp.zeros(env.action_space().shape)
    )
    assert len(calls) == 1
