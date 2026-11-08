"""Attempt 3's new physics and its environment, from the equations, written before the code.

Contract, from conservation, the sources and the owner's score, not from any code:
- the reversal, averaged: each checker node takes half its flue duty and half its air duty,
  the first-order average of the half-cycle equations. Air from ambient T_a through N nodes
  of effectiveness e, all at T_n, leaves at T_n - (1 - e)^N (T_n - T_a);
- the firing gap, averaged: 1 - (dead / period) (1 - floor), the time mean of the clock;
- the canal reads the leaving glass, weighted by its flow, and the glass leaves evenly
  over depth;
- what the current carries along the tank keeps its bounds: a share stays in [0, 1] past
  a sharp front, as a total-variation-diminishing scheme guarantees (Sweby);
- the refiner's cooling is a flow out of the books, and colder glass reaches the canal;
- fining: glass releases the share of its sulfate that its temperature allows over the
  sourced band, 1420 to 1500 C, keeps what it released, and carries it with the flow;
  batch glass arrives with none. The canal reads the leaving glass's share by its flow;
- reward in money: fuel at the gas price; the glass charged while the canal is under its
  target or under-fined is lost at its price, on the pull actually charged. The silica
  crown's 1620 C (degradation) is the plant's own override, not a price (owner,
  2026-11-08): whatever the valve asks, the hottest face stays under it;
- the agent reads what operators read (REALITY.md): the fining share only at the cold
  end, a transit later and with noise; the cullet ratio the batch plant holds; the batch
  line on its camera's cadence; and the pull plan as far ahead as its notice;
- the episode is established operation (rule 17): the walks start where the seed puts
  them, at their stationary spread, and planned pull changes follow their plan;
- the observation carries readings only: the hidden walks don't enter it;
- each walk is stationary with its stated spread and correlation time.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace import physical_env as E
from target_gym.glass_furnace.env import N_REGEN_NODES, firing_fraction


@pytest.fixture(scope="module")
def plant():
    g, p = P.Grid(), P.PhysicalParams()
    return g, p, P.initial_fields(g, p)


def test_uniform_checkers_preheat_by_the_node_cascade(plant):
    _, p, _ = plant
    T_n, m_air, m_gas = 1200.0, 8.0, 8.5
    nodes = jnp.full((N_REGEN_NODES,), T_n)
    Q, air, flue = P.averaged_stack(nodes, T_n, m_air, m_gas, p)
    e, Ta = p.eps_regen_node, p.T_ambient
    expected = T_n - (1.0 - e) ** N_REGEN_NODES * (T_n - Ta)
    assert float(air) == pytest.approx(expected, rel=1e-5)
    assert float(flue) == pytest.approx(
        T_n, rel=1e-6
    )  # flue at the nodes gives nothing
    # Half the cycle on air: the nodes give half the air's heat.
    assert float(jnp.sum(Q)) == pytest.approx(
        -0.5 * m_air * p.c_p_air * (expected - Ta), rel=1e-5
    )


def test_averaged_checkers_keep_their_books(plant):
    _, p, _ = plant
    nodes = jnp.array([1250.0, 1000.0, 800.0, 500.0])
    m_air, m_gas, T_in = 8.0, 8.5, 1500.0
    Q, air, flue = P.averaged_stack(nodes, T_in, m_air, m_gas, p)
    given = 0.5 * m_gas * p.c_p_gas * (T_in - flue)
    taken = 0.5 * m_air * p.c_p_air * (air - p.T_ambient)
    assert float(jnp.sum(Q)) == pytest.approx(float(given - taken), rel=1e-5)


def test_mean_firing_is_the_clocks_average(plant):
    _, p, _ = plant
    hand = 1.0 - p.reversal_dead_time / p.reversal_period * (
        1 - p.reversal_firing_floor
    )
    assert float(P.mean_firing(p)) == pytest.approx(hand, rel=1e-6)
    # The shipped clock, sampled each second over a whole double cycle.
    fine = p.replace(delta_t=1.0)
    n = int(2 * p.reversal_period)
    clock = np.mean(np.asarray(firing_fraction(jnp.arange(n), fine)))
    assert clock == pytest.approx(hand, rel=1e-4)


def test_canal_reads_the_leaving_glass_by_its_flow(plant):
    g, p, s = plant
    a = P.layout(g)
    profile = jnp.linspace(1500.0, 1300.0, g.n_depth)
    share = jnp.linspace(1.0, 0.2, g.n_depth)
    s = s.replace(T=s.T.at[:, -1].set(profile), fining=s.fining.at[:, -1].set(share))
    weigh = lambda x: float(jnp.sum(x * a.dz) / jnp.sum(a.dz))
    read = P.sensors(s, g, p)
    assert float(read["canal"]) == pytest.approx(weigh(profile), rel=1e-6)
    assert float(read["fining"]) == pytest.approx(weigh(share), rel=1e-6)


def test_a_carried_share_stays_bounded_past_a_sharp_front(plant):
    g, p, s = plant
    a = P.layout(g)
    front = (jnp.arange(a.nc) < a.nm // 2).astype(float)
    s = s.replace(fining=jnp.broadcast_to(front, s.T.shape))

    def f(s, _):
        s, _ = P.plant_step(s, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
        return s, (s.fining.min(), s.fining.max())

    lo, hi = jax.jit(lambda: jax.lax.scan(f, s, None, length=200)[1])()
    assert float(lo.min()) >= -1e-6 and float(hi.max()) <= 1.0 + 1e-6


def _run(s, g, p, steps, cooling=0.0):
    def f(s, _):
        new, b = P.plant_step(s, 0.46, 5.79, 0.25, 0.2, cooling, g, p)
        return new, (b["in"] - b["out"], P.sensors(new, g, p)["canal"])

    return jax.lax.scan(f, s, None, length=steps)


def test_a_step_keeps_its_books_with_cooling(plant):
    # Without cooling, test_glass_physical.py's books test covers it.
    g, p, start = plant
    cooling = 2.0e6
    day = int(86400 / p.delta_t)
    # Past the start, where a foam's top carried one step sums out. 0.5 % of the firing.
    s = jax.jit(lambda: _run(start, g, p, day, cooling)[0])()
    new, (flows, _) = jax.jit(lambda: _run(s, g, p, 100, cooling))()
    change = float(P.stored_energy(new, g, p) - P.stored_energy(s, g, p))
    assert change == pytest.approx(
        p.delta_t * float(jnp.sum(flows)),
        abs=0.005 * 100 * p.delta_t * 0.46 * p.LHV,
    )


def test_cooling_the_refiner_cools_the_canal(plant):
    g, p, start = plant
    hours = int(6 * 3600 / p.delta_t)
    warm = jax.jit(lambda: _run(start, g, p, hours, 0.0)[1][1])()
    cool = jax.jit(lambda: _run(start, g, p, hours, 2.0e6)[1][1])()
    assert float(cool[-1]) < float(warm[-1]) - 1.0


def test_glass_past_the_band_has_finished_fining(plant):
    g, p, s = plant
    hot = s.replace(T=jnp.full_like(s.T, p.T_fine_high + 100.0))
    new, _ = P.plant_step(hot, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    assert float(P.sensors(new, g, p)["fining"]) == pytest.approx(1.0, abs=1e-3)


def test_glass_under_the_band_brings_no_fining_to_the_canal(plant):
    g, p, s = plant
    cold = s.replace(
        T=jnp.full_like(s.T, p.T_fine_low - 100.0), fining=jnp.zeros_like(s.T)
    )
    new, _ = P.plant_step(cold, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    assert float(P.sensors(new, g, p)["fining"]) == 0.0


def test_reward_prices_fuel_and_glass_lost_on_the_charged_pull():
    ep = E.EnvParams()
    dt = ep.plant.delta_t
    fuel_cost = 0.45 * dt * ep.plant.LHV / 3.6e9 * ep.gas_price  # EUR/MWh on energy
    pull = 1.1 * ep.pull  # a pull change in progress
    glass = pull * dt / 1000.0 * ep.lost_glass_price  # EUR/t on the tonnes charged
    good = dict(canal=ep.canal_target + 0.5, fining=ep.fining_pass + 0.01, pull=pull)
    ok = E.reward_terms(0.45, **good, ep=ep)
    assert set(ok) == {"fuel", "lost_glass"}
    assert float(ok["fuel"]) == pytest.approx(fuel_cost, rel=1e-6)
    assert float(ok["lost_glass"]) == 0.0
    for bad in (dict(canal=ep.canal_target - 0.5), dict(fining=ep.fining_pass - 0.01)):
        cost = E.reward_terms(0.45, **{**good, **bad}, ep=ep)
        assert float(cost["lost_glass"]) == pytest.approx(glass, rel=1e-6)


def _quiet(ep):
    return ep.replace(walk_sd=(0.0,) * len(E.WALKS))


def test_crown_override_holds_the_hottest_face_under_1620_c(plant):
    g, p, s = plant
    ep = _quiet(E.EnvParams())
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)

    def f(st, k):
        _, st, _, _ = E.step(k, st, ep.fuel_max, ep)
        return st, P.sensors(st.fields, g, p)["hot"]

    keys = jax.random.split(jax.random.PRNGKey(1), int(2 * 86400 / p.delta_t))
    hot = np.asarray(jax.jit(lambda: jax.lax.scan(f, state, keys)[1])())
    later = hot[int(6 * 3600 / p.delta_t) :]  # past the start's own transient
    assert later.max() <= ep.crown_high


def test_override_leaves_the_valve_alone_under_the_limit(plant):
    _, _, s = plant
    ep = _quiet(E.EnvParams())
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)
    # A cold crown no longer calls for more fuel: the low override is gone.
    for face in (1773.15, 1600.0):  # 1500 C, and 1327 C, under silica's 1400 C
        cold = state.fields.replace(T_roof=state.fields.T_roof.at[0].set(face))
        _, new, _, _ = E.step(
            jax.random.PRNGKey(2), state.replace(fields=cold), 0.30, ep
        )
        assert float(new.fuel) == pytest.approx(0.30, rel=1e-6)


def test_observation_carries_readings_not_the_hidden_walks(plant):
    _, _, s = plant
    ep = E.EnvParams()
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)
    other = state.replace(walk=state.walk + 0.03)
    np.testing.assert_array_equal(E.observe(state, ep), E.observe(other, ep))
    assert len(E.OBS_NAMES) == E.observe(state, ep).shape[0]


def test_walks_start_where_the_seed_puts_them(plant):
    _, _, s = plant
    ep = E.EnvParams()
    first = lambda k: E.reset(jax.random.PRNGKey(k), s, ep)[1].walk
    np.testing.assert_array_equal(first(3), first(3))
    assert not np.allclose(first(3), first(4))
    # Over many seeds, each walk starts at its stationary spread; 2000 draws give the sd
    # to about 2 %, so 10 % is the bound.
    starts = np.asarray(jax.vmap(lambda k: E.reset(k, s, ep)[1].walk)(
        jax.random.split(jax.random.PRNGKey(5), 2000)))  # fmt: skip
    np.testing.assert_allclose(starts.std(0), ep.walk_sd, rtol=0.1)


def _trace(state, ep, fuel, steps, key=1):
    g, p = ep.grid, ep.plant

    def f(st, k):
        obs, st, _, _ = E.step(k, st, fuel, ep)
        return st, (obs, P.sensors(st.fields, g, p)["fining"], st.pull, st.time)

    keys = jax.random.split(jax.random.PRNGKey(key), steps)
    return [np.asarray(x) for x in jax.jit(lambda: jax.lax.scan(f, state, keys)[1])()]


def test_quality_is_read_a_transit_later(plant):
    _, _, s = plant
    ep = _quiet(E.EnvParams()).replace(quality_noise=0.0)
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)
    d = int(round(ep.quality_delay_h * 3600 / ep.plant.delta_t))
    obs, share, _, _ = _trace(state, ep, 0.47, 3 * d + 10)
    q = E.OBS_NAMES.index("fining share read at the cold end")
    np.testing.assert_allclose(obs[d:, q], share[:-d], rtol=1e-6)


def test_a_planned_pull_change_follows_its_plan_and_shows_its_notice(plant):
    _, _, s = plant
    ep = _quiet(E.EnvParams()).replace(pull_change=0.1, pull_change_window_h=(6.0, 6.0))
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)
    H = 3600 / ep.plant.delta_t
    steps = int((6.0 + ep.pull_ramp_h + 2.0) * H)
    obs, _, pull, time = _trace(state, ep, 0.47, steps)
    hours = time * ep.plant.delta_t / 3600
    plan = obs[:, E.OBS_NAMES.index("pull planned, kg/s")]
    assert np.allclose(pull[hours < 6.0], ep.pull)
    assert np.allclose(pull[hours > 6.0 + ep.pull_ramp_h], 1.1 * ep.pull)
    early = hours < 6.0 - ep.pull_notice_h - 0.1
    assert np.allclose(plan[early], ep.pull)
    assert np.allclose(
        plan[hours >= 6.0 - ep.pull_notice_h + ep.pull_ramp_h], 1.1 * ep.pull
    )


def test_batch_line_is_read_on_its_cadence(plant):
    _, _, s = plant
    ep = _quiet(E.EnvParams())
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)
    every = int(round(ep.batch_line_every_h * 3600 / ep.plant.delta_t))
    obs, _, _, time = _trace(state, ep, 0.40, 4 * every)
    line = obs[:, E.OBS_NAMES.index("batch line, m")]
    blocks = line.reshape(4, every)
    assert np.all(blocks == blocks[:, :1])  # held between readings
    assert not np.all(blocks[:, 0] == blocks[0, 0])  # and read anew at each


def test_cullet_ratio_is_read_as_charged(plant):
    _, _, s = plant
    ep = E.EnvParams()
    _, state = E.reset(jax.random.PRNGKey(0), s, ep)
    _, new, _, _ = E.step(jax.random.PRNGKey(1), state, 0.47, ep)
    c = E.observe(new, ep)[E.OBS_NAMES.index("cullet ratio")]
    assert float(c) == pytest.approx(
        ep.cullet + float(new.walk[E.WALKS.index("cullet")])
    )


def test_walks_have_their_stated_spread_and_memory():
    ep = E.EnvParams()
    steps = int(2000 * 86400 / ep.plant.delta_t)
    w = np.asarray(E.walks(jax.random.PRNGKey(3), steps, ep))
    # Each column: sd as stated, and correlation e^-1 one correlation time apart. About
    # 1000 independent stretches, so the sd is good to about 3 %; 10 % is the bound.
    for i, (sd, hours) in enumerate(zip(ep.walk_sd, ep.walk_hours)):
        lag = int(hours * 3600 / ep.plant.delta_t)
        x = w[:, i]
        assert x.std() == pytest.approx(sd, rel=0.1)
        r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        assert r == pytest.approx(np.exp(-1.0), abs=0.06)
