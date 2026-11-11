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
- fining: a time-temperature integral (US6673730), not the peak. In the sourced band,
  1420 to 1500 C, glass fines at a rate: over a time dt its unfined share falls by
  exp(-dt f / fining_time), f the band's share at its temperature (0 at 1420 C, 1 at
  1500 C). Cooled glass keeps what it fined, since the gas it released is spent. The
  share is carried with the flow, and batch glass arrives with none. The canal reads
  the leaving glass's share by its flow;
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

from target_gym.glass_furnace import env_jax as E
from target_gym.glass_furnace import physical as P


@pytest.fixture(scope="module")
def plant():
    g, p = P.Grid(), P.PhysicalParams()
    return g, p, P.initial_fields(g, p)


def test_uniform_checkers_preheat_by_the_node_cascade(plant):
    _, p, _ = plant
    T_n, m_air, m_gas = 1200.0, 8.0, 8.5
    nodes = jnp.full((P.N_REGEN_NODES,), T_n)
    Q, air, flue = P.averaged_stack(nodes, T_n, m_air, m_gas, p)
    e, Ta = p.eps_regen_node, p.T_ambient
    expected = T_n - (1.0 - e) ** P.N_REGEN_NODES * (T_n - Ta)
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


def test_glass_past_the_band_fines_at_its_rate_not_at_once(plant):
    """One step over the band's top from no fining: 1 - exp(-dt / fining_time), 0.0328
    at 120 s on 1 h, not the whole share at once."""
    g, p, s = plant
    hot = s.replace(
        T=jnp.full_like(s.T, p.T_fine_high + 100.0), fining=jnp.zeros_like(s.T)
    )
    new, _ = P.plant_step(hot, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    expected = 1.0 - np.exp(-p.delta_t / p.fining_time)
    assert float(P.sensors(new, g, p)["fining"]) == pytest.approx(expected, rel=1e-3)


def _held(temperatures_c, p, start=0.0):
    X = jnp.asarray(start)
    for T in temperatures_c:
        X = P.fining_after(X, T + 273.15, p.delta_t, p)
    return float(X)


def test_fining_follows_the_time_in_the_band_not_the_peak(plant):
    """Two hours swung step by step between 1440 and 1480 C fine as much as two hours
    held at 1460 C: the band's share averages 0.5 either way, so both reach
    1 - exp(-0.5 * 2 h / fining_time)."""
    _, p, _ = plant
    n = int(round(2 * 3600.0 / p.delta_t))
    expected = 1.0 - np.exp(-0.5 * 2 * 3600.0 / p.fining_time)
    assert _held([1460.0] * n, p) == pytest.approx(expected, rel=1e-4)
    assert _held([1440.0, 1480.0] * (n // 2), p) == pytest.approx(expected, rel=1e-4)


def test_cooled_glass_keeps_the_fining_it_has_done(plant):
    _, p, _ = plant
    cold = float(p.T_fine_low) - 273.15 - 50.0
    assert _held([cold] * 30, p, start=0.6) == pytest.approx(0.6, abs=1e-6)


def test_glass_under_the_band_brings_no_fining_to_the_canal(plant):
    g, p, s = plant
    cold = s.replace(
        T=jnp.full_like(s.T, p.T_fine_low - 100.0), fining=jnp.zeros_like(s.T)
    )
    new, _ = P.plant_step(cold, 0.46, 5.79, 0.25, 0.2, 0.0, g, p)
    assert float(P.sensors(new, g, p)["fining"]) == 0.0


ENV = E.GlassFurnace()
P0 = ENV.default_params


def _quiet(params=P0):
    return params.replace(walk_sd=(0.0,) * len(E.WALKS))


def _act(fuel, params=P0):
    return E.to_action(jnp.asarray(fuel), params)


def test_reward_prices_fuel_and_glass_lost_on_the_charged_pull():
    p = P0
    dt = p.delta_t
    fuel_cost = 0.45 * dt * p.plant.LHV / 3.6e9 * p.gas_price  # EUR/MWh on energy
    pull = 1.1 * p.pull  # a pull change in progress
    glass = pull * dt / 1000.0 * p.lost_glass_price  # EUR/t on the tonnes charged
    good = dict(canal=p.canal_target + 0.5, fining=p.fining_pass + 0.01, pull=pull)
    ok = E.costs(0.45, **good, params=p)
    assert set(ok) == {"running", "tracking"}  # fuel, and lost glass
    assert float(ok["running"]) == pytest.approx(fuel_cost, rel=1e-6)
    assert float(ok["tracking"]) == 0.0
    for bad in (dict(canal=p.canal_target - 0.5), dict(fining=p.fining_pass - 0.01)):
        cost = E.costs(0.45, **{**good, **bad}, params=p)
        assert float(cost["tracking"]) == pytest.approx(glass, rel=1e-6)


def test_the_registered_task_is_this_plant_with_its_scenarios():
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    spec = REGISTRY["glass_furnace"]
    assert type(spec.make_env()) is E.GlassFurnace
    shares = scenarios.weights(spec)
    assert set(shares) == {"normal", "pull_up", "pull_down", "gas_rich", "gas_lean"}
    up = scenarios.params_for(spec, scenarios.suite(spec)[1])
    assert up.pull_change > 0.0


def test_crown_override_holds_the_hottest_face_under_1620_c():
    p = _quiet()
    _, state = ENV.reset_env(jax.random.PRNGKey(0), p)

    def f(st, k):
        _, st, _, _, _ = ENV.step_env(k, st, _act(p.fuel_max, p), p)
        return st, P.sensors(st.plant, p.grid, p.plant)["hot"]

    keys = jax.random.split(jax.random.PRNGKey(1), int(2 * 86400 / p.delta_t))
    hot = np.asarray(jax.jit(lambda: jax.lax.scan(f, state, keys)[1])())
    later = hot[int(6 * 3600 / p.delta_t) :]  # past the start's own transient
    assert later.max() <= p.crown_high


def test_override_leaves_the_valve_alone_under_the_limit():
    p = _quiet()
    _, state = ENV.reset_env(jax.random.PRNGKey(0), p)
    # A cold crown no longer calls for more fuel: the low override is gone.
    for face in (1773.15, 1600.0):  # 1500 C, and 1327 C, under silica's 1400 C
        cold = state.plant.replace(T_roof=state.plant.T_roof.at[0].set(face))
        _, new, _, _, _ = ENV.step_env(
            jax.random.PRNGKey(2), state.replace(plant=cold), _act(0.30, p), p
        )
        assert float(new.fuel) == pytest.approx(0.30, rel=1e-5)


def test_observation_carries_readings_not_the_hidden_walks():
    _, state = ENV.reset_env(jax.random.PRNGKey(0), P0)
    other = state.replace(walk=state.walk + 0.03)
    np.testing.assert_array_equal(ENV.get_obs(state, P0), ENV.get_obs(other, P0))
    assert len(E.OBS_NAMES) == ENV.get_obs(state, P0).shape[0]


def test_walks_start_where_the_seed_puts_them():
    first = lambda k: ENV.reset_env(jax.random.PRNGKey(k), P0)[1].walk
    np.testing.assert_array_equal(first(3), first(3))
    assert not np.allclose(first(3), first(4))
    # Over many seeds, each walk starts at its stationary spread; 2000 draws give the sd
    # to about 2 %, so 10 % is the bound.
    starts = np.asarray(jax.vmap(lambda k: ENV.reset_env(k, P0)[1].walk)(
        jax.random.split(jax.random.PRNGKey(5), 2000)))  # fmt: skip
    np.testing.assert_allclose(starts.std(0), P0.walk_sd, rtol=0.1)


def _trace(state, p, fuel, steps, key=1):
    def f(st, k):
        obs, st, _, _, _ = ENV.step_env(k, st, _act(fuel, p), p)
        return st, (
            obs,
            P.sensors(st.plant, p.grid, p.plant)["fining"],
            st.pull,
            st.time,
        )

    keys = jax.random.split(jax.random.PRNGKey(key), steps)
    return [np.asarray(x) for x in jax.jit(lambda: jax.lax.scan(f, state, keys)[1])()]


def _reading(channel):
    return next(r for r in ENV.readings if r.channel == channel)


def test_quality_is_read_a_transit_later():
    p = _quiet().replace(sensor_noise_scale=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(0), p)
    q = E.OBS_NAMES.index("fining share read at the cold end")
    d = int(round(_reading(q).dead_time / p.delta_t))
    assert d > 1
    obs, share, _, _ = _trace(state, p, 0.47, 3 * d + 10)
    np.testing.assert_allclose(obs[d:, q], share[:-d], rtol=1e-6)


def test_a_planned_pull_change_follows_its_plan_and_shows_its_notice():
    p = _quiet().replace(pull_change=0.1, pull_change_window_h=(6.0, 6.0))
    _, state = ENV.reset_env(jax.random.PRNGKey(0), p)
    H = 3600 / p.delta_t
    steps = int((6.0 + p.pull_ramp_h + 2.0) * H)
    obs, _, pull, time = _trace(state, p, 0.47, steps)
    hours = time * p.delta_t / 3600
    plan = obs[:, E.OBS_NAMES.index("pull planned, kg/s")]
    assert np.allclose(pull[hours < 6.0], p.pull)
    assert np.allclose(pull[hours > 6.0 + p.pull_ramp_h], 1.1 * p.pull)
    early = hours < 6.0 - p.pull_notice_h - 0.1
    assert np.allclose(plan[early], p.pull)
    assert np.allclose(
        plan[hours >= 6.0 - p.pull_notice_h + p.pull_ramp_h], 1.1 * p.pull
    )


def test_batch_line_is_read_on_its_cadence():
    p = _quiet().replace(sensor_noise_scale=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(0), p)
    b = E.OBS_NAMES.index("batch line, m")
    every = int(round(_reading(b).interval / p.delta_t))
    obs, _, _, time = _trace(state, p, 0.40, 4 * every)
    line, block = obs[:, b], time // every
    for k in np.unique(block):
        assert np.all(line[block == k] == line[block == k][0])  # held between readings
    firsts = [line[block == k][0] for k in np.unique(block)]
    assert len(set(np.round(firsts, 6))) > 1  # and read anew at each


def test_cullet_ratio_is_read_as_charged():
    _, state = ENV.reset_env(jax.random.PRNGKey(0), P0)
    _, new, _, _, _ = ENV.step_env(jax.random.PRNGKey(1), state, _act(0.47), P0)
    c = ENV.get_obs(new, P0)[E.OBS_NAMES.index("cullet ratio")]
    assert float(c) == pytest.approx(
        P0.cullet + float(new.walk[E.WALKS.index("cullet")])
    )


def test_walks_have_their_stated_spread_and_memory():
    ep = P0
    steps = int(2000 * 86400 / ep.delta_t)
    w = np.asarray(E.walks(jax.random.PRNGKey(3), steps, ep))
    # Each column: sd as stated, and correlation e^-1 one correlation time apart. About
    # 1000 independent stretches, so the sd is good to about 3 %; 10 % is the bound.
    for i, (sd, hours) in enumerate(zip(ep.walk_sd, ep.walk_hours)):
        lag = int(hours * 3600 / ep.delta_t)
        x = w[:, i]
        assert x.std() == pytest.approx(sd, rel=0.1)
        r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        assert r == pytest.approx(np.exp(-1.0), abs=0.06)


def test_scenario_weights_give_the_normal_day_most_of_the_time():
    # The family runs a float line steadily for most of its time [S5]; gas swings
    # [S19; S20] and planned pull changes [S1; S9] are rarer windows (REALITY.md).
    from target_gym.glass_furnace.physical_scenarios import suite

    shares = {s.name: s.weight for s in suite()}
    total = sum(shares.values())
    assert all(w > 0 for w in shares.values())
    assert shares["normal"] / total > 0.5
    assert max(shares, key=shares.get) == "normal"
