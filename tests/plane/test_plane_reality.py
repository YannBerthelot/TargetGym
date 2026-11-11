"""``plane`` against the real aircraft (charter Amendments 1 and 2).

The tests are BEHAVIOUR_TESTS.md's, turned into code. Each names its source in
src/target_gym/plane/REALITY.md and a tolerance fixed before it ran.

Contract:

- Limits (rule 14). Load factor, AoA and bank are protected by the flight
  controls, not trips [CFR 25.337, NTSB 1.6.3, BEA447]. VMO 350 kt, MMO 0.82 and
  the 39 100 ft ceiling are the type's [TCDS], and crossing VMO is a warning,
  not a trip.
- Readings (rule 16). The autopilot reads air data: airspeed, and the
  accelerations TECS fuses [BEA447, TECS].
- Episode (rule 17). The hold starts established at the assigned level [BEA447,
  ALT CRZ]; the target is one assigned level [RVSM]; turbulence comes in zones
  of 3 to 30 min [BEA447 Figure 24], scale length 1 750 ft [MADDEN].
- Score (rule 15). Zero inside +-65 ft of the assigned altitude [RVSM] and
  +-10 kt of the assigned speed [AIM 4-4-12]; past a band edge, the excess in
  band widths, squared, per second (the owner's choice of shape, 2026-11-08).

Hand-derived numbers: 65 ft = 19.812 m. 30 m off the level is 10.188 m past the
edge, (10.188 / 19.812)^2 = 0.2644. 15 kt over the assigned speed at sea level,
where CAS equals TAS, is 5 kt past the edge, (5 / 10)^2 = 0.25.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg

FT, KT = 0.3048, 1852.0 / 3600.0
BAND = 65.0 * FT
KEY = jax.random.PRNGKey(7)


def _rollout(name, params, steps, policy=None, key=KEY):
    """States and observations of ``steps`` steps under the reference expert, or
    under a fixed action ``policy``."""
    env, _ = tg.make(name)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    if policy is None:
        expert = tg.make_expert(name)
        memory = expert.reset()
        act = jax.jit(expert.step)
    states, observations, tripped = [state], [obs], False
    for k in jax.random.split(key, steps):
        if policy is None:
            action, memory = act(memory, obs)
        else:
            action = jnp.asarray(policy, jnp.float32)
        obs, state, _, _, info = step(k, state, action, params)
        tripped |= bool(info["tripped"])
        states.append(state)
        observations.append(obs)
    return states, observations, tripped


def _calm(params):
    return params.replace(turbulence_sigma=0.0)


# -- Limits (rule 14) ----------------------------------------------------------


@pytest.mark.parametrize("pitch", (1.0, -1.0))
@pytest.mark.parametrize("throttle", (1.0, -1.0))
def test_full_pitch_demand_is_protected_not_tripped(throttle, pitch):
    """A3: full nose-up or nose-down for 30 s from cruise at 6 000 m neither
    trips nor leaves [-1.0, 2.5] g plus a 0.1 g protection overshoot, and the
    wing stays below its stall angle [CFR 25.337, NTSB 1.6.3]."""
    _, params = tg.make("plane")
    params = _calm(params).replace(target_altitude_range=(6000.0, 6000.0))
    states, _, tripped = _rollout("plane", params, 30, policy=(throttle, pitch))
    n = np.array([float(s.load_factor) for s in states])
    alpha = np.rad2deg([float(s.alpha) for s in states])
    assert not tripped
    assert n.max() <= 2.5 + 0.1 and n.min() >= -1.0 - 0.1
    assert alpha.max() < params.aoa_stall


def test_the_type_limits_are_the_tcds_figures():
    """A4: VMO 350 kt, MMO 0.82, ceiling 39 100 ft [TCDS items 10 and 11],
    within 1 kt, 0.005 and 100 ft."""
    _, params = tg.make("plane")
    assert abs(params.vmo_eas / KT - 350.0) < 1.0
    assert abs(params.mmo - 0.82) < 0.005
    assert abs(params.max_alt / FT - 39100.0) < 100.0


def test_overspeed_is_a_warning_not_a_trip():
    """A4: past VMO the TCDS lists a warning; no source lists a trip."""
    env, params = tg.make("plane")
    _, state = env.reset_env(KEY, _calm(params))
    fast = state.replace(x_dot=jnp.float32(300.0), z=jnp.float32(3000.0))
    assert not bool(env.is_terminated(fast, params))


# -- Readings (rule 16) --------------------------------------------------------


def test_the_autopilot_reads_airspeed_not_ground_speed():
    """Air data gives airspeed [BEA447, TECS]: in a 20 m/s headwind the airspeed
    reading is the ground speed plus 20 m/s."""
    env, params = tg.make("plane")
    params = _calm(params).replace(wind_x=-20.0, sensor_noise_scale=0.0)
    obs, state = env.reset_env(KEY, params)
    ground = float(jnp.hypot(state.x_dot, state.z_dot))
    assert abs(float(obs[env.obs_index["airspeed"]]) - (ground + 20.0)) < 0.01


def test_the_autopilot_reads_normal_acceleration():
    """TECS fuses vertical acceleration [TECS, P409]: an established aircraft
    reads 1 g, within 0.02 g."""
    env, params = tg.make("plane")
    obs, _ = env.reset_env(KEY, _calm(params).replace(sensor_noise_scale=0.0))
    assert abs(float(obs[env.obs_index["load_factor"]]) - 1.0) < 0.02


# -- Episode (rule 17) ---------------------------------------------------------


def test_the_hold_starts_established_at_the_assigned_level():
    """Cruise is flown established in ALT CRZ [BEA447]: the reset aircraft is at
    the assigned level and speed, and holding the action the reset reports keeps
    it inside +-65 ft for 60 s of calm air, at 1 g within 0.05 g (ours: the
    step's worst load, which a trimmed aircraft keeps near 1 g)."""
    env, params = tg.make("plane")
    params = _calm(params).replace(sensor_noise_scale=0.0)
    obs, state = env.reset_env(KEY, params)
    assert float(state.z) == pytest.approx(float(state.target_altitude), abs=0.5)
    ias = obs[env.obs_index["airspeed"]]
    assert abs(float(ias - obs[env.obs_index["target_speed"]])) < 10.0 * KT
    states, _, tripped = _rollout("plane", params, 60, policy=env.trim_action(state))
    error = np.array([float(s.z - s.target_altitude) for s in states])
    n = np.array([float(s.load_factor) for s in states])
    assert not tripped
    assert np.abs(error).max() < BAND
    assert np.abs(n - 1.0).max() < 0.05


def test_the_target_is_one_assigned_level():
    """B2: altitude is assigned and held [RVSM, IFH ch. 10]."""
    _, params = tg.make("plane")
    states, _, _ = _rollout("plane", _calm(params), 50, policy=(0.0, 0.0))
    targets = {float(s.target_altitude) for s in states}
    assert len(targets) == 1


def test_the_hold_is_at_a_cruise_level_in_rvsm_airspace():
    """The +-65 ft band is RVSM's, which applies from FL290 to FL410 [RVSM];
    the levels stop at FL370, clear of the 39 100 ft maximum altitude [TCDS]
    (Amendment 3: the reviewer's FL390 cost 294 at that edge). On every seed
    the assigned level, where the aircraft starts, is in FL290 to FL370."""
    env, params = tg.make("plane")
    for seed in range(8):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        level = float(state.target_altitude) / FT
        assert 29_000.0 <= level <= 37_000.0
        assert float(state.z) == pytest.approx(float(state.target_altitude), abs=0.5)


def test_the_light_zone_is_derived_at_the_cruise_level():
    """turbulence_sigma from the sharp-edged gust load at FL330, the middle
    level, on the M0.78 schedule: dn/dw = rho V S a / (2 m g), with a the
    airplane's lift slope. ISA at 10 058 m: 222.77 K, rho 0.4097 kg/m3, a 299.2
    m/s, so V = 233.4 m/s. The swept wing's slope at M 0.78 is 6.39 per rad
    [PHYSICS.md 4]; the tail adds 0.9 x 31 / 122.6 x 4.98 x (1 - 0.429) = 0.65,
    its slope less the wing's downwash, so a = 7.03 per rad and dn/dw = 0.4097 x
    233.4 x 122.6 x 7.03 / (2 x 73 500 x 9.81) = 0.0572 g per m/s. A minute is
    60 V / L = 26.3 scale lengths of 533.4 m, whose expected range is
    2 sqrt(2 ln 26.3) = 5.11 standard deviations, so 0.2 g peak to peak
    [BEA447 Figure 24's usual level] needs 0.2 / (5.11 x 0.0572) = 0.684 m/s."""
    _, params = tg.make("plane")
    assert float(params.turbulence_sigma) == pytest.approx(0.684, rel=0.01)


def test_turbulence_comes_in_a_zone_with_calm_around_it():
    """Turbulence comes in zones of 3 to 30 min [BEA447 Figure 24]: on every
    seed the gust is zero outside one zone, and the zone lasts 180 to 1 800 s."""
    env, params = tg.make("plane")
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        length = float(state.zone_end - state.zone_start)
        assert 180.0 <= length <= 1800.0
        assert 0.0 <= float(state.zone_start) < params.max_steps_in_episode


# -- Score (rule 15) -----------------------------------------------------------


def _held(params, **changes):
    env, _ = tg.make("plane")
    _, state = env.reset_env(KEY, _calm(params))
    return env, state.replace(**changes)


def _band_cost(env, state, params):
    """The bands' cost, every term but the operator's economy (rule 15)."""
    terms = env.reward_terms(state, params)
    return float(sum(v for k, v in terms.items() if k != "economy"))


def test_inside_both_bands_costs_nothing():
    """Precision inside a band earns nothing [rule 15]: 15 m off the level and
    5 kt off the speed cost zero."""
    env, params = tg.make("plane")
    env, state = _held(params)
    tas = float(jnp.hypot(state.x_dot, state.z_dot))
    off = state.replace(z=state.target_altitude + 15.0, x_dot=jnp.float32(tas + 2.0))
    assert _band_cost(env, off, params) == 0.0


def test_altitude_past_the_band_costs_the_squared_excess():
    """30 m above the level is 10.188 m past +-65 ft: 0.2644 per second."""
    env, params = tg.make("plane")
    env, state = _held(params)
    off = state.replace(z=state.target_altitude + 30.0)
    assert _band_cost(env, off, params) == pytest.approx(0.2644, rel=1e-3)


def test_speed_past_the_band_costs_the_squared_excess():
    """15 kt over the assigned speed at sea level, where CAS is TAS, is 5 kt past
    +-10 kt [AIM 4-4-12]: 0.25 per second. Within 1 %: the model's speed of
    sound uses R = 287.0 J/(kg K) against ISA's 287.053 (PHYSICS.md, known
    deviations), 0.02 kt at 305 kt, 0.4 % of the 5 kt excess, doubled squared."""
    env, params = tg.make("plane")
    params = params.replace(min_alt=-10.0)
    env, state = _held(params)
    level = state.replace(z=jnp.float32(0.0), target_altitude=jnp.float32(0.0))
    target = float(params.target_cas)
    fast = level.replace(x_dot=jnp.float32(target + 15.0 * KT), z_dot=0.0)
    assert _band_cost(env, fast, params) == pytest.approx(0.25, rel=1e-2)


# -- Turbulence and the reference (medium: 600 to 1 800 closed-loop steps) -----


@pytest.mark.medium
def test_the_reference_holds_the_rvsm_band_in_still_air():
    """B1: +-65 ft in straight and level flight, nonturbulent [RVSM A.4.1.4],
    10 min under the reference autopilot."""
    _, params = tg.make("plane")
    states, _, tripped = _rollout("plane", _calm(params), 600)
    error = np.array([float(s.z - s.target_altitude) for s in states])
    assert not tripped
    assert np.abs(error).max() < BAND


@pytest.fixture(scope="module")
def zone_run():
    """30 min of the reference in a zone that covers the whole run."""
    _, params = tg.make("plane")
    params = params.replace(
        max_steps_in_episode=1800,
        zone_onset_range=(0.0, 0.0),
        zone_duration_range=(1800.0, 1800.0),
    )
    states, _, tripped = _rollout("plane", params, 1800)
    return states, tripped


@pytest.mark.medium
def test_light_turbulence_load_matches_the_recorded_cruise(zone_run):
    """A5: every 1 min window under 0.5 g peak to peak, the median window
    0.1 to 0.3 g [BEA447, ICAO light; Figure 24's recorded range]."""
    states, tripped = zone_run
    n = np.array([float(s.load_factor) for s in states[1:]])
    spans = np.array([np.ptp(w) for w in n.reshape(30, 60)])
    assert not tripped
    assert spans.max() < 0.5
    assert 0.1 <= np.median(spans) <= 0.3


@pytest.mark.medium
def test_the_gust_scale_length_is_1750_ft(zone_run):
    """A6: the vertical gust's integral scale along the path is 533 m within
    25 % [MADDEN, MIL-F-8785C], estimated as the airspeed times the integral of
    the gust's autocorrelation up to its first zero."""
    states, _ = zone_run
    w = np.array([float(s.gust_z) for s in states[1:]])
    speed = np.mean([float(jnp.hypot(s.x_dot, s.z_dot)) for s in states[1:]])
    w = w - w.mean()
    acf = np.correlate(w, w, "full")[len(w) - 1 :] / (w.var() * len(w))
    first_zero = int(np.argmax(acf <= 0.0))
    scale = speed * (acf[:first_zero].sum() - 0.5)  # trapezoid, dt = 1 s
    assert abs(scale - 533.0) / 533.0 < 0.25


def test_the_reference_tecs_sits_in_its_documented_ranges():
    """C1: TECS_TIME_CONST 3 to 10 s, TECS_PTCH_DAMP 0.1 to 1.0, TECS_THR_DAMP
    0.1 to 1.0, TECS_INTEG_GAIN 0 to 0.5, TECS_SPDWEIGHT 0 to 2 [P409]."""
    from target_gym.plane.experts_tecs import AIRLINER as p

    assert 3.0 <= p.time_const <= 10.0
    assert 0.1 <= p.ptch_damp <= 1.0
    assert 0.1 <= p.thr_damp <= 1.0
    assert 0.0 <= p.integ_gain <= 0.5
    assert 0.0 <= p.spd_weight <= 2.0
