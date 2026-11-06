"""The lead's wake on the patrol family.

The contract, read before any of this was written:

- Blake and Multhopp, AIAA-98-4343: the lead's lift is a horseshoe vortex, its legs
  ``pi b / 4`` apart at circulation ``L / (rho V b0)``, with Burnham's core. An
  upwash ``w`` rotates a section's lift forward by ``w / V``, which is the drag
  saving: nothing needs adding for it.
- Dogan, Venkataramanan and Blake, J. Aircraft 2005: the follower takes the
  non-uniform field as an effective wind and an effective gradient. Here those are
  weighted by an elliptic lift distribution, so the gradient is the plant's own
  ``p_g``: a wing rolling at ``p`` sees ``w = p y``, which that weight gives back.
- Holzapfel and Steen, AIAA J. 2007: the pair sinks at ``w0 = Gamma / (2 pi b0)``,
  on the time scale ``t0 = 2 pi b0^2 / Gamma``.
- NASA's Autonomous Formation Flight flight data (AFF overview deck, 2 F/A-18s, M
  0.56 at 25,000 ft) sets the targets, read off its charts: induced drag 40 to 52%
  lower near a 13% tip overlap, 10 to 20% lower at 25% tip separation; rolling
  moment peaking near 0.01 away from the lead, reversing near 27% overlap.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym.eval import scored_burn_in
from target_gym.patrol.env import (
    GUST_MEMORY,
    PatrolParams,
    initial_wake,
    lead_cruise_speed,
    make_cruise_state,
    shed_wake,
    slot_error,
    step_lead,
    wake_air,
    wake_sample,
)
from target_gym.plane.dynamics import compute_air_density_from_altitude
from target_gym.plane3d.dynamics import compute_acceleration_3d

KEY = jax.random.PRNGKey(0)

# The trials' F/A-18 at M 0.56 and 25,000 ft, with a 16 t weight (ours, a mid-fuel
# figure), which puts its lift coefficient at 0.51.
FA18 = PatrolParams(wingspan=11.43, initial_mass=16_000.0, wings_surface=37.16)
FA18_Z, FA18_V = 7620.0, 174.0
#: The share of an elliptic wing's induced drag a low aspect ratio wing keeps (ours).
#: NASA gives no span efficiency, and this is the usual range for an AR of 3.5.
FA18_E = 0.7
#: Roll damping of an F/A-18 (ours, the usual size for a wing of this aspect ratio).
FA18_CLP = -0.25


def _pair(params, z, speed):
    """A lead flying straight along +x, its wake laid out behind it."""
    lead = make_cruise_state(0.0, 0.0, z, 0.0, speed, params, z, 0.0)
    return lead, initial_wake(lead, params)


def _read(history, params, z, speed, right, up, back=0.0):
    """Upwash, sidewash to the lead's right, and rolling gust, on one follower.
    On a lead flying along +x, its right is -y."""
    delay = jnp.array([back / (speed * float(params.delta_t))])
    gusts, rolls = wake_air(
        history,
        jnp.array([-back]),
        jnp.array([-right]),
        jnp.array([z + up]),
        delay,
        params,
    )
    return float(gusts[0, 2]), float(-gusts[0, 1]), float(rolls[0])


def _fa18(tip_overlap, up=0.0, back=0.0):
    """``(induced drag change, rolling moment away from the lead)``, for a tip
    separation and height in spans, on the trials' two F/A-18s."""
    b = FA18.wingspan
    _, history = _pair(FA18, FA18_Z, FA18_V)
    w, _, p = _read(history, FA18, FA18_Z, FA18_V, (1 + tip_overlap) * b, up * b, back)
    q_s = 0.5 * float(compute_air_density_from_altitude(FA18_Z)) * FA18_V**2 * 37.16
    cl = FA18.initial_mass * FA18.gravity / q_s
    aspect = b**2 / 37.16
    induced = -(w / FA18_V) * np.pi * aspect * FA18_E / cl
    return induced, -p * b / (2 * FA18_V) * -FA18_CLP


# ─── The flight data ────────────────────────────────────


def test_the_induced_drag_saving_peaks_where_nasa_measured_it():
    """Flight, Z about 0: 40 to 52% near a 13% overlap, with the peak inside 8 to
    20%. The same chart has 20% at about +17% and 10% at about +33%."""
    overlaps = np.arange(-0.4, 0.3, 0.02)
    change = np.array([_fa18(y)[0] for y in overlaps])
    best = overlaps[np.argmin(change)]
    assert -0.20 <= best <= -0.08
    assert -0.52 <= change.min() <= -0.40
    assert -0.20 <= _fa18(0.25)[0] <= -0.10


def test_the_rolling_moment_has_nasas_shape():
    """Flight, Z about 0 to -13%: away from the lead, about 0.008 to 0.010 near a 12%
    overlap and 0.0025 at +25%, nothing by +50%, and reversing near a 27% overlap.
    Strip theory's peak runs high, so the bound is the chart's peak plus half."""
    overlaps = np.arange(-0.40, 0.5, 0.01)
    roll = np.array([_fa18(y)[1] for y in overlaps])
    assert 0.008 <= roll.max() <= 0.015
    assert -0.20 <= overlaps[np.argmax(roll)] <= -0.08
    assert 0.0015 <= _fa18(0.25)[1] <= 0.004
    assert abs(_fa18(0.5)[1]) <= 0.002
    assert -0.33 <= overlaps[np.argmin(np.abs(roll[:40]))] <= -0.20
    assert roll[0] < 0.0  # deep overlap rolls the other way


def test_the_saving_falls_with_distance_as_nasa_measured():
    """The chart's induced drag band falls from 43-52% at 275% of a span aft to
    31-40% at 950%, a ratio of about 0.75, which the sinking pair gives by itself:
    at 950% it has sunk about 18% of a span, and the follower stays at Z = -5%."""
    near = _fa18(-0.13, up=-0.05, back=2.75 * FA18.wingspan)[0]
    far = _fa18(-0.13, up=-0.05, back=9.5 * FA18.wingspan)[0]
    assert 0.6 <= far / near <= 0.9


# ─── The vortex pair ────────────────────────────────────


def test_far_away_the_wake_is_the_lead_lift_and_nothing_else():
    """Ten spans out the pair is a doublet: ``w = L / (2 pi rho V y^2)``, which holds
    no span. It follows the lift in proportion."""
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    y = 10.0 * params.wingspan
    lift = float(params.initial_mass * params.gravity)
    rho = float(compute_air_density_from_altitude(z))
    _, history = _pair(params, z, speed)
    w, _, _ = _read(history, params, z, speed, y, 0.0)
    assert w == pytest.approx(lift / (2 * np.pi * rho * speed * y**2), rel=0.03)
    _, doubled = _pair(params.replace(initial_mass=2 * params.initial_mass), z, speed)
    assert _read(doubled, params, z, speed, y, 0.0)[0] == pytest.approx(2 * w, rel=1e-4)


@pytest.mark.parametrize("right, up", [(30.0, 0.0), (8.0, 5.0), (60.0, -20.0)])
def test_the_effective_wind_and_gradient_match_a_fine_integration(right, up):
    """A slow reference, from the formulas: Burnham's profile summed over 200,000
    stations of an elliptic wing, against the plant's quadrature."""
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    b, b0 = float(params.wingspan), float(np.pi / 4 * params.wingspan)
    core2 = (float(params.wake_core_ratio) * b) ** 2
    lift = float(params.initial_mass * params.gravity)
    gamma = lift / (float(compute_air_density_from_altitude(z)) * speed * b0)
    y = np.linspace(-b / 2, b / 2, 200_001)[1:-1]
    load = np.sqrt(1 - (2 * y / b) ** 2)
    load /= np.trapezoid(load, y)
    w = v = 0.0
    for sign, leg in ((1.0, b0 / 2), (-1.0, -b0 / 2)):
        along = right + y - leg
        r2 = along**2 + up**2 + core2
        w = w + sign * gamma / (2 * np.pi) * along / r2
        v = v - sign * gamma / (2 * np.pi) * up / r2
    want = (
        np.trapezoid(w * load, y),
        np.trapezoid(v * load, y),
        16 / b**2 * np.trapezoid(w * load * y, y),
    )
    _, history = _pair(params, z, speed)
    np.testing.assert_allclose(
        _read(history, params, z, speed, right, up), want, rtol=0.01
    )


def test_the_two_sides_of_the_lead_mirror_each_other():
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    _, history = _pair(params, z, speed)
    w, v, p = _read(history, params, z, speed, 30.0, 8.0)
    mirrored = _read(history, params, z, speed, -30.0, 8.0)
    np.testing.assert_allclose(mirrored, (w, -v, -p), rtol=1e-4)


def test_the_pair_sinks_and_decays_with_age():
    """Read at age ``t`` a distance ``w0 t`` lower, the wake is the one at age 0,
    weaker by the lidar median's ``exp(-0.2 t / t0)``, Holzapfel and Steen's figure 3
    as we fit it."""
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    b0 = np.pi / 4 * float(params.wingspan)
    lift = float(params.initial_mass * params.gravity)
    gamma = lift / (float(compute_air_density_from_altitude(z)) * speed * b0)
    w0, t0 = gamma / (2 * np.pi * b0), 2 * np.pi * b0**2 / gamma
    back = 2.0 * speed
    _, history = _pair(params, z, speed)
    young = _read(history, params, z, speed, 30.0, 0.0)
    old = _read(history, params, z, speed, 30.0, -w0 * 2.0, back)
    # On the pair's own line the sidewash is zero, hence the absolute tolerance.
    want = np.array(young) * np.exp(-0.2 * 2.0 / t0)
    np.testing.assert_allclose(old, want, rtol=0.01, atol=1e-4)


def test_the_wake_lies_where_the_lead_flew_not_where_it_is():
    """In a turn the pair stays on the lead's past track. A follower abeam the lead's
    position two steps ago, by that moment's right, meets it as on a straight path:
    the delay the gust uses reaches the wake too."""
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    lead, straight = _pair(params, z, speed)
    want = _read(straight, params, z, speed, 30.0, 0.0, back=2.0 * speed)
    history, turn, x, y, psi = straight, 0.05, 0.0, 0.0, 0.0
    places = []
    for _ in range(GUST_MEMORY):
        places.append((x, y, psi))
        here = lead.replace(x=x, y=y, psi=psi)
        history = shed_wake(history, wake_sample(here, 722_000.0, speed, params))
        psi, x, y = psi + turn, x + speed * np.cos(psi), y + speed * np.sin(psi)
    x, y, psi = places[-3]
    right = np.array([np.sin(psi), -np.cos(psi)])
    gusts, rolls = wake_air(
        history,
        jnp.array([x + 30.0 * right[0]]),
        jnp.array([y + 30.0 * right[1]]),
        jnp.array([z - 0.0]),
        jnp.array([2.0]),
        params,
    )
    assert float(gusts[0, 2]) == pytest.approx(want[0], rel=0.05)
    assert float(rolls[0]) == pytest.approx(want[2], rel=0.05)


def test_in_a_banked_turn_the_pair_tilts_with_the_lift():
    """The lead banked: a follower along its wing, at the same distance, meets the
    same wake, turned with it. A positive bank puts the right wing up."""
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    lead, level = _pair(params, z, speed)
    w, v, p = _read(level, params, z, speed, 30.0, 0.0)
    phi = 0.5
    banked = initial_wake(lead.replace(phi=phi), params)
    tilted = _read(banked, params, z, speed, 30.0 * np.cos(phi), 30.0 * np.sin(phi))
    assert tilted[2] == pytest.approx(p, rel=1e-3)
    np.testing.assert_allclose(
        tilted[:2],
        (w * np.cos(phi) + v * np.sin(phi), v * np.cos(phi) - w * np.sin(phi)),
        rtol=1e-3,
    )


def test_a_follower_ahead_of_the_lead_is_clear_of_its_wake():
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    _, history = _pair(params, z, speed)
    gusts, rolls = wake_air(
        history,
        jnp.array([50.0]),
        jnp.array([-30.0]),
        jnp.array([z]),
        jnp.array([1.0]),
        params,
    )
    np.testing.assert_array_equal(gusts, 0.0)
    np.testing.assert_array_equal(rolls, 0.0)


# ─── On the plant ───────────────────────────────────────


def _slot(name, right, up=0.0, **overrides):
    """A task in still air with its follower exactly in a slot 200 m back."""
    env, params = tg.make(name)
    params = params.replace(
        turbulence_sigma=0.0,
        follower_spawn_noise=0.0,
        lead_turn_rate_range=(0.0, 0.0),
        sensor_noise_scale=0.0,
        slot_back_range=(200.0, 200.0),
        slot_right_range=(right, right),
        slot_up_range=(up, up),
        target_altitude_range=(5000.0, 5000.0),
        **overrides,
    )
    return env, params, *env.reset_env(KEY, params)


def test_upwash_saves_the_drag_the_rotated_lift_gives_back():
    """Blake and Multhopp's forward rotation of the lift is ``L w / V``. At one angle
    of attack, the upwash the wake puts on a follower near the sweet spot gains it
    ``g w / V`` along its path, and nothing but the wind carries it: the plant's lift
    lies across the air-relative velocity, which the upwash tilts. That is the sweet
    spot's drag saving, so it has no term of its own."""
    params = PatrolParams()
    z, speed = 5000.0, 190.0
    _, history = _pair(params, z, speed)
    w = _read(history, params, z, speed, 0.87 * params.wingspan, 0.0, 200.0)[0]
    assert w > 0.5

    def along_path(wind_z, theta):
        velocities = jnp.array([speed, 0.0, 0.0, 0.0, 0.0])
        positions = jnp.array([0.0, 0.0, z, theta, 0.0])
        accel, metrics = compute_acceleration_3d(
            velocities, positions, (40e3, 0.0, 0.0), params.replace(wind_z=wind_z)
        )
        return float(accel[0]), float(metrics[1])

    theta = 0.03
    clear, lift = along_path(0.0, theta)
    gained = along_path(w, theta - w / speed)[0] - clear
    assert gained == pytest.approx(lift * w / (speed * params.initial_mass), rel=0.02)


def test_a_slot_astride_the_wake_rolls_a_follower_away_from_the_lead():
    """Right of the lead with its inner wing in the upwash: the left wing rises, so
    the bank goes negative, right wing down."""
    env, params, _, state = _slot("patrol", 0.87 * float(PatrolParams().wingspan))
    assert float(state.follower.phi) == 0.0
    step, action = jax.jit(env.step_env), jnp.array([0.0, 0.1, 0.0])
    for t in range(2):
        _, state, _, _, _ = step(jax.random.PRNGKey(t), state, action, params)
    assert float(state.follower.phi) < -1e-3
    off = params.replace(wake_strength=0.0)
    _, calm = env.reset_env(KEY, off)
    for t in range(2):
        _, calm, _, _, _ = step(jax.random.PRNGKey(t), calm, action, off)
    assert float(calm.follower.phi) == 0.0


def test_the_lead_flies_clear_of_its_own_wake():
    env, params, _, state = _slot("patrol", 30.0)
    _, state, _, _, _ = env.step_env(KEY, state, jnp.zeros(3), params)
    assert float(state.lead.gust_z) == 0.0
    assert float(state.follower.gust_z) != 0.0


def test_the_wake_comes_from_the_lift_of_each_step():
    """A reset aircraft carries much less than its weight until it pitches up, so the
    wake it sheds then is weaker than its weight's. Ten spans out it is ``L / (2 pi
    rho V y^2)`` on the lift it made that step."""
    env, params, _, state = _slot("patrol", 30.0)
    lift = float(step_lead(state, params)[2])
    assert lift < 0.8 * float(state.lead.m * params.gravity)
    _, new, _, _, _ = env.step_env(KEY, state, jnp.zeros(3), params)
    y = 10.0 * params.wingspan
    gusts, _ = wake_air(
        new.wake_history,
        jnp.atleast_1d(new.lead.x + y * jnp.sin(new.lead.psi)),
        jnp.atleast_1d(new.lead.y - y * jnp.cos(new.lead.psi)),
        jnp.atleast_1d(new.lead.z),
        jnp.zeros(1),
        params,
    )
    rho = float(compute_air_density_from_altitude(new.lead.z))
    speed = float(jnp.hypot(new.lead.x_dot, new.lead.y_dot))
    want = lift / (2 * np.pi * rho * speed * y**2)
    assert float(gusts[0, 2]) == pytest.approx(want, rel=0.05)


def test_the_wake_stays_with_the_lead_down_the_route():
    """Six seconds on, over a kilometre from where it began, the follower in its slot
    still has the wake on it."""
    env, params, _, state = _slot("patrol", 40.0)
    step = jax.jit(env.step_env)
    for t in range(6):
        action = jnp.array([0.0, 0.1, 0.0])
        _, state, _, _, _ = step(jax.random.PRNGKey(t), state, action, params)
    assert float(state.lead.x**2 + state.lead.y**2) > 1000.0**2
    assert abs(float(state.follower.gust_z)) > 0.2


def test_each_wingman_meets_the_wake_on_its_own_side():
    """The V's wings mirror each other, so in still air they see one upwash."""
    env, params = tg.make("formation")
    params = params.replace(
        turbulence_sigma=0.0,
        follower_spawn_noise=0.0,
        lead_turn_rate_range=(0.0, 0.0),
        slot_up_range=(0.0, 0.0),
    )
    _, state = env.reset_env(KEY, params)
    _, state, _, _, _ = env.step_env(KEY, state, jnp.zeros(9), params)
    gust = np.asarray(state.wingmen.gust_z)
    assert gust[0] > 0.0
    assert gust[1] == pytest.approx(gust[0], rel=1e-3)


def test_with_no_wake_the_plant_is_the_one_before_it():
    """``wake_strength`` 0 turns it off, which is how a run measures what it costs."""
    env, params, _, state = _slot("patrol", 30.0, wake_strength=0.0)
    _, state, _, _, _ = env.step_env(KEY, state, jnp.zeros(3), params)
    assert float(state.follower.gust_z) == 0.0


def test_the_step_is_differentiable_through_a_vortex_core():
    """Dead on a leg, where a line vortex would be singular: a planner's gradient
    stays finite and moves."""
    b0 = np.pi / 4 * float(PatrolParams().wingspan)
    env, params, _, state = _slot("patrol", b0 / 2)

    def cost(action):
        return -env.step_env(KEY, state, action, params)[2]

    grad = jax.grad(cost)(jnp.array([0.0, 0.1, 0.0]))
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


# ─── The expert in the wake ─────────────────────────────


@pytest.mark.parametrize("altitude", [3000.0, 8000.0])
def test_the_aileron_can_hold_the_worst_of_the_wake(altitude):
    """Across the wake's cross-plane 200 m back, the largest rolling moment stays
    under full aileron's, so a slot in it can be flown, as NASA's pilots flew it. The
    plant's rolling gust stands for ``-C_lp p b / 2V`` and its aileron for ``6 m^2 x
    0.08786 /deg x 0.4 x 25 deg x 14 m / (S b)``, which is 0.0168."""
    params = PatrolParams()
    speed = float(lead_cruise_speed(altitude))
    _, history = _pair(params, altitude, speed)
    right, up = np.meshgrid(np.arange(-80.0, 80.0), np.arange(-30.0, 30.0, 2.0))
    n = right.size
    _, rolls = wake_air(
        history,
        jnp.full(n, -200.0),
        jnp.asarray(-right.ravel()),
        jnp.asarray(altitude + up.ravel()),
        jnp.full(n, 200.0 / speed),
        params,
    )
    moment = float(jnp.abs(rolls).max()) * params.wingspan / (2 * speed) * -params.C_lp
    assert 0.002 < moment < 0.0168


def _hold(right, strength, seed):
    """``(trips, RMS slot error after the burn-in)``: the shipped expert in a slot
    ``right`` metres off the lead's path, 200 m back and level with it."""
    env, params = tg.make("patrol")
    params = params.replace(
        slot_right_range=(right, right),
        slot_up_range=(0.0, 0.0),
        slot_back_range=(200.0, 200.0),
        wake_strength=strength,
    )
    expert = tg.make_expert("patrol")
    step, act = jax.jit(env.step_env), jax.jit(expert.step)
    memory = expert.reset()
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset_env(key, params)
    trips, errors = 0, []
    for _ in range(int(params.max_steps_in_episode)):
        key, sub = jax.random.split(key)
        action, memory = act(memory, obs)
        obs, state, _, _, info = step(sub, state, action, params)
        trips += int(info["tripped"])
        errors.append(float(slot_error(state)))
    burn_in = scored_burn_in("patrol", params)
    return trips, float(np.sqrt(np.mean(np.square(errors[burn_in:]))))


@pytest.mark.large
@pytest.mark.xfail(
    strict=True, reason="PR 9: the hold has no allowance for a slot in the wake"
)
@pytest.mark.parametrize("right", [13.0, 31.0])
def test_the_expert_flies_a_slot_in_the_wake(right):
    """NASA held position in the vortex ("Controllable flight in vortex is possible")
    and the aileron has the authority, so the wake should not cost the expert the
    slot: no more trips than the same slot without it, and within 1.5 times its hold.
    Against itself, since the hold's 8.5 m is PR 9's on any slot."""
    seeds = (200, 201, 202)
    clear = [_hold(right, 0.0, seed) for seed in seeds]
    waked = [_hold(right, 1.0, seed) for seed in seeds]
    assert sum(t for t, _ in waked) <= sum(t for t, _ in clear)
    assert np.mean([r for _, r in waked]) <= 1.5 * np.mean([r for _, r in clear])
