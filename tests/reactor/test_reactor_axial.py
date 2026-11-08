"""The two-point core: axial offset, its xenon mode, and the boron line.

Contract, from sources other than the code:

- Mastroianni (VT thesis, 1979), the Onega and Kisner two-point model, eqs 2.15 to
  2.18, Table III-1 for its numbers. The flux tilt is algebraic in the xenon tilt,
  top and bottom are the two halves of the core, and the Oconee II test at 75 %
  power was a 0.18 %dk/k rod move, held 2.5 h: flux amplitude 1.19 at its end,
  turning at about 23, 38 and 51.5 h.
- Pedersen, Demaziere and Vinai, Prog. Nucl. Energy 202 (2027): the axial offset of
  Ringhals-4 oscillates at 28.3 h with a decay of 0.05 per hour.
- Barbier et al. (OAPS, ICAPP 2025): boron reaches the core 5 to 15 min late, at
  10 kg/s for dilution and 3 kg/s for boration, and that limits Mode A to 1 to 2
  %/min at the beginning of a cycle.
- Krsko (NENE 2022, paper 911): AO = (PT - PB) / (PT + PB), delta I = PT - PB, the
  band is +-5 % of rated, and a down-power shifts the power toward the top.
- The APR1400 paper (KNS 24A-359): soluble boron has a negligible effect on the
  axial power index.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.reactor import Reactor, ReactorParams
from target_gym.reactor.env import (
    LAMBDA_IODINE,
    LAMBDA_XENON,
    flux_tilt,
    moderator_tilt,
    rod_tilt,
    rod_worth,
    tilt_rates,
    xenon_stiffness,
)


# The thesis's linear model, written out from its printed equations (2.15, 2.16,
# 2.18) with its own numbers, so that it shares nothing with the env's code.
def thesis_mode(sigma_x, flux=2.02e13):
    lx, li = LAMBDA_XENON, LAMBDA_IODINE
    gx, gi, Sf = 0.0046, 0.061, 0.14
    X0 = (gi + gx) * Sf * flux / (lx + sigma_x * flux * np.pi / 4)
    S = sigma_x * X0
    stiffness = 45 * np.pi**3 / (8 * 365.8**2) * 0.405 - S + 7 * flux * 3.64e-16 * 0.39
    tilt = -4 * S / stiffness  # A = tilt * B, small B, from A = v - sqrt(v^2 + 1)
    g, h, b = gx * Sf * flux / X0, gi * Sf * flux / X0, sigma_x * flux
    matrix = np.array([[g * tilt - lx - 2 / 3 * b * (tilt + 1), h], [li * tilt, -li]])
    root = np.linalg.eigvals(matrix)
    root = root[np.argmax(root.real)]
    return root.real * 3600, 2 * np.pi / abs(root.imag) / 3600


def env_mode(n, sigma_x=2.72e-18, **changes):
    """Growth (1/h) and period (h) of the env's xenon tilt at a level ``n``, read off
    its rates by a finite difference around the balanced core."""
    p = ReactorParams(
        sigma_x=sigma_x, sigma_phi0=sigma_x * 2.02e13 / 0.75 * np.pi / 4, **changes
    )
    lam = LAMBDA_XENON + p.sigma_phi0
    mean_xe = float(lam * n / (LAMBDA_XENON + p.sigma_phi0 * n))

    def rate(c, b):
        return np.array(tilt_rates(n, n, mean_xe, c, b, 0.0, p))

    h = 1e-4
    matrix = np.stack([(rate(h, 0) - rate(-h, 0)) / 2 / h,
                       (rate(0, h) - rate(0, -h)) / 2 / h], axis=1)  # fmt: skip
    root = np.linalg.eigvals(matrix)
    root = root[np.argmax(root.real)]
    return root.real * 3600, 2 * np.pi / abs(root.imag) / 3600


def test_thesis_equations_reproduce_the_oconee_period_and_the_printed_boundary():
    """Sigma_a = nu x Sigma_f = 2.44 x 0.16 reproduces the thesis's own results,
    which its printed 1.53 does not: undamped near 3.02e-18 cm2 at about 27 to 31 h
    (section 3.3), and 28.5 h from the plant's two troughs."""
    _, period = thesis_mode(2.72e-18)
    assert 27.0 < period < 31.0
    sigmas = np.linspace(2.6e-18, 3.4e-18, 161)
    growth = np.array([thesis_mode(s)[0] for s in sigmas])
    assert sigmas[np.argmin(np.abs(growth))] == pytest.approx(3.02e-18, rel=0.04)


THESIS = {"D_diffusion": 0.405}  # Table III-1; the env now couples the halves harder


def test_env_tilt_matches_the_thesis_at_its_own_flux():
    # Same flux as the test: 75 % of full, on the thesis's own constants. The env's
    # two nodes conserve the mean, which the thesis fixes by hand, so the decay is
    # checked by its sign and the period to 3 %.
    growth, period = env_mode(0.75, **THESIS)
    want_growth, want_period = thesis_mode(2.72e-18)
    assert period == pytest.approx(want_period, rel=0.03)
    assert growth < 0 and want_growth < 0


def test_full_power_mode_is_a_damped_oscillation_near_the_published_figures():
    # Owner's decision of 2026-11-08 (docs/proposals/reactor-coupling.md): with the
    # halves coupled as Ringhals-4's 283 pcm, the period moves from the measured 28.3 h
    # into the range the Chalmers models give, 31.5 to 39.3 h.
    growth, period = env_mode(1.0)
    assert 28.3 < period < 39.3
    assert -0.08 < growth < -0.02  # 0.05 per hour, Ringhals-4


def test_the_halves_are_coupled_as_ringhals_4():
    """The first axial harmonic's eigenvalue separation, one group, cosine modes:
    D x 3 pi^2 / (H^2 nu Sigma_f), 283 pcm in Pedersen et al.'s homogeneous model
    (Progress in Nuclear Energy 202, table 3)."""
    p = ReactorParams()
    evs = p.D_diffusion * 3.0 * np.pi**2 / (p.H_core**2 * p.nu_Sigma_f)
    assert evs * 1e5 == pytest.approx(283.0, rel=1e-3)


def test_half_power_mode_is_damped():
    """Lowering the power weakens the feedback that holds the mode (DP-1239). With
    the halves coupled as Ringhals-4 the half-power mode still decays."""
    growth, _ = env_mode(0.5)
    assert growth < 0


def test_flux_tilt_matches_the_printed_root():
    """A = v - sqrt(v^2 + 1), v = stiffness / (8 S B): eq 2.17 at no disturbance."""
    p = ReactorParams(**THESIS)
    n, mean_xe, B = 0.8, 0.95, 0.07
    S = xenon_stiffness(p)
    N = 45 * np.pi**3 / (8 * 365.8**2) * 0.405 - S * mean_xe
    N = N + 7 * n * p.Phi_full * 3.64e-16 * (2.44 * 0.16)
    v = N / (8 * S * B)
    assert float(flux_tilt(B, mean_xe, n, 0.0, p)) == pytest.approx(
        v - np.sqrt(v * v + 1), rel=1e-5
    )


def test_oconee_pulse_leaves_the_plants_tilt():
    """The test at 75 % power: a 0.25 % change of absorption, up in one half and
    down in the other, for 2.5 h, standing in for the rod move. Once it is off, the
    flux amplitude read 1.19 (fig. III-1), so a tilt of 0.19 either way."""
    p = ReactorParams(**THESIS)
    n, mean_xe = 0.75, 0.918  # the balanced xenon at 75 %
    eps = 0.0025 * p.nu_Sigma_f
    c, b = 0.0, 0.0
    for _ in range(150):  # 2.5 h of 60 s Euler steps
        dc, db = tilt_rates(n, n, mean_xe, c, b, eps, p)
        c, b = c + 60.0 * float(dc), b + 60.0 * float(db)
    assert abs(float(flux_tilt(b, mean_xe, n, 0.0, p))) == pytest.approx(0.19, abs=0.04)


def test_tilt_is_odd_and_stays_a_fraction():
    p = ReactorParams()
    up = float(flux_tilt(0.2, 1.0, 1.0, 0.0, p))
    down = float(flux_tilt(-0.2, 1.0, 1.0, 0.0, p))
    assert up == pytest.approx(-down, rel=1e-5)
    assert up < 0
    assert -1.0 < float(flux_tilt(5.0, 1.0, 0.5, 0.0, p)) < 0.0


def test_rod_worth_is_the_s_curve():
    """Integral worth y - sin(2 pi y) / (2 pi): half at half the stroke, and flat at
    the ends, so a shallow bank is worth little."""
    p = ReactorParams()
    full = float(rod_worth(1.0, p))
    assert full == pytest.approx(p.rho_ext_min, rel=1e-5)
    assert float(rod_worth(0.5, p)) == pytest.approx(0.5 * full, rel=1e-5)
    assert float(rod_worth(0.0, p)) == pytest.approx(0.0, abs=1e-9)
    assert abs(float(rod_worth(0.1, p))) < 0.02 * abs(full)


XE_FULL = {"initial_xenon_power_range": (1.0, 1.0)}


def _reset(load_flat=True, **changes):
    env = Reactor()
    p = ReactorParams(demand_sigma=0.0, **changes)
    _, s = env.reset_env(jax.random.PRNGKey(3), p)
    if load_flat:
        far = 1e7
        s = s.replace(ramp_down_at=jnp.float32(far), ramp_up_at=jnp.float32(far))
    return env, p, s


def _run(env, p, s, action, steps):
    action = jnp.asarray(action, jnp.float32)

    def step(s, _):
        _, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, action, p)
        return s, s

    return jax.lax.scan(step, s, None, length=steps)


def _hold(s, p):
    """The action that leaves the rods where they are, with no boron."""
    return [1.0 - 2.0 * float(s.rod_in), 0.0]


def test_reset_is_balanced_with_the_rods_parked():
    env, p, s = _reset(**XE_FULL)
    assert float(s.rod_in) == pytest.approx(3.0 / 228.0, rel=1e-3)  # 225 steps
    assert float(s.boron) == pytest.approx(p.boron_ref, rel=1e-3)
    end, _ = _run(env, p, s, _hold(s, p), 120)
    assert float(end.n) == pytest.approx(1.0, abs=2e-3)
    assert abs(float(end.delta_I)) < 1e-3


def test_rod_insertion_pushes_power_down_the_core_and_boron_leaves_it():
    env, p, s = _reset(**XE_FULL)
    # 40 steps in over 10 minutes, an hour held: Tavg falls and the offset goes negative.
    rods, _ = _run(env, p, s, [1.0 - 2.0 * 43.0 / 228.0, 0.0], 120)
    assert float(rods.delta_I) < -0.02
    # The same reactivity by boron, against the rods' 30 min: the offset barely moves.
    boron, _ = _run(env, p, s, [_hold(s, p)[0], 0.3], 120)
    assert float(boron.boron) > float(s.boron) + 1.0
    assert abs(float(boron.delta_I)) < 0.1 * abs(float(rods.delta_I))


@pytest.mark.parametrize("delay", [10, 30])
def test_boron_reaches_the_core_late(delay):
    env, p, s = _reset(**XE_FULL)
    s = s.replace(boron_delay=jnp.int32(delay))
    before, _ = _run(env, p, s, [_hold(s, p)[0], 1.0], delay - 1)
    assert float(before.boron) == pytest.approx(float(s.boron), abs=1e-3)
    after, _ = _run(env, p, s, [_hold(s, p)[0], 1.0], delay + 2)
    assert float(after.boron) > float(s.boron)


def test_dilution_loses_strength_with_the_concentration():
    """Feed and bleed removes a fraction of what is there (OAPS, section 2.1)."""
    env, p, s = _reset(**XE_FULL)
    s = s.replace(boron_delay=jnp.int32(1))
    high, _ = _run(env, p, s, [_hold(s, p)[0], -1.0], 20)
    env, half, low_state = _reset(boron_ref=0.5 * p.boron_ref, **XE_FULL)
    low_state = low_state.replace(boron_delay=jnp.int32(1))
    low, _ = _run(env, half, low_state, [_hold(s, half)[0], -1.0], 20)
    fall_high = float(s.boron) - float(high.boron)
    fall_low = float(low_state.boron) - float(low.boron)
    assert fall_low == pytest.approx(0.5 * fall_high, rel=0.05)


def test_flow_limits_give_the_published_mode_a_rate():
    """At 1690 ppm, 10 kg/s of water into 250 t takes out about 32 pcm/min, so a
    power defect near 1000 pcm covers 50 % of power at about 2 %/min."""
    p = ReactorParams()
    pcm_per_min = (
        p.dilution_flow / p.primary_mass * p.boron_ref * abs(p.boron_worth) * 1e5 * 60
    )
    rate = 50.0 / (961.0 / pcm_per_min)
    assert 1.0 <= rate <= 2.5


def test_part_load_shifts_the_offset_up_the_core_near_the_published_state():
    """Krsko figs 3, 5, 6: at 50 % power, D bank at 194 steps, xenon at 110 of 96,
    delta I near +5 %. The coolant's smaller rise pushes power up and the bank
    pushes it down, so the sign is the published one. The model lands near twice
    the figure, so the check is the sign and a factor of three."""
    p = ReactorParams()
    n = 0.5
    rise = p.P_thermal_ref * n / p.m_dot_cp
    eps = rod_tilt(34.0 / 228.0, p) + moderator_tilt(rise, 0.0, p)
    delta_I = n * float(flux_tilt(0.0, 110.0 / 96.0, n, eps, p))
    assert 0.05 / 3.0 < delta_I < 0.05 * 3.0


def test_reward_charges_past_the_band_only():
    env, p, s = _reset(**XE_FULL)
    terms = env.reward_terms(s.replace(delta_I=jnp.float32(0.049)), p)
    outside = env.reward_terms(s.replace(delta_I=jnp.float32(-0.06)), p)
    base = env.reward_terms(s, p)
    assert float(terms["tracking"]) == pytest.approx(float(base["tracking"]))
    # 1 % past the band: 1 % of rated derated, $1000/h, over a 5 s sub-step.
    extra = float(outside["tracking"]) - float(base["tracking"])
    assert extra == pytest.approx(1000.0 * 5.0 / 3600.0, rel=1e-3)


# Contract: the reference's rod loop on delta I is stable at every point the day
# holds. Its plant settles inside a step, so the loop is discrete: with g the gain
# times Kc, z^2 - (1 - 2g) z - g = 0 reaches -1, a swing of delta I every other step,
# at g = 2/3. A swing over the band (0.05) between two steps is that cycle, since
# nothing else in the plant moves delta I that fast. The day's hard point is the
# half-power hold; seeds 100, 102 and 103 cycled there under a full-power tune.
@pytest.mark.parametrize("seed", [100, 102, 103])
def test_the_reference_does_not_cycle_the_offset_in_the_half_power_hold(seed):
    from target_gym.reactor.experts import make_reactor_pid_controller

    env = Reactor()
    p = ReactorParams(demand_sigma=0.0, sensor_noise_scale=0.0)
    expert = make_reactor_pid_controller()
    obs, s = env.reset_env(jax.random.PRNGKey(seed), p)

    def step(carry, key):
        s, obs, memory = carry
        a, memory = expert.act(memory, obs)
        obs, s, _, _, _ = env.step_env(key, s, a, p)
        return (s, obs, memory), s.delta_I

    keys = jax.random.split(jax.random.PRNGKey(seed + 7), 1440)  # half a day
    di = np.asarray(jax.lax.scan(step, (s, obs, expert.init_state), keys)[1])
    assert np.max(np.abs(np.diff(di))) < p.ao_band
