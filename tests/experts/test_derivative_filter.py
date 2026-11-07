"""The shared PID's derivative filter.

Contract, from docs/proposals/measurement-noise.md and the textbook form
(Astrom and Hagglund, 2006): the derivative term is ``Kd s / (1 + T_f s)``,
with ``T_f = T_d / 10`` and ``T_d = |Kd / Kp|``, discretised by backward
differences over a step ``h``:

    D[k] = T_f / (T_f + h) * D[k-1] + Kd / (T_f + h) * (e[k] - e[k-1])

Every expected value below is worked from that recursion by hand.
"""

import jax
import numpy as np
import pytest

from target_gym.experts.pid import filtered_derivative, siso_pid


def _derivative_terms(errors, Kp, Kd, dt):
    """What the loop adds to ``Kp * e``, read through the public controller."""
    pid = siso_pid(Kp, 0.0, Kd, dt, 0, 1, action_min=-1e9, action_max=1e9).host()
    pid.reset()
    return [
        float(pid(np.array([0.0, e], dtype=np.float32))[0]) - Kp * e for e in errors
    ]


def test_a_step_in_the_error_decays_geometrically():
    """Kp = 2, Kd = 1, h = 0.5: T_d = 0.5, T_f = 0.05, T_f + h = 0.55.

    D[1] = 1 / 0.55 = 1.81818; D[k] = D[k-1] * 0.05 / 0.55 = D[k-1] / 11.
    The unfiltered term would be 2 and then 0.
    """
    terms = _derivative_terms([1.0, 1.0, 1.0], 2.0, 1.0, 0.5)
    np.testing.assert_allclose(terms, [1.81818, 0.165289, 0.0150263], rtol=1e-4)


def test_a_ramp_reaches_the_derivative_it_is_filtering():
    """On e = k * h the true derivative is 1. The term is ``Kd`` times that once
    the filter's transient has gone: a filter on D must not bias a steady slope.
    Kp = 1, Kd = 2, h = 1: T_f = 0.2, and the transient decays by 1/6 a step."""
    terms = _derivative_terms([float(k) for k in range(1, 12)], 1.0, 2.0, 1.0)
    assert terms[-1] == pytest.approx(2.0, rel=1e-4)


def test_white_noise_on_the_error_is_attenuated_by_the_filter_gain():
    """Measured white noise, the reason for the filter. For ``T_f = h`` the
    recursion's pole is a = 0.5 and its variance gain is
    ``(Kd / (T_f + h))^2 * 2 / (1 + a)``: one sixth of the raw difference's
    ``2 (Kd / h)^2``. 20000 draws put the sample variance within
    ``4 sqrt(2 / 20000)``, 6%, ignoring the filter's small autocorrelation."""
    h = 1.0
    Kd = 1.0
    Kp = Kd / (10.0 * h)  # T_d = 10 h, so T_f = h
    errors = np.random.default_rng(0).standard_normal(20001)
    terms = np.array(_derivative_terms(errors.tolist(), Kp, Kd, h))[1000:]
    raw_variance = 2.0 * (Kd / h) ** 2
    assert terms.var() == pytest.approx(raw_variance / 6.0, rel=0.06)


def test_without_derivative_action_the_loop_is_the_pi_it_was():
    """``first_order``, ``glass_furnace`` and ``hvac`` run Kd = 0."""
    assert _derivative_terms([1.0, -2.0, 0.5], 2.0, 0.0, 0.5) == [0.0, 0.0, 0.0]


def test_no_derivative_time_means_no_nan():
    """Kp = 0 gives no derivative time. Both gains at 0 is a gain policy's
    corner, and the loop must stay finite there."""
    terms = _derivative_terms([1.0, 2.0], 0.0, 0.0, 1.0)
    assert np.all(np.isfinite(terms))
    derivative, filtered = filtered_derivative(1.0, 0.0, 0.0, 0.0, 1.0)
    assert np.isfinite(derivative) and np.isfinite(filtered)


def test_sign_of_the_gains_does_not_change_the_filter_time():
    """Negative gains occur on reverse-acting loops. T_d is a time, so the
    filter uses its size: the term flips with Kd and keeps its shape."""
    plus = _derivative_terms([1.0, 1.0], 2.0, 1.0, 0.5)
    minus = _derivative_terms([1.0, 1.0], -2.0, -1.0, 0.5)
    np.testing.assert_allclose(minus, [-t for t in plus], rtol=1e-6)


def test_the_traced_and_host_loops_stay_one_function():
    """The scored controller and the trainable one share the recurrence."""
    pid = siso_pid(2.0, 0.1, 1.0, 0.5, 0, 1, action_min=-5.0, action_max=5.0)
    host = pid.host()
    host.reset()
    state = pid.init_state
    step = jax.jit(pid.act)
    for e in [1.0, 0.5, -0.25, 2.0]:
        obs = np.array([0.0, e], dtype=np.float32)
        traced, state = step(state, obs)
        assert np.asarray(host(obs)).dtype == np.float32
        np.testing.assert_allclose(host.state.prev_error, state.prev_error, rtol=1e-6)


def test_the_filter_helper_serves_a_derivative_on_the_measurement():
    """A task's own loop calls the same helper: first call from rest, as the
    step test's first term, then one more sample of a held value."""
    derivative, memory = filtered_derivative(1.0, 0.0, 2.0, 1.0, 0.5)
    assert float(derivative) == pytest.approx(2.0 * 1.0 / 1.1, rel=1e-5)  # 1 / 0.55
    derivative, memory = filtered_derivative(1.0, memory, 2.0, 1.0, 0.5)
    assert float(derivative) == pytest.approx(2.0 / 1.1 / 11.0, rel=1e-5)
