"""PJM's regulation performance score, for ``battery``'s credibility check.

Contract, from PJM Manual 12, revision 57, section 4.4.5: on 10 s samples,

    denominator = 0.5 * mean(|signal|) + 0.5 * AReg, over the interval
    score = 1 - mean(|response - signal| / denominator)

with a 10 s latency allowed. Section 4.4.8 scores from 10 minutes after the start,
and an interval is 30 minutes. Expected values are worked by hand in each test.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "pjm_regulation_score.py"
_SPEC = importlib.util.spec_from_file_location("pjm_regulation_score", _PATH)
pjm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pjm)


def test_following_the_signal_exactly_scores_one():
    signal = np.array([0.5, -0.5, 1.0, -1.0])
    assert pjm.interval_score(signal, signal, areg=1.0) == pytest.approx(1.0)


def test_doing_nothing_on_a_full_range_signal_scores_zero():
    """mean(|signal|) = 1 and AReg = 1, so the denominator is 1 and each error 1."""
    signal = np.array([1.0, -1.0, 1.0, -1.0])
    assert pjm.interval_score(np.zeros(4), signal, areg=1.0) == pytest.approx(0.0)


def test_the_denominator_blends_the_mean_signal_and_the_assignment():
    """Denominator 0.5 * 2 + 0.5 * 2 = 2, and an error of 1 over it is 0.5."""
    signal = np.full(4, 2.0)
    assert pjm.interval_score(signal - 1.0, signal, areg=2.0) == pytest.approx(0.5)


def test_a_small_signal_is_judged_against_the_assignment_too():
    """Mean |signal| 0.1, AReg 1: denominator 0.55, so an error of 0.11 costs 0.2."""
    signal = np.full(4, 0.1)
    score = pjm.interval_score(signal + 0.11, signal, areg=1.0)
    assert score == pytest.approx(0.8)


def test_the_score_has_no_floor_at_zero():
    """The manual states none: an error of 3 on a denominator of 1 scores -2."""
    assert pjm.interval_score(np.full(4, 4.0), np.ones(4), areg=1.0) == pytest.approx(
        -2
    )


def test_one_sample_of_latency_is_forgiven_and_two_are_not():
    signal = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    late_one = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    late_two = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    forgiven = {"areg": 1.0, "step_s": 10.0, "skip_s": 0.0}
    assert pjm.score(late_one, signal, **forgiven) == pytest.approx(1.0)
    assert pjm.score(late_two, signal, **forgiven) < 1.0


def test_the_first_ten_minutes_are_not_scored():
    """60 samples of 10 s are unscored, and the rest follow exactly."""
    signal = np.ones(180 + 60)
    response = signal.copy()
    response[:59] = 0.0
    assert pjm.score(
        response, signal, areg=1.0, step_s=10.0, latency=0
    ) == pytest.approx(1.0)


def test_intervals_score_on_their_own_and_are_averaged():
    """Interval means differ: 1 in the first 30 minutes, 0 in the second, so 0.5."""
    signal = np.concatenate([np.ones(60), np.ones(180), np.ones(180)])
    response = signal.copy()
    response[240:] = 0.0
    assert pjm.score(
        response, signal, areg=1.0, step_s=10.0, latency=0
    ) == pytest.approx(0.5)


def test_five_second_steps_are_read_every_ten_seconds():
    """Only every second sample counts, so the wrong ones at odd steps never show."""
    signal = np.ones(2 * (180 + 60))
    response = signal.copy()
    response[1::2] = 0.0
    assert pjm.score(
        response, signal, areg=1.0, step_s=5.0, latency=0
    ) == pytest.approx(1.0)
