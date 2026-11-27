"""The scoring seeds are fixed (owner, 2026-11-27, "Fixed seeds, run-level
bootstrap", on the outside review's H3, docs/proposals/rl-research-review.md).

Contract: J4 scores on the ten anchor seeds 200 to 209 in every scenario, once.
The stop rule of 2026-11-24, blocks of ten seeds up to fifty, stopping at the
first block whose 95 % interval excluded 1, is dropped: five looks at an
unadjusted interval declared a false win 10 to 11.5 % of the time against a
nominal 2.5 %. No rule adds scoring seeds after a look at a result.
"""

from target_gym import charter


def test_the_scoring_seeds_are_the_ten_anchor_seeds():
    assert charter.ANCHOR_SEEDS == tuple(range(200, 210))


def test_no_rule_adds_scoring_seeds_after_a_look():
    for name in ("stop_seeds", "STOP_BLOCK", "STOP_MAX", "STOP_RESAMPLES"):
        assert not hasattr(charter, name), f"charter.{name} is the dropped stop rule"
