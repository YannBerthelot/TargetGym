"""The statistics an agent's verdict is read with (charter J4, docs/rl-protocol.md).

Contract, from the owner's decision of 2026-11-27 ("Fixed seeds, run-level
bootstrap") on the outside review (docs/proposals/rl-research-review.md):

- Selection happens on a validation block, episode seeds 300 to 349 in every
  scenario, apart from the scoring seeds 200 to 209 and every other set (M6).
- One policy against the reference on the scoring seeds: the saving
  ``1 - J_policy / J_reference``, ``J = sum_s w_s mean_s``, with a 95 % t interval on
  the paired per-seed values (H3). With no real saving and a 5 % spread per seed,
  one look declares the policy cheaper 2.5 % of the time, within Monte Carlo error,
  and at most that on the review's own simulation, where the review measured 5.5 %
  for the percentile bootstrap it replaces.
- An agent is its training runs, each scored on the same seeds: the saving of the
  mean over runs, with a two-level bootstrap over runs and then seeds within each
  scenario, and each run's saving beside it (H4). Its false-win rate stays at the
  nominal 2.5 % or under, whether the runs or the seeds carry the spread.
- Scenarios run on the same seeds are resampled together (L1).
- A cost may be negative; the saving is defined while ``J_reference`` is above 0 (H2).
- A seed within the tie tolerance, the task's noise floor, is a tie, never a win (H3).
- Valve travel and reversals are reported beside the headline as ratios and never
  decide it (M8). pH's off-spec band is its permit (M7).

Expected values are worked by hand in each test, or are the nominal rates.
"""

import math

import numpy as np
import pytest

from target_gym import charter, scoring
from target_gym import eval as E
from target_gym.registry import REGISTRY
from target_gym.scoring import headline

FIVE = {"normal": 0.7, "a": 0.1, "b": 0.1, "c": 0.05, "d": 0.05}
NOMINAL = 0.025  # one side of a 95 % interval


def side(policy, reference, **extra):
    return {"policy": policy, "reference": reference, **extra}


def within_mc(rate, sims, nominal=NOMINAL):
    """Within three Monte Carlo standard errors of ``nominal``."""
    return abs(rate - nominal) <= 3.0 * math.sqrt(nominal * (1 - nominal) / sims)


def test_validation_seeds_are_a_block_of_their_own():
    seeds = charter.VALIDATION_SEEDS
    assert seeds == tuple(range(300, 350))
    assert not set(seeds) & set(charter.ANCHOR_SEEDS)
    assert not set(seeds) & set(range(100, 110))  # the ladder's, J1 and J3
    assert max(seeds) < charter.J2_SEED_FLOOR  # J2's hash seeds


def _false_wins(weights, gap, sims=2000):
    """The share of ``sims`` looks at ten seeds where the interval puts the policy
    cheaper, with no real saving: the reference's per-seed cost lognormal, the
    policy's that cost moved by ``gap(b, d)``, ``d`` normal with sd 5 %."""
    rng, wins = np.random.default_rng(12345), 0
    for _ in range(sims):
        costs = {}
        for s in weights:
            b = rng.lognormal(0.0, 0.5, 10)
            costs[s] = side(gap(b, rng.normal(0.0, 0.05, 10)), b)
        wins += headline(costs, weights)["saving_ci"][0] > 0.0
    return wins / sims


@pytest.mark.parametrize("weights", [{"s": 1.0}, FIVE], ids=["one", "five"])
def test_one_look_at_the_scoring_seeds_declares_a_false_win_at_the_nominal_rate(
    weights,
):
    # Paired differences normal, sd 5 % of a reference of order 1: the t interval's
    # own assumption, where it is exact.
    rate = _false_wins(weights, lambda b, d: b + d)
    assert within_mc(rate, 2000), f"false wins {rate:.4f}"


@pytest.mark.parametrize("weights", [{"s": 1.0}, FIVE], ids=["one", "five"])
def test_the_reviews_simulation_declares_a_false_win_at_most_at_the_nominal_rate(
    weights,
):
    # The review's design: the policy's cost the reference's times 1 + d. The
    # differences then grow with the seed's cost, which the t interval meets on the
    # safe side (2.1 % in 20,000 looks); the percentile bootstrap it replaces gave
    # 5.5 % on one scenario and 4.9 % on five.
    rate = _false_wins(weights, lambda b, d: b * (1.0 + d))
    assert rate <= NOMINAL or within_mc(rate, 2000), f"false wins {rate:.4f}"


@pytest.mark.medium
@pytest.mark.parametrize(
    "run_sd, seed_sd, noise_sd",
    [(0.05, 0.0, 0.05), (0.01, 0.05, 0.01)],
    ids=["runs carry the spread", "seeds carry the spread"],
)
def test_thirty_runs_declare_a_false_win_at_most_at_the_nominal_rate(
    run_sd, seed_sd, noise_sd
):
    # Thirty runs on the same ten seeds of five scenarios, no real saving: each
    # run's own offset, each seed's offset shared by every run, and noise.
    sims, rng, wins = 1000, np.random.default_rng(7), 0
    for _ in range(sims):
        costs, run = {}, rng.normal(0.0, run_sd, (30, 1))
        for s in FIVE:
            b = rng.lognormal(0.0, 0.5, 10)
            shift = run + rng.normal(0.0, seed_sd, 10)
            costs[s] = side(b * (1.0 + shift + rng.normal(0.0, noise_sd, (30, 10))), b)
        wins += headline(costs, FIVE)["saving_ci"][0] > 0.0
    rate = wins / sims
    assert rate <= NOMINAL or within_mc(rate, sims), f"false wins {rate:.4f}"


def test_an_agents_saving_is_the_mean_over_its_runs_with_each_runs_beside_it():
    # Two runs on two seeds, the reference 10 on both. Run 1 costs 8 and 9 (mean
    # 8.5, saving 0.15), run 2 costs 10 and 11 (mean 10.5, saving -0.05). The
    # agent's J is (8.5 + 10.5) / 2 = 9.5, a saving of 0.05.
    result = headline({"s": side([[8.0, 9.0], [10.0, 11.0]], [10.0, 10.0])}, {"s": 1})
    assert result["saving"] == pytest.approx(0.05)
    assert result["run_savings"] == pytest.approx([0.15, -0.05])


def test_runs_cheaper_by_the_same_share_on_every_seed_have_a_tight_interval():
    # Every run costs 0.9 of the reference on every seed: any resample of runs
    # and seeds gives exactly 0.1.
    reference = np.array([8.0, 10.0, 12.0, 15.0])
    runs = np.tile(0.9 * reference, (5, 1))
    result = headline({"s": side(runs, reference)}, {"s": 1.0})
    assert result["saving_ci"] == pytest.approx((0.1, 0.1))


@pytest.mark.parametrize("runs", [None, 3], ids=["one policy", "three runs"])
def test_scenarios_that_share_seeds_are_resampled_together(runs):
    # The reference costs 1 everywhere. On seed i the policy costs 1 + e_i in one
    # scenario and 1 - e_i in the other, at equal weights, so each seed's weighted
    # cost is the reference's: every joint resample gives a saving of exactly 0.
    # Resampling each scenario on its own would not cancel them.
    e = np.array([0.3, -0.1, 0.2, -0.4, 0.0, 0.1, -0.2, 0.3, -0.1, -0.1])
    up, down = 1.0 + e, 1.0 - e
    if runs:
        up, down = np.tile(up, (runs, 1)), np.tile(down, (runs, 1))
    ones = np.ones(10)
    result = headline({"a": side(up, ones), "b": side(down, ones)}, {"a": 1, "b": 1})
    assert result["saving_ci"] == pytest.approx((0.0, 0.0), abs=1e-9)


def test_a_negative_cost_is_scored_while_the_reference_costs_more_than_nothing():
    # hvac's edge charge left one seed of the expert below 0 (H2). J_reference =
    # (0.006 - 0.0002 + 0.005) / 3 and J_policy = (0.005 - 0.0004 + 0.004) / 3, so
    # the saving is 1 - 8.6 / 10.8.
    result = headline(
        {"normal": side([0.005, -0.0004, 0.004], [0.006, -0.0002, 0.005])},
        {"normal": 1.0},
    )
    assert result["saving"] == pytest.approx(1 - 8.6 / 10.8)
    low, high = result["saving_ci"]
    assert low < result["saving"] < high


def test_a_reference_that_costs_less_than_nothing_leaves_no_saving():
    # J_reference = (0.1 - 0.3) / 2 = -0.1: the saving is not defined.
    result = headline({"s": side([0.1, -0.2], [0.1, -0.3])}, {"s": 1.0})
    assert math.isnan(result["saving"])


def test_a_seed_within_the_tie_tolerance_is_a_tie_not_a_win():
    reference = np.array([10.0, 10.0, 10.0, 10.0])
    # 0.5 % cheaper on every seed against a 1 % tolerance: four ties, no win.
    tied = headline({"s": side(0.995 * reference, reference)}, {"s": 1.0}, tie=0.01)
    assert tied["probability_of_improvement"] == 0.0
    assert tied["ties"] == 1.0
    # 2 % cheaper on two seeds and 2 % dearer on two: half won, none tied.
    split = reference * np.array([0.98, 0.98, 1.02, 1.02])
    mixed = headline({"s": side(split, reference)}, {"s": 1.0}, tie=0.01)
    assert mixed["probability_of_improvement"] == 0.5
    assert mixed["ties"] == 0.0


def _job_episode(cost):
    n = len(cost)
    return E.Episode(
        cost=np.asarray(cost, float),
        in_band=np.ones(n, bool),
        target_change=np.zeros(n, bool),
        failed=np.zeros(n, bool),
        terms={"energy": np.asarray(cost, float)},
    )


def test_a_controller_against_itself_through_another_path_wins_no_seed():
    # The review: score_policy gave the cstr expert a win rate of 1 against its own
    # pid slot, the two within 1e-6 of each other on every seed. Under a 1e-4 tie
    # tolerance those seeds are ties.
    nan = float("nan")
    expert = [_job_episode(np.full(50, 4.0)) for _ in range(3)]
    policy = [_job_episode(np.full(50, 4.0 * (1 - 1e-6))) for _ in range(3)]
    m = scoring.compare(policy, expert, nan, burn_in=10, settle=5, band=nan, tie=1e-4)
    assert m["win_rate"] == 0.0


def _weights(task):
    from target_gym import scenarios, scoring_runs

    if task in scoring_runs.PROTOCOLS:  # the reactor's folded twins
        return scoring_runs.weights(task)
    return scenarios.weights(REGISTRY[task])


@pytest.mark.parametrize("task", charter.FLAGSHIP)
def test_each_scored_task_publishes_the_noise_floor_it_ties_on(task):
    floor = charter.noise_floor(task)
    assert floor != charter.NOT_MEASURED, f"{task}: no noise floor measured"
    # The floor is the largest relative gap, over every scenario's scoring seeds,
    # between one controller's costs through two code paths.
    gaps = floor["per_scenario"]
    assert set(gaps) == set(_weights(task))
    assert all(len(v) == len(charter.ANCHOR_SEEDS) for v in gaps.values())
    assert floor["floor"] == max(max(v) for v in gaps.values())
    assert charter.tie_tolerance(task) == floor["floor"]


def test_travel_and_reversals_are_reported_beside_the_headline_and_decide_nothing():
    # normal (0.75): travel 4 against 2; upset (0.25): 8 against 2. The ratio is
    # (0.75 * 4 + 0.25 * 8) / (0.75 * 2 + 0.25 * 2) = 5 / 2. Reversals 30 against 3
    # in both: 10. The policy is 10 % cheaper everywhere and trips no more, so it
    # beats the reference whatever it does to the valve.
    reference = np.array([8.0, 10.0, 12.0, 15.0])

    def scenario(travel):
        return side(
            0.9 * reference,
            reference,
            policy_travel=[travel] * 4,
            reference_travel=[2.0] * 4,
            policy_reversals=[30.0] * 4,
            reference_reversals=[3.0] * 4,
            policy_trips=[0] * 4,
            reference_trips=[0] * 4,
        )

    result = headline(
        {"normal": scenario(4.0), "upset": scenario(8.0)},
        {"normal": 0.75, "upset": 0.25},
    )
    assert result["travel_ratio"] == pytest.approx(2.5)
    assert result["reversal_ratio"] == pytest.approx(10.0)
    assert result["beats_reference"] is True


@pytest.mark.parametrize("low, high, band", [(6.0, 9.0, 1.5), (5.5, 9.5, 2.0)])
def test_phs_off_spec_band_is_its_permit(low, high, band):
    # The observation's target is the permit's centre, so a half-width of
    # (high - low) / 2 about it is the permit itself.
    spec = REGISTRY["ph_neutralization"]
    params = spec.make_test_params().replace(permit_low=low, permit_high=high)
    np.testing.assert_allclose(scoring.spec_band("ph_neutralization", params), [band])


@pytest.mark.records
@pytest.mark.parametrize("task", charter.FLAGSHIP)
def test_a_noise_floor_describes_the_plant_and_expert_as_they_are(task):
    # Measured on another plant or with other gains, the floor is stale:
    # re-measure with ``scripts/record_charter.py --tasks <task> --noise-floor``.
    from target_gym.provenance import environment_fingerprint, expert_fingerprint

    floor, spec = charter.noise_floor(task), REGISTRY[task]
    assert floor["stamp"] == environment_fingerprint(spec)
    assert floor["expert_fingerprint"] == expert_fingerprint(spec)


@pytest.mark.medium
def test_the_anchor_line_records_each_seeds_travel_and_reversals():
    # M8 needs the reference's side: ``scripts/record_charter.py``'s anchor part
    # keeps, on each scoring seed, the full strokes and the changes of direction
    # beside the cost, and the line still validates. On the fixture task.
    import importlib
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    record = importlib.import_module("record_charter")
    ladder = importlib.import_module("reference_ladder")
    task = record.Task("first_order", {})
    task.record = {"reference": {"kept": next(iter(ladder.build(task.name, task.row)))}}
    block = record.part_anchor(task)
    for v in block["line"]["scenarios"].values():
        n = len(charter.ANCHOR_SEEDS)
        assert len(v["costs"]) == len(v["travel"]) == len(v["reversals"]) == n
        assert all(x >= 0 for x in v["travel"]) and all(r >= 0 for r in v["reversals"])
    weights = task.weights
    entry = {"task": task.name, "version": {}, "weights": weights}
    assert charter.validate(entry | {"anchor": block | {"fingerprint": "f"}}) == []
