"""Every charter task's record (docs/proposals/task-charter.md, J2 to J4) is
present, schema-valid and current. Charter tasks: every row in
scripts/ladder_rows less the fixture first_order and the Extra tasks, plus the
reactor, whose reference is its shipped expert
(``target_gym.charter.charter_tasks``). An Extra task's record is kept but no
longer held current (owner's decision, 2026-11-15).

Marked ``records``: a stale record fails here alone, and
``scripts/record_charter.py --tasks T`` refreshes it.
"""

from __future__ import annotations

import pytest

from target_gym import charter


@pytest.mark.records
@pytest.mark.parametrize("task", charter.charter_tasks())
def test_each_charter_task_has_a_current_valid_record(task):
    record = charter.load(task)
    assert record is not None, f"{task}: no record in {charter.DATA}"
    assert charter.validate(record) == []
    assert charter.stale_parts(record, charter.current(task)) == []


#: The J3 lines' seeds (reference_ladder.SEEDS): the MPC line's costs per scenario are
#: in this order; a ceiling file lists its own seeds among them.
LINE_SEEDS = tuple(range(100, 110))
#: A replayed plan's cost per step is float32: the same plan replayed twice in the
#: benchmark's batched loop agrees to about 1e-6 relative, and a cost near zero to
#: well under 1e-9 per step.
RTOL, ATOL = 1e-5, 1e-9


@pytest.mark.records
@pytest.mark.parametrize("task", charter.charter_tasks())
def test_the_ceiling_never_costs_more_than_the_oracle_on_the_same_seed(task):
    """Charter Words "Ceiling" (owner, 2026-11-13): the ceiling plans with the true
    model, the true state and the whole future, which is all the receding-horizon
    oracle sees and more, so on each seed and scenario it costs no more than the
    oracle's own run. Skipped where either line is not measured."""
    stored, record = charter.load_ceiling(task), charter.load(task)
    mpc = (record or {}).get("mpc", {}).get("line")
    if stored is None or not isinstance(mpc, dict):
        pytest.skip(f"{task}: ceiling or MPC line not measured")
    worse = []
    for s, v in stored["scenarios"].items():
        costs = v.get("ceiling_costs", v.get("ceiling_per_seed"))
        oracle = mpc["scenarios"][s]["costs"]
        for seed, c in zip(v["seeds"], costs, strict=True):
            o = oracle[LINE_SEEDS.index(seed)]
            if c > o + RTOL * abs(o) + ATOL:
                worse.append(f"{s} seed {seed}: ceiling {c:.6g} > oracle {o:.6g}")
    assert not worse, f"{task}: " + "; ".join(worse)
