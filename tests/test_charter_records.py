"""Every scored task's record (docs/proposals/task-charter.md, J2 to J4) is
present, schema-valid and current. The scored tasks are the flagship
(``target_gym.charter.FLAGSHIP``). The record of a task outside it, Extra or
under review, is kept but not held current (owner's decisions, 2026-11-15 for
Extra and 2026-11-24 for the rest: boiler_drum, distillation, glass_furnace
and wind_turbine left this check that day).

Marked ``records``: a stale record fails here alone, and
``scripts/record_charter.py --tasks T`` refreshes it.
"""

from __future__ import annotations

import pytest

from target_gym import charter


@pytest.mark.records
@pytest.mark.parametrize("task", charter.FLAGSHIP)
def test_each_charter_task_has_a_current_valid_record(task):
    record = charter.load(task)
    assert record is not None, f"{task}: no record in {charter.DATA}"
    assert charter.validate(record) == []
    assert charter.stale_parts(record, charter.current(task)) == []


@pytest.mark.records
@pytest.mark.parametrize("task", charter.FLAGSHIP)
def test_a_scored_task_has_its_lines_measured_and_is_accepted(task):
    """The bar a task passes to be scored (owner, 2026-11-24): its best null
    policy, its reference and its ceiling are measured, so its room is, and it
    passed acceptance (J2) with the sweep (B7) run. The MPC line alone may read
    "not measured" in 0.8.0 (owner, the same day): cstr ships no planner and
    pH's oracle is not rebuilt, and both get the line in a later update."""
    record = charter.load(task)
    assert record is not None, f"{task}: no record in {charter.DATA}"
    lines = charter.j3(record)
    unmeasured = [
        k
        for k in ("best_null", "reference", "ceiling")
        if lines[k] == charter.NOT_MEASURED
    ]
    assert not unmeasured, f"{task}: not measured: {unmeasured}"
    room = charter.room(lines["reference"], lines["ceiling"], record["weights"])
    assert room != charter.NOT_MEASURED
    j2 = record.get("j2")
    assert j2 is not None, f"{task}: acceptance (J2) has not run"
    assert j2["sweep"] != charter.NOT_MEASURED, f"{task}: the sweep has not run"
    assert j2["accepted"] is True, f"{task}: {j2['reasons']}"


#: The J3 lines' seeds (reference_ladder.SEEDS): the MPC line's costs per scenario are
#: in this order; a ceiling file lists its own seeds among them.
LINE_SEEDS = tuple(range(100, 110))
#: A replayed plan's cost per step is float32: the same plan replayed twice in the
#: benchmark's batched loop agrees to about 1e-6 relative, and a cost near zero to
#: well under 1e-9 per step.
RTOL, ATOL = 1e-5, 1e-9


@pytest.mark.records
@pytest.mark.parametrize("task", charter.FLAGSHIP)
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
