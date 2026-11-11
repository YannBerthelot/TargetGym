"""Every charter task's record (docs/proposals/task-charter.md, J2 to J4) is
present, schema-valid and current. Charter tasks: every row in
scripts/ladder_rows less the fixture first_order, plus the reactor, whose
reference is its shipped expert (``target_gym.charter.charter_tasks``).

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
