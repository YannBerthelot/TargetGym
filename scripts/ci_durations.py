"""Rewrite tests/ci_durations.json from pytest ``--durations`` reports.

CI's test jobs print every test of 3 s or more (``--durations=0
--durations-min=3``). Feed this the logs of a push run and it records each
test's setup, call and teardown together, keeping the tests of 3 s or more;
tests/conftest.py balances the shards on them. A test the logs do not mention
keeps its recorded time; ``--fresh`` drops those instead.

    gh run view <run> --log > run.log
    python scripts/ci_durations.py run.log
"""

import argparse
import json
import pathlib
import re
from collections import defaultdict

PATH = pathlib.Path(__file__).resolve().parent.parent / "tests" / "ci_durations.json"
#: Shorter tests are left to the hash: together they even out across shards.
MIN_SECONDS = 3.0
_LINE = re.compile(r"(?:^|\s)([\d.]+)s (?:setup|call|teardown)\s+(\S+::\S+)")


def parse(text):
    """Seconds per test id in one pytest report, phases summed. A test reported
    twice (two columns in one log) keeps its longest run."""
    runs = defaultdict(float)
    for line in text.splitlines():
        if m := _LINE.search(line):
            runs[m.group(2)] += float(m.group(1))
    return dict(runs)


def merge(recorded, logs):
    """``recorded`` updated with every test of the logs, then filtered."""
    out = dict(recorded)
    seen = {}
    for text in logs:
        for test, seconds in parse(text).items():
            seen[test] = max(seen.get(test, 0.0), seconds)
    out.update(seen)
    return {t: round(s, 1) for t, s in sorted(out.items()) if s >= MIN_SECONDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    recorded = {} if args.fresh else json.loads(PATH.read_text())
    merged = merge(recorded, [p.read_text(errors="replace") for p in args.logs])
    PATH.write_text(json.dumps(merged, indent=0, sort_keys=True) + "\n")
    print(f"{len(merged)} tests of {MIN_SECONDS:g} s or more in {PATH.name}")


if __name__ == "__main__":
    main()
