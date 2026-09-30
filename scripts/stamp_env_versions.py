"""Record what each environment version currently means.

An environment's version is a promise: a number published against `plane-v1`
should keep meaning what it meant. That promise is only worth something if
changing the environment without bumping the version is *caught*, which is what
`src/target_gym/data/env_versions.json` and `tests/test_env_versions.py` are for.

The stamp is `provenance.environment_fingerprint`, which covers the
environment's own modules, the shared physics and the parameter values, but not
the controllers. Re-tuning a PID does not change what the environment is.

    uv run python scripts/stamp_env_versions.py            # stamp / re-stamp
    uv run python scripts/stamp_env_versions.py --check    # CI: fail if stale

When the test reports a mismatch you have two honest options, and the note field
is where you say which one you took:

1. The behaviour changed. Bump `version` on the `EnvSpec`, then re-stamp. The
   old entry stays in the file as a record of what that version was.
2. The behaviour cannot have changed, and the fingerprint moved for another
   reason: a rename, a refactor, a constant pulled into a shared module. The
   fingerprint is taken over source, so it reports staleness that is not real
   rather than freshness that is not real. Re-stamp *in place* and write a note
   saying why the change is behaviour-preserving.

Never re-stamp in place to make a red test green without deciding which of those
two it is. That is the only way this file can lie.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from target_gym.provenance import environment_fingerprint  # noqa: E402
from target_gym.registry import REGISTRY  # noqa: E402

PATH = ROOT / "src" / "target_gym" / "data" / "env_versions.json"


def current() -> dict[str, str]:
    return {
        spec.versioned_name: environment_fingerprint(spec) for spec in REGISTRY.values()
    }


def load() -> dict:
    if not PATH.exists():
        return {}
    return json.loads(PATH.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if any stamp is stale")
    ap.add_argument("--note", default="", help="why a re-stamp is behaviour-preserving")
    args = ap.parse_args()

    now, stored = current(), load()

    if args.check:
        bad = [
            f"{name}: recorded {stored[name]['fingerprint']}, tree has {fp}"
            for name, fp in now.items()
            if name in stored and stored[name]["fingerprint"] != fp
        ]
        missing = [name for name in now if name not in stored]
        if bad or missing:
            for line in bad:
                print(f"  changed: {line}")
            for name in missing:
                print(f"  unstamped: {name}")
            print(f"\nrun {pathlib.Path(__file__).name}, and read its docstring first")
            return 1
        print(f"{len(now)} environment versions are current")
        return 0

    out = dict(stored)
    stamped = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    for name, fp in now.items():
        previous = stored.get(name, {})
        if previous.get("fingerprint") == fp:
            continue
        entry = {"fingerprint": fp, "stamped": stamped}
        if args.note:
            entry["note"] = args.note
        elif previous:
            entry["note"] = "re-stamped without a note"
        out[name] = entry
        print(f"  stamped {name}")

    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {PATH.relative_to(ROOT)} ({len(out)} versions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
