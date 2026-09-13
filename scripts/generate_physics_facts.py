"""Write the computable facts into each ``PHYSICS.md``, so they cannot drift.

    uv run python scripts/generate_physics_facts.py            # write
    uv run python scripts/generate_physics_facts.py --check    # CI: exit 1 if stale

``docs/environments.md`` has never drifted, because a generator writes it and CI
checks it. Nothing else got that treatment, and the drift found by reviewing all
twenty-one environments was concentrated exactly where it was missing: two
registry comments quoting episode lengths from before an audit changed them,
eleven parameters describing a reward that does not read them, and observation
sections that had to be written by hand.

Only facts that can be *computed* go in the block. Everything a contract says
about why a number is what it is, what the plant is, what is deliberately
omitted and where the model stops being valid, stays hand-written, because none
of that can be derived and all of it is the point of the document.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "target_gym"
BEGIN = "<!-- BEGIN GENERATED FACTS -->"
END = "<!-- END GENERATED FACTS -->"


def _duration(steps: int, dt: float) -> str:
    total = steps * dt
    if total >= 7200:
        return f"{total / 3600:.1f} h"
    if total >= 120:
        return f"{total / 60:.0f} min"
    return f"{total:.0f} s"


def _tup(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


def _facts(names: list[str]) -> str:
    from target_gym.registry import REGISTRY, control_step_seconds

    rows = []
    for name in names:
        spec = REGISTRY[name]
        env = spec.make_env()
        p = spec.make_test_params()
        # Seconds per env step, not ``delta_t``: two plants keep minutes and
        # the reactor runs ten physics sub-steps per step (see
        # ``registry.control_step_seconds``).
        dt = control_step_seconds(env, p)
        steps = int(p.max_steps_in_episode)
        space = env.action_space(p)
        shape = space.shape or (1,)
        lo = np.broadcast_to(np.asarray(space.low, float), shape)
        hi = np.broadcast_to(np.asarray(space.high, float), shape)
        obs, state = env.reset_env(__import__("jax").random.PRNGKey(0), p)
        n_state = sum(
            int(np.asarray(getattr(state, f)).size)
            for f in state.__dataclass_fields__
            if f != "time" and np.asarray(getattr(state, f)).dtype.kind == "f"
        )
        rows.append(
            f"| `{name}` | {steps} | {dt:g} | {_duration(steps, dt)} | "
            f"{shape[0]} in [{lo.min():g}, {hi.max():g}] | {int(np.size(obs))} | {n_state} |"
        )

    return "\n".join(
        [
            BEGIN,
            "",
            "<!-- Written by scripts/generate_physics_facts.py. Do not edit by hand:",
            "     `make ci-docs` fails if this block does not match the code. Prose",
            "     about *why* these numbers are what they are belongs outside it. -->",
            "",
            "### Facts, generated from the code",
            "",
            "| environment | steps | step (s) | episode | action | obs | float state |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "`float state` counts the scalar and array float fields the state carries,",
            "`time` excluded; the gap between it and `obs` is what the controller cannot",
            "see. Episode lengths are `EnvSpec.test_params`, which is what the recorded",
            "baselines use.",
            "",
            END,
        ]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    from target_gym.registry import REGISTRY

    by_pkg: dict[pathlib.Path, list[str]] = {}
    for name, spec in REGISTRY.items():
        # The params class knows which module it lives in, and its package is
        # where the contract sits. Several packages serve more than one
        # environment -- the 2D aircraft three, the 3D four, patrol two -- so a
        # block can carry several rows.
        module = getattr(spec.params_cls, "_module", "")
        rel = module.replace("target_gym.", "").replace(".", "/")
        doc = (SRC / rel).parent / "PHYSICS.md"
        if doc.exists():
            by_pkg.setdefault(doc, []).append(name)

    stale = []
    for doc, names in sorted(by_pkg.items()):
        block = _facts(sorted(names))
        text = doc.read_text()
        if BEGIN in text and END in text:
            i, j = text.index(BEGIN), text.index(END) + len(END)
            new = text[:i] + block + text[j:]
        else:
            new = text.rstrip() + "\n\n---\n\n" + block + "\n"
        if new == text:
            continue
        stale.append(doc.relative_to(ROOT))
        if not args.check:
            doc.write_text(new)

    if args.check:
        for d in stale:
            print(f"  {d} is out of date")
        print(f"{len(stale)} stale block(s); run scripts/generate_physics_facts.py")
        return 1 if stale else 0
    print(f"wrote {len(stale)} block(s) across {len(by_pkg)} contracts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
