"""Fingerprints that say whether a recorded measurement still describes the code.

Measuring how well the shipped controllers actually *control* costs about forty
minutes of CPU, because a single aircraft MPC step optimises a 90-step rollout
fifty times over. Paying that on every merge is wasteful: the answer only moves
when the physics, the controllers or their gains move, and most merges touch
none of the three.

So the measurement is recorded to ``src/target_gym/data/baseline_returns/`` by
``scripts/record_baselines.py`` and CI compares the recorded numbers instead of
reproducing them. That trade is only safe if a stale record cannot pass
silently, which is what this module is for: each record carries a fingerprint of
everything that determines it, and the test refuses to read a record whose
fingerprint no longer matches the tree.

A record has two parts, the expert's rows (``pid_returns``, ``pid``) and the
oracle's (``mpc_returns``, ``mpc``, ``target_gym.records.PARTS``), and each
carries its own fingerprint (:func:`expert_fingerprint`,
:func:`oracle_fingerprint`), so retuning an expert re-records only expert rows
and rebuilding an oracle only oracle rows.

Why the fingerprint is taken over *source* rather than over behaviour
--------------------------------------------------------------------
The obvious alternative -- roll the plant forward a few steps and hash the
trajectory -- is more direct, and it does not survive the matrix. Those numbers
are float32 results of ``exp``, ``sin`` and ``tanh``, whose last bits are not
guaranteed identical between macOS arm64 (where a maintainer regenerates) and
Linux x86 (where CI checks), nor across libm versions. A fingerprint that
disagrees with itself across platforms would fail every CI run and teach people
to regenerate on red rather than on change, which is worse than no check.

Hashing the source has the opposite failure mode, and it is the safe one: it can
report staleness that is not real (a rename invalidates a record the behaviour
would have kept), never freshness that is not real. The cost of a false stale is
one regeneration command; the cost of a false fresh is CI certifying a claim
about code that no longer exists.

Comments and docstrings are stripped before hashing, because this repository
edits prose constantly and none of it changes a number.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent

# Inside the package, not beside it, so they resolve from a source checkout and
# from an installed wheel alike.
GAINS_PATH = _ROOT / "data" / "pid_gains.json"
#: The recorded returns, one file per task (``target_gym.records``).
BASELINES_PATH = _ROOT / "data" / "baseline_returns"

# Shared controller code, one file per part: a change here can move any task's
# expert rows (pid.py) or oracle rows (mpc.py).
_EXPERT_SOURCES = (_ROOT / "experts" / "pid.py",)
_ORACLE_SOURCES = (_ROOT / "experts" / "mpc.py",)
#: The shared solver (``target_gym.oracles``), in place of experts/mpc.py in
#: the oracle part of every task whose oracle stands on it (``oracle_data``).
_SOLVER_SOURCES = tuple(sorted((_ROOT / "oracles").glob("*.py")))
#: The keys of a data file that only describe the run that wrote it.
_RUN_KEYS = frozenset({"generated", "cpu", "seconds"})

#: The two parts of a record, each with a fingerprint of its own.
PARTS = ("expert", "oracle")

# Shared *physics*: what a learned policy's score depends on, which is a
# narrower set. A learned policy never runs a PID or an MPC, so re-tuning a
# controller must not invalidate a training run that cost GPU-hours -- but a
# change to the integrator, or to the reward helper every environment scores
# through, absolutely must.
_PHYSICS_SOURCES = (
    _ROOT / "utils.py",
    _ROOT / "integration.py",
    _ROOT / "reward.py",
    _ROOT / "base.py",
)


#: Module-level names that only document the code, skipped by the fingerprint.
_DOC_ONLY_NAMES = frozenset({"PARAM_UNITS"})


def _is_doc_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False
    return all(isinstance(t, ast.Name) and t.id in _DOC_ONLY_NAMES for t in targets)


def _stable_ast_digest(source: str) -> str:
    """Hash a module's structure, ignoring comments, docstrings and formatting.

    ``ast.dump`` is not used: its output gained fields between Python versions
    (``type_params`` in 3.12, for one), so the same file would fingerprint
    differently across the CI matrix. This walks the tree and emits only node
    type names and the handful of value-bearing fields, which have been stable
    across every version this package supports.
    """
    tree = ast.parse(source)
    out: list[str] = []

    def emit(node: ast.AST) -> None:
        out.append(type(node).__name__)
        if isinstance(node, ast.Constant):
            out.append(repr(node.value))
        elif isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.arg):
            out.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)

        body = getattr(node, "body", None)
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            for i, child in enumerate(items):
                if not isinstance(child, ast.AST):
                    continue
                # Drop documentation-only module constants (``PARAM_UNITS``):
                # they describe a value and cannot change behaviour.
                if isinstance(node, ast.Module) and _is_doc_constant(child):
                    continue
                # Drop docstrings: the leading bare string of a module, class or
                # function body.
                if (
                    field == "body"
                    and i == 0
                    and body is not None
                    and isinstance(child, ast.Expr)
                    and isinstance(getattr(child, "value", None), ast.Constant)
                    and isinstance(child.value.value, str)
                    and isinstance(
                        node,
                        (
                            ast.Module,
                            ast.ClassDef,
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    )
                ):
                    continue
                emit(child)

    emit(tree)
    return hashlib.sha256("\x00".join(out).encode()).hexdigest()


def _digest_paths(paths) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        if not path.exists():
            continue
        h.update(path.name.encode())
        h.update(_stable_ast_digest(path.read_text()).encode())
    return h.hexdigest()


def _env_sources(spec, controllers: bool = True) -> list[pathlib.Path]:
    """Every Python module in the package the environment is defined in, plus
    the files the spec declares in ``fingerprint_sources``.

    ``controllers=False`` leaves out the controller files (``_side``): every
    task keeps its expert in its package's ``experts.py`` and its oracle in
    its ``oracle.py``. They belong in their part's fingerprint but not in the
    version stamp, since retuning a controller does not change the
    environment.
    """
    module = type(spec.make_env()).__module__
    package = module.rsplit(".", 1)[0]
    directory = _ROOT.parent / pathlib.Path(*package.split("."))
    paths = sorted(directory.glob("*.py")) if directory.is_dir() else []
    for rel in spec.fingerprint_sources:
        path = _ROOT / rel
        # _digest_paths skips a missing path silently; a typo here must not.
        if not path.is_file():
            raise FileNotFoundError(
                f"{spec.name}: declared fingerprint source {rel!r} does not exist"
            )
        paths.append(path)
    # Any rendering module, not just the one named exactly "rendering.py".
    # Drawing code cannot change a return, so hashing it only produces false
    # stales; the 2D aircraft's second renderer, rendering_console.py, was
    # being hashed purely because the filter matched on an exact filename
    # rather than on the role.
    return [
        p
        for p in paths
        if not p.name.startswith("rendering") and (controllers or not _side(p))
    ]


def _side(path: pathlib.Path) -> str | None:
    """The part a controller file belongs to, or ``None`` for any other file:
    ``oracle*`` files are the oracle's, ``experts*`` files the expert's."""
    if path.name.startswith("oracle"):
        return "oracle"
    return "expert" if path.name.startswith("experts") else None


def expert_sources(spec) -> list[pathlib.Path]:
    """The files the expert part of ``spec``'s records depends on: the
    environment's non-controller modules, the shared physics, experts/pid.py
    and the task's expert modules."""
    own = [p for p in _env_sources(spec) if _side(p) == "expert"]
    return sorted(
        {*_env_sources(spec, controllers=False), *_PHYSICS_SOURCES}
        | {*_EXPERT_SOURCES, *own},
        key=str,
    )


def oracle_sources(spec) -> list[pathlib.Path]:
    """The files the oracle part depends on: the environment's non-controller
    modules, the shared physics, the shared machinery (experts/mpc.py, or the
    shared solver's modules for a task with ``oracle_data``), the task's
    oracle modules, and the expert code its oracle still reads
    (``spec.oracle_reads``)."""
    own = [p for p in _env_sources(spec) if _side(p) == "oracle"]
    reads = []
    for rel in spec.oracle_reads:
        if not (_ROOT / rel).is_file():
            raise FileNotFoundError(f"{spec.name}: oracle_reads names {rel!r}")
        reads.append(_ROOT / rel)
    return sorted(
        {*_env_sources(spec, controllers=False), *_PHYSICS_SOURCES}
        | {*(_SOLVER_SOURCES if spec.oracle_data else _ORACLE_SOURCES), *own, *reads},
        key=str,
    )


def gains_entries(spec, allgains: dict) -> dict:
    """The entries of ``pid_gains.json`` a task's expert rows depend on.

    Every key that starts with the task's name (a controller may read a
    differently named key of its own, as ``plane_cascaded`` is the 2D
    aircraft's), the key named by ``spec.tuned_gains_key``, and every key the
    spec declares in ``gains_keys``. Only the first used to be taken, so a
    task whose controllers read another task's key had a record no retune of
    that key could stale: the ``plane_sine`` and ``plane_energy`` PID
    baselines read ``plane_cascaded``, which reached their fingerprints only
    through the copies their oracle reads. A declared key that is not in the
    file raises, as a missing ``fingerprint_sources`` file does, so a typo
    cannot drop it silently. The prefix rule can take a key only the oracle
    reads (``plane`` takes ``plane_cascaded``), which stales expert rows
    needlessly, never oracle rows silently.
    """
    keys = {k for k in allgains if k.startswith(spec.name)}
    if spec.tuned_gains_key is not None and spec.tuned_gains_key in allgains:
        keys.add(spec.tuned_gains_key)
    return _entries(spec, allgains, keys | set(spec.gains_keys))


def _entries(spec, allgains: dict, keys) -> dict:
    for key in keys:
        if key not in allgains:
            raise KeyError(
                f"{spec.name}: declared gains key {key!r} is not in {GAINS_PATH.name}"
            )
    return {k: allgains[k] for k in sorted(keys)}


def _fingerprint(spec, part: str, sources, gains_of, extra=()) -> str:
    """One part's fingerprint: its ``sources``, the gains ``gains_of`` picks,
    ``extra``, and what both parts share: the parameter values the
    measurement is taken at, the burn-in the protocol scores it from and the
    fixed settle.

    The burn-in is the capped value the protocol and the oracles use
    (``eval.scored_burn_in``: the task's row in ``hold_measurements/``,
    capped at half the episode), not the raw row. A change to the row or to
    the cap that moves the scored window stales the record, and one the cap
    absorbs does not. The rest of the hold row is a measurement taken with
    the controllers and is left out; the settle is a per-task constant
    (``eval.FIXED_SETTLE``).
    """
    # Imported here, as experts.mpc.protocol_burn_in does, so that importing
    # this module stays standard-library only (eval imports numpy).
    from target_gym.eval import fixed_settle, scored_burn_in

    params = spec.make_test_params()
    gains = {}
    if GAINS_PATH.exists():
        gains = gains_of(spec, json.loads(GAINS_PATH.read_text()))
    h = hashlib.sha256()
    h.update(f"{part}-v1".encode())
    h.update(_digest_paths(sources).encode())
    h.update(json.dumps(_values(params), sort_keys=True).encode())
    h.update(json.dumps(gains, sort_keys=True).encode())
    h.update(json.dumps(list(extra)).encode())
    h.update(json.dumps(scored_burn_in(spec.name, params)).encode())
    h.update(json.dumps(fixed_settle(spec.name)).encode())
    return h.hexdigest()[:16]


def expert_fingerprint(spec) -> str:
    """Everything that determines the expert's rows of ``spec``'s records:
    ``expert_sources``, its gains entries (``gains_entries``) and what both
    parts share (``_fingerprint``). Rendering is excluded, since it cannot
    change a return."""
    return _fingerprint(spec, "expert", expert_sources(spec), gains_entries)


def _data_digest(rel: str) -> str:
    """A data file an oracle reads (``spec.oracle_data``), as ``rel`` and the
    hash of its values, the keys that describe its run left out."""
    path = _ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"oracle_data names {rel!r}, which does not exist")
    row = {k: v for k, v in json.loads(path.read_text()).items() if k not in _RUN_KEYS}
    return (
        f"{rel}:{hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}"
    )


def oracle_fingerprint(spec) -> str:
    """Everything that determines the oracle's rows: ``oracle_sources``, the
    gains the oracle still reads (``spec.oracle_gains_keys``), its noise
    fields, the values of the data files it reads (``spec.oracle_data``: an
    oracle on the shared solver's horizon and smoothing widths), and what
    both parts share (``_fingerprint``).

    The noise fields are which parameters the planner zeroes for its own
    model. Not a parameter *value*, yet they decide whether the MPC plans on
    the mean disturbance or on one invented realisation of it -- worth 350.4
    against 151.8 on the battery. A record taken under one and read under the
    other is exactly the silent staleness this guards.
    """
    return _fingerprint(
        spec,
        "oracle",
        oracle_sources(spec),
        lambda s, allgains: _entries(s, allgains, set(s.oracle_gains_keys)),
        sorted(spec.noise_fields) + [_data_digest(rel) for rel in spec.oracle_data],
    )


def fingerprints(spec) -> dict[str, str]:
    """``{part: fingerprint}`` for the parts ``spec`` records: the oracle's
    only where it ships one."""
    out = {"expert": expert_fingerprint(spec)}
    if spec.make_mpc is not None:
        out["oracle"] = oracle_fingerprint(spec)
    return out


def _values(params) -> dict:
    return {
        k: repr(v)
        for k, v in sorted(vars(params).items())
        if isinstance(v, (int, float, bool, str, tuple))
    }


def environment_fingerprint(spec) -> str:
    """Everything a *learned* policy's score depends on.

    Deliberately narrower than the record fingerprints: it covers the
    environment's own modules, the shared physics, and the parameter values,
    but not the controllers. An agent never calls a PID, so re-tuning one
    should not invalidate a training run that cost GPU-hours; changing the
    integrator or the reward every environment scores through should.
    """
    h = hashlib.sha256()
    h.update(b"env-v1")
    h.update(_digest_paths(_env_sources(spec, controllers=False)).encode())
    h.update(_digest_paths(_PHYSICS_SOURCES).encode())
    h.update(json.dumps(_values(spec.make_test_params()), sort_keys=True).encode())
    return h.hexdigest()[:16]


def load_recorded_baselines() -> dict:
    """The recorded measurements by task, or an empty mapping if none exist
    yet."""
    from target_gym import records

    return records.load("baseline_returns")
