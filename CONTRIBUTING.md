# Contributing to TargetGym

Contributions are welcome — bug reports, new environments, better baselines,
or corrections to the physics.

## Setting up

```bash
git clone https://github.com/YannBerthelot/TargetGym.git
cd TargetGym
uv sync --group dev          # creates .venv with runtime, test and lint deps
uv run pre-commit install    # optional: runs the CI checks on each commit
```

## Running the checks

```bash
make ci          # everything CI runs: ruff, black --check, fast tests
make test        # fast tests only, in parallel
make test-all    # adds the slow closed-loop controller checks
```

The suite runs under [pytest-xdist](https://pytest-xdist.readthedocs.io/).
`tests/conftest.py` holds each worker to one compute thread so they do not
compete for cores, and forces a headless SDL and matplotlib backend. Pass
`-n0` when you want readable output from a single test, or `--pdb`.

Note that `--durations` under `-n auto` reports wall time inflated by worker
contention. Profile with `-n0` before concluding a test is slow.

Two markers matter: tests are unmarked by default, and `@pytest.mark.slow`
covers the closed-loop controller contracts, which run on merges to `main`
rather than on every pull request.

CI runs the fast suite against Python 3.11, 3.12, 3.13 and 3.14 in parallel,
plus a lint job and, on merges to `main`, the slow suite on one interpreter.
If you add a dependency, check it resolves across that whole window --
`uv lock --check --python 3.14` is the quickest way to find out. Note that
`pygame` itself only ships wheels through 3.13, so on 3.14 the project pulls
`pygame-ce`, the maintained fork of the same code under the same import name.

## Style

`ruff` and `black`, both enforced in CI and configured in `pyproject.toml`.
`make format` applies both. The ruff rule set is deliberately narrow — real
defects, import errors and import ordering — so that anything it reports is
worth acting on; the reasoning for what is excluded is in `pyproject.toml`
next to the `select` list.

`mypy` is available via `make mypy` but is not enforced: the tree does not
currently pass it.

## Adding an environment

An environment lives in its own package under `src/target_gym/`, following the
shape of an existing one such as `src/target_gym/boiler_drum/`:

| file | holds |
|---|---|
| `env.py` | parameters, state, dynamics, reward, termination |
| `env_jax.py` | the gymnax `Environment` subclass |
| `rendering.py` | a dashboard built on `target_gym.render_kit` |
| `PHYSICS.md` | the physics contract (see below) |

Then add an `EnvSpec` to `src/target_gym/registry.py`. That entry is what makes
the environment real to the rest of the repo: `tests/test_env_conformance.py`
parametrises **every** conformance test over the registry, so registering an
environment immediately subjects it to the shared contracts — determinism,
observation and action space agreement, disturbances that behave like
disturbances under a constant PRNG key, and a PID that beats the best constant
action. Most defects in a new environment surface there before you write a
single environment-specific test.

`EnvSpec`'s docstring documents each field. Two are easy to overlook:
`disturbance_fields` (state entries holding zero-mean noise, which the
conformance suite checks do not ratchet) and `baselines_note` (why a PID or
MPC is absent, so a missing baseline is a documented gap rather than a silent
one).

### The physics contract

Every environment carries a `PHYSICS.md` with a sourced parameter table,
published validation targets that tests assert, and quantified known
deviations. `docs/PHYSICS_METHODOLOGY.md` explains the approach; the short
version is that a test must assert an **emergent consequence** of the model,
not restate the formula the code already contains. A test that re-implements
`compute_next_state` and compares confirms only that you typed it twice.

Where the model knowingly departs from the literature, record it as a numbered
deviation in `PHYSICS.md` and, where it is quantifiable, pin it with a
`strict` xfail so that fixing it later fails loudly rather than passing
silently.

## Baselines

Environments ship a PID and, where tractable, an MPC, so a learned policy has
something real to beat. PID gains are tuned by `scripts/tune_pid.py` and cached
in `data/pid_gains.json`. When the gradient-based MPC is unusable — the cement
kiln's adjoint overflows through its transport delay — use the sampling
(CEM) MPC instead.

## Pull requests

Keep the physics and the code in one change: a dynamics edit that moves a
validation number should update `PHYSICS.md` in the same commit. Say in the
description what you measured, not only what you changed.
