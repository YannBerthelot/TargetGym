# Contributing to TargetGym

Contributions are welcome: bug reports, new environments, better baselines,
or corrections to the physics. Everyone taking part is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Setting up

```bash
git clone https://github.com/YannBerthelot/TargetGym.git
cd TargetGym
uv sync --group dev          # creates .venv with runtime, test and lint deps
uv run pre-commit install    # optional: runs the CI checks on each commit
```

## Running the checks

```bash
make ci          # everything CI runs: ruff, black --check, docs, fast tests
make test        # fast tests only, in parallel
make test-all    # adds the slow closed-loop controller checks
make mypy        # type-check the enforced modules
make coverage    # tests with the coverage threshold
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
plus a lint job, a docs job (`make ci-docs`) and, on merges to `main`, the slow
suite on one interpreter.
If you add a dependency, check it resolves across that whole window --
`uv lock --check --python 3.14` is the quickest way to find out. Note that
`pygame` itself only ships wheels through 3.13, so on 3.14 the project pulls
`pygame-ce`, the maintained fork of the same code under the same import name.

## Style

`ruff` and `black`, both enforced in CI and configured in `pyproject.toml`.
`make format` applies both. The ruff rule set is deliberately narrow (real
defects, import errors and import ordering) so that anything it reports is
worth acting on; the reasoning for what is excluded is in `pyproject.toml`
next to the `select` list.

`mypy` runs over a deliberately small set of modules, the ones that pass
today, and CI enforces it. The rest of the tree does not pass, for a reason
recorded next to the config: the environments annotate `struct.dataclass`
fields as `float` while holding traced JAX `Array` values, which is the
ordinary flax idiom and produces hundreds of mismatches that are shorthand
rather than bugs. `make mypy-all` shows the whole picture; widen the enforced
list by making a module pass and adding it.

Coverage sits above 90%, enforced at 89 by `make coverage` and on one CI
column. It is a ratchet: raise it as coverage improves, never lower it to make
a red build green. Nothing is excluded from the measurement. The tuners and
the figure runner are things users run, so leaving them out would measure a
smaller library than the one that ships.

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
environment immediately subjects it to the shared contracts: determinism,
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
in `src/target_gym/data/pid_gains.json`. When the gradient-based MPC is unusable, as on the
cement kiln whose adjoint overflows through its transport delay, use the sampling
(CEM) MPC instead.

### Re-recording the baselines

How well each controller actually controls is measured by hand and committed,
not measured in CI: a full run takes hours, nearly all of it in the aircraft,
and the answer only moves when the physics, the controllers or their gains do.

```bash
uv run python scripts/record_baselines.py                  # everything
uv run python scripts/record_baselines.py --envs plane cstr # just these
```

The headline table in `docs/baselines.md` and the per-environment pages both
show those records, so regenerate them afterwards:

```bash
uv run python scripts/generate_baseline_table.py
uv run python scripts/generate_env_pages.py
```

`make ci-docs` and the CI docs job fail while either is stale.

Each record carries a fingerprint of what determined it: the environment's
modules, the shared controller and integration code, the gains, the parameter
values. The suite refuses to read one whose fingerprint no longer matches
the tree. So if you change any of those, a test will tell you which records went
stale; re-record them and commit the result with the change that invalidated it.
`src/target_gym/provenance.py` explains why the fingerprint is taken over source
rather than over behaviour, and what that trade buys.

The long-run protocol costs in `src/target_gym/data/protocol_results.json` work
the same way. They are what the floor check in `tests/test_reward_contract.py`
compares each measured floor against, and they carry the same fingerprint, so
the same change that makes a baseline stale makes its row stale too. Re-run the
protocol for the environments the test names, and if the numbers moved, paste
the table it prints into the protocol section of `docs/baselines.md`:

```bash
uv run python scripts/evaluate_baselines.py --envs plane cstr
```

Two things to know before starting a run. It reads `src/target_gym/data/baseline_returns.json`
and merges its results into whatever the file holds *at the time it writes*, so
a second run started later will not clobber it, but two runs recording the same
environment will still race, and the last one wins. And gains changes invalidate
records, so tune first, record second.

## Documentation

`docs/` holds the guides; each environment's physics contract lives in a
`PHYSICS.md` beside its module. Two parts of it are checked rather than
trusted, by `tests/test_docs.py`:

- **Every runnable example in `docs/` is executed.** A fenced `python` block
  runs unless its first line is `# doc: skip`. If you change a signature, the
  docs fail with the code.
- **The environment pages are generated** from the registry:
  `docs/environments.md` by `scripts/generate_env_reference.py`, and the
  per-environment pages under `docs/environments/` by
  `scripts/generate_env_pages.py`. The recorded baseline table in
  `docs/baselines.md` is generated as well, by
  `scripts/generate_baseline_table.py` from
  `src/target_gym/data/baseline_returns.json`. Adding an environment or a
  baseline means regenerating all three, and so does re-recording baselines,
  since the pages and the table show the recorded numbers. `make ci-docs` runs
  each with `--check` and then `mkdocs build --strict`, which fails on a
  dangling link, including one that resolves on GitHub but not on the built
  site, since anything outside `docs/` (a `PHYSICS.md` under `src/`,
  `CONTRIBUTING.md`) has to be linked absolutely.

  Action meanings on those pages are read out of each environment class's
  docstring, from a line like `Action (2,): [power, stick], raw in [-1, 1]`.
  Write one, or the page shows bounds with a blank meaning. The generator will
  not invent labels for you.

  The gallery mosaics come from `scripts/make_gallery_mosaic.py`, which builds
  its sets from the registry groups, so a new environment appears in its group's
  mosaic once it has a clip under `videos/`.

A new environment also needs a `PHYSICS.md`, and that is asserted too. The
README's claim that every environment carries one is backed by a test.

## Pull requests

Keep the physics and the code in one change: a dynamics edit that moves a
validation number should update `PHYSICS.md` in the same commit. Say in the
description what you measured, not only what you changed.
