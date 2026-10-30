# Testing

Tests come in three sizes, set by what a test uses rather than by how long it
happens to take. The size decides where it runs.

| size | marker | what a test may use | where it runs | budget |
| --- | --- | --- | --- | --- |
| small | none | one process, memory, `tmp_path` | a plain `pytest`; CI on every push, Python 3.11 to 3.13 | the whole tier within 60 s on a laptop |
| medium | `medium` | another process, video files, a compile that needs gigabytes | CI on every push, Python 3.11 and 3.12, after the small tier | small and medium within 10 min on CI |
| large | `large` | closed-loop rollouts at the scored length, many seeds | CI on merges to `main`, one interpreter | each test within 15 min |

`records` is not a size. Those tests check that the recorded baselines still
describe this tree, and they run in a job of their own, so a stale record fails
that job alone and a failing test job always means a real bug.

```bash
pytest                   # small, on every core
pytest -m medium -n 2    # medium; two workers keep two big compiles in memory
pytest -m large          # large
pytest -m records        # records
pytest -m ""             # everything
```

A later `-m` replaces the default one, and `-n0` runs serially, which `--pdb`
needs.

Every run ends with its 15 slowest tests, so a test that outgrows its size
shows up there before it shows up in the budget.

## Which size a new test is

Start small. A small test that is slow is a design problem first, and the usual
cause is stepping a plant from Python: run it inside `jax.lax.scan` or
`jax.lax.while_loop`, jit the step, and shorten episodes and shapes to what the
property needs. That is how the glass furnace and boiler drum tests went from
125 s and 44 s to a few seconds without changing an assertion.

Mark it `medium` when it starts another process (`subprocess`, or moviepy
running ffmpeg to write a video), or when it compiles an aircraft planner or
gradient tuner. Those compiles peak at 2.5 to 5 GB and take up to three
minutes on a cold cache; four at once shut CI runners down on 2026-10-27.

Mark it `large` when the property only shows at the scored length: a closed
loop that holds its target, a PID that beats the best constant action, a check
over many seeds.

## The expensive measurement is recorded, not reproduced

Checking that each shipped MPC really is an upper bound on its PID meant rolling
sixteen controllers out on five seeds: about forty minutes of CPU, and
`[plane]` alone was **836 s**. Because `pytest-xdist` distributes across tests
and not within one, that single test set roughly 70% of the large job's
wall-clock floor and came close to its 30-minute timeout on CI's slower cores.

It is now recorded by hand into `src/target_gym/data/baseline_returns/` and asserted from
there, guarded by a fingerprint that refuses a record whose code has moved. See
[Baselines](baselines.md) for the design and the reasoning behind fingerprinting
source rather than behaviour.

```bash
make baselines           # re-measure everything, ~40 min
make baselines-plane     # or just one environment
```

Run it when a test tells you a record is stale, and commit the result with the
change that invalidated it.

## XLA compilations are cached, in the process and between runs

What remained was not stepping the plants but *compiling* them. Reverse-mode
through an RK4 aircraft builds a large graph, and it showed: a tuner smoke test
that takes **two** gradient steps still cost 45 s, and shrinking its problem from
36 target pairs and 2000 steps to 2 and 100 barely moved it. The graph was the
cost, not the rollout.

JAX can persist compiled executables to disk, keyed on the computation itself.
`tests/conftest.py` points it at `.jax_cache`, which does two things at once:
under xdist it stops every worker compiling the same graph, and in CI
`actions/cache` restores it between runs, so compilation becomes a
once-per-code-change cost rather than once-per-push.

Measured on the whole fast suite: **217 s cold, 34.5 s warm**. On one tuner test
alone, 44.9 s against 3.4 s. The cache is about 51 MB.

A stale entry is never wrongly reused -- entries are keyed by the jaxpr, the
backend and the JAX version, so a changed computation simply misses. The CI key
includes a hash of `src/`, with a looser `restore-keys` prefix so a code change
still seeds from the previous cache instead of starting empty.

## Conventions

- Coverage is gated at 89% over the small tier alone, so moving a test to
  `medium` or `large` removes it from the gate. Check the number before doing
  it.
- `strict=True` on every xfail. A known defect that gets fixed must fail the
  suite until its annotation is updated, so the record cannot quietly rot.
