# Changelog

Notable changes to TargetGym. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/).

Tags before this file were named `vX.Y.Z`, except `0.5.0`; future tags use the
`vX.Y.Z` form.

## [Unreleased]

The work between `0.5.0` and 1.0. Grouped by what it changes for a user rather
than by commit.

### Added

- **Seven environments**: building HVAC, pH neutralisation, Skogestad's
  distillation Column A, the NREL 5 MW reference wind turbine, a grid battery,
  a boiler drum, and a cement kiln. Eighteen in total.
- **A central registry** (`target_gym.registry`) describing every environment,
  its parameters and its baselines, and a **shared conformance suite** that runs
  the same contracts against all of them -- PRNG hygiene, determinism, the
  gymnax six-value step API, and a PID that beats the best constant action.
- **Physics contracts.** Every environment carries a `PHYSICS.md` with a sourced
  parameter table, published validation targets asserted by tests, and numbered
  known deviations. The method is written up in `docs/PHYSICS_METHODOLOGY.md`.
- **Documentation** under `docs/`: getting started, a public API contract, a
  baselines guide, and an environment reference generated from the registry.
  Every runnable example is executed by the test suite.
- `CONTRIBUTING.md`, `LICENSE`, `CITATION.cff`, and a pre-commit configuration.
- `tracked_names`, `obs_value_index` and `obs_target_index` on every
  environment, so generic tooling can find the tracked variable and its
  setpoint without naming the environment.
- Actuator lag parameters for the aircraft (`power_response_rate`,
  `stick_response_rate`, `aileron_response_rate`), previously literals.
- `CHANGELOG.md`, issue templates and a pull-request template.

### Changed

- **Migrated to the gymnax 1.0 six-value step API.** `step_env` now reports
  natural termination alone; the time limit is gymnax's, via `step`.
- **Every renderer rebuilt** on a shared control-room toolkit
  (`target_gym.render_kit`), and the README gallery regenerated and extended to
  every environment.
- **Seven per-environment figure/video runners consolidated** into one
  registry-driven module, which covers all eighteen environments rather than
  eight.
- Python support is 3.11 through 3.14, tested on all four in CI.
- The test suite runs in parallel; full-suite wall time went from about
  fifteen minutes to under two.

### Fixed

- **Four-tank**: the target range sat entirely above what the plant can reach,
  so every episode was unwinnable. The range, the loop pairing (the RGA puts
  λ11 at −0.067, so the loops must be crossed) and the tuner objective were all
  corrected.
- **Aircraft lift curve**: `cl_alpha` was 54 % below what the wing's own aspect
  ratio implies, and the stall clamp was applied before the Prandtl--Glauert
  factor, so peak lift *rose* with Mach instead of falling past `M_crit`.
- **Reactor renderer** produced no frames for short episodes and never reset its
  history between episodes: it advances `state.time` by a control period, so the
  `time == 1` episode-start signal never fired.
- **Bearing-only patrol** gained a baseline, via a lead-state estimator feeding
  the same pursuit law the full-observation variant uses.
- The glass furnace setpoint band was narrowed and given a working tuner.
- **Four-tank gradients.** Outflow goes as the square root of the level and a
  tank can sit empty; written as `sqrt(max(h, 0))` the forward value is right
  but the reverse-mode derivative is NaN at zero, which made gradient-based PID
  tuning return NaN gains from a loss that evaluated perfectly well. Forward
  results are unchanged.

### Removed

- Dead modules carrying no importers: `experts/degradation.py`,
  `experts/cpg.py` and `experts/pd.py` (the latter two were Brax/MuJoCo
  locomotion experts for environments this project does not have), and
  `scripts/benchmark_integration.py`, which imported an undeclared dependency
  and could not run.

### Known gaps

- Both patrol variants ship a PID, but it holds formation only loosely --
  roughly 139 m of settled slot error against a 60 m tolerance, pinned by six
  `strict` xfail cases.
- Three of the seven gradient PID tuners (`plane`, `plane3d_heading`,
  `plane3d_circle`) return NaN gains, pinned by `strict` xfails. Relay
  autotuning, which produced the shipped gains, is unaffected.
- No published RL baseline results yet.
- The tested configuration pins `gymnax` to upstream `main`, so it cannot be
  reproduced from PyPI alone.

[Unreleased]: https://github.com/YannBerthelot/TargetGym/compare/0.5.0...HEAD
