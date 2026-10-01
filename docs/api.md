# Public API

What this page promises: names listed as **stable** will not change
incompatibly without a major version bump and a deprecation period. Names
listed as **provisional** are usable, documented and tested, but may still
change shape before they settle.

## Stable

### Environment and parameter classes

The forty-four names in `target_gym.__all__` are stable. They are every
environment class and its matching `Params` class, plus the Gymnasium wrapper.
To check the count:

```python
import target_gym
print(len(target_gym.__all__))
```

### The environment interface

Every environment implements the [gymnax](https://github.com/RobertTLange/gymnax)
`Environment` interface. These are the members to rely on:

| Member | Meaning |
|---|---|
| `reset_env(key, params) -> (obs, state)` | Fresh episode; the target is sampled here |
| `step_env(key, state, action, params)` | One step, reporting **natural termination only** |
| `step(key, state, action, params)` | gymnax's six-value step, which also applies the time limit |
| `observation_space(params)`, `action_space(params)` | gymnax spaces |
| `get_obs(state, params)` | The observation for a state |
| `default_params` | A ready-made parameter set |
| `render(...)`, `save_video(...)` | The control-room dashboard |

Two conventions make generic code possible across environments, and are
themselves stable:

| Attribute | Meaning |
|---|---|
| `obs_value_index` | Observation slot(s) holding the tracked variable, an `int` or a tuple for multi-loop plants |
| `obs_target_index` | Slot(s) holding its setpoint, in the same order |
| `tracked_names` | Human-readable name and unit per tracked slot |

`step_env` reporting natural termination alone is deliberate: the time limit is
gymnax's business, and conflating the two is what makes an agent learn that
running out of clock is a failure state.

### The registry

`target_gym.registry.REGISTRY` maps a name to an `EnvSpec`. The spec's fields
(`make_env`, `params_cls`, `make_pid`, `make_mpc`, `test_params`,
`disturbance_fields`, `baselines_note`, ...) are documented on the class and
are stable.

```python
from target_gym.registry import REGISTRY, GROUPS

for name, spec in REGISTRY.items():
    assert spec.name == name
print(len(REGISTRY), "environments in", len(GROUPS), "groups")
```

## Provisional

| Module | Why it is not yet stable |
|---|---|
| `target_gym.experts.pid`, `.mpc` | The per-environment factories are many and their signatures still vary. Reach them through `EnvSpec.make_pid` / `make_mpc`, which is stable. |
| `target_gym.runners` | Figure and video generation. A tool, not a library surface. |
| `target_gym.render_kit` | The dashboard toolkit. Stable enough to build on, but its primitives are still moving. |
| `target_gym.utils` | A grab-bag; parts of it will move or go. |
| `target_gym.benchmark` | One policy interface and a paired runner over every task (`Policy`, `TaskInfo`, `run_policy`, `run_policy_on_benchmark`, `shipped_policy`). New; its `TaskInfo.extras` conventions are still settling with its first downstream user. |

## Not public

Anything beginning with an underscore, and the environment modules' internal
`env.py` helpers (`compute_next_state`, `compute_reward`, ...). These are
imported directly by the test suite because tests are allowed to know more
than users; that is not a promise about them.

## Versioning

Two things are versioned here, and they move independently.

### The package

Semantic versioning. The version is derived from the git tag by `hatch-vcs`, so
`target_gym.__version__` reflects the release you installed. The classifier in
`pyproject.toml` says what maturity to expect and is kept honest rather than
aspirational.

### The environments

Every environment carries a version, and `spec.versioned_name` gives the name a
published result should cite: `plane-v1`, `cstr-v1`. Versioning starts at the
0.6 release, where everything ships as `v1`. Nothing before that is versioned,
because the package had no users and so no published numbers to keep meaningful.

Registry keys stay unversioned, since they are an internal handle used for
gains, recorded baselines and file paths. `REGISTRY["plane"]` is how you load
it; `plane-v1` is what you cite.

The version changes when the environment does: its dynamics, its reward, the
parameters it is measured at, or its observation layout. Re-tuning a controller
is not such a change, and does not bump anything.

That promise is enforced rather than asserted. `src/target_gym/data/env_versions.json` records
the fingerprint each version was stamped at, and `tests/test_env_versions.py`
fails when the tree no longer matches. So an environment cannot change under a
name that has already been published against without CI saying so.
