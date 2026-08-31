# Public API

What this page promises: names listed as **stable** will not change
incompatibly without a major version bump and a deprecation period. Names
listed as **provisional** are usable, documented and tested, but may still
change shape before they settle.

## Stable

### Environment and parameter classes

The thirty-seven names in `target_gym.__all__` -- every environment class and
its matching `Params` class, plus the Gymnasium wrapper:

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
| `obs_value_index` | Observation slot(s) holding the tracked variable -- an `int`, or a tuple for multi-loop plants |
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

## Not public

Anything beginning with an underscore, and the environment modules' internal
`env.py` helpers (`compute_next_state`, `compute_reward`, ...). These are
imported directly by the test suite because tests are allowed to know more
than users; that is not a promise about them.

## Versioning

Semantic versioning. The version is derived from the git tag by
`hatch-vcs`, so `target_gym.__version__` reflects the release you installed.

Before 1.0, the classifier in `pyproject.toml` says what maturity to expect,
and it is kept honest rather than aspirational.
