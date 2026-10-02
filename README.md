<h1 align="center">TargetGym</h1>

<h3 align="center">
  23 JAX environments for setpoint tracking,<br/>
  with tuned PID and MPC baselines.
</h3>

<p align="center"><i>Reach the target. Then hold it.</i></p>

<p align="center">
  <a href="https://yannberthelot.github.io/TargetGym/"><img alt="Documentation" src="https://img.shields.io/badge/docs-yannberthelot.github.io%2FTargetGym-blue"></a>
  <a href="https://pypi.org/project/target-gym/"><img alt="PyPI" src="https://img.shields.io/pypi/v/target-gym?color=blue"></a>
  <a href="https://pypi.org/project/target-gym/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/target-gym"></a>
  <a href="https://github.com/YannBerthelot/TargetGym/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/YannBerthelot/TargetGym/python-app.yml?branch=main&label=tests"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/pypi/l/target-gym"></a>
  <img alt="JAX" src="https://img.shields.io/badge/JAX-jit%20%7C%20vmap%20%7C%20scan-orange">
  <a href="https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center">
  <img src="videos/mosaic_flagship.webp" width="100%"/><br/>
  <sub>One example task from each family, under PID control.</sub>
</p>

TargetGym provides reinforcement learning environments for **target MDPs**:
tasks where the objective is to reach a setpoint and hold it against
disturbances, rather than to reach a goal state once. The environments model
real plants, including an A320-like aircraft, a glass furnace, a nuclear
reactor, a cement kiln, a grid battery and a wind turbine.

- **Baselines included.** Every environment ships a tuned PID and an MPC
  with full state access, which serves as an upper bound. Both are recorded
  over ten seeds in `src/target_gym/data/baseline_returns.json`.
- **Validated physics.** Each environment carries a `PHYSICS.md` with a sourced
  parameter table, published validation targets asserted by tests, and its
  documented approximations.
- **JAX throughout.** `jit`, `vmap` and `scan` compatible, end-to-end GPU,
  0.6 M to 700 M steps/s on CPU depending on the plant.
- **gymnax API**, with a Gymnasium wrapper for non-JAX libraries and a JaxMARL
  interface for the multi-agent patrol task.

Learned-policy results are not published yet. The measurement protocol is
defined in [docs/rl-protocol.md](docs/rl-protocol.md).

---

## Installation

```bash
pip install target-gym
```

## Quickstart

Also available as a [Colab notebook](https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb).

```python
import jax
import numpy as np
from target_gym import Plane

env = Plane()
pid = env.make_pid()            # the shipped baseline, tuned

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

total = 0.0
for _ in range(env.default_params.max_steps_in_episode):
    action = pid(np.asarray(obs))
    obs, state, reward, terminated, truncated, _ = env.step(key, state, action)
    total += float(reward)
    if terminated or truncated:
        break

print("PID return:", total)
```

`reset` and `step` follow the
[gymnax](https://github.com/RobertTLange/gymnax) API and take an optional
`params`; each environment exports its parameter class (`PlaneParams`, and so
on) for custom configurations. Every environment also exposes `make_pid()`,
`make_mpc()` and `save_video()`.

<details>
<summary>Non-JAX libraries, e.g. stable-baselines3</summary>

End-to-end GPU requires a JAX-based library.

```python
# doc: skip (trains for 10 000 steps; tests/plane/test_agent.py covers this path)
from target_gym import GymnasiumPlane
from stable_baselines3 import SAC

env = GymnasiumPlane()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

</details>

### Comparing against the baselines

For most environments the defaults *are* the scored configuration, so a plain
rollout is comparable to the published numbers. The exceptions are the plants
that host several task variants: `PlaneParams` is shared by `plane`,
`plane_sine` and `plane_energy`, which are scored over 280, 480 and 1200 steps,
so one class cannot default to all three. `EnvSpec.test_params` carries the
per-variant settings, and `spec.make_test_params()` resolves them:

```python
import numpy as np

from target_gym.provenance import load_recorded_baselines
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy, rollout

spec = REGISTRY["plane"]
params = spec.make_test_params()                     # the scored configuration
pid = baseline_policy(spec, "pid", params)

print("PID:", float(np.sum(rollout(spec, params, pid, seed=0)[2])))
print("MPC:", load_recorded_baselines()["plane"]["mpc_returns"][0])
```

The PID reads `obs` only, as a plant controller does. The MPC reads the full
state, including quantities the observation withholds, which is what makes it
an upper bound rather than a peer; [docs/baselines.md](docs/baselines.md)
quantifies the resulting advantage.

Vectorised rollouts, the registry API, the patrol interface and the wind model
are covered in [docs/getting-started.md](docs/getting-started.md).

---

## Environments

| Family | Count | Environments |
|---|---|---|
| **Aircraft** | 9 | A320-like 2D aircraft on three reference patterns; four 3D path-following tasks; two formation-patrol variants |
| **Process control** | 6 | CSTR, first-order lag, four-tank, pH neutralisation, binary distillation, CSTR on its unstable steady state |
| **Industrial / energy** | 6 | Glass furnace, nuclear reactor, building HVAC, boiler drum, cement kiln, compressor held off its surge line |
| **Renewable energy** | 2 | Wind turbine, grid battery |

**[Full environment reference →](docs/environments.md)** with observation and
action shapes, tracked variables, baseline returns and physics contracts.

The CSTR, first-order and four-tank models are adapted from
[PC-gym](https://github.com/MaximilianB2/pc-gym) and checked against its source
term by term, as each `PHYSICS.md` provenance line records.

The two patrol environments are multi-agent: wingmen hold a slot on a lead
flying its own route, so the reference is another aircraft and collision ends
the episode.

Environments span six difficulty tiers, graded on dynamics (linearity,
coupling, stiffness) and on the RL side (dimensionality, horizon, partial
observability), from a first-order lag at tier 1 to the cement kiln and the
multi-agent patrol at tier 6. See the
**[complexity ladder](docs/complexity.md)**.

Each environment renders a control-room dashboard: plant schematic, gauges with
limit and setpoint markers, strip charts, and explicit marking of quantities the
controller cannot measure. See the **[rendering guide](docs/rendering.md)**.

---

## Why setpoint tracking

Holding a setpoint indefinitely exposes failure modes that episodic goal-reaching
does not:

| Property | In the suite | Why it is hard to learn |
|---|---|---|
| **Irrecoverable states** | A drum that carries water into the turbine, a reactor past runaway, a kiln gone cold | Exploration that reaches them ends the episode permanently |
| **Deep partial observability** | The furnace hides 6 of 9 states, the reactor 7 of 11, the kiln 64 behind 8 measurements | The policy has to infer what it cannot measure |
| **Wrong-way-first response** | Opening the steam valve makes drum level *rise* before it falls, as steam bubbles expand | A controller following the immediate trend pushes the loop the wrong way |
| **Transport delay** | Half the kiln's response to a fuel change arrives a 25-minute residence time later | Credit assignment spans hundreds of steps |
| **Multi-timescale dynamics** | Millisecond neutronics against hour-long xenon; sub-second flame gas against 30-hour glass residence | One control interval cannot serve both ends |
| **Finite budgets** | A battery's charge window trips the pack at either edge, though exact dispatch tracking stays inside it for the whole 30-minute episode | Within an episode, tracking is priced against wear at every step, not against tracking later |

Also modelled: actuator lag, competing objectives, and scheduled setpoints that
reward anticipation (the building's night setback, the furnace's crown schedule,
the battery's dispatch blocks). Aircraft fly in steady wind, altitude shear and
Ornstein-Uhlenbeck turbulence, unobservable by default.

---

## Baselines

Every environment ships a tuned PID and an MPC. On `patrol_bearing_only`, which
withholds the slot error a planner would read, the MPC slot holds `patrol`'s
oracle reading the true state, labelled a full-state bound.

Controller structure is chosen per plant:

- **Boiler drum**: three-element control, so feedwater tracks measured steam
  flow and shrink-and-swell cannot mislead the level loop.
- **Cement kiln**: cascade, because integral action on a 25-minute-old
  measurement oscillates at the delay period.
- **Four-tank**: crossed loops, since the negative RGA element makes the
  diagonal pairing unstable.
- **Unstable CSTR**: cascade, an outer PI on concentration setting the
  temperature an inner PD holds, since no P or PI loop on concentration alone
  can stabilise the reactor's middle steady state.
- **Compressor surge**: a pressure PI on the drive's speed and an anti-surge PI
  on the recycle valve, with a full-opening override near the surge line. This
  is the pairing industrial anti-surge systems use, and the recycle valve gets
  a loop of its own because the margin it protects is not a tracked output.

Five kinds of controller fill the MPC slot: CasADi/IPOPT where a symbolic model
exists, gradient-based planning through the JAX dynamics elsewhere,
cross-entropy sampling for the cement kiln, on the battery a feedforward of the
scheduled dispatch level, which is the best causal action there, on the wind
turbine a feedback law (generator torque solved so that the next step's power
meets the target, and pitch from the PID with its command slew capped), and on
the patrol tasks the lead's own autopilot flown on the follower's state with a
short residual planner on top. Solver convergence is recorded alongside every
result.

Each baseline must beat the best constant action on its environment, a
deliberately low bar that a mis-wired controller fails. Weak baselines are
documented as such on their environment page.

> **A PID losing on a task says something about that task, not about PID
> control.** These environments are selected for problems where anticipation
> pays. Where reacting to the reference and the disturbances is sufficient, a
> PID is optimal or close enough that the difference does not appear in a
> return. Two environments were corrected after measurement showed exactly
> this: a battery whose dispatch signal was so noisy that no controller could
> exceed 0.43 of the reward ceiling, and a patrol follower tracking a lead at
> one fixed turn rate, which a single feedforward term cancels.

**[docs/baselines.md](docs/baselines.md)** covers the MPC implementations,
tuning and caching, solver reporting and per-environment coverage.

---

## Physics validation

Each environment carries a `PHYSICS.md` stating what it models and where that
holds, a parameter table citing or deriving every constant (flagged
`TUNED - not sourced` when neither applies), the published numbers it must
reproduce, and its known deviations.

Tests assert consequences rather than formulas: ISA table values, L/D ratios,
thermal time constants, energy balances, equilibria. A test that recomputes the
implementation's own expression would pass on a wrong one.

All 23 environments are covered by seventeen contracts, since aircraft variants
share a plant. A shared conformance suite additionally checks determinism, PRNG
handling, `jit`/`vmap`/`scan` compatibility and numerical health over full
episodes.

**[docs/PHYSICS_METHODOLOGY.md](docs/PHYSICS_METHODOLOGY.md)** documents the
method and lists what each model is validated against.

---

## Documentation

| | |
|---|---|
| **[Getting started](docs/getting-started.md)** | Episodes, vectorised rollouts, the registry, Gymnasium |
| **[Target MDPs](docs/target-mdp.md)** | The formal setting |
| **[Environment reference](docs/environments.md)** | All 23: shapes, tracked variables, baselines, contracts |
| **[Public API](docs/api.md)** | Stable and provisional surface |
| **[Baselines](docs/baselines.md)** | PID and MPC controllers, tuning, solver reporting |
| **[RL protocol](docs/rl-protocol.md)** | Measuring a learned policy |
| **[Reward shaping](docs/reward-shaping.md)** | The tracking reward and why it has that shape |
| **[Complexity ladder](docs/complexity.md)** | Six tiers, for curriculum use |
| **[Rendering](docs/rendering.md)** | Dashboards, toolkits, regenerating media |
| **[Throughput](docs/performance.md)** | Steps per second per environment |
| **[Physics methodology](docs/PHYSICS_METHODOLOGY.md)** | Sourcing, validation, bounds |
| **[Model review checklist](docs/model-review-checklist.md)** | Thirteen checks for any plant model |
| **[Testing](docs/testing.md)** | Suite organisation |

Rendered with search and navigation at
**[yannberthelot.github.io/TargetGym](https://yannberthelot.github.io/TargetGym/)**.
The links above point at the Markdown in this repository, which reads on
GitHub and is rewritten for the site at build time.

---

## Related projects

- **[gymnax](https://github.com/RobertTLange/gymnax)**: the JAX environment API
  TargetGym implements.
- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)**: supported
  through a wrapper for non-JAX libraries.
- **[PC-gym](https://github.com/MaximilianB2/pc-gym)**: process-control
  environments; source of the CSTR, first-order and four-tank models.
- **[safe-control-gym](https://github.com/utiasDSL/safe-control-gym)**: closest
  in intent, comparing classical control, MPC and RL on shared tasks. It covers
  more controllers, TargetGym covers more plants.
- **[JaxMARL](https://github.com/FLAIROX/JaxMARL)**: the multi-agent API the
  patrol formation task follows.

---

## Roadmap

Rewards, physics, baselines and the measurement protocol are settled.
Outstanding before 1.0: published learned-policy results and hosted
documentation. Known gaps are recorded rather than omitted.

**[Full roadmap →](docs/roadmap.md)**

## Contributing

Bug reports, new environments, improved baselines and physics corrections are
welcome.

```bash
git clone https://github.com/YannBerthelot/TargetGym.git
cd TargetGym
uv sync --group dev
make ci                # ruff, black --check, docs, fast tests
```

Further targets: `make test`, `make test-all`, `make figures`, `make videos`,
`make tuning`.

Adding an environment means registering an `EnvSpec`, which inherits every
shared check, and writing a `PHYSICS.md` sourcing its numbers.
**[CONTRIBUTING.md](CONTRIBUTING.md)** has the details.

---

## Citation

```bibtex
@misc{targetgym2025,
  title        = {TargetGym: Reinforcement Learning Environments for Target MDPs},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/TargetGym},
  note         = {Lightweight physics-based RL environments for aircraft, process control, and industrial systems}
}
```

## License

MIT.
