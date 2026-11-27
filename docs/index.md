---
hide:
  - toc
---

# TargetGym

<p align="center">
  <b>Reach a setpoint. Hold it forever. Against disturbances.</b><br/>
  25 JAX environments for <i>target MDPs</i>, the control problems industry actually has.
</p>

<p align="center">
  <img src="videos/mosaic_flagship.webp" width="100%"/><br/>
  <sub>One example task from each family, under PID control.</sub>
</p>

**25 environments**: 11 aircraft, 6 process control, 6 industrial / energy, 2 renewable energy. Every one of them is in the
[gallery](environments.md) with its own page and clip.

**Four of them are scored:** `reactor`, `hvac`, `compressor_surge` and
`ph_neutralization`. Each has a measured reference to beat and a ceiling, and
their numbers are in [Baseline results](baseline-results.md). The other
twenty-one run the same way and are not scored yet: each one's page says where
it stands.

---

```bash
pip install target-gym
```

!!! note "Releases and `main`"
    TargetGym is in active development: PyPI releases are the stable line, and
    `main` can change quickly between them. This site is versioned: the menu
    at the top switches between releases and `dev`, which follows `main` and
    installs with `pip install "git+https://github.com/YannBerthelot/TargetGym"`.

Or try it in the browser: [Colab quickstart](https://colab.research.google.com/github/YannBerthelot/TargetGym/blob/main/notebooks/quickstart.ipynb).

```python
import jax
import target_gym as tg

env, params = tg.make("plane")          # the scored configuration
expert = tg.make_expert("plane")        # the shipped expert
obs, state = env.reset(jax.random.PRNGKey(0), params)
memory = expert.reset()                 # carried like the env's state

for t in range(200):
    action, memory = expert.step(memory, obs)
    obs, state, reward, terminated, truncated, info = env.step(
        jax.random.PRNGKey(t), state, action, params
    )
    if terminated or truncated:
        break
```

[Browse the 25 environments →](environments.md){ .md-button .md-button--primary }
[Getting started →](getting-started.md){ .md-button }

---

## Why these environments

Holding a setpoint forever breaks differently than reaching a goal once, and
these are the failure modes that come with it:

| | |
|---|---|
| **Irrecoverable states** | A boiler drum that carries water into the turbine, a reactor past runaway, a kiln that has gone cold |
| **Partial observability** | The furnace hides 6 of 9 states, the reactor 7 of 11, the kiln 64 behind 8 measurements |
| **Non-minimum phase** | Drum level rises as mass *leaves*; the four-tank's obvious loop pairing is unstable |
| **Transport delay** | Half the kiln's response to a fuel change takes a full 25-minute residence time |
| **Multi-timescale** | Millisecond neutronics against hour-long xenon; sub-second flame gas against 30 h glass residence |
| **A trip the reward does not track** | A compressor trips when its flow margin to surge runs out. No reward term reads that margin, and the recycle valve that protects it has no tracked output of its own |

Every environment ships a tuned PID, and 20 of the 25 ship an MPC (on
`patrol_range_bearing` it reads the true state, a full-state bound), so a learned
policy has something real to beat. The five without one, `plane`, `plane_energy`,
`plane3d_heading`, `plane3d_racetrack` and `wake_surf`, say why on their pages. And **where a baseline is weak, the docs say how weak**.

## Documentation

<div class="grid cards" markdown>

- **Use it**

    [Getting started](getting-started.md) ·
    [Environments](environments.md) ·
    [API reference](api.md)

- **Beat the reference**

    [Comparing an agent with the reference](comparing.md) ·
    [Baseline results](baseline-results.md) ·
    [RL protocol](rl-protocol.md) ·
    [RL results](rl-baselines.md)

- **Trust the numbers**

    [Physics methodology](PHYSICS_METHODOLOGY.md) ·
    [Model review checklist](model-review-checklist.md) ·
    [Reward shaping](reward-shaping.md) ·
    [Testing](testing.md)

- **Contribute**

    [Contributing](https://github.com/YannBerthelot/TargetGym/blob/main/CONTRIBUTING.md) ·
    [Roadmap and known gaps](roadmap.md) ·
    [Functional structure](functional-structure.md)

</div>
