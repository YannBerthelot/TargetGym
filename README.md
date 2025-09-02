# ✈️ Plane: Reinforcement Learning Environment for Aircraft Control

![Demo of Plane environment](demo.gif)

**Plane** is a lightweight yet realistic **reinforcement learning environment** simulating a 2D side view of an Airbus A320-like aircraft.
It’s designed for **fast, end-to-end training on GPU with JAX** while staying **physics-based** and **realistic enough** to capture the core challenges of aircraft control.

Plane allows you to benchmark RL agents on **delays, irrecoverable states, partial observability, and competing objectives** — challenges that are often ignored in standard toy environments.

---

## ✨ Features

* 🏎 **Fast & parallelizable** thanks to JAX — scale to thousands of parallel environments on GPU/TPU.
* 📐 **Physics-based**: Dynamics are derived from airplane modeling equations (not arcade physics).
* 🧪 **Reliable**: Covered by unit tests to ensure stability and reproducibility.
* 🎯 **Challenging**: Captures real-world aviation control problems (momentum, delays, irrecoverable states).
* 🔄 **Compatible with multiple interfaces**: Designed to work with JAX-based environments.
* 🌟 **Upcoming features**: Environmental perturbations (e.g., wind) will be available in future releases.

---

## 📊 Stable Altitude vs. Power & Pitch

Below is an example of how stable altitude changes with engine power and pitch:

![Stable altitude graph](altitude_vs_power_and_stick.png)

This highlights the **multi-stability** phenomenon: holding a constant power setting can lead the plane to naturally converge to a stable altitude.

---

## 🚀 Installation

Once released on PyPI, you can install Plane with:

```bash
# Using pip
pip install plane-env

# Or with Poetry
poetry add plane-env
```

---

## 🎮 Usage

Here’s a minimal example of running an episode and saving a video:

```python
from plane_env.env_jax import Airplane2D, EnvParams

# Create env
env = Airplane2D()
seed = 42
env_params = EnvParams(max_steps_in_episode=1_000)

# Simple constant policy with 80% power and 0° stick input.
action = (0.8, 0.0)

# Save the video
env.save_video(lambda o: action, seed, folder="videos", episode_index=0, params=env_params, format="gif")
```

Of course you can also directly use it to train an agent using your favorite RL library (here: stable-baselines3)

```python
from plane_env.env_gymnasium import Airplane2D, EnvParams
from stable_baselines3 import SAC

# Create env
env = Airplane2D()
# Model training (adapted from https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)


model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, log_interval=4)
model.save("sac_plane")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_plane")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```


---

## 🛩️ Environment Overview (Reinforcement Learning Perspective)

**State (`EnvState`)**: 13-dimensional vector representing aircraft dynamics:

| Variable          | Description                             |
| ----------------- | --------------------------------------- |
| `x`               | Horizontal position (m)                 |
| `x_dot`           | Horizontal speed (m/s)                  |
| `z`               | Altitude (m)                            |
| `z_dot`           | Vertical speed (m/s)                    |
| `theta`           | Pitch angle (rad)                       |
| `theta_dot`       | Pitch angular velocity (rad/s)          |
| `alpha`           | Angle of attack (rad)                   |
| `gamma`           | Flight path angle (rad)                 |
| `m`               | Aircraft mass (kg)                      |
| `power`           | Normalized engine thrust (0–1)          |
| `stick`           | Control stick input for pitch (–1 to 1) |
| `fuel`            | Remaining fuel (kg)                     |
| `t`               | Current timestep                        |
| `target_altitude` | Desired target altitude (m)             |

The state also provides **derived properties** like air density, Mach number, and speed of sound.

The agent currently observes all of the state, minus **x** and **t** (as they should be irrelevant for control), as well as fuel which is currently not used.

**Action Space**: Continuous 2D vector `[power_requested, stick_requested]` controlling engine thrust and pitch.

**Reward Function**:

* Encourages maintaining **target altitude**.
* Terminal altitude violations (`z < min_alt` or `z > max_alt`) incur `-max_steps_in_episode`.
* Otherwise, reward is sthe quared normalized difference to target altitude:

$`r_t = \left( \frac{\text{max\_alt} - | \text{target\_altitude} - z_t |}{\text{max\_alt} - \text{min\_alt}} \right)^2`$



**Episode Termination**:

* **Altitude limits exceeded** → terminated
* **Maximum episode length reached** → truncated

**Time step**: `delta_t = 0.5 s`, `max_steps_in_episode = 1,000`.

---

## 🧩 Challenges Modeled

Plane is designed to test RL agents under **realistic aviation challenges**:

* ⏳ **Delay**: Engine power changes take time to fully apply.
* 👀 **Partial observability**: Some forces cannot be directly measured.
* 🏁 **Competing objectives**: Reach target altitude fast while minimizing fuel and overshoot.
* 🌀 **Momentum effects**: Control inputs show delayed impact due to physical inertia.
* ⚠️ **Irrecoverable states**: Certain trajectories inevitably lead to failure (crash).

> Environmental perturbations (wind, turbulence) are coming in a future release.

---

## 📦 Roadmap

* [ ] Add perturbations (wind with varying speeds and directions) to model the non-stationarity of the dynamics.
* [ ] Add an easier interface to create partially-observable versions of the environment.
* [ ] Provide ready-to-use benchmark results for popular RL baselines.
* [ ] Add fuel consumption.

---

## 🤝 Contributing

Contributions are welcome!
Please open an issue or PR if you have suggestions, bug reports, or new features.

---

## 📜 License

MIT License – feel free to use it in your own research and projects.
