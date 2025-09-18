# 🎯 TargetGym: Reinforcement Learning Environments for Target MDPs



**TargetGym** is a lightweight yet realistic collection of **reinforcement learning environments** designed around **target MDPs** — tasks where the objective is to **reach and maintain a specific subset of states** (target states).

Environments are built to be **fast, parallelizable, and physics-based**, enabling large-scale RL research while capturing the core challenges of real-world control systems such as **delays, irrecoverable states, partial observability, and competing objectives**.

Currently included environments:

* 🛩 **Plane** – control of a 2D Airbus A320-like aircraft - Stable-Target-MDP
* 🚗 **Car** – maintain a desired speed on a road - Stable-Target-MDP
* 🚲 **Bike** – stabilize and steer a 2D bicycle model - Unstable-Target-MDP (from [Randlov et al.](https://gwern.net/doc/reinforcement-learning/model-free/1998-randlov.pdf)
)

![Demo of Plane environment](videos/plane/output.gif)
![Demo of Car environment](videos/car/output.gif)
![Demo of Bike environment](videos/bike/output.gif)

---

## ✨ Features

* ⚡ **Fast & parallelizable** with JAX — scale to thousands of parallel environments on GPU/TPU.
* 📐 **Physics-based**: Derived from modeling equations, not arcade physics.
* 🧪 **Reliable**: Unit-tested for stability and reproducibility.
* 🎯 **Target MDP focus**: Each task is about reaching and maintaining target states.
* 🌀 **Challenging dynamics**: Captures irrecoverable states, and momentum effects.
* 🔄 **Compatible with RL libraries**: Works with JAX-based and Gymnasium-style interfaces.
* 🌟 **Upcoming features**: Environmental perturbations (wind, turbulence, bumpy road) and fuel consumption.

---

## 📊 Example: Stable Altitude in Plane

Below is an example of how stable altitude changes with engine power and pitch in the **Plane** environment:

![Stable altitude graph](figures/plane/power_trajectories.png)

This illustrates **multi-stability**: with fixed power and pitch, the aircraft naturally converges to a stable altitude. Similar properties can be found in Car environment

---

## 🚀 Installation

Once released on PyPI, install with:

```bash
# Using pip
pip install target-gym

# Or with Poetry
poetry add target-gym
```

---

## 🎮 Usage

Here’s a minimal example of running an episode in the **Plane** environment and saving a video:

```python
from target_gym import Plane, PlaneParams

# Create env
env = Plane()
seed = 42
env_params = PlaneParams(max_steps_in_episode=1_000)

# Simple constant policy with 80% power and 0° stick input
action = (0.8, 0.0)

# Save the video
env.save_video(lambda o: action, seed, folder="videos", episode_index=0, params=env_params, format="gif")
```

Or train an agent using your favorite RL library (example with stable-baselines3):

```python
from target_gym import PlaneGymnasium, PlaneParams
from stable_baselines3 import SAC

env = PlaneGymnasium()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, log_interval=4)
model.save("sac_plane")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

---

## 🧩 Challenges Modeled

TargetGym tasks are designed to expose RL agents to **realistic control challenges**:

* ⏳ **Delays**: Inputs (like engine power) take time to apply.
* 👀 **Partial observability**: Some forces cannot be directly measured.
* 🏁 **Competing objectives**: Reach the target state quickly while minimizing overshoot or cost.![Demo of Plane environment](videos/plane/output.gif)
* 🌀 **Momentum effects**: Physical inertia delays control effectiveness.
* ⚠️ **Irrecoverable states**: Certain trajectories inevitably lead to failure.

---

## 📦 Roadmap

* [ ] Add perturbations (wind, turbulence, uneven terrain) for non-stationary dynamics.
* [ ] Easier interface for creating partially-observable variants.
* [ ] Provide benchmark results for popular RL baselines.
* [ ] Add fuel consumption and resource constraints.

---

## 🤝 Contributing

Contributions are welcome!
Open an issue or PR if you have suggestions, bug reports, or new features.

---


## 📖 Citation

If you use **TargetGym** in your research or project, please cite it as:

```bibtex
@misc{targetgym2025,
  title        = {TargetGym: Reinforcement Learning Environments for Target MDPs},
  author       = {Yann Berthelot},
  year         = {2025},
  url          = {https://github.com/YannBerthelot/TargetGym},
  note         = {Lightweight physics-based RL environments for aircraft, car, and bike control}
}
```


---

## 📜 License

MIT License – free to use in research and projects.


