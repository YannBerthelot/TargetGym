import time
import jax
import jax.numpy as jnp
from functools import partial
import diffrax
import matplotlib.pyplot as plt

from plane_env.plane.dynamics import (
    compute_velocity_and_pos_from_acceleration_integration,
    compute_acceleration,
)
from plane_env.plane.env import EnvParams

# --- Wrap compute_acceleration to match expected signature ---
def make_acceleration_fn(action, params):
    def accel_fn(velocities, positions):
        return compute_acceleration(velocities, positions, action, params)
    return accel_fn

# --- Diffrax baseline ---
def diffrax_baseline(velocities, positions, acceleration_fn, delta_t_total):
    def ode_fn(t, y, args):
        v, p = y[:3], y[3:]
        a, _ = acceleration_fn(v, p)
        dydt = jnp.concatenate([a, v])
        return dydt

    y0 = jnp.concatenate([velocities, positions])
    term = diffrax.ODETerm(ode_fn)
    solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term,
        solver=solver,
        t0=0.0,
        t1=delta_t_total,
        dt0=0.1,
        y0=y0,
        max_steps=100_000
    )
    v_traj = sol.ys[:, :3]
    p_traj = sol.ys[:, 3:]
    return v_traj, p_traj

# --- Benchmark function ---
def benchmark_aircraft(params, initial_action, steps=5000, delta_t=0.1):
    velocities0 = jnp.array([200.0, 0.0, 0.0])
    positions0 = jnp.array([0.0, 3000.0, 0.0])

    acceleration_fn = make_acceleration_fn(initial_action, params)

    methods = [
        ("rk4_1", "RK4_1"),
        ("euler_10", "Euler_10"),
        ("euler_20", "Euler_20"),
    ]

    results = {}

    # --- Diffrax baseline ---
    print("Computing high-precision baseline (Diffrax Tsit5)...")
    total_time = steps * delta_t
    t0 = time.time()
    v_gt, p_gt = diffrax_baseline(velocities0, positions0, acceleration_fn, total_time)
    elapsed_gt = time.time() - t0
    results["Ground truth"] = {"v_traj": v_gt, "p_traj": p_gt, "time_s": elapsed_gt}
    print(f"Ground truth done in {elapsed_gt:.4f}s")

    # --- JIT each method ---
    for method_str, label in methods:
        @partial(jax.jit, static_argnames=("method_str",))
        def step(v, p, method_str=method_str):
            return compute_velocity_and_pos_from_acceleration_integration(
                v, p, delta_t, acceleration_fn, method=method_str
            )

        # Warm-up
        v, p, _ = step(velocities0, positions0)
        v.block_until_ready()
        p.block_until_ready()

        # Store trajectories
        v_traj = [v]
        p_traj = [p]

        # Run simulation
        v, p = velocities0, positions0
        start_time = time.time()
        for _ in range(steps):
            v, p, _ = step(v, p)
            v_traj.append(v)
            p_traj.append(p)
        v.block_until_ready()
        p.block_until_ready()
        elapsed = time.time() - start_time

        results[label] = {"v_traj": jnp.stack(v_traj),
                          "p_traj": jnp.stack(p_traj),
                          "time_s": elapsed}

    # --- Compute L2 errors at final step ---
    gt_v = results["Ground truth"]["v_traj"][-1]
    gt_p = results["Ground truth"]["p_traj"][-1]

    for label in ["RK4_1", "Euler_10", "Euler_20"]:
        v_err = jnp.linalg.norm(results[label]["v_traj"][-1] - gt_v)
        p_err = jnp.linalg.norm(results[label]["p_traj"][-1] - gt_p)
        print(f"{label}: time={results[label]['time_s']:.4f}s, "
              f"L2 velocity error={v_err:.6f}, L2 position error={p_err:.6f}")

    print("Ground truth time:", results["Ground truth"]["time_s"])

    # --- Plot trajectories ---
    plt.figure(figsize=(12, 5))
    for label in ["Ground truth", "RK4_1", "Euler_10", "Euler_20"]:
        traj = results[label]["p_traj"]
        plt.plot(traj[:,0], traj[:,1], label=label)
    plt.xlabel("x (horizontal position)")
    plt.ylabel("z (altitude)")
    plt.title("Aircraft trajectory: x vs z")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Run benchmark ---
if __name__ == "__main__":
    params = EnvParams()
    initial_action = (0.8, 0.2)
    benchmark_aircraft(params, initial_action, steps=5000, delta_t=0.1)
