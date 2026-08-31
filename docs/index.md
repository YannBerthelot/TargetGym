# TargetGym documentation

JAX reinforcement learning environments for **target MDPs** -- tasks where the
objective is to reach and hold a subset of the state space against
disturbances, not to reach a goal and stop. Holding a setpoint, forever, is
what industrial control actually is.

| | |
|---|---|
| **[Getting started](getting-started.md)** | Install, run an episode, plug into Gymnasium or a JAX training loop |
| **[Environment reference](environments.md)** | All eighteen: shapes, tracked variables, baselines, physics contracts |
| **[Public API](api.md)** | What is stable, what is not, and what changes at 1.0 |
| **[Baselines](baselines.md)** | The shipped PID and MPC controllers, and how to tune them |
| **[Reward shaping](reward-shaping.md)** | Why the tracking rewards have the shape they do, with the measurements |
| **[Physics methodology](PHYSICS_METHODOLOGY.md)** | How each environment's physics is sourced, validated and bounded |
| **[Contributing](../CONTRIBUTING.md)** | Tests, style, and what adding an environment involves |

## What makes these environments different

Every environment's physics is a **documented, tested contract** rather than a
claim. Each carries a `PHYSICS.md` beside its module giving a sourced parameter
table, published validation targets that the test suite asserts, and quantified
known deviations from the literature. A deviation that cannot be fixed today is
recorded and pinned with a strict xfail, so fixing it later fails loudly
instead of passing unnoticed.

They also target the failure modes that make real control hard:

| | |
|---|---|
| **Irrecoverable states** | A boiler drum that carries water into the turbine, a reactor past runaway, a kiln that has gone cold |
| **Partial observability** | The furnace hides 6 of 9 states, the reactor 7 of 11, the kiln 64 behind 8 measurements |
| **Non-minimum phase** | Drum level rises as mass *leaves*; the four-tank's obvious loop pairing is unstable |
| **Transport delay** | Half the kiln's response to a fuel change takes a full 25-minute residence time |
| **Multi-timescale** | Millisecond neutronics against hour-long xenon; sub-second flame gas against 30 h glass residence |
| **Finite budgets** | A battery whose tracking *now* costs the ability to track later |

Every environment ships a PID baseline and most also ship an MPC, so a learned
policy has something real to beat -- and where a baseline is missing, the
registry records why.
