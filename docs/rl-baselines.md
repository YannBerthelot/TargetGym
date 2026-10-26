# Learned-policy results

Every environment ships a tuned PID and an MPC upper bound so that a learned
policy has something real to beat. This page is the other half of that
comparison, and the honest state of it is: **the harness is in place and no
results are published yet.**

## Why the numbers live in a file

Training costs GPU-hours, and the answer only moves when the environment or the
agent moves, so learned results are recorded once and checked, like the
baselines.

Results go in `src/target_gym/data/rl_results.json`, written through
`target_gym.rl_results.record_result`, and the suite checks three things about
each entry: that it is well formed, that it is scored over the environment's own
episode so it sits in the same column as the PID and MPC numbers, and that it
still describes the environment it claims to.

Nothing asserts that RL beats a PID. Whether it does is the question this library
exists to ask, and a test that presumed the answer would be worthless.

## What invalidates a result, and what does not

The guard is `environment_fingerprint`, which is deliberately *narrower* than the
one the shipped baselines use. An agent never calls a PID, so re-tuning a
controller must not throw away a training run that cost GPU-hours. A change to
the dynamics, the reward, the parameters, the integrator or the shared reward
helper must.

The agent's own configuration -- learning rate, network width, timesteps -- is
stored verbatim and is deliberately *not* fingerprinted: two runs of the same
agent at different hyperparameters are different results, not stale ones, and
both are worth keeping. `tag` separates them.

The experimental design -- which agents, which hyperparameters, how many seeds,
what the expert-based arm actually is -- is fixed in advance in
[the measurement protocol](rl-protocol.md). It is written to be one a reader who
wanted the opposite conclusion would accept.

## Running the training

Training happens outside this package. The dependency runs one way -- from the RL
library to here -- so installing TargetGym never drags in an RL framework, and
there is no cycle to untangle.

[Ajax](https://github.com/YannBerthelot/Ajax) is the intended runner. It is
JAX-native, already depends on this package, and already handles the two things
that would otherwise bias every number on these tasks: termination against
truncation, and preserving the terminal observation at a time-limit truncation
so the value bootstrap is correct. These episodes truncate at `max_steps`
constantly, so getting that wrong would quietly distort every result.

```python
# doc: skip -- the call shape; it needs a trained agent's returns to run
from target_gym.rl_results import record_result

record_result(
    env="plane",
    algorithm="SAC",
    library="ajax",
    library_version=ajax.__version__,
    returns=per_seed_episode_returns,   # undiscounted, one per seed
    episode_steps=280,                  # plane's scored episode
    total_timesteps=1_000_000,
    config={"n_envs": 64, "lr": 3e-4},
)
```

`returns` must be undiscounted episode returns measured the way the shipped
baselines are. A normalised score or a training curve is not on the same axis as
the numbers it would sit beside, and the suite rejects an episode longer than
the environment's own.

## Reading the result, when there is one

Before trusting a win or a loss, run the checks in
[How to read the result](rl-protocol.md#12-how-to-read-the-result): cross-check
a loss against a standard implementation, and check that a win was not against
a defective baseline or a simulator exploit.
