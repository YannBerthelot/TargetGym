# Baselines

Every environment ships a controller so that a learned policy has something
real to beat. A benchmark whose only reference point is a random policy tells
you an agent learned *something*; one with a tuned PID tells you whether it
learned anything worth having.

Reach a baseline through the registry rather than by importing a factory:

```python
from target_gym.registry import REGISTRY

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.params_cls()

pid = spec.make_pid()
pid.reset()

mpc = spec.make_mpc(env, params)
mpc.reset()
```

## Coverage

All eighteen environments ship a PID. Sixteen also ship an MPC; the two
`patrol` variants do not, and `EnvSpec.baselines_note` records why -- the
follower's plant is the full 3D aircraft and its reference is a *manoeuvring
lead*, so an MPC needs the lead's future trajectory as a time-varying
parameter, which is not yet wired.

A missing baseline is a documented gap rather than a silent one: the
conformance suite reads `baselines_note` and skips with that reason, so a
baseline cannot quietly disappear.

## PID

The PID baselines are stateful objects with `reset()` and `__call__(obs)`.
They range from a single loop to gain-scheduled and cascaded structures, and
for the multi-loop plants a MIMO form with a deliberate pairing -- the
four-tank's loops are **crossed**, because its relative gain array puts
λ11 at −0.067 and the obvious pairing is unstable.

Gains are tuned by `scripts/tune_pid.py` and cached in `data/pid_gains.json`:

```bash
uv run python scripts/tune_pid.py --envs cstr    # or `make tuning-cstr`
make clear-tuning                                # drop the cache and retune
```

## MPC

Three implementations, chosen per environment by what its dynamics allow:

| Implementation | Used by | When it applies |
|---|---|---|
| `CasadiMPC` subclasses | 7 environments | A direct nonlinear program over an explicit model; the sharpest when the model can be written in CasADi |
| `GradientMPC` | 8 environments | Differentiates the JAX dynamics directly and descends the objective |
| `SamplingMPC` | cement kiln | Cross-entropy sampling, for when gradients are unusable |

The cement kiln uses sampling because its adjoint overflows: half its response
to a fuel change takes a full 25-minute residence time, and differentiating
back through that transport delay does not survive in floating point.

An MPC objective must share the **minimiser** of the environment's reward, not
its shape. A reward with a flat or clipped region is fine to score against but
useless to descend, so the MPC objectives are written to be smooth where the
reward is not.

MPC rollouts are expensive, so episodes are cached under `data/mpc_cache/`:

```bash
make clear-mpc     # drop the MPC trajectory cache
```

## What the suite guarantees

Two contracts, asserted for every registered environment, run in the `slow`
job:

- the PID beats the **best constant action** -- a weak bar, but exactly the
  one a mis-indexed setpoint fails;
- the MPC is at least as good as the PID.

These are what stop a baseline from silently degrading into a controller that
is well-formed but not controlling.
