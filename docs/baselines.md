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

One contract, asserted for every registered environment, runs in the `slow`
job: the PID must beat the **best constant action**. A weak bar, but exactly
the one a mis-indexed setpoint fails -- it caught a furnace PID tracking fuel
percentage as its temperature setpoint.

There is deliberately **no** "MPC beats PID" contract, and it would not pass if
there were: on the 2D aircraft the shipped MPC settles about 2 400 m from the
target altitude where the PID settles about 390 m. Writing the contract would
therefore be asserting something untrue. It is recorded here instead, and in
the roadmap, as work rather than as a guarantee.

## Structure over gains

The right structure usually matters more than the numbers, and the shipped
baselines are chosen to show it:

- **Three-element control on the boiler drum.** Feedwater tracks measured steam
  flow as a feedforward, so shrink-and-swell cannot fool the level loop: the
  drum level *rises* when steam demand increases, and a naive level controller
  responds by cutting feedwater at exactly the wrong moment.
- **A cascade on the cement kiln.** Integral action on a measurement half an
  hour old oscillates at the delay period; an inner loop on a faster
  measurement is what makes the outer loop tractable.
- **Crossed loops on the four-tank.** Its relative gain array puts λ11 at
  −0.067, so pairing each pump with the tank beneath it -- the obvious choice --
  is unstable. The shipped PID pairs them the other way.
- **A cascaded autopilot for aircraft altitude** (altitude → vertical speed →
  pitch → elevator) with attitude limiting and angle-of-attack protection. A
  single loop mapping altitude error straight to elevator departs controlled
  flight on large climbs.

## The MPC objective is not the reward

An MPC objective must share the reward's *minimiser*, not its shape. Copying a
clipped tracking reward gives the optimiser no gradient exactly where it is
needed; dropping the clip makes large errors score better than they should.
Both failures happened here before the objectives became plain quadratics.

The cement kiln is the clearest case for choosing the implementation to fit the
plant: its free lime depends on temperature through a 280 kJ/mol Arrhenius term
that is then advected down the kiln, so reverse-mode gradients overflow to NaN
after about eight steps while finite differences on the same objective stay
clean. Hence cross-entropy sampling rather than a gradient method.
