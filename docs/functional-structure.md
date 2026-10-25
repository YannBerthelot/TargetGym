# Functional structure, and what monadic patterns would buy

A study of whether this library should adopt monadic patterns, written after
measuring what it already does. The short answer: **the useful half is already
here and unnamed, the code should lean on it much harder, and the half that
looks most attractive is the half JAX cannot support.**

## What is already monadic here

Three of the library's central shapes are monadic without saying so.

`step_env` is `(key, state, action, params) -> (obs, state', reward, terminated,
info)`. Not `done`: gymnax 1.0 splits that into natural termination, which
`step_env` reports, and truncation, which the base `step` derives from the
step count.
That is a state transformer that also emits an output — State plus Writer, in
the usual naming. `jax.lax.scan`, which the runners already use, *is* the fold
of that monad: it threads the carried state and collects the outputs, which is
precisely `mapM` over a State computation.

Every expert has the same shape: `step_fn(params, state, obs) -> (action, state')`.
That is `obs -> State PIDState action`, written out by hand.

So the question is not whether to introduce monads. It is whether to name what
is already there and take the consequences.

## The measured problem

`src/target_gym/experts/pid.py` is 3312 lines, and the PID recurrence — advance
the integral, difference the error, sum three terms, clip, undo the integral if
saturated — appears in **about thirty places**. Counting the anti-windup line
alone, the one step every transcription needs, finds 30 sites.

That is not merely repetitive; it has already cost something. Check 12 of
[the model review checklist](model-review-checklist.md) exists because a fix
applied to the three `plane3d_*_pid_step` functions changed the benchmark by
*nothing at all* — identical to the decimal — since `EnvSpec.make_pid` returns a
different, hand-written implementation of the same three control laws. The
heading law exists in three copies, the circle law in three, the figure-8 in two.

**The environments already solved this and the experts did not.** Physics here is
written once and read twice:

```python
# doc: skip -- the shape of the convention, not a runnable snippet
def compute_reward(state, params, xp=jnp): ...
```

`xp` is the array module. Pass `numpy` and it runs on the host in a Python loop;
pass `jax.numpy` and the same source traces into a jit. Forty-five functions take
that parameter across seventy-seven call sites. It is a tagless-final
interpreter: one expression of the model, two evaluators.

The experts instead keep two hand-written copies — `jnp` for the traced form,
`np` for the host form — and the reason is real. Converting the host-side
controllers from `jnp` to `np` took `tests/experts` from 143 s to 37 s, because
a Python loop calling `jnp` pays dispatch on every scalar. So the split must
survive any refactor. The `xp` parameter is exactly how the environments keep
both without keeping two copies.

## What to adopt

**State, made explicit.** Write each control law once, pure, over `xp`. Then two
runners interpret it: the existing `FunctionalExpertPolicy` for traced use, and
one generic adapter for host use, replacing fourteen hand-written `Stateful*`
classes:

```python
# doc: skip -- proposed, not yet in the package
class Stateful:                      # the "run the State monad" adapter
    def __init__(self, step_fn, params, init, xp=np):
        self._step, self._params, self._init, self._xp = step_fn, params, init, xp
        self.reset()

    def reset(self):
        self._state = self._init(self._params, self._xp)

    def __call__(self, obs):
        action, self._state = self._step(self._params, self._state, obs, self._xp)
        return action
```

The mutable object stops being a second implementation and becomes what it
should always have been: a place to keep the carried state between calls.

**Reader, which is what `xp` and `params` already are.** Both are threaded
through every function unchanged. That is Reader, and the existing style — an
explicit trailing parameter — is the right encoding in Python. It needs no
machinery, only consistency.

## What not to adopt, and why it is the tempting part

The obvious attraction is `Maybe`/`Either` for the things that fail: an episode
that terminates, an MPC solve that does not converge, parameters that do not
validate. Short-circuiting a pipeline on failure is the canonical monadic win.

**It cannot work inside traced code, and the reason is structural rather than a
missing library.** Monadic bind decides *whether to run the rest* by inspecting
the value. Under `jax.jit` that value is a tracer, so branching on it raises
`ConcretizationTypeError`; under `vmap` it is worse than an error, because
different batch elements would need different control flow and there is no such
thing. `lax.cond` under `vmap` does not short-circuit either — it converts to a
`select`, evaluating both branches and discarding one. A `Maybe` that always
runs both sides is not a `Maybe`; it is masking with extra syntax.

Which is exactly what the code already does: `jnp.where(done, ...)`. That is not
a failure to find the elegant abstraction. It is the only thing the execution
model permits.

**And the abstraction would have actively hidden a real bug.** The wind turbine's
MPC needed a differentiable overspeed barrier because masking on a boolean
carries a *cost* but no *gradient* — the planner could see that tripping was bad
and had no derivative pointing away from it, so it drove into the trip at every
horizon from 60 to 200. A `Maybe` presents short-circuiting as free and total.
Here it is neither free nor differentiable, and the whole difficulty is in that
gap. Making it look clean would have made it harder to find, not easier.

Monad transformer stacks are a second no, for a plainer reason: every layer of
indirection between the author and the traced function makes it harder to see
what jit is given. This library has already paid for that kind of distance —
`jnp.asarray(1.0)` producing a weakly-typed scalar that silently recompiled a
cache, and bound methods retaining an executable per instance for the life of
the process. Both were found by reading what was actually traced.

`Maybe` and `Either` remain the right tools **outside** the traced region — in
the tuners, the scripts and the runners, where Python control flow exists and a
solve can genuinely be abandoned. That boundary, traced versus host, is the line
to draw, and it is the same line `xp` already draws.

## Pilot

The smallest instance is done and shipped. `pid_update(e, integral, prev_error,
Kp, Ki, Kd, dt, action_min, action_max, xp)` holds the recurrence once; both
`pid_step` (with `jnp`) and `StatefulPID` (with `np`) call it. The two copies
had even spelled the anti-windup differently — `where(u == u_clipped, new, old)`
against `where(u != u_clipped, integral - e*dt, integral)` — which are equal,
though nothing said so and nothing checked it. 1227 tests pass unchanged.

## Staging the rest

The thirty sites are not thirty copies of one thing, and a single sweeping
rewrite would be the wrong move. There are two distinct anti-windup idioms:

- most loops gate windup on **their own** saturation (`u == u_clipped`);
- the cascades gate on a **downstream** actuator (`abs(aileron) >= 1.0`), holding
  the outer integral when the inner loop is pinned.

The second is not a copy of the first; it is a different and correct piece of
control engineering. A kernel has to express both — plausibly by taking the
saturation flag as an argument rather than deriving it — and that design should
be settled on paper before twenty call sites are moved onto it.

The order that keeps every step verifiable:

1. `pid_update` for the plain-saturation sites. **Done for the SISO pair.**
2. The same for the gain-scheduled and MIMO variants, which share the recurrence
   exactly.
3. A two-phase kernel for the cascade-gated sites, once its shape is agreed.
4. The generic `Stateful` adapter, retiring the hand-written classes law by law,
   each with the benchmark re-measured — because check 12's lesson is that a
   controller refactor which changes *nothing* has usually missed its target,
   and an identical number is stronger evidence of that than a plausible one.

## Status (2026-10-25)

Step 4 is done for the PIDs. `Controller(params, init_state, step)` in
`experts/pid.py` holds one pure `step(params, state, obs, xp)`. `.host()` runs
it with `np`, which is the `Stateful` adapter above under the name
`HostController`, and `.act` and `.batched()` run it with `jnp` for jit, vmap
and GPU. All 23 shipped PIDs are written this way in their packages'
`experts.py`. The hand-written classes are gone, except the compressor surge
and unstable CSTR ones, which their MPCs still read. The gain-scheduled
variants were removed rather than ported, since no environment shipped them.

Every baseline and protocol return re-recorded identical. Here that is the
goal rather than check 12's warning sign: the registry now builds every PID
from the new step functions, so the recording ran the new code, and a change
meant to preserve behaviour should alter nothing. The jitted form agrees with the
host to within 1.3e-4 of the action range, the worst case being the unstable
CSTR, whose host form runs in float64 and whose jitted form runs in float32.
