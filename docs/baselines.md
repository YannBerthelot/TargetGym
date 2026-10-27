# How the baselines work

Every environment ships two reference controllers, a PID and an MPC, so that a
learned policy has something real to beat. The numbers are on
[Baseline results](baseline-results.md). This page says what each controller
is, what to know before quoting one, and how the numbers are kept current.

## Read the results as facts about these tasks

The suite collects tasks where **anticipation pays**: lookahead, constraint
handling or a model earn their keep. It is not evidence that PID control is
inferior, and no number in it should be quoted that way.

A PID is optimal, or close enough that the gap cannot be measured, whenever the
reference and the disturbances can be reacted to rather than foreseen. Twice
this project found a task where no controller could beat the PID, and both times
the task was at fault: the battery tracked a dispatch signal that was mostly
noise, and the patrol lead flew one constant turn that feedforward cancels
outright. Both tasks were changed (see their `PHYSICS.md`).

So where a PID sits close to the MPC, the usual reason is that a good reactive
controller can solve that task, which is often the right answer in practice. A
PID is cheap, transparent, certifiable and runs on a microcontroller; none of
that shows up in a return.

## Calling a baseline like an agent

Both baselines are available as one callable, so a single evaluation loop
serves the PID, the MPC and a learned policy written to the same shape:

```python
import jax, numpy as np
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
policy = baseline_policy(spec, "pid", params)      # or "mpc", or your agent

key, reset_key = jax.random.split(jax.random.PRNGKey(0))
obs, state = env.reset_env(reset_key, params)
total = 0.0
for _ in range(int(params.max_steps_in_episode)):
    key, step_key = jax.random.split(key)   # a fresh key every step
    action = policy(np.asarray(obs), state)
    obs, state, reward, terminated, _ = env.step_env(step_key, state, action, params)
    total += float(reward)
    if bool(terminated):
        break
```

Both take `(obs, state)`, but they read different things. **The PID reads only
`obs`**, as a plant controller does. **The MPC reads only `state`**, the true
simulator state, so it is an upper bound rather than a peer to a policy that
sees what a plant instruments.

To compare your own agent, evaluate it on the seeds the baselines were recorded
on, 0 to 9. The comparison is then paired, and the baseline returns can be read
from `src/target_gym/data/baseline_returns.json` instead of re-run.
[Using the PIDs](using-the-pids.md) walks through this, and through using the
PID in training.

## The PID

Each PID reads the observation and is tuned once per environment. Most are a
single loop or two; where the plant needs it, the structure does the work that
gains cannot:

- **Three-element control on the boiler drum.** Feedwater follows measured
  steam flow, so shrink-and-swell cannot fool the level loop. The drum level
  *rises* when steam demand increases, and a plain level controller would cut
  feedwater at exactly the wrong moment.
- **A cascade on the cement kiln.** Integral action on a measurement half an
  hour old oscillates at the delay period; an inner loop on a faster
  measurement makes the outer loop tractable.
- **Crossed loops on the four-tank.** Its relative gain array puts λ11 at
  −0.067, so pairing each pump with the tank beneath it is unstable. The
  shipped PID pairs them the other way.
- **A fixed-wing autopilot for the aircraft**, as PX4 and ArduPilot fly it:
  TECS for height and speed over a pitch loop, and on the 3D tasks a bank loop
  under a course hold or L1 path guidance. A single loop from altitude error
  to elevator departs controlled flight on large climbs.

The aircraft experts are tuned by `scripts/tune_aircraft_guidance.py`, on
training seeds and never on the scored ones, and their numbers live beside
them in code. The other gains are tuned by `scripts/tune_pid.py` (relay
autotuning on the process plants) and live in
`src/target_gym/data/pid_gains.json`:

```bash
uv run python scripts/tune_pid.py --envs cstr    # or `make tuning-cstr`
```

The script skips environments already in the file, so naming one with `--envs`
is how you re-tune it. Before keeping a re-tune:

- **Measure it against the shipped gains.** The search is stochastic and
  sensitive to the operating point, and a fresh "best" can be worse.
- **Check the winner is not on the edge of the grid.** A score that rises all
  the way to the last grid point is a boundary, not an optimum.
- **Check the task is feasible first.** No gain flies a circle that needs more
  bank than the aircraft allows.

## The MPC

The MPC slot holds the best controller we can build for each task, as an upper
bound for the others. Where the best causal action has a closed form, the slot
holds that instead of a planner. Each task's controller lives in its package's
`experts.py`:

| Implementation | Used by | When it applies |
|---|---|---|
| `CasadiMPC` | first_order, ph_neutralization, glass_furnace, reactor, hvac, unstable_cstr, compressor_surge | A nonlinear program over an explicit model, solved by IPOPT |
| `ShrinkingHorizonNLP` | cstr, four_tank | A deterministic plant with an episode short enough to plan whole: one NLP over every remaining step, on the environment's own step |
| `GradientMPC` | distillation, boiler_drum, plane_energy, the four 3D aircraft tasks | Differentiates the JAX dynamics and descends the objective |
| Two gradient planners with a handover | plane, plane_sine | Both actuators planned to capture the initial altitude, then the elevator only, with the throttle on the PID's airspeed loop |
| `SamplingMPC` | cement_kiln | Cross-entropy sampling: gradients through the kiln's transport delay overflow |
| Feedforward | battery | Commands the scheduled level of the next block, which leaves only the dispatch noise |
| Feedback law plus a residual planner | patrol, patrol_bearing_only, formation | Flies the lead's own autopilot on the follower's state, and a 30-step planner adds a correction; on formation one planner corrects every seat, the lead's too |
| Feedback law | wind_turbine | Generator torque solved so the next step's power meets the target; pitch from the PID with its slew capped |

The MPC has no learned parameters. It solves its problem online at every step
(the four-tank every 25), and carries only a warm start and a few estimator or
fallback states from step to step. **Call `reset()` between episodes**, or that
memory leaks into the next one.

### Four things to know before quoting an MPC number

**It reads the true state.** It knows the reactor's xenon inventory, the
column's interior profile and the turbine's rotor-effective wind, which are
exactly what those environments hide. So part of every MPC lead is
information rather than planning, and the results are not a comparison at equal
information. On `patrol_bearing_only` the slot holds `patrol`'s oracle reading
the true state, so its score there measures what the hidden lead heading is worth
plus what the PID leaves on the table.

**It plans on the mean disturbance.** The planners zero the noise amplitudes the
spec declares in `noise_fields` for their own copy of the params, so they plan
on the mean path: certainty equivalence, not a robust formulation.

**It does not always minimise the reward exactly.** See the next section.

**It is slow**, from about 0.1 ms per step for the feedforward and feedback
laws to about 1.3 s for the 3D aircraft. The CasADi planners call IPOPT, outside
JAX, so they do not jit or vmap. Use it as a reference to measure against, not
inside a training loop; the recorded returns save you running it at all.

### Why the MPC does not minimise the reward

An MPC objective must share the reward's **minimiser**, not necessarily its
shape, and must give the optimiser a gradient.

- **The gradient planners** descend the plant's own cost in floor units. Where
  the tracking cost is linear in the error, they square it, which keeps the
  minimiser and gives a gradient that vanishes at the optimum. A trip is a
  boolean with no gradient, so they add a smooth barrier on the approach,
  weighted at the failure cost.
- **The CasADi planners** minimise a quadratic in a per-plant error band
  (`tracking_band`, `comfort_band` and similar). These bands are controller
  constants, not reward parameters. The quadratic shares the reward's
  minimiser but not its trade-offs against a running cost, so a badly sized
  band can settle the plant off target to save fuel.
- **The CSTR and four-tank NLPs** minimise the reward's own tracking term,
  weighted toward the scored window, and reach the plant's optimum to float32
  rounding.

### Solver limits, and how often they bit

A real MPC has a sample period and applies the best iterate it holds when time
runs out, so `CasadiMPC` caps IPOPT at 150 iterations (`IPOPT_MAX_ITER`), with a
CPU-time backstop far above any healthy step. The iteration cap is the one
meant to bind because it is deterministic, so a baseline recorded on one
machine reproduces on another. `ShrinkingHorizonNLP` is capped at 500.

A capped solve is applied; any other failure (infeasible, restoration failed)
returns an iterate that means nothing, so the controller holds its previous
action. The unstable CSTR's and the compressor's MPCs hand such a step to their
PIDs instead, because holding the last action could trip either plant. Every
record counts these, so a baseline cannot quietly stop being an upper bound:

| field | meaning |
| --- | --- |
| `solver_calls` | solves performed across all seeds |
| `solver_failures` | solves that did not reach a converged status |
| `solver_capped` | of those, the ones stopped by the iteration or time cap |
| `solver_mean_iters` | mean IPOPT iterations per solve |

Read `solver_failures` before quoting an MPC number;
`scripts/record_baselines.py` prints the success rate as it records.

### What the MPC does not guarantee

- **No stability certificate.** No planner has a terminal set, and among the
  CasADi planners only the unstable CSTR's and the compressor's weight the end
  of their horizon, so there is no nominal stability guarantee in the Mayne
  sense. The baselines are measured, not certified.
- **Some aircraft horizons are short.** `scripts/audit_mpc_horizons.py` checks
  each horizon against the time a viable controller needs to close the error.
  On `plane`, `plane3d_heading` and `plane3d_circle` the 30-step horizon covers
  0.75 to 0.81 of it. A longer horizon helps only at several times the size and
  about 9 times the compute, so this is an open item rather than a tuning knob.

## How the numbers are kept current

Rolling out every controller takes about 40 minutes of CPU, and the answer only
moves when the physics, the controllers or their gains move. So the rollouts
are recorded by hand and committed, and CI checks the record instead:

```bash
make baselines              # everything, ~40 min
make baselines-plane        # or one environment
```

Or add the `re-record` label to a pull request: CI records every stale
environment on GitHub's runners and commits the result to the branch. A weekly
run re-records everything, to catch a dependency bump that moves the numbers
without touching the code.

**A stale record cannot pass.** Each record carries a fingerprint of everything
that determines it: the environment's modules, the shared controller and
integration code, its tuned gains, the parameters it was measured at, and the
protocol's burn-in. A test refuses a record whose fingerprint no longer matches
the tree, and names the command that regenerates it. These checks run as CI's
`records` check, apart from the other tests, so a stale record never hides a
real failure. Comments and docstrings
are excluded, so prose edits never stale a record. `src/target_gym/provenance.py`
explains why the fingerprint is taken over source rather than behaviour.

**What CI asserts from the record**, on every push:

- The PID beats the best constant action. A weak bar, but exactly the one a
  mis-indexed setpoint fails.
- The MPC returns no less than the PID's mean return minus 10% of its size.
  That tolerance is a tripwire against an MPC that has stopped controlling, not
  the published comparison. An MPC that runs but loses to its PID would be
  recorded in `EnvSpec.mpc_degraded`, and none currently is.
