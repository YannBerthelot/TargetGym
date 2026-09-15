# Baselines

Every environment ships a controller so that a learned policy has something
real to beat. A benchmark whose only reference point is a random policy tells
you an agent learned *something*; one with a tuned PID tells you whether it
learned anything worth having.

## What a PID losing here does and does not mean

This suite is built to collect tasks where **anticipation pays**. It is not
evidence that PID control is inferior, and no number in it should be quoted
that way.

A PID is optimal, or so close that the difference is unmeasurable, whenever
the reference and the disturbances are things you can react to rather than
things you must foresee. Twice now this project has had to learn that from its
own measurements rather than from principle, and both times the environment was
at fault, not the controller:

- The **battery** tracked a dispatch signal whose one-step innovation had a
  standard deviation of 63.6 kW against a 150 kW tracking band. The best
  attainable tracking reward was 0.429; the PID scored 0.447 and the MPC 0.430.
  Both were pinned on an irreducible noise floor. Nothing could have beaten
  that PID, because there was nothing left to control.
- The **patrol follower** chased a lead that held one constant turn rate for a
  whole episode. Adding a single feedforward term took its settled error from
  77.8 m to 2.4 m, against a 3 m reward precision floor. A constant, exactly
  observable rate is cancelled outright by feedforward -- that is what
  feedforward *is* -- so the PID was near-optimal and the MPC had nothing to
  anticipate.

In both cases the honest finding was that the task was not measuring control
skill, and the fix was to the environment: a scheduled dispatch signal for the
battery, a routed lead for the patrol. Both changes are described in the
relevant `PHYSICS.md`.

So read the table below as a statement about **these tasks**, chosen because
lookahead, constraint handling or a model earn their keep in them. Where a PID
sits close to the MPC, the usual explanation is that the task is one a good
reactive controller can solve, which is a fact about the problem and often the
right answer in practice. A PID is cheap, transparent, certifiable and runs on
a microcontroller; none of those properties appear anywhere in a return.

## Calling a baseline like an agent

Both baselines are available as one uniform callable, so a single evaluation
loop serves the PID, the MPC and a learned policy written to the same shape:

```python
import jax, numpy as np
from target_gym.registry import REGISTRY
from target_gym.runners.runners import baseline_policy

spec = REGISTRY["cstr"]
env, params = spec.make_env(), spec.make_test_params()
policy = baseline_policy(spec, "pid", params)      # or "mpc", or your agent

obs, state = env.reset_env(jax.random.PRNGKey(0), params)
total = 0.0
for _ in range(int(params.max_steps_in_episode)):
    action = policy(np.asarray(obs), state)
    obs, state, reward, terminated, _ = env.step_env(
        jax.random.PRNGKey(0), state, action, params
    )
    total += float(reward)
    if bool(terminated):
        break
```

`baseline_policy` returns `None` where an environment does not ship that
baseline, which for the MPC is the two `patrol` variants.

**Both take `(obs, state)`, and the asymmetry underneath is the point.** The PID
ignores `state`: it reads the observation, as a plant controller does. The MPC
ignores `obs`: it reads the true state, because it is a full-state upper bound
rather than a peer to a policy that sees only what a plant instruments. Giving
each the signature it happens to need made that easy to miss, and a benchmark
whose ceiling quietly sees more than its contestants should say so loudly.

To compare your own agent, write it to the same shape and swap it in. Evaluate
on the same episode seeds the baselines were recorded on, 0 to 9, which is what
makes the comparison paired -- and means you can read the baseline numbers out
of `src/target_gym/data/baseline_returns.json` rather than paying to regenerate them.

The underlying objects are reachable directly if you need them, through
`spec.make_pid()` and `spec.make_mpc(env, params)`; both are stateful and want
`reset()` between episodes.

## Coverage

All twenty-one environments ship a PID. Twenty also ship an MPC; only
`patrol_bearing_only` does not, and `EnvSpec.baselines_note` records why -- it
withholds the decomposed slot error a planner would read, which is the point of
the variant, so it needs a planner built on its estimator rather than the
full-observation one.

The obstacle once recorded against a `patrol` MPC -- that its reference is a
manoeuvring lead, so the lead's future trajectory would have to be wired in as
a time-varying parameter -- is real for a CasADi model and irrelevant for a
gradient planner. The lead is scripted and deterministic, so a planner that
differentiates the true `step_env` propagates it for free, exactly as it
propagates the follower.

A missing baseline is a documented gap rather than a silent one: the
conformance suite reads `baselines_note` and skips with that reason, so a
baseline cannot quietly disappear.

## PID

The PID baselines are stateful objects with `reset()` and `__call__(obs)`.
They range from a single loop to gain-scheduled and cascaded structures, and
for the multi-loop plants a MIMO form with a deliberate pairing -- the
four-tank's loops are **crossed**, because its relative gain array puts
λ11 at −0.067 and the obvious pairing is unstable.

Gains are tuned by `scripts/tune_pid.py` and cached in `src/target_gym/data/pid_gains.json`:

```bash
uv run python scripts/tune_pid.py --envs cstr    # or `make tuning-cstr`
make clear-tuning                                # drop the cache and retune
```

The script skips any environment already present in the cache, so re-running it
without `--envs` or `make clear-tuning` is a no-op for everything already tuned.

Two caveats worth knowing before you re-tune anything:

- **Re-tuning can make an environment worse.** The searches are stochastic and
  the relay experiment is sensitive to the operating point, so a fresh "best"
  is not automatically better than what is shipped. Measure both before keeping
  one: re-running the tuner over the whole registry produced a genuinely better
  `cstr` and a distinctly worse `first_order` in the same pass.
- **A tuned gain cannot rescue an infeasible task.** The circle expert's radial
  gains were searched to convergence against a task that, on a third of its own
  radius range, no gain could fly: holding 230 m/s around an 8.4 km circle needs
  33.8 deg of bank against a 30 deg limit, so the aircraft sat pinned at the
  limit for the whole episode. Giving it the airspeed the radius admits was
  worth 31% of the return, against nothing at all from further tuning. Check
  feasibility before searching gains.

- **Check the winner is not the last point in the grid.** The glass furnace's
  search is a grid over (Kp, Ki, Kd), and it returned `Kp=0.040` -- the largest
  value in `kp_grid`, with the score rising monotonically across the entire Kp
  column right up to it. That is the signature of a boundary solution, not an
  optimum, and it is invisible in the output unless you compare the winner
  against the grid's own bounds. Probing outward found the real turn at
  `Kp~5`, roughly 125x further out, and `Ki` was then pinned at *its* edge as
  well. Widening both grids moved the furnace PID from 144.1 to 149.1 over ten
  seeds. The grids now bracket their optima on both sides, and the chosen
  `Ki=0.15` was verified interior by direct probe rather than assumed.

  Beware the interaction: `Ki` and `Kd` had been chosen while `Kp` was pinned,
  so all three had to be re-examined at the new operating point rather than
  just the one that hit the wall.

- **The aircraft are tuned by coordinate descent, not by the relay.** Both of the
  other methods fail on these plants, and structurally rather than by bad luck:
  the relay reports *"every operating point failed (no zero-crossings)"* because
  the altitude/power loop will not sustain a bang-bang oscillation, and the
  gradient tuner returns NaN gains (a documented xfail in
  `tests/experts/test_pid_tuning.py`, hardened against the obvious causes and
  still not localised). Coordinate descent on episode return needs neither an
  oscillation nor a derivative, only forward rollouts.

  It scores **return**, not tracking error. Scoring one term of a
  multi-objective reward is the mistake that made the MPC baselines look broken
  for a long time; the same trap applies here.

  Two bugs were fixed alongside it. Every tuner in both systems pinned
  `integration_method="rk4_1"`, so they would have tuned against the plant as it
  was before the integration order was corrected. And the 2D aircraft's tuner
  wrote the `plane` key while its shipped autopilot reads `plane_cascaded` --
  a key that had never existed in the file. `make tuning-plane` therefore ran,
  reported success, wrote gains nothing loaded, and left the controller on its
  constructor defaults. The tuner now seeds from those defaults and writes the
  key the controller actually reads.

  Gains improved on seeds the search never saw, which is the only number worth
  quoting (search on seeds 0-2, held out 3-9):

  | | shipped | tuned | |
  | --- | --- | --- | --- |
  | plane | 194.62 | 285.20 | +47% |
  | plane3d_heading | 60.93 | 80.25 | +32% |
  | plane3d_circle | 67.23 | 84.59 | +26% |
  | plane3d_figure8 | 34.15 | 81.58 | +139% |

  Those four returns were measured under the previous reward, which charged a
  flat -200 for a crash; the ratios are what the table is for, and the gains
  themselves were re-checked against the current reward and did not move.

## MPC

Three implementations, chosen per environment by what its dynamics allow:

| Implementation | Used by | When it applies |
|---|---|---|
| `CasadiMPC` subclasses | 7 environments | A direct nonlinear program over an explicit model; the sharpest when the model can be written in CasADi |
| `GradientMPC` | 11 environments | Differentiates the JAX dynamics directly and descends the objective |
| `SamplingMPC` | cement kiln | Cross-entropy sampling, for when gradients are unusable |


### The MPC is not trained, and you can just run it

It solves an optimisation problem online, from the current state, at every
step. There are no learned parameters, so there is nothing that could be
specific to an episode or a seed. What it has instead is a model, a horizon and
solver settings, and an objective, all chosen once per environment the way a
controller structure is. Between episodes it carries only a warm start, and on
the glass furnace an offset-free bias integrator; `reset()` clears both.

The irony is that the **PID** is the trained one here. Its gains come from a
search on seeds 0 to 2 and are reported on held-out seeds. The MPC has never
seen a seed before it runs.

Four things to know before you use it.

**It reads the true state, not the observation.** Quote it as a ceiling, not as
an opponent: it knows the reactor's xenon inventory, the column's interior
profile and the turbine's rotor-effective wind.

**It is slow**, from 3 ms to 600 ms per step against environments that step in
microseconds. A reference to measure against, not something to put inside a
training loop.

**It does not vmap or jit** on the CasADi plants, which call IPOPT, a solver
outside JAX. The eleven `GradientMPC` environments do batch, which is how the
recording parallelises across seeds.

**`reset()` between episodes**, or the furnace's bias integrator carries a
correction into an episode where it is a standing error.

And you may not need to run it at all: `src/target_gym/data/baseline_returns.json` holds ten
seeds of both baselines per environment, on the same episodes an agent is
evaluated on.

### Why the MPC does not minimise the reward

Every MPC here optimises a **quadratic surrogate** in a per-plant error band,
not the environment's own reward. That is deliberate, standard, and measured.

It is the difference between *economic* MPC, which optimises the true
objective, and *tracking* MPC, which optimises a quadratic around the setpoint;
quadratic stage costs are the overwhelming norm in practice. Here there are two
independent reasons. The log-scaled reward's gradient is
`-1/((f + e)·log1p(E/f))`, which decays like `1/e`: the pull toward the setpoint
is weakest exactly where the controller is furthest from it. A quadratic in the
normalised error has the same minimiser and a gradient that instead *grows*
with the error. Measured on the wind turbine over six seeds, that difference is
worth almost everything, 341.9 against 172.1 for the reward itself. Separately,
for the CasADi plants a quadratic is far better conditioned than a log, whose
curvature is unbounded at the floor.

So each plant declares an error band the planner normalises by:
`tracking_band` on the four-tank, the column, the pH loop and the glass
furnace, `power_band` on the turbine and the battery, plus `comfort_band`,
`lime_band`, `level_band`, `pressure_band` and `reward_band`.

**These are controller constants, not reward parameters**, and it is worth
saying so loudly because they did not always look like it. Several once carried
comments claiming the reward reached zero, or halved, at the band. It does not:
the rewards normalise by an operating envelope and a `precision_floor`.

The failure mode is specific. Surrogate and reward agree on the *minimiser*,
but not on trade-offs against any **second** term. The glass furnace is the
worked example: its band was 40 K, inherited from a reward the environment had
stopped using, and against a 0.1 fuel weight that made a 3.3 K standing error
the optimum of what the controller was asked to minimise. It sat 6 K cold with
fuel at minimum 80% of the time and trailed its own PID by 16% on ten seeds out
of ten. With the running costs zeroed for this release line there is no second
term anywhere, so no band can currently do that damage. Restoring any weight
re-arms it, which is why the roadmap item on running cost and the one on
deriving these bands are the same piece of work.

The cement kiln uses sampling because its adjoint overflows: half its response
to a fuel change takes a full 25-minute residence time, and differentiating
back through that transport delay does not survive in floating point.

An MPC objective must share the **minimiser** of the environment's reward, not
its shape. A reward with a flat or clipped region is fine to score against but
useless to descend, so the MPC objectives are written to be smooth where the
reward is not.

### Three things to know before quoting an MPC number

These are properties of the baselines as they stand, not defects being hidden.
They matter because they all inflate the MPC side of the comparison below, and
a reader deciding whether their own controller is competitive needs them.

**The MPC sees the full simulator state. The PID sees only the observation.**
Every planner's entry point is `step(obs, state)` with `obs` ignored, and
`runners.mpc_policy` hands it the state object. So on the pH CSTR the MPC reads
the reaction invariants `Wa` and `Wb`; on the glass furnace it reads the glass
and checker temperatures and the pull-rate disturbance; on the reactor it reads
the xenon and iodine inventories and the fuel temperature. Those are exactly
the quantities each environment hides on purpose, and several environments are
built as genuine POMDPs on the strength of that. The PID, and any learned
policy, gets the observation vector alone. **The table below is therefore not a
controller-class comparison at equal information**, and part of every MPC lead
is the hidden state rather than the planning.

**The gradient and sampling planners do not plan on the mean disturbance.**
Both roll the true environment forward internally, and the environments derive
their process noise as `fold_in(key, state.time)`. The planners pass a fixed
`PRNGKey(0)`, so they simulate one specific pseudo-random disturbance
trajectory, consistent across an episode and unrelated to the realisation the
environment will actually produce. That is neither certainty equivalence, which
would use the mean, nor a robust or scenario formulation. It has not been
measured against the alternatives; it is recorded here so nobody assumes
otherwise from the word "MPC".

**The CasADi objectives are quadratic proxies, not the environment's reward.**
The shipped rewards are log-scaled and clip to zero outside the tracking band,
which is fine to be scored on and useless to descend: on the pH CSTR, IPOPT
optimised the only term with a live gradient, the reagent cost, railed the
valve shut and settled at about 3.9 pH of mean error. The proxies share the
reward's minimiser and have a usable gradient everywhere. The consequence is
that these controllers are not optimising the quantity they are scored on, and
because the reward is not quadratic, the plan that minimises the proxy is not
in general the plan that maximises expected reward.

MPC rollouts are expensive, so episodes are cached under `data/mpc_cache/`:

```bash
make clear-mpc     # drop the MPC trajectory cache
```

### The solver is capped, and every record says how often the cap bit

IPOPT ships with a limit of 3000 iterations and no time limit at all. In a
receding-horizon loop that is not a safety net, it is a hang. One badly
conditioned step runs for half an hour while its neighbours take a tenth of a
second, and the episode never ends. It happened here: nine glass-furnace seeds
finished in about three and a half minutes each and the tenth was still going
after seventy, holding up a whole re-record.

A real MPC has a sample period and returns the best iterate it holds when the
clock runs out, so `CasadiMPC` does the same. `IPOPT_MAX_ITER` is 150, against a
healthy furnace step that converges in 21 iterations and a worst healthy step of
38. The iteration cap is the one meant to bind, because it is deterministic: a
baseline recorded on one machine reproduces on another, which a wall-clock cap
could not promise. `IPOPT_MAX_CPU_TIME` is a backstop against a solve that is
pathological rather than merely hard, and sits far above anything a healthy step
needs.

Capping alone would not be enough, because do-mpc neither raises nor warns when
IPOPT gives up. It stores the failed iterate, hands it back as the action, and
warm-starts the next step from it, so nothing downstream can tell a failure from
a converged solve. An MPC baseline can therefore quietly stop being an upper
bound. Every record now carries the count:

| field | meaning |
| --- | --- |
| `solver_calls` | solves performed across all seeds |
| `solver_failures` | solves that did not reach a converged status |
| `solver_capped` | of those, the ones stopped by the iteration or time cap |
| `solver_mean_iters` | mean IPOPT iterations per solve |

A capped solve is still applied: IPOPT was converging and we stopped it, which
is the whole point. Any *other* failure -- infeasible, restoration failed,
invalid number -- returns an iterate that means nothing, so the controller holds
its previous action and restores the previous warm start rather than planning
from the wreckage.

Read `solver_failures` before quoting a number. All seven CasADi environments
currently record 100%.

### Conditioning, constraints, and what is deliberately absent

Three related pieces of standard MPC practice, and where this suite stands on
each.

**Variable scaling.** IPOPT auto-scales the objective and the constraints but
not the decision variables, so step norms, bound handling and the warm start all
run in whatever units the model happens to use. Left alone the reactor handed it
a vector spanning `rho_ext` around 0.0016 up to a precursor concentration around
377, a factor of 605 000 measured over a PID episode, with hard bounds on the
smallest entry. Every `CasadiMPC` subclass now declares a `SCALING` table of
typical magnitudes, taken from the mean of `|x|` over a PID episode and rounded
to one figure.

**Hard bounds on inputs, soft bounds on states.** The optimiser owns the inputs
and can always satisfy their bounds, so those stay hard. A state bound is a
different animal: the initial state comes from the plant, and if the plant walks
it onto the bound the NLP is infeasible at `x0` and IPOPT answers with a
restoration phase and hundreds of iterations instead of an action. Most
environments cannot reach that -- the reactor clips `rho_ext` to its bounds and
the pH plant bisects its algebraic variable on `[0, 14]` -- but the four-tank
does not clip, it *ends the episode* when a level touches `h_min` or `h_max`.
Those two bounds are now soft, and `h_max` is now present at all; before this the
controller was blind to half of a termination condition it is scored on.

**No terminal ingredients.** `mterm` is the stage cost everywhere, so there is
no terminal cost or terminal set and therefore no nominal stability guarantee
in the Mayne sense. These horizons are long relative to the closed-loop
transient they have to cover, which is checked separately by
`scripts/audit_mpc_horizons.py`, and the baselines are measured rather than
certified. It is recorded here so nobody assumes the guarantee exists.

### Horizons, and which ones are too short

`scripts/audit_mpc_horizons.py` checks each MPC's horizon against `tau_close`,
the time a *viable* controller needs to bring the tracking error to 1/e and keep
it there. A receding-horizon controller can only optimise what it can see, so
`horizon * mpc_dt` has to cover that transient. Most environments pass with room
to spare; two groups do not:

| Environment | horizon | `tau_close` | ratio | |
|---|---|---|---|---|
| `plane` | 30 | 37 | 0.81 | myopic |
| `plane3d_heading` | 30 | 40 | 0.75 | myopic |
| `plane3d_circle` | 30 | 40 | 0.75 | myopic |
| `four_tank` | 5 | 198 | 0.03 | myopic |

The aircraft cases are **not** fixed by nudging the horizon to meet the
criterion. Measured on `plane3d_heading` over 150 steps, horizon 30 and horizon
40 both leave the altitude error *larger* than it started (3228 m and 3168 m
against an initial 2623 m) for 17% more compute -- the difference is noise. An
earlier measurement at horizon 80 did help substantially (921 m against 1741 m),
so the horizon really is the binding constraint, but the useful size is several
times the audit's minimum and costs roughly 9x. These are `GradientMPC`
instances, which roll out `step_env` itself, so covered time cannot be bought
with a coarser `mpc_dt` the way the CasADi controllers allow.

`four_tank` is the CasADi case where that trick does apply: at ratio 0.03 it is
the worst in the suite, and a coarser prediction step would buy the covered time
at the same optimisation cost.

Both are open items rather than tuning knobs, and neither is affected by the
reward shape -- `GradientMPC` sums the environment's reward directly, so it
picks up reward changes without any objective to re-derive.

## Regenerating the figures and videos

```bash
make figures          # or figures-<env>
make videos           # or videos-<env>
make short-gifs       # lightweight *_short.gif copies, inputs to the mosaics
```

**What is committed, and what is not.** Only the five gallery mosaics
(`videos/mosaic_*.webp`) are tracked, because they are the only media a
published page embeds: the README and the environment index carry them. The
per-environment clips are rendered by the `docs-deploy` workflow before it
builds the site, so a clean checkout is light and the published pages still
have their pictures.

This matters because it used to be the other way round and silently broken.
`.gitignore` excluded `videos/**/*.gif` but made an exception for
`*_short.gif`, while `scripts/generate_env_pages.py` deliberately embeds
`pid_output.gif`. The repository therefore carried 55 MB of shorts that nothing
published referenced, and lacked every file the environment pages actually
pointed at. A local `mkdocs build --strict` passed anyway, because the working
tree happened to have the clips; a build from a clean checkout would have
published twenty-one pages of broken images.

A few things still do not round-trip through the targets above, and are worth
knowing before regenerating anything:

- The runner writes `sweep.png`, `pid_response.png` and `comparison.png`, none of
  which are tracked. The five tracked `figures/**/*.png` come from an older
  script and are not reproduced by `make figures`.
- `scripts/make_gallery_clips.py` re-quantises the console clips; `make
  short-gifs` only trims frames. The mosaics prefer `*_short.gif`, so rebuilding
  them needs the shorts present locally.

## What the suite guarantees

One contract, asserted for every registered environment, runs in the `slow`
job: the PID must beat the **best constant action**. A weak bar, but exactly
the one a mis-indexed setpoint fails -- it caught a furnace PID tracking fuel
percentage as its temperature setpoint.

The MPC now has a contract too, in the same job: it must not end the episode in
a terminal state, and must not return materially less than the PID. Until it
existed, `tests/experts/test_mpc_baselines.py` asserted only that a controller
built and emitted finite, in-bounds actions -- which is exactly what a
controller that has given up does, so an MPC returning -0.02 against a PID's 393
passed for as long as it was there.

### The measurement is recorded, not reproduced

Rolling every shipped controller out to check that claim costs about forty
minutes of CPU, and it was being paid on every merge. Profiled with
`--durations`, one parametrisation -- `[plane]` -- took **836 s** of a 19:47 job,
against 415 / 376 / 344 for the three 3D tasks and 94 or less for everything
else. That is structural rather than careless: the 2D aircraft's MPC plans a
horizon of 30 and then holds its last action for another 60 steps, so choosing a
single action optimises a 90-step rollout fifty times over -- 4500 simulated
steps per control step.

It also could not be parallelised away. `pytest-xdist` distributes across tests
and not within one, so the job's wall clock can never fall below its longest
single test: `[plane]` alone set roughly 70% of the floor, and on GitHub's four
slower cores it approached the job's own 30-minute timeout.

But the answer only moves when the physics, the controllers or their gains move,
and most merges touch none of them. So the rollouts are run by hand and the
result committed:

```bash
make baselines              # everything, ~40 min
make baselines-plane        # or one environment
```

`scripts/record_baselines.py` writes `src/target_gym/data/baseline_returns.json`, and the
contract is asserted from that. Reading a number rather than producing it makes
the check *stronger*: it now runs in the fast job on every push and across the
whole Python matrix, where before it ran once per merge to main on a single
interpreter.

**What it bought.** The slow job went from 19:47 to **42 s**, and the contract
moved into the fast job where it now runs on every push and on every interpreter
in the matrix. It also removed the reason the measurement had been compromised:
episodes had been capped at 250 steps and seeds at five to fit a CI budget, and
neither cap is needed once the rollouts happen by hand. Both were lifted, which
is what exposed the glass furnace's MPC (see below).

**What it costs.** The fingerprint covers the shared controller modules, so
editing `experts/mpc.py` invalidates all sixteen records even when the change
provably touches one environment. That is deliberate. A finer, symbol-level
fingerprint would have to resolve `_pid("make_glass_furnace_stateful_pid")` --
a string lookup -- and a miss there produces a record that is stale and *looks*
fresh, which is the one direction this design refuses. The price is a
re-measurement after controller work; the alternative price is a false green.

**What stops a stale record from passing.** Each entry carries a fingerprint of
everything that determines it -- the environment's own modules, the shared
controller and integration code, that environment's tuned gains, and the
parameter values the measurement was taken at. A test compares it against the
tree and refuses a record that no longer describes the code, naming the command
that regenerates it.

The fingerprint is taken over *source*, not over behaviour, and that is
deliberate. Hashing a short trajectory would be more direct and does not survive
the matrix: those numbers are float32 results of `exp`, `sin` and `tanh`, whose
last bits are not guaranteed identical between the arm64 machine a maintainer
regenerates on and the x86 runner that checks. A fingerprint that disagreed with
itself across platforms would fail every run and teach people to regenerate on
red rather than on change. Source hashing fails the other way, which is the safe
one: it can report staleness that is not real -- a rename invalidates a record
the behaviour would have kept -- but never freshness that is not real. One costs
a command; the other costs CI certifying a claim about code that no longer
exists.

Comments and docstrings are excluded, because this repository edits prose
constantly and none of it moves a number. `ast.dump` is deliberately not used to
build the digest: its fields gained members between Python versions, so the same
file would fingerprint differently across the 3.11-3.14 matrix.

### What the tolerance means

The bar is deliberately loose (10% of the PID's return, five seeds, episodes
capped at 250 steps). It is a tripwire against gross regression, not
the published comparison: the numbers below were measured over ten seeds, and
the tolerance was set from the real defects rather than chosen. Verified by
reverting each fix: the wind turbine (terminates at step 20 of 400) and the
battery (-27.5%) are caught; the glass furnace is *not*, and that is the
contract's honest limit -- its bug costs -17.6% over ten seeds but only -4.4%
over five, against +3.7% when fixed, and no sane threshold separates those.
Subtle objective errors are below its resolution; this table is what finds
them. Those three percentages were measured under the previous reward and have
not been re-derived -- reverting each fix again costs hours and would restate a
conclusion about the contract's *resolution*, which the reward change does not
alter. No environment is currently recorded as `EnvSpec.mpc_degraded`. The field
exists so that a baseline which runs but does not beat its own PID is an
explicit, measured gap rather than a silently bad benchmark number, and two
entries have been retired from it by fixing the cause rather than the wording.

The glass furnace MPC was 16.0% behind its PID and lost 10 of 10 seeds; it now
leads on all ten, and the last piece was variable scaling. The battery MPC lost
on 9 of 10 with its mean carried entirely by seed 0, where the planner happened
to share the plant's PRNG key and so knew the future noise exactly; closing that
leak and reshaping the dispatch signal took it to 8 of 10 on merit.

## What a longer episode exposed

Lengthening the benchmark episodes to satisfy
`N >= max(10 * tau_actuator, 1 * T_period)` where a settling time exists at
all -- see the episode-length section of
[the RL protocol](rl-protocol.md) -- immediately found a defect that the short
ones had been hiding, which is the argument for having done it.

The **glass furnace MPC was 16.0% behind its PID over ten seeds and lost on 10
of 10.** At the previous 240-step episode the two scored within 1.3% and the
contract passed. Split into deciles they were *identical* over the first half of
a 1600-step episode -- both still on their way to the setpoint, which is all the
old episode ever measured -- and from the sixth the PID converged to 0.0-0.5 K
of crown-temperature error while the MPC plateaued at 2-6 K.

That was a steady-state offset with a structural cause, and three things fed it.
The objective normalised its error by 40 K, a constant inherited from a reward
the environment had stopped using, so against a 0.1 fuel weight a 3.3 K standing
error was the optimum of what the controller was being asked to minimise. The
fuel weight is zero for this release line. And the planner's regenerator
disagreed with the plant's, so it optimised against a reduced model, and a
finite-horizon MPC with plant-model mismatch settles with a bias that a PID's
integrator removes; an offset-free correction, a clamped integral of the
measured error shifting the solver's setpoint and dropped at each schedule step,
took it from 19.1% behind to 16.0% and is kept. Horizon was never the cause:
going from 0.45 to 1.52 open-loop time constants was worth 1.1 points at three
times the solve cost.

**It now leads, 1513.4 against 1443.5, winning 10 of 10 seeds**, and the
`EnvSpec.mpc_degraded` flag has been removed. The last piece was variable
scaling: the planner had been handing IPOPT crown temperatures around 1600 K
next to a fuel fraction around 0.6, and its mean iteration count over 16 000
solves is now 16.8.


The same change moved the two path-following tasks from being scored over **less
than one lap** -- 0.76 for the circle, 0.91 for the figure-8 -- to three. Their
MPC leads over the PID went to +48.7% and +347.5% per step, because holding a
path is what those tasks are for and a sub-lap episode never asked for it.

## MPC against PID, ten seeds

> **Version 1.** The hand-written table in this section was measured under
> the version-1 log-scaled reward and is kept as the record of that release;
> the sections after it that describe "the shipped reward" as log-scaled and
> capped describe version 1 too. The current numbers, under the
> floor-normalised cost of version 2 (docs/reward-shaping.md), are the
> generated table at the end of this page and the protocol table after it.

Return, paired per seed, on each environment's own episode. Every number here
was re-measured after the reward unification -- returns are not comparable
across that change, so the previous table was discarded rather than patched.

Both the mean and the median are given. They disagree on two rows, in opposite
directions, and either one alone would misreport the pair.

Read this alongside the three caveats above. The MPC has the full state and the
PID does not, so this is a comparison of two controllers with different
information, not two controllers with different algorithms.

| environment | mean | median | seeds won |
| --- | --- | --- | --- |
| plane3d_figure8 | **+450.8** | +449.6 | 10/10 |
| plane3d_heading | **+236.9** | +314.1 | 8/10 |
| plane3d_circle | +145.1 | +200.1 | 6/10 |
| four_tank | +56.9 | +64.6 | 10/10 |
| boiler_drum | +51.4 | +54.9 | 10/10 |
| plane | +33.8 | +30.2 | 8/10 |
| ph_neutralization | +33.4 | +29.8 | 9/10 |
| distillation | +29.2 | +26.6 | 10/10 |
| reactor | +26.6 | +28.1 | 10/10 |
| wind_turbine | +11.8 | +13.9 | 9/10 |
| cement_kiln | +9.4 | +9.3 | 10/10 |
| hvac | +8.4 | +8.4 | 10/10 |
| cstr | +5.3 | +4.5 | 10/10 |
| first_order | +2.5 | +2.3 | 10/10 |
| glass_furnace | -7.0 | -5.2 | 2/10 |
| battery | +14.0 | **-4.1** | 1/10 |

The MPC is the upper bound on **fourteen of the sixteen**, and behind on the
glass furnace and the battery. Both shortfalls are inside the contract's 10%
tolerance (4.7% and 2.6%), so this is a documented gap rather than a failure.

**The battery's mean is the wrong statistic.** It is carried by a single seed
where lookahead pays enormously -- 350 against the PID's 164 -- while the
controller trails on the other nine for a median of -4.1 and one win in ten.
Horizon, iterations and step size were all swept without closing it.

**The glass furnace lost this row to a better opponent, not to a regression.**
Its MPC is unchanged and scores exactly what it scored before (142.09). What
moved was the PID: its gains had been pinned at the edge of the tuner's search
grid, and widening the grid took it from 144.1 to 149.1. That was enough to
turn a split -- MPC ahead on 7 of 10 seeds -- into a clear PID win at 2 of 10.
Strengthening a baseline is supposed to be able to do this, and reporting the
flip is the point of tuning the baseline honestly in the first place.

The aircraft rows now carry win counts. The previous table quoted their margins
as a difference of means with no per-seed count, because the PID column had been
re-tuned while the MPC column had not, and re-running the MPC cost hours. Both
columns are current here, so the counts are real: the MPC leads on all four.

The circle row moved after the table was first measured, and not because of the
MPC, which is unchanged at 361.23. Its PID gained 31% -- 165.5 to 216.1 -- when
the expert was given the ability to trade speed for turn radius, without which a
third of the task's own radius range is unflyable at the cruise it holds (check 9
in the model review checklist). A stronger baseline narrows the MPC's lead from
+195.8 to +145.1 and its win count from 8 of 10 to 6, which is what a better
opponent is supposed to do.

**The aircraft rows are the second measurement.** On the previous dynamics the
2D plane scored -171 and the 3D heading task -34, each winning most seeds and
losing the average to three terminations worth -600 apiece. A great deal of
effort went into those crashes -- a stall-margin barrier, an altitude barrier,
tails out to 240 steps, a crash charge matched to the environment's own penalty,
and making terminations visible to the planner -- and the one that helped fixed
a single seed. None of it was the cause. The integrator was: at one RK4 substep
the plant the MPC plans against and the plant it is stepping through disagree
enough to fly into the ground. At two substeps both controllers win 10 of 10
with no terminations at all.

The machinery was then re-measured rather than left to rot. The 2D aircraft's
terminal cost -- `n_tail=60`, which holds the last action and scores the flight
that follows -- still earns its place: five seeds out of five with it, four
without, and 61 more return. Those figures predate the reward unification, but
the argument for it got *stronger*, not weaker. Removing the environment's flat
crash charge means forgone reward is now the entire cost of a crash, and a
planner can only see forgone reward by looking past its own horizon. Its
stall-margin barrier does not earn its place: it adds 1.3%, inside this
machine's noise, and it existed only to fight the crashes, so it has been
removed.

The wind turbine's overspeed barrier and the battery and furnace surrogates
were re-verified against the new reward rather than assumed. The barrier is
still the difference between controlling and tripping (172.1 with it, 52.4
without, and 6 of 6 episodes ending early on the overspeed trip). The two
surrogates survive for a reason that changed: they were written to route around
a clipped tracking term that was exactly flat outside its band, and no reward in
the library has such a term any more. Log-scaling fixes the *value*, though, not
the gradient -- see "The MPC objective is not the reward" below.

Two seeds would misreport almost everything. Measuring on two produced three
wrong conclusions during this work -- the wind turbine at "98% of the PID", the
2D aircraft at "no crashes", and an original verdict of MPC ahead on 14 of 16 --
each overturned by widening the seed count. Nothing here is quoted below ten.

### Why these failed, which was never tuning

Every MPC that lost to its PID was given an objective it could not descend, or
one that did not share the reward's minimiser:

- **wind_turbine** and **battery** scored tracking as `clip(1 - err/band, 0, 1)**2`.
  One step outside the band leaves that term flat, so the only surviving gradient
  belongs to the *penalty* terms -- and the planner is then correctly guided to
  stop acting. Both were fixed by a smooth surrogate with the same minimiser.
- **glass_furnace** normalised its error by the crown's whole 250 K envelope
  where the reward uses 40 K, six times too flat against an unchanged fuel
  penalty, and its term turned back upward past the band so that beyond twice it
  the objective preferred *more* error.
- **plane** and **plane3d_heading** crash, and a crash penalty behind
  `where(terminated, ...)` is a boolean: it carries a cost but no gradient away
  from the boundary. A differentiable barrier on the approach is what works; it
  fixed the wind turbine's overspeed trip and one of the plane's two failing
  seeds.

The one time the *optimiser* was improved instead -- swapping the aircraft's
gradient descent for Adam, which tripled the predicted return -- closed-loop
performance got dramatically worse (737 m to 4527 m, with new crashes). A weak
optimiser was masking the myopia; pursuing a truncated-horizon objective harder
just exploited it. Fix the objective first.

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

Unifying the rewards on a log scale removed every clipped plateau, so the
obvious next step was to delete the two surrogates and let the planners descend
the reward itself. Measured, that is clearly wrong: the turbine scores 341.9 on
its surrogate against 172.1 on the reward, with the same barrier in both.

The reason is worth stating, because it is a property of log-scaled rewards in
general and not a defect in these two plants. A log-scaled reward is
scale-free in *value* -- every halving of the error is worth the same increment,
which is exactly what makes it good to learn from. Its gradient is not
scale-free. Differentiating

    r(e) = 1 - log1p(e / floor) / log1p(envelope / floor)

gives `-1 / ((floor + e) * log1p(envelope / floor))`, which decays like `1/e`:
the pull toward the setpoint is *weakest* precisely where the controller is
furthest from it. A quadratic in the normalised error has the same minimiser and
a gradient that instead grows with the error.

So the surrogates are no longer workarounds for a broken reward. They are
planner-side reformulations of a reward that is now correct -- which is an
ordinary thing for an MPC to carry, and the distinction matters for anyone
reading them as evidence that the reward needs fixing.

The cement kiln is the clearest case for choosing the implementation to fit the
plant: its free lime depends on temperature through a 280 kJ/mol Arrhenius term
that is then advected down the kiln, so reverse-mode gradients overflow to NaN
after about eight steps while finite differences on the same objective stay
clean. Hence cross-entropy sampling rather than a gradient method.

<!-- BEGIN GENERATED BASELINE TABLE -->

<!-- Written by scripts/generate_baseline_table.py from
     data/baseline_returns.json. Do not edit by hand. -->

| environment | steps | PID cost/step | MPC cost/step | MPC saves | MPC wins | trips |
| --- | --- | --- | --- | --- | --- | --- |
| `plane3d_figure8` | 400 | 4.695e+04 | 6.772 | 1.000 | 10/10 | 0 |
| `plane3d_racetrack` | 650 | 1.78e+05 | 758.8 | 0.996 | 10/10 | 0 |
| `plane_energy` | 1200 | 1366 | 73.21 | 0.946 | 10/10 | 0 |
| `reactor` | 864 | 24.37 | 1.373 | 0.944 | 10/10 | 0 |
| `boiler_drum` | 400 | 641.9 | 42.4 | 0.934 | 10/10 | 0 |
| `distillation` | 200 | 262.3 | 31.91 | 0.878 | 10/10 | 0 |
| `plane3d_circle` | 300 | 1207 | 235.1 | 0.805 | 10/10 | 0 |
| `plane3d_heading` | 200 | 3.15e+04 | 6734 | 0.786 | 10/10 | 0 |
| `plane_sine` | 480 | 5184 | 1251 | 0.759 | 10/10 | 0 |
| `cement_kiln` | 700 | 6.368 | 1.705 | 0.732 | 10/10 | 0 |
| `four_tank` | 500 | 1167 | 344.5 | 0.705 | 10/10 | 0 |
| `ph_neutralization` | 300 | 336.3 | 100.5 | 0.701 | 10/10 | 0 |
| `plane` | 280 | 1.04e+04 | 3276 | 0.685 | 10/10 | 0 |
| `patrol` | 200 | 11.22 | 3.56 | 0.683 | 10/10 | 0 |
| `battery` | 360 | 0.003457 | 0.001196 | 0.654 | 10/10 | 0 |
| `glass_furnace` | 1600 | 1.352 | 0.4943 | 0.634 | 10/10 | 0 |
| `hvac` | 720 | 0.01966 | 0.01064 | 0.459 | 10/10 | 0 |
| `wind_turbine` | 400 | 4.541e-05 | 3.557e-05 | 0.217 | 9/10 | 0 |
| `cstr` | 100 | 6319 | 5803 | 0.082 | 10/10 | 0 |
| `first_order` | 100 | 1018 | 1002 | 0.015 | 10/10 | 0 |

`cost/step` is minus the mean return over the episode length: tracking in
floor-widths plus avoidable consumption (dimensionless plants) or dollars /
euros per step (reactor, battery, wind turbine, HVAC). `MPC saves` is the
fraction of the PID's cost the MPC removes. Episode returns mix the reach
transient with the hold; the protocol table below separates them.

`MPC wins` counts seeds where the MPC out-scored the PID, paired.
A ⚠️ marks an environment where it loses more often than it wins, which
means it is not the upper bound this table presents it as; those carry an
`EnvSpec.mpc_degraded` note saying why.

`trips` counts the MPC's trips over the ten windows: a trip never ends a
window, the plant is down at the failure cost and restarts, or stays
down (`base.failure_kernel`). A permanent zero can mean the controller
is safe or that the plant cannot leave its envelope; `first_order` is
the latter.

<!-- END GENERATED BASELINE TABLE -->

## The protocol numbers: hold and reach, separately

Written from `data/protocol_results.json` (`scripts/evaluate_baselines.py`;
re-run it after any change and paste the table). Gain is the mean cost per
step over every step after the plant's burn-in (three cost-bearing time
constants, capped at half the test episode), transients included, split into
tracking and running cost -- no settling time enters it, so the two
controllers are compared on the same steps. Hold is the same over the settled
steps only, each controller's transient measured on its own cycles (the first
step from which the cost stays within twice the cycle's late level). Reach is
the summed cost of that transient above the level, per target change (one
cycle, hence ~0, where the target drifts continuously: reactor, HVAC) -- the
bias B of Theorem 4 -- with the transient's absolute summed cost in
parentheses. B is relative to each controller's *own* hold level, so it does
not compare two controllers whose holds differ (a PID holding hundreds of
floor-widths off shows a small B because its level swallows its transient:
patrol); the absolute transient cost does. A negative B (†) means the cost is
still rising at the window's end -- no hold was reached, the "hold" level is
above the transient -- and is a finding about the window, not a cheap
transient. NEA is `(PID - MPC) / (PID - floor)` on the gain: 1 at the floor,
0 at PID parity; the floor is 0 on the deterministic plants, where the
documented resolution is a scale and exact hold is achievable. Units are
floor-widths (squared where p = 2) on the dimensionless plants, dollars or
euros per step on the priced ones; nothing here is comparable across rows
except NEA. Three seeds on every plant -- the aircraft draw their targets
per seed too, and a one-seed run had hidden the 2D aircraft MPC losing its
hold on seed 1. `fail` is the trip rate per cycle.

Reading across the rows: the aircraft now fly in light turbulence with no
altitude dead zone, so their holds are real -- the 2D aircraft MPC holds 1.4
floor-widths-squared of altitude against the PID's 6, and pays for it in
airspeed (2.1 against 0.17), which is the trade the two-cost split exists to
show; the aircraft PIDs remain structurally inadequate on the
moving-reference tasks (thousands of floor-widths while "holding" the
racetrack, the figure-8 and the heading), where the MPC sits within a few;
the glass furnace's MPC is 26x its own long-run hold cost on the 13 h test
episode (0.80 against a 0.03 reference) because the episode is still in the
transient of a 30 h plant; on the battery the two controllers hold within 10%
of each other and the MPC's advantage is in the transients after each
dispatch block; on the building, priced at EUR 0.03/K^2 h of discomfort, the
MPC now spends less gas *and* less comfort than the PID (0.0036 against
0.0086, 0.0053 against 0.0099 EUR per step), where at EUR 0.2 the gas term
never bound; and no controller tripped a plant in any window.

| plant | floor ρ* | PID gain (track / run) | MPC gain (track / run) | NEA(MPC) | PID hold | MPC hold | PID reach B (transient) | MPC reach B (transient) | fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `battery` | 0.000222 | 0.00346 (0.00288 / 0.000589) | 0.00112 (0.000548 / 0.000567) | 0.724 | 0.00132 | 0.00117 | 0.0497 (0.0292) | 0.00358 (0.00128) | 0 |
| `boiler_drum` | 1.32 | 524 (524 / 0.0276) | 23.8 (23.8 / 0.0323) | 0.957 | 460 | 24.7 | 3.28e+04 (3.84e+04) | 3e+03 (4.81e+03) | 0 |
| `cement_kiln` | 0.468 | 4.75 (4.68 / 0.0732) | 1.33 (1.27 / 0.0535) | 0.8 | 5.75 | 1.37 | 910 (2.46e+03) | 318 (585) | 0 |
| `cstr` | 0 | 41.2 (41.2 / —) | 0.317 (0.317 / —) | 0.992 | 5.96e-07 | 4.21e-05 | 6.64e+05 (6.6e+05) | 6.12e+05 (6.12e+05) | 0 |
| `distillation` | 0.127 | 62.6 (62.6 / 0.00219) | 0.297 (0.284 / 0.0123) | 0.997 | 62.6 | 0.297 | 4.05e+03 (1.33e+04) | 1.65e+03 (1.68e+03) | 0 |
| `first_order` | 0 | 3.57e-06 (3.57e-06 / —) | 0 (0 / —) | 1 | 7.89e-11 | 0 | 1.19e+05 (1.19e+05) | 1.17e+05 (1.17e+05) | 0 |
| `four_tank` | 0 | 30.3 (30.3 / —) | 0.00741 (0.00741 / —) | 1 | 30.3 | 0.00741 | 6.5e+05 (6.58e+05) | 1.92e+05 (1.92e+05) | 0 |
| `glass_furnace` | 0.0306 | 2.09 (2.02 / 0.0724) | 0.803 (0.759 / 0.0437) | 0.625 | 2.09 | 0.803 | -225† (1.14e+03) | -99.5† (405) | 0 |
| `hvac` | 0.001 | 0.0185 (0.00991 / 0.00863) | 0.00884 (0.00525 / 0.00359) | 0.553 | 0.0154 | 0.00368 | 0.367 (0.913) | 0.377 (0.514) | 0 |
| `patrol` | 1.03 | 10.1 (10.1 / —) | 3.57 (3.57 / —) | 0.721 | 10.1 | 3.57 | 818 (1.78e+03) | 170 (377) | 0 |
| `patrol_bearing_only` | 1.03 | 11.1 (11.1 / —) | — | — | 7.65 | — | 692 (20.6) | — | 0 |
| `ph_neutralization` | 0.64 | 12.6 (12.6 / 0.00681) | 3.06 (3.06 / 0.00669) | 0.797 | 9.73 | 1.79 | 9.07e+04 (9.12e+04) | 2.34e+04 (2.33e+04) | 0 |
| `plane` | 0.706 | 6.21 (6.04 / 0.169) | 3.54 (1.43 / 2.11) | 0.484 | 6.21 | 3.54 | 2.73e+06 (2.73e+06) | 1.19e+06 (1.19e+06) | 0 |
| `plane3d_circle` | 2 | 98.6 (98.6 / —) | 9.84 (9.84 / —) | 0.919 | 98.6 | 9.84 | 4.18e+05 (4.33e+05) | 8.38e+04 (8.53e+04) | 0 |
| `plane3d_figure8` | 1 | 2.5e+04 (2.5e+04 / —) | 2.1 (2.1 / —) | 1 | 2.5e+04 | 2.1 | 8.57e+06 (1.28e+07) | 739 (1.15e+03) | 0 |
| `plane3d_heading` | 1 | 1.79e+04 (1.79e+04 / —) | 5.05 (5.05 / —) | 1 | 1.79e+04 | 5.05 | 4.01e+06 (5.45e+06) | 1.53e+06 (1.53e+06) | 0 |
| `plane3d_racetrack` | 2 | 3.18e+05 (3.18e+05 / —) | 7.04 (7.04 / —) | 1 | 3.18e+05 | 6.86 | 6.82e+04 (3.56e+06) | 6.28e+05 (6.3e+05) | 0 |
| `plane_energy` | 1 | 780 (780 / 0.321) | 38.8 (35.4 / 3.39) | 0.951 | 51.7 | 25.4 | 1.23e+05 (1.28e+05) | 9.5e+03 (1.28e+04) | 0 |
| `plane_sine` | 1 | 1.52e+03 (1.52e+03 / 0.261) | 2.99 (2.24 / 0.749) | 0.999 | 1.49e+03 | 2.86 | 1.88e+06 (2.14e+06) | 7.61e+05 (7.62e+05) | 0 |
| `reactor` | 1.25 | 23.7 (23.3 / 0.385) | 1.38 (1.38 / 2.41e-05) | 0.994 | 23.7 | 1.38 | 1.64e+03 (1.24e+04) | 4.77 (597) | 0 |
| `wind_turbine` | 1.17e-05 | 2.64e-05 (2.43e-05 / 2.12e-06) | 2.1e-05 (1.55e-05 / 5.46e-06) | 0.365 | 2.64e-05 | 2.1e-05 | 0.00854 (0.0132) | 0.00687 (0.011) | 0 |

Reach B is each controller's transient cost above its *own* hold level (Theorem 4's bias), so it is not comparable between two controllers whose holds differ: a controller holding far off shows a small B because its level swallows its transient. The number in parentheses is the transient's summed cost, not relative to anything, and is the one to compare across controllers.
† cost still rising at the end of the window (no hold reached, so the transient is cheaper than the "hold" level and B is negative): PID on `glass_furnace`, MPC on `glass_furnace`.

