# Capability profile

One score per task says how a policy compares with the expert. A profile says
where it is strong and where it is weak: every task is tagged with the
difficulties it poses (an unstable plant, a transport delay, a hidden state),
and the policy's scores are averaged per difficulty and drawn as a radar, as
in [bsuite](https://github.com/google-deepmind/bsuite).

## Run it

`make_policy(name)` returns a fresh `policy(obs)` for the task `name`. Here
the expert stands in for your policy:

```python
import target_gym as tg
from target_gym.suite import plot_profile, references, run_suite

out = run_suite(tg.make_expert, seeds=2, tasks=["first_order", "cstr"])
out["tasks"]["cstr"]["transient_ratio"]  # 1.0: the expert against itself
fig = plot_profile({**references(), "mine": out["profile"]})
fig.savefig("profile.png")
```

Leave out `tasks` to run all 25, and use 10 seeds for a result you report.
`references()` gives the expert and the oracle profiles from the recorded
results, so they cost nothing to draw.

## Read the two radars

Each spoke is a capability, and its point is the mean over the tasks that
probe it. These are the expert and the oracle, drawn from the recorded
results and redrawn whenever they are re-recorded:

<!-- ``../``: this page is served at /capability-profile/, and MkDocs does
     not rewrite paths inside raw HTML. -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/capability-profile-dark.svg">
  <img alt="The expert and the oracle on each capability, hold and reach" src="../images/capability-profile.svg" width="680">
</picture>

- **The dashed ring at 0.5 is the expert.** Outside it the policy is better
  than the shipped controller; inside it, worse.
- **Hold** scores tracking once the target is reached. The edge is the
  ceiling: every tracked quantity within its instrument's resolution. A point
  at 0.75 has covered half the distance from the expert to the ceiling, on the
  log scale of the [holding score](rl-protocol.md#4-metrics).
- **Reach** scores the moves: the approach from the reset and every target
  change. Moves have no ceiling, so the edge is the best known, the oracle's
  recorded ratio to the expert. So on Reach the oracle is the edge itself,
  and what differs between tasks is how much room that edge stands for
  (`oracle_transient_ratio`). A policy that beats it is drawn on the edge,
  and its number in the table says by how much.
- **0 at the centre** is as far below the expert as the edge is above it.
- **The Hold edge may be out of reach.** Some plants have a known bound above
  the ceiling, so no controller gets there. `s_max` gives the most a task allows.
- **A grayed label** means no task on that spoke leaves room. On Hold, an
  expert that already holds within the resolution leaves none. On Reach, a
  task counts only where the oracle reaches new targets at least 1.25 times
  cheaper than the expert; closer than that, seed noise would decide the
  point. `first_order` has room on neither.
- **The oracle** reads the full state, which the policy cannot. It shows what
  is reachable, not what is fair.

## The numbers behind each point

`out["tasks"][name]` holds what `target_gym.scoring.score_policy` returns.
For the moves:

| Key | Meaning |
|---|---|
| `transient_ratio` | The expert's cost per move divided by the policy's, per seed: 3 means it reaches new targets 3 times cheaper. The cost is summed over the plant's fixed settle after each move. |
| `transient_ratio_ci`, `transient_win_rate` | Its 95% bootstrap interval and the share of seeds the policy wins. |
| `settling_ratio` | How many times sooner the policy settles: its tracking cost enters the band and stays there until the next move. The band is every tracked error at its floor, the cost the reward charges as 1 per term. |
| `expert_settling`, `policy_settling` | Those settling times, in steps per move. |
| `oracle_transient_ratio`, `oracle_settling_ratio` | The oracle's recorded ratios, the best known. NaN until the protocol records moves. |

`beats_expert`, `ratio`, `win_rate` and `score` are the verdict and the holding
score of the [protocol](rl-protocol.md#4-metrics), which also lists the trip,
travel, reversal and off-spec numbers.

## Why each task has its tags

Each tag comes from the task's PHYSICS.md (line numbers in brackets).

| Task | Capabilities | From the physics |
|---|---|---|
| `first_order` | none | A sanity fixture with no disturbance, nothing hidden and nothing stochastic (73, 90), so it is in no aggregate. |
| `cstr` | basic, nonlinearity | Arrhenius gain (68); nothing hidden (119). |
| `unstable_cstr` | none | Extra since 2026-11-15 (owner's decision): its oracle tracks exactly on 4 of 5 scenarios, so its room measures a perfect state reading, not control, and it is in no aggregate. Its reason is in `pc_gym/unstable_cstr/PHYSICS.md`, Status. |
| `ph_neutralization` | nonlinearity, disturbance | The gain varies 45 times across the range (94); the buffer flow is unmeasured (97). |
| `four_tank` | none | Extra since 2026-11-08: a lab rig with no industrial job, so it is in no aggregate (an Extra task carries no tag, owner's decision of 2026-11-24). Its reason is in `pc_gym/four_tank/PHYSICS.md`, Status. |
| `distillation` | coupling, delay and slow | Ill-conditioned, relative gain about 52 (4, 58); dominant time constant about 194 min (123). |
| `boiler_drum` | none | Extra since 2026-11-24 (owner's decision): solved as scored, its reference costs nothing on normal days, so it is in no aggregate. Its reason is in `boiler_drum/PHYSICS.md`, Status. |
| `cement_kiln` | delay and slow, partial observability, nonlinearity | The suite's transport-delay problem (26); the axial profile is hidden (144); free lime is steeply temperature-sensitive. |
| `glass_furnace` | delay and slow, partial observability | A pure transport delay on fuel (153) and glass time constants of about 30 h; melt and regenerators are hidden (23 to 28). |
| `hvac` | delay and slow, economic, partial observability | A heavyweight building with hours of memory; comfort and gas priced in euros (216, 217); surface temperature and weather hidden (119). |
| `reactor` | delay and slow, partial observability, economic, moving target | Xenon is a multi-hour memory; most of the state is hidden (148); priced at the spot price (217); follows a drifting demand. |
| `compressor_surge` | safety, nonlinearity, disturbance | Trips below the surge line (9); the Moore-Greitzer cubic characteristic; the consumers' demand ramps. |
| `wind_turbine` | disturbance, economic, safety | The rotor-effective wind is hidden in turbulence (87); an imbalance price (198); trips on over- and under-speed (107). |
| `battery` | none | Extra since 2026-11-15 (owner's decision): it failed acceptance (J2), so it is in no aggregate. Its reason is in `energy/battery/PHYSICS.md`, Status. |
| `plane` | disturbance | The gusts are hidden, which is the point (180). |
| `plane_energy` | moving target, coupling | A ladder of altitudes with airspeed scored too, which removes the spare degree of freedom (193, 194). |
| `plane_sine` | moving target | A sinusoid, as a frequency probe (193). |
| `plane3d_heading` | coupling | Banking costs lift: altitude and heading share the controls. |
| `plane3d_circle` | none | Extra since 2026-11-24 (owner's decision): no airliner duty, and solved after its wings-level start, so it is in no aggregate. Its reason is in `plane3d/PHYSICS.md`, Status. |
| `plane3d_racetrack` | moving target | A path. |
| `plane3d_figure8` | none | Extra since 2026-11-24 (owner's decision): no airliner duty and no sourced manoeuvre, and solved after its wings-level start, so it is in no aggregate. Its reason is in `plane3d/PHYSICS.md`, Status. |
| `patrol` | none | Extra since 2026-11-24 (owner's decision): solved, its reference already holds the source's 10 ft after the start, so it is in no aggregate. Its reason is in `patrol/PHYSICS.md`, Status. |
| `patrol_range_bearing` | none | Extra since 2026-11-24 (owner's decision), with `patrol`: its job seen through a sensor a third of whose cost has no source, so it is in no aggregate. Its reason is in `patrol/PHYSICS.md`, Status. |
| `formation` | none | Extra since 2026-11-24 (owner's decision): one policy flies three aircraft, outside the charter's one plant, one agent, so it is in no aggregate. Its reason is in `formation/PHYSICS.md`, Status. |
| `wake_surf` | economic, safety | Fuel is priced: the follower's cost falls only where it finds the lead's upwash; a bank past 60 deg is an upset, a trip. |

## Limits

- The Reach edge stands for different room on different tasks, from 1.25
  times cheaper than the expert upwards. `oracle_transient_ratio` gives it.
- The fixed settle is 0 steps on `hvac`, 1 on `reactor` and 4 on `battery`, so
  their move cost covers a step or a few. Read `settling_ratio` there.
- Instability, inverse response and multi-agent are not on the radar. Only
  `boiler_drum` and `four_tank` carried the first two and only `formation` the
  third; all three are Extra, and an Extra task carries no tag. Each comes back
  when a core task is shown to carry it.
- Every task counts the same on a spoke, however long its episode.
