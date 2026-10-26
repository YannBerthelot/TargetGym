# Multi-agent formation

The `formation` task puts a learner in every seat of a three-ship formation.
The lead flies a patrol route: an altitude, and a heading that turns onto a new
rate every minute. Two wingmen hold their slots in a V behind it, one on each
side. All three share one cost, so the lead flying its route well is worth as
much as the wingmen keeping station.

It is the patrol with the scripted lead replaced by a learner. If you want a
single follower behind a lead you do not control, use
[`patrol`](environments/patrol.md) instead.

## Two views of one task

| view | use it for | actions | observations | reward |
|---|---|---|---|---|
| `Formation` | single-agent algorithms (one controller flies every aircraft), the baselines and the protocol | one array of 9 | one array of 74 | one team reward |
| `FormationMARL` | multi-agent algorithms (one policy per aircraft, or shared parameters) | a dict, 3 per agent | a dict, per agent | the team reward, given to every agent |

Both run the same step, so a team scores the same in either view. The
registered task is `Formation`, and its scored episode is 200 steps of 1 s:

```python
import jax
from target_gym import Formation, registry

spec = registry.get("formation")
env, params = spec.make_env(), spec.make_test_params()   # 200-step episode

obs, state = env.reset_env(jax.random.PRNGKey(0), params)   # obs.shape == (74,)
action = env.action_space(params).sample(jax.random.PRNGKey(1))   # (9,)
obs, state, reward, terminated, truncated, info = env.step(
    jax.random.PRNGKey(2), state, action, params
)
```

The multi-agent view follows the [JaxMARL](https://github.com/FLAIROX/JaxMARL)
`MultiAgentEnv` conventions: agents are `"lead"`, `"wingman_0"` and
`"wingman_1"`, `step` resets every agent when `dones["__all__"]` is true, and
everything is `jit`, `vmap` and `scan` compatible.

```python
import jax
from target_gym import FormationMARL, registry

env = FormationMARL()
params = registry.get("formation").make_test_params()
key = jax.random.PRNGKey(0)

obs, state = env.reset(key, params)   # {"lead": (22,), "wingman_0": (26,), "wingman_1": (26,)}
actions = {a: env.action_space(a).sample(key) for a in env.agents}
obs, state, rewards, dones, info = env.step(key, state, actions, params)
# rewards: the same team reward for every agent
# info["costs"]: each aircraft's own cost this step
```

## What each aircraft controls and sees

Every aircraft acts with `[power, stick, aileron]`, each in [-1, 1]. In the
joint view the lead's three come first, then each wingman's.

| seat | observation | what it contains |
|---|---|---|
| lead | 22 values | its own flight state (15 values, the 3D heading task's), whose target is the heading the route commands this step; the route's current turn rate; each wingman's slot error (behind, right, up, in the lead's frame) |
| each wingman | 26 values | the patrol's observation: its own flight state, the lead's position and velocity relative to it, its slot, and its slot error |

A wingman sees exactly what the patrol's follower sees, so a policy trained on
`patrol` can fly a wingman seat unchanged. The wingmen do not see each other;
the V puts them at least 120 m apart, and only a wingman far out of its slot
comes near the other.

The joint observation is the lead's 22 values followed by each wingman's 26.
The tracked values are the lead's altitude (index 2) and each wingman's slot
error (indices 46 and 72).

## The reward

One team cost, charged every step and given to every agent:

- the lead's altitude error, in units of 1 m (an altimeter's resolution), squared;
- the lead's heading error against its route, in units of 0.5 deg, squared;
- each wingman's distance from its slot, in units of 3 m (relative GPS), squared;
- each wingman's heading difference from the lead, in units of 0.5 deg, squared.

The reward is minus their sum, the same contract as every TargetGym task (see
[Reward shaping](reward-shaping.md)). In the multi-agent view
`info["costs"]` splits it by aircraft, for credit assignment: the costs add up
to minus the reward on every step without a trip.

## Trips and restarts

The formation trips when any aircraft leaves the altitude band, any two
aircraft come within 25 m of each other, or a wingman ends up 1500 m from its
slot. The step that trips is charged the trip cost and sets `info["tripped"]`
and `info["trip_cost"]`, and the whole formation restarts on the same episode
clock. The episode does not end.

For episodic training, set `continuing=False` in the params: a trip then ends
the episode, and `dones["__all__"]` is true in the multi-agent view.

## Baselines

Two baselines ship, recorded on the [Baseline results](baseline-results.md)
page.

- **PID team** (`env.make_pid()`, or `env.expert_policy` for batched JAX): the
  patrol's controllers, seat for seat. The lead seat flies the patrol's lead
  autopilot, and each wingman seat flies the patrol's pursuit PID. It never
  trips, but the autopilot leaves the lead about 140 m off its altitude, which
  dominates its cost. That is the easiest gain for a learner: the 3D heading
  task's own PID holds the lead to about 38 m, but the patrol PID's wingmen
  collide behind it.
- **Oracle** (`env.make_mpc()`): each seat flies a law, the patrol's lead
  autopilot or the patrol's twin-autopilot law, and one planner adds a
  correction to every seat over a 30-step horizon, on the true team cost. It
  holds the lead to under 1 m and each slot to about 0.3 m. Like every oracle
  in the suite it reads the true state rather than the observations, and it
  can coordinate the team only as far as its horizon reaches.

## Other formation sizes

`Formation(num_wingmen=k)` and `FormationMARL(num_wingmen=k)` take 1 to 4
wingmen. The lead observes 16 + 3k values and each wingman 26; the joint action
has 3(k + 1). Wingmen alternate sides, right first, and each pair sits one rank
further back and out. Only the two-wingman formation is scored. For another
size, build the controllers with the size:

```python
from target_gym import Formation
from target_gym.formation.experts import (
    make_formation_mpc,
    make_formation_pid_controller,
)

env = Formation(num_wingmen=4)
pid = make_formation_pid_controller(num_wingmen=4).host()
oracle = make_formation_mpc(env, env.default_params)
```
