# Vision

*Adopted by the owner on 2026-11-27.*

TargetGym is a benchmark of realistic plant-control tasks for reinforcement
learning. Each task is a reduced-order model of real equipment. It is scored in
money or product, it has the limits that trip a real plant, and it ships with
the controller a competent engineer would deploy.

## Who it is for

- RL researchers who want control problems with real stakes and a credible
  human baseline.
- Control engineers who want to know whether RL beats what they ship.

## What you get

For each task that has reached v1:

- `pip install target-gym`;
- the task through `gymnasium.make("TargetGym/<task>-v1")` or the JAX API;
- one number, the agent's saving over the expert;
- published baselines: doing nothing, the expert, an MPC, and PPO and SAC at
  one cheap budget;
- a promise that the v1 task never changes.

A task reaches v1 when it meets the [task protocol](task-protocol.md). Tasks
reach it one at a time, as each passes. Until then a task has no Gymnasium ID
and loads with `tg.make` under its development name. On 2026-11-27 no task has
reached v1.

## Out of scope

- An arcade track.
- Suite-wide scores before per-task numbers.

`first_order` stays as a labelled, unscored debugging task, for checking that an
agent learns at all.

The study behind these choices is `docs/proposals/benchmark-adoption-study.md`
in the repository.
