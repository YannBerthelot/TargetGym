# Known limits

TargetGym 0.8.0 is a beta. Its four scored tasks, their references and the
comparison `tg.score` makes are settled for this release. Three things are not,
and a result quoted from this release should say so where they matter.

## The room is an upper bound

A task's room is the reference's cost divided by the ceiling's. The ceiling is
the best plan found with the true model, the true state and the future: an
achieved cost, so the true optimum may be lower still, but one no policy that
reads only the observations can be expected to reach. The room therefore bounds
what an agent can save from above. It does not say how much of it is reachable.

## Part of it needs information a policy does not have

Each scored task's ceiling reads facts its policy never observes, such as the
true surge line of the compressor or the acid strength on the pH plant. Each
task card names them: [reactor](tasks/reactor.md), [hvac](tasks/hvac.md),
[compressor_surge](tasks/compressor_surge.md) and
[ph_neutralization](tasks/ph_neutralization.md). The share of the room that
needs them is not measured.

## No causal line or RL result yet

No line sits between the reference and the ceiling that reads only what a
policy reads. An output-feedback MPC with an estimator would be one, and would
say how much of the room a causal controller can reach. It is not built.

No learned policy's result is published yet either. The
[RL protocol](rl-protocol.md) says how one is measured: training runs scored
with `tg.score` on the scoring seeds, choices made on the validation seeds.
