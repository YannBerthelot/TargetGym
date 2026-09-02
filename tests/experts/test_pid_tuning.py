"""The gradient PID tuners are runtime, and are reachable automatically.

``experts.pid`` runs ``tune_all_and_save`` whenever ``data/pid_gains.json`` is
missing, so a user who clears the cache -- or installs without it -- gets these
tuners rather than an error. They had no test at all.

Each tuner is run for a couple of gradient steps: enough to build its loss,
differentiate it and apply an update, which is where the failures live. What is
checked is that the gains that come back are usable numbers, because their
output is cached and then shipped as the environment's baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

import target_gym.experts.pid_tuning as tuning

SMOKE_STEPS = 2

# Three aircraft tuners return NaN gains. The mechanism is the one that was
# found and fixed for the four-tank -- an operation whose forward value is fine
# but whose reverse-mode derivative is NaN, which then poisons every gain -- but
# the specific operation has not been localised for these, so they are pinned
# rather than guessed at. strict=True means a fix flips them to passing.
NAN_TUNERS = {
    "tune_plane_pid",
    "tune_plane3d_heading_pid",
    "tune_plane3d_circle_pid",
}

TUNERS = [
    "tune_first_order_pid",
    "tune_cstr_pid",
    "tune_four_tank_pid",
    "tune_plane_pid",
    "tune_plane3d_heading_pid",
    "tune_plane3d_circle_pid",
    "tune_plane3d_figure8_pid",
]


def _flatten(gains) -> list[float]:
    """Tuners return a triple, a pair of triples, or a pair plus a scalar."""
    out: list[float] = []
    if isinstance(gains, (tuple, list)):
        for item in gains:
            out.extend(_flatten(item))
    else:
        out.append(float(gains))
    return out


@pytest.mark.parametrize(
    "name",
    [
        (
            pytest.param(
                n,
                marks=pytest.mark.xfail(
                    strict=True,
                    reason=(
                        "returns NaN gains: the loss evaluates but its gradient does "
                        "not. The four-tank had the same class of defect -- "
                        "sqrt(max(h, 0)) is exact forward and NaN in reverse at "
                        "h = 0 -- and is fixed. The aircraft equivalent is still not "
                        "localised, but it is not the plant: a plain differentiable "
                        "rollout of these dynamics gives finite gradients out to 200 "
                        "steps, so it is somewhere in this module's own loss. Relay "
                        "autotuning no longer covers for it either -- the relay fails "
                        "on all three 3D tasks with no zero-crossings -- and the "
                        "shipped aircraft gains now come from coordinate descent on "
                        "episode return (scripts/tune_pid.py, _tune_aircraft_search)."
                    ),
                ),
            )
            if n in NAN_TUNERS
            else n
        )
        for n in TUNERS
    ],
)
def test_tuner_returns_usable_gains(name):
    gains = getattr(tuning, name)(n_grad_steps=SMOKE_STEPS, verbose=False)
    values = _flatten(gains)

    assert values, f"{name} returned no gains"
    assert np.all(np.isfinite(values)), f"{name} returned non-finite gains: {gains}"


def test_four_tank_loss_is_differentiable():
    """Pins the fix: the four-tank loss must have a finite gradient.

    Its outflow goes as sqrt of the level and a tank can sit empty, so the
    naive ``sqrt(max(h, 0))`` gave an exact forward value and a NaN derivative.
    That made every gain NaN while the loss itself looked healthy, which is
    what made it hard to see.
    """
    import jax

    from target_gym import FourTank, FourTankParams

    env = FourTank(integration_method="rk4_1")
    params = FourTankParams()
    loss = tuning.make_mimo_pid_loss_fn(
        env,
        params,
        reset_fn=lambda key, p, t1, t2: env.reset_env(key, p)[1].replace(
            target_h1=t1, target_h2=t2
        ),
        obs_indices=((0, 4), (1, 5)),
        target_ranges=(tuple(params.target_h1_range), tuple(params.target_h2_range)),
        n_targets=2,
        n_steps=int(params.max_steps_in_episode),
    )

    value = float(loss(5.0, 1.0, 0.0, 5.0, 1.0, 0.0))
    grad = float(jax.grad(lambda kp: loss(kp, 1.0, 0.0, 5.0, 1.0, 0.0))(5.0))

    assert np.isfinite(value), "the loss itself is non-finite"
    assert np.isfinite(grad), "the loss has a NaN gradient -- see _safe_sqrt"
