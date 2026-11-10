"""The aircraft models the renderers draw, and which task is drawn with which.

Contract (owner, the 2026-11-10 aircraft review): ``patrol``,
``patrol_range_bearing`` and ``formation`` fly an F/A-18 Hornet and are drawn as
one; ``wake_surf`` and the ``plane3d`` tasks fly the A320 and keep its model.
Each model is drawn to true scale, so its span is its airframe's: the A320's
35.8 m (``PlaneParams3D.wingspan``) and the Hornet's 37.42 ft [HARV Table 1],
and the Hornet is 56 ft nose to tail, AFF's figure.

The models are swappable: a projection draws whichever model it is handed.
"""

import numpy as np
import pytest

from target_gym import render_aircraft as ra
from target_gym.registry import REGISTRY

FT = 0.3048


@pytest.mark.parametrize(
    "task, airframe",
    [
        ("patrol", "hornet"),
        ("patrol_range_bearing", "hornet"),
        ("formation", "hornet"),
        ("wake_surf", "a320"),
        ("plane3d_heading", "a320"),
    ],
)
def test_each_task_is_drawn_as_the_airframe_it_flies(task, airframe):
    params = REGISTRY[task].make_test_params()
    assert ra.model_for(params).name == airframe


@pytest.mark.parametrize("name, span", [("a320", 35.8), ("hornet", 37.42 * FT)])
def test_models_span_their_airframes_wingspan(name, span):
    # 5 cm: the meshes are placed by hand to the centimetre.
    assert ra.aircraft_model(name).span == pytest.approx(span, abs=0.05)


def test_hornet_is_56_ft_nose_to_tail():
    assert ra.aircraft_model("hornet").length == pytest.approx(56.0 * FT, abs=0.1)


def _drawn_width(model, scale_px):
    """Pixel width of a model drawn top-down, nose up, on an empty surface."""
    surf = ra.pygame.Surface((400, 400), ra.pygame.SRCALPHA)
    ra.draw_plane_topdown(surf, 200, 200, 0.0, 0.0, np.pi / 2, scale_px, model=model)
    alpha = ra.pygame.surfarray.pixels_alpha(surf)
    cols = np.flatnonzero(alpha.max(axis=1) > 0)
    return cols[-1] - cols[0] + 1


def test_a_projection_draws_the_model_it_is_handed():
    # Nose up, a model's drawn width is its span, plus the nav-light halos at
    # the wingtips (a few pixels each). At the same scale the Hornet is about a
    # third as wide as the A320: 11.41 / 35.8 = 0.32.
    hornet = _drawn_width(ra.aircraft_model("hornet"), 4.0)
    a320 = _drawn_width("a320", 4.0)
    assert hornet / a320 == pytest.approx(11.41 / 35.8, abs=0.05)
