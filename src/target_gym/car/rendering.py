import time

import numpy as np
import pygame
from jax import grad
from pygame import gfxdraw

from target_gym.car.env import road_profile


def road_slope(x, road_profile):
    """Return the slope (angle) of the road at position x."""
    dzdx = grad(road_profile)
    slope = dzdx(x)
    return slope  # radians


import numpy as np
import pygame
from jax import grad

grad_fn = grad(road_profile)


def render_car_scene_centered_with_milestones(
    screen_width,
    screen_height,
    state,
    params,
    road_profile,
    positions_history=None,
    milestones=None,
    road_buffer=50,
    speed_scale=1.0,  # inflated y-scale for better slope visualization
    visible_range=1_000,
    milestone_spacing=1_000,  # meters
):
    """Render car centered, milestones every 100m, keep frames for video."""
    if positions_history is None:
        positions_history = []

    if milestones is None:
        milestones = []

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((135, 206, 235))  # sky blue

    car_screen_x = screen_width // 2
    car_screen_y = screen_height // 2

    # Vectorized road points
    xs = np.linspace(
        state.x - visible_range / 2, state.x + visible_range / 2, screen_width
    )
    road_heights = road_profile(xs)
    road_center = road_profile(state.x)
    ys = car_screen_y - speed_scale * (road_heights - road_center)
    road_points = list(zip(range(screen_width), ys.astype(int)))

    # Fill area below road
    pygame.draw.polygon(
        surf,
        (0, 180, 0),
        [(0, screen_height)] + road_points + [(screen_width - 1, screen_height)],
    )

    # Draw road line
    pygame.draw.lines(surf, (50, 50, 50), False, road_points, 3)

    # Compute slope at car with finite difference
    slope = -grad_fn(state.x)

    # Car body
    car_width = 40
    car_height = 20
    wheel_radius = 5
    wheel_base_y = car_screen_y
    body_offset_y = -car_height

    corners = np.array(
        [
            [-car_width / 2, body_offset_y],
            [car_width / 2, body_offset_y],
            [car_width / 2, body_offset_y + car_height],
            [-car_width / 2, body_offset_y + car_height],
        ]
    )
    cos_t = np.cos(slope * speed_scale)
    sin_t = np.sin(slope * speed_scale)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated_corners = (corners @ rotation.T) + np.array([car_screen_x, wheel_base_y])
    pygame.draw.polygon(surf, (255, 0, 0), rotated_corners)
    pygame.draw.polygon(surf, (0, 0, 0), rotated_corners, 2)

    # Wheels
    wheel_offsets = [(-car_width / 3, 0), (car_width / 3, 0)]
    for dx_, dy_ in wheel_offsets:
        wheel_pos = np.array([dx_, dy_]) @ rotation.T + np.array(
            [car_screen_x, wheel_base_y]
        )
        pygame.draw.circle(surf, (0, 0, 0), wheel_pos.astype(int), wheel_radius)

    # Trail
    positions_history.append((state.x, road_center))
    trail_points = positions_history[::5]
    for px, py in trail_points:
        screen_px = car_screen_x + (px - state.x) * (screen_width / visible_range)
        screen_py = wheel_base_y - speed_scale * (py - road_center)
        pygame.draw.circle(surf, (0, 0, 0), (int(screen_px), int(screen_py)), 2)

    # Milestones every `milestone_spacing` meters
    last_milestone_x = milestones[-1] if milestones else 0
    next_milestone_x = last_milestone_x + milestone_spacing
    while next_milestone_x < state.x + visible_range / 2:
        milestones.append(next_milestone_x)
        next_milestone_x += milestone_spacing

    # Draw milestones
    for mx in milestones:
        screen_x = car_screen_x + (mx - state.x) * (screen_width / visible_range)
        if 0 <= screen_x < screen_width:  # only draw if on screen
            y = wheel_base_y - speed_scale * (road_profile(mx) - road_center)
            pygame.draw.rect(
                surf, (255, 255, 255), (int(screen_x) - 2, int(y) - 10, 4, 20)
            )

    # Clean up milestones behind the screen to avoid growing list indefinitely
    milestones = [m for m in milestones if m > state.x - visible_range / 2]

    # Reward
    max_v_diff = params.max_velocity - params.min_velocity
    reward = (
        (max_v_diff - abs(state.target_velocity - state.velocity)) / max_v_diff
    ) ** 2

    # HUD
    font = pygame.font.SysFont("arial", 16)
    padding = 8
    line_height = 20

    hud_texts = [
        f"Speed: {state.velocity*3.6:.1f} km/h",
        f"Target: {state.target_velocity*3.6:.1f} km/h",
        f"Reward: {reward:.2f}",
        f"Slope: {-np.rad2deg(slope):.1f}°",
        f"Power: {state.throttle*100:.0f}%",
    ]

    # Draw HUD background
    hud_width = 200
    hud_height = len(hud_texts) * line_height + 2 * padding
    hud_rect = pygame.Rect(screen_width - hud_width - 10, 10, hud_width, hud_height)
    pygame.draw.rect(surf, (0, 0, 0), hud_rect, border_radius=6)
    pygame.draw.rect(surf, (255, 255, 255), hud_rect.inflate(-4, -4), border_radius=6)

    # Draw text
    for i, text in enumerate(hud_texts):
        txt_surf = font.render(text, True, (0, 0, 0))
        surf.blit(
            txt_surf, (hud_rect.x + padding, hud_rect.y + padding + i * line_height)
        )

    # Draw throttle bar (flipped horizontally)
    bar_width = 150
    bar_height = 20
    bar_x = hud_rect.x
    bar_y = hud_rect.y + hud_height + 10

    # Draw bar background (half green right, half red left)
    pygame.draw.rect(
        surf, (0, 180, 0), (bar_x + bar_width // 2, bar_y, bar_width // 2, bar_height)
    )
    pygame.draw.rect(surf, (180, 0, 0), (bar_x, bar_y, bar_width // 2, bar_height))

    # Fill current throttle
    if state.throttle >= 0:
        fill_width = int(bar_width // 2 * state.throttle)
        pygame.draw.rect(
            surf,
            (144, 238, 144),  # lighter green
            (bar_x + bar_width // 2, bar_y, fill_width, bar_height),
        )
    else:
        fill_width = int(bar_width // 2 * -state.throttle)
        pygame.draw.rect(
            surf,
            (255, 160, 122),  # lighter red
            (bar_x + bar_width // 2 - fill_width, bar_y, fill_width, bar_height),
        )

    return surf, positions_history, milestones


def _render(cls, screen, state, params, frames, clock):
    """Render function for Car2D environment with milestones."""
    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state

    # Initialize screen if needed
    if screen is None:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((cls.screen_width, cls.screen_height))
        cls.positions_history = []
        cls.milestones = []
        cls.screen = screen

    # Initialize clock if needed
    if clock is None:
        clock = pygame.time.Clock()
        cls.clock = clock

    # Render car scene with milestones
    surf, cls.positions_history, cls.milestones = (
        render_car_scene_centered_with_milestones(
            cls.screen_width,
            cls.screen_height,
            state,
            params,
            road_profile=road_profile,
            positions_history=cls.positions_history,
            milestones=cls.milestones,
            road_buffer=50,
            speed_scale=1.0,
        )
    )

    # Blit to screen and capture frame
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    frame = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    frames.append(frame)
    cls.frames = frames

    return frames, screen, clock
