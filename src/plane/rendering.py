import time

import numpy as np
import pygame
from pygame import gfxdraw


def draw_dashed_line(surface, color, start_pos, end_pos, dash_length=5, space_length=5):
    x1, y1 = start_pos
    x2, y2 = end_pos

    total_length = x2 - x1
    num_dashes = total_length // (dash_length + space_length)

    for i in range(int(num_dashes) + 1):
        start_x = x1 + i * (dash_length + space_length)
        end_x = min(start_x + dash_length, x2)
        pygame.draw.line(surface, color, (start_x, y1), (end_x, y2))


def draw_cloud(
    surface,
    center_x,
    center_y,
    scale=1.0,
    seed=42,
    color=(255, 255, 255),
    outline_color=(0, 0, 0),
    outline_thickness=3,
):
    rnd = np.random.default_rng(seed)

    num_circles = rnd.integers(10, 20)
    circles = []
    for _ in range(num_circles):
        dx = rnd.integers(-40, 40) * scale
        dy = rnd.integers(-20, 20) * scale
        r = rnd.integers(10, 20) * scale
        circles.append((dx, dy, r))

    surf_w, surf_h = int(160 * scale), int(100 * scale)

    def make_cloud_surf(extra_radius, fill_color):
        temp_surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA).convert_alpha()
        for dx, dy, r in circles:
            x = int(surf_w // 2 + dx)
            y = int(surf_h // 2 + dy)
            pygame.gfxdraw.filled_circle(
                temp_surf, x, y, int(r + extra_radius), fill_color
            )
        return temp_surf

    outline_surf = make_cloud_surf(outline_thickness, outline_color)
    surface.blit(outline_surf, (center_x - surf_w // 2, center_y - surf_h // 2))

    body_surf = make_cloud_surf(0, color)
    surface.blit(body_surf, (center_x - surf_w // 2, center_y - surf_h // 2))


def render_plane_scene(
    screen_width,
    screen_height,
    state,
    params,
    positions_history,
    cloud_positions,
    max_steps,
    ground_buffer=500,
):
    """Main rendering function for the airplane scene"""
    # Scaling
    world_width = 343 * max_steps
    scale_x = screen_width / world_width
    scale_y = screen_height / (params.max_alt + ground_buffer)

    # Create surface
    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((135, 206, 235))  # Sky blue

    # Ground
    ground_y = int(screen_height - (0 + ground_buffer) * scale_y)
    pygame.draw.rect(
        surf, (0, 255, 0), (0, ground_y, screen_width, screen_height - ground_y)
    )

    # Plane position
    planex = int(state.x * scale_x)
    planey = int(screen_height - (state.z + ground_buffer) * scale_y)

    # Plane dimensions
    plane_length = 80
    plane_height = 12

    # Define plane parts
    fuselage = [
        (-plane_length // 2, -plane_height // 2),
        (plane_length // 2, -plane_height // 2),
        (plane_length // 2, plane_height // 2),
        (-plane_length // 2, plane_height // 2),
    ]

    # Nose points
    nose_radius = plane_height // 2
    nose_center = (plane_length // 2, 0)
    nose_points = [
        (
            nose_center[0] + nose_radius * np.cos(theta),
            nose_center[1] + nose_radius * np.sin(theta),
        )
        for theta in np.linspace(-np.pi / 2, np.pi / 2, 20)
    ]

    plane_body = [fuselage[0], fuselage[1]] + nose_points + [fuselage[2], fuselage[3]]
    offset = 2
    wing = [
        (-15, plane_height // 4 + offset),
        (15, plane_height // 4 - 5 + offset),
        (15, plane_height // 4 - 2 + offset),
        (-15, plane_height // 4 + 1 + offset),
    ]

    engine_height = 2
    engine_width = 10
    engine = [
        (0, plane_height // 4 + 2 + offset),
        (engine_width, plane_height // 4 + offset),
        (engine_width, plane_height // 4 + 1 + engine_height + 4 + offset),
        (0, plane_height // 4 + 1 + 5 + engine_height + offset),
    ]

    stabilizer = [
        (-plane_length // 2, -plane_height // 3),
        (-plane_length // 2 + 8, -plane_height // 3),
        (-plane_length // 2 + 6, -plane_height // 3 - 16),
        (-plane_length // 2, -plane_height // 3 - 16),
    ]

    hstab = [
        (-plane_length // 2 + 0, plane_height // 4 - plane_height // 2),
        (-plane_length // 2 + 12, plane_height // 4 - plane_height // 2),
        (-plane_length // 2 + 12, plane_height // 4 + 3 - plane_height // 2),
        (-plane_length // 2 + 0, plane_height // 4 + 3 - plane_height // 2),
    ]

    # --- Passenger windows ---
    num_windows = 15  # number of windows along the fuselage
    window_radius = 1  # radius of each window

    # Place windows along the fuselage (relative to plane center)
    fus_x_start = -plane_length // 4  # start behind nose
    fus_x_end = plane_length // 2 - 5  # end before nose
    window_x_positions = np.linspace(fus_x_start, fus_x_end, num_windows)
    window_y_position = -1  # centered along fuselage vertically

    # Combine into list of points
    passenger_windows_shape = [(x, window_y_position) for x in window_x_positions]
    # Rotate & translate all windows like other plane parts

    # --- Rotate and translate as a group ---
    def rotate_and_translate_shape(points, theta, center_x, center_y):
        """Rotate and translate all points in a shape."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotated_translated = [
            (x * cos_t - y * sin_t + center_x, x * sin_t + y * cos_t + center_y)
            for x, y in points
        ]
        return rotated_translated

    # Example usage with plane position
    passenger_windows_points = rotate_and_translate_shape(
        passenger_windows_shape, -state.theta, planex, planey
    )

    # Rotation functions
    def rotate_point(x, y, theta):
        theta = -theta
        return (
            x * np.cos(theta) - y * np.sin(theta),
            x * np.sin(theta) + y * np.cos(theta),
        )

    def rotate_and_translate(points):
        return [
            (
                rotate_point(x, y, state.theta)[0] + planex,
                rotate_point(x, y, state.theta)[1] + planey,
            )
            for (x, y) in points
        ]

    # Transform all plane parts
    plane_body = rotate_and_translate(plane_body)
    wing = rotate_and_translate(wing)
    stabilizer = rotate_and_translate(stabilizer)
    hstab = rotate_and_translate(hstab)
    engine = rotate_and_translate(engine)

    # Draw clouds behind plane
    for cx, cy, scale, shape in cloud_positions[:5]:
        draw_cloud(surf, cx, cy, scale=scale, seed=shape)

    # Draw plane parts
    gfxdraw.filled_polygon(surf, plane_body, (255, 255, 255))
    gfxdraw.aapolygon(surf, plane_body, (0, 0, 0))

    gfxdraw.filled_polygon(surf, wing, (180, 180, 180))
    gfxdraw.aapolygon(surf, wing, (0, 0, 0))

    gfxdraw.filled_polygon(surf, stabilizer, (255, 255, 255))
    gfxdraw.aapolygon(surf, stabilizer, (0, 0, 0))

    gfxdraw.filled_polygon(surf, hstab, (180, 180, 180))
    gfxdraw.aapolygon(surf, hstab, (0, 0, 0))

    gfxdraw.filled_polygon(surf, engine, (180, 180, 180))
    gfxdraw.aapolygon(surf, engine, (0, 0, 0))

    # Draw passenger windows
    for wx, wy in passenger_windows_points:
        pygame.draw.circle(surf, (0, 0, 0), (int(wx), int(wy)), window_radius)

    # Draw remaining clouds
    for cx, cy, scale, shape in cloud_positions[5:]:
        draw_cloud(surf, cx, cy, scale=scale, seed=shape)

    # Draw target altitude line
    target_y = int(screen_height - (state.target_altitude + ground_buffer) * scale_y)
    draw_dashed_line(
        surf,
        (20, 20, 20),
        (0, target_y),
        (screen_width, target_y),
        dash_length=10,
        space_length=10,
    )

    # Draw ground line and trail
    ground_y = int(screen_height - ground_buffer * scale_y)
    gfxdraw.hline(surf, 0, screen_width, ground_y, (0, 0, 0))

    # Draw trail
    for x, y in positions_history[0::20]:
        gfxdraw.circle(surf, x, y, 2, (0, 0, 0))
        gfxdraw.circle(surf, x, y, 1, (255, 255, 255))

    # Draw HUD
    padding = 8
    line_height = 22
    font = pygame.font.SysFont("arial", 16)

    # Prepare text surfaces
    text_target = font.render(
        f"Target Alt: {int(state.target_altitude * 3.281)} ft", True, (255, 0, 0)
    )
    text_altitude = font.render(
        f"Altitude: {int(state.z * 3.281)} ft - {int(state.z)} m", True, (0, 0, 255)
    )
    text_distance = font.render(
        f"Distance:  {int(state.x*0.539957/1000)} nm - {int(state.x/1000)} km",
        True,
        (0, 0, 255),
    )
    text_velocity = font.render(
        f"Speed: {int(state.x_dot * 1.944)} kt - {int(state.x_dot * 3.6)} km/h",
        True,
        (0, 0, 255),
    )
    text_pitch = font.render(
        f"Pitch: {np.rad2deg(state.theta):.1f}°", True, (0, 0, 255)
    )
    text_power = font.render(f"Power: {state.power*100:.0f}%", True, (0, 0, 255))
    text_stick = font.render(
        f"Stick: {np.rad2deg(state.stick):.0f}°", True, (0, 0, 255)
    )
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(state.t))
    text_time = font.render(f"Time: {time_elapsed}", True, (0, 0, 255))
    # reward = 1 if abs(state.z - state.target_altitude) < 1_000 else 0
    max_alt_diff = params.max_alt - params.min_alt
    done1 = state.t >= params.max_steps_in_episode
    if done1:
        reward = -1.0 * params.max_steps_in_episode
    else:
        reward = (
            (max_alt_diff - abs(state.target_altitude - state.z)) / max_alt_diff
        ) ** 2
    text_reward = font.render(f"Reward: {reward:.2f}", True, (0, 0, 255))

    # Organize columns
    left_column = [text_altitude, text_distance, text_velocity]
    middle_column = [text_pitch, text_power, text_stick]
    right_column = [text_time, text_reward]

    def rounded_width(column_texts, bucket=20):
        max_w = max(text.get_width() for text in column_texts)
        return ((max_w + bucket - 1) // bucket) * bucket

    # Calculate column widths
    left_width = rounded_width(left_column)
    middle_width = rounded_width(middle_column)
    right_width = rounded_width(right_column)

    col_spacing = [
        left_width + padding * 2,
        middle_width + padding * 2,
        right_width + padding * 2,
    ]

    # HUD dimensions
    num_lines = max(len(left_column), len(middle_column), len(right_column))
    hud_width = sum(col_spacing)
    hud_height = num_lines * line_height + 2 * padding

    # Draw HUD box
    hud_x = (screen_width - hud_width) // 2
    hud_y = 10
    hud_rect = pygame.Rect(hud_x, hud_y, hud_width, hud_height)

    outline_thickness = 3
    pygame.draw.rect(surf, (0, 0, 0), hud_rect, border_radius=6)
    inner_rect = hud_rect.inflate(-2 * outline_thickness, -2 * outline_thickness)
    pygame.draw.rect(surf, (255, 255, 255), inner_rect, border_radius=6)

    # Draw text
    col_x = hud_x + padding
    for i, text in enumerate(left_column):
        surf.blit(text, (col_x, hud_y + padding + i * line_height))
    col_x += col_spacing[0]
    for i, text in enumerate(middle_column):
        surf.blit(text, (col_x, hud_y + padding + i * line_height))
    col_x += col_spacing[1]
    for i, text in enumerate(right_column):
        surf.blit(text, (col_x, hud_y + padding + i * line_height))

    # Target altitude text
    surf.blit(text_target, (10, target_y - 20))

    return surf


def _render(cls, screen, state, params, frames, clock):
    if state is None:
        if cls.state is None:
            raise ValueError("No state provided")
        state = cls.state
    if screen is None:
        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((cls.screen_width, cls.screen_height))
        cls.positions_history = []
        cloud_positions = []
        seed = 42  # for reproducibility
        rng = np.random.default_rng(seed)
        N_CLOUDS = 8
        for _ in range(N_CLOUDS):
            cx = rng.integers(0, cls.screen_width)
            cy = rng.integers(50, cls.screen_height // 2)
            scale = rng.uniform(0.5, 1.5)
            shape = rng.integers(0, cls.max_steps)
            cloud_positions.append((cx, cy, scale, shape))
        cls.cloud_positions = cloud_positions
        cls.screen = screen

    if clock is None:
        clock = pygame.time.Clock()
        cls.clock = clock

    if state is None:
        return None

    # Use the refactored rendering function
    surf = render_plane_scene(
        cls.screen_width,
        cls.screen_height,
        state,
        params,
        cls.positions_history,
        cls.cloud_positions,
        max_steps=cls.max_steps,
    )

    # Update position history
    planex = int(state.x * cls.screen_width / (343 * cls.max_steps))
    planey = int(
        cls.screen_height - (state.z + 500) * cls.screen_height / (params.max_alt + 500)
    )
    cls.positions_history.append((planex, planey))

    # Display and capture frame
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    frame = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    frames.append(frame)
    cls.frames = frames

    return frames, screen, clock
