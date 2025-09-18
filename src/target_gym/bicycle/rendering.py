import jax.numpy as jnp
import numpy as np
import pygame

from target_gym.bicycle.env import compute_reward


def render_bicycle_top_view_fixed(
    screen_width,
    screen_height,
    state,
    params,
    positions_history=None,
    scale=50.0,
    trail_spacing=3,
    world_bounds=None,
):
    if positions_history is None:
        positions_history = []

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((200, 200, 200))

    # --- Compute reward ---
    reward = float(compute_reward(state, params))

    # --- World to screen mapping ---
    if world_bounds is None:
        xs = [state.x_b] + [p[0] for p in positions_history]
        ys = [state.y_b] + [p[1] for p in positions_history]
        x_min, x_max = min(xs) - 2.0, max(xs) + 2.0
        y_min, y_max = min(ys) - 2.0, max(ys) + 2.0
    else:
        (x_min, x_max), (y_min, y_max) = world_bounds

    def world_to_screen(x, y):
        sx = int((x - x_min) / (x_max - x_min) * screen_width)
        sy = int(screen_height - (y - y_min) / (y_max - y_min) * screen_height)
        return sx, sy

    # --- Trail ---
    positions_history.append((state.x_b, state.y_b))
    n = len(positions_history)
    for i, (xb, yb) in enumerate(positions_history[::trail_spacing]):
        sx, sy = world_to_screen(xb, yb)
        color = (50 + int(205 * i / max(1, n - 1)), 50, 50)
        pygame.draw.circle(surf, color, (sx, sy), 2)

    # --- Compute bike frame (scaled for visibility) ---
    scale_factor = 1.0
    dx = (state.x_f - state.x_b) * scale_factor
    dy = (state.y_f - state.y_b) * scale_factor
    xb, yb = world_to_screen(state.x_b, state.y_b)
    xf, yf = world_to_screen(state.x_b + dx, state.y_b + dy)

    # --- Draw bike frame ---
    pygame.draw.line(surf, (50, 50, 200), (xb, yb), (xf, yf), 6)

    # --- Draw wheels as thin lines perpendicular to frame ---
    wheel_length = 16
    frame_angle = np.arctan2(yf - yb, xf - xb)

    for x, y in [(xb, yb), (xf, yf)]:
        wx1 = x + np.cos(frame_angle) * wheel_length / 2
        wy1 = y + np.sin(frame_angle) * wheel_length / 2
        wx2 = x - np.cos(frame_angle) * wheel_length / 2
        wy2 = y - np.sin(frame_angle) * wheel_length / 2
        pygame.draw.line(surf, (0, 0, 0), (wx1, wy1), (wx2, wy2), 4)

    # --- Draw handlebars at front wheel ---
    handle_length = 60
    theta = -state.theta
    handle_angle = frame_angle + theta
    hx1 = xf + np.cos(handle_angle + np.pi / 2) * handle_length / 2
    hy1 = yf + np.sin(handle_angle + np.pi / 2) * handle_length / 2
    hx2 = xf + np.cos(handle_angle - np.pi / 2) * handle_length / 2
    hy2 = yf + np.sin(handle_angle - np.pi / 2) * handle_length / 2
    pygame.draw.line(surf, (200, 0, 0), (hx1, hy1), (hx2, hy2), 3)

    # --- HUD ---
    font = pygame.font.SysFont("Arial", 16)
    padding = 8
    line_height = 18

    trajectory_length = 0.0
    if len(positions_history) > 1:
        traj = np.array(positions_history)
        diffs = np.diff(traj, axis=0)
        trajectory_length = np.sum(np.linalg.norm(diffs, axis=1))

    left_texts = [
        f"Lean ω: {np.rad2deg(state.omega):.1f}°",
        f"Steer θ: {np.rad2deg(state.theta):.1f}°",
        f"Step: {state.t}",
    ]
    right_texts = [
        f"Heading ψ: {np.rad2deg(state.psi):.1f}°",
        f"Trajectory: {trajectory_length:.1f} m",
        f"Reward: {reward:.3f}",
    ]

    left_surfs = [font.render(t, True, (0, 0, 0)) for t in left_texts]
    right_surfs = [font.render(t, True, (0, 0, 0)) for t in right_texts]

    def column_width(surfs):
        return max(s.get_width() for s in surfs) + padding

    left_width = column_width(left_surfs)
    right_width = column_width(right_surfs)
    hud_width = left_width + right_width + 3 * padding
    hud_height = max(len(left_surfs), len(right_surfs)) * line_height + 2 * padding
    hud_rect = pygame.Rect(10, 10, hud_width, hud_height)

    pygame.draw.rect(surf, (0, 0, 0), hud_rect, border_radius=6)
    pygame.draw.rect(surf, (255, 255, 255), hud_rect.inflate(-4, -4), border_radius=6)

    for i, s in enumerate(left_surfs):
        surf.blit(s, (hud_rect.x + padding, hud_rect.y + padding + i * line_height))
    for i, s in enumerate(right_surfs):
        surf.blit(
            s,
            (
                hud_rect.x + left_width + 2 * padding,
                hud_rect.y + padding + i * line_height,
            ),
        )

    # --- Torque & displacement bars ---
    a_T, a_d = state.torque, state.displacement
    bar_width = 150
    bar_height = 15
    bar_x = hud_rect.right + 20
    bar_y = hud_rect.y + 15

    def draw_bar(center_value, label):
        # Draw label above bar
        lbl_surf = font.render(label, True, (0, 0, 0))
        surf.blit(lbl_surf, (bar_x, bar_y - line_height, bar_width, line_height))

        # Background: red left, green right
        pygame.draw.rect(surf, (180, 0, 0), (bar_x, bar_y, bar_width // 2, bar_height))
        pygame.draw.rect(
            surf,
            (0, 180, 0),
            (bar_x + bar_width // 2, bar_y, bar_width // 2, bar_height),
        )

        # Fill based on positive/negative value
        if center_value >= 0:
            fill_width = int(bar_width // 2 * np.clip(center_value, 0, 1))
            pygame.draw.rect(
                surf,
                (144, 238, 144),
                (bar_x + bar_width // 2, bar_y, fill_width, bar_height),
            )
        else:
            fill_width = int(bar_width // 2 * np.clip(-center_value, 0, 1))
            pygame.draw.rect(
                surf,
                (255, 160, 122),
                (bar_x + bar_width // 2 - fill_width, bar_y, fill_width, bar_height),
            )

    draw_bar(a_T, "Torque")
    bar_y += bar_height + 20
    draw_bar(a_d, "Displacement")

    return surf, positions_history


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
        cls.screen = screen

    if clock is None:
        clock = pygame.time.Clock()
        cls.clock = clock

    surf, cls.positions_history = render_bicycle_top_view_fixed(
        cls.screen_width,
        cls.screen_height,
        state,
        params,
        positions_history=cls.positions_history,
        scale=50.0,
    )

    screen.blit(surf, (0, 0))
    pygame.display.flip()

    frame = np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
    frames.append(frame)
    cls.frames = frames
    return frames, screen, clock
