from plane_env.env_jax import Airplane2D, EnvParams
    
# Create env
env = Airplane2D()
seed = 42
env_params = EnvParams(max_steps_in_episode=1_000)

# Simple constant policy with 80% power and 0° stick input.
action = (0.5, 0.0)

# Save the video
env.save_video(lambda o: action, seed, folder="videos", episode_index=0, params=env_params, format="gif")
