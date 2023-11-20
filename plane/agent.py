from jaxppo.ppo import PPO, LoggingConfig

from plane.env import Airplane2D, EnvParams

logging_config = LoggingConfig(  # TODO : automate generation of this
    project_name="plane test",
    group_name="test",
    run_name=f"test 1",
    config={"agent": f"jax", "action_stacking": False},
    mode="online",
)
env = Airplane2D()
agent = PPO(
    total_timesteps=int(1e6),
    num_steps=2048,
    num_envs=4,
    env_id=env,
    learning_rate=1e-4,
    env_params=EnvParams(),
    logging_config=logging_config,
)
agent.train(seed=42, test=False)
