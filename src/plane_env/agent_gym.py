import os

from stable_baselines3 import PPO

from plane_env.env_gymnasium import Airplane2D


def evaluate_and_save_video(
    model, env_id, video_folder, seed, env_kwargs, episode_index=0
):
    """Run evaluation episode and save video"""
    eval_env = Airplane2D(render_mode="rgb_array_list", **env_kwargs)
    obs, _ = eval_env.reset(seed=seed)
    done = False
    frames = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = eval_env.step(action)
        frames = eval_env.render()
        done = terminated or truncated

    # Save video
    from gymnasium.utils.save_video import save_video

    save_video(
        frames,
        video_folder,
        fps=eval_env.metadata["render_fps"],
    )
    eval_env.close()
    return os.path.join(video_folder, "rl-video-episode-0.mp4")


class TrainingCallback:
    def __init__(
        self,
        video_folder: str,
        eval_freq: int,
        total_timesteps: int,
        seed: int,
        mode: str = "power_and_stick",
    ):
        self.video_folder = video_folder
        self.eval_freq = eval_freq
        self.seed = seed
        self.step_count = 0
        self.total_timesteps = total_timesteps
        self.video_count = 0  # Add counter for video indexing
        self.env_kwargs = {"mode": mode}

    def __call__(self, locals: dict, globals: dict) -> bool:
        self.step_count += 1

        # Check if we should evaluate
        if self.step_count >= self.eval_freq * (self.video_count + 1):
            model = locals["self"]
            progress = self.step_count / self.total_timesteps

            # Use integer index for save_video
            video_path = evaluate_and_save_video(
                model,
                env_id="Airplane2D",
                video_folder=self.video_folder,
                seed=self.seed,
                episode_index=self.video_count,
                env_kwargs=self.env_kwargs,
            )
            print(
                f"Saved evaluation video at {progress*100:.1f}% progress to {video_path}"
            )

            # Create descriptive name after saving
            progress_name = f"seed{self.seed}_progress{int(progress*100)}"
            os.rename(
                video_path,
                os.path.join(self.video_folder, f"rl-video-{progress_name}.mp4"),
            )

            self.video_count += 1

        return True


def main():
    # Create base environment
    mode = "stick_only"
    env = Airplane2D(mode=mode)
    video_folder = f"training_videos_{mode}"
    os.makedirs(video_folder, exist_ok=True)

    total_timesteps = int(1e6)
    # Calculate eval_freq to get 10 evaluations during training
    eval_freq = total_timesteps // 100

    for seed in range(10):
        print(f"Training seed {seed}")
        # Create and train PPO model
        model = PPO(
            "MlpPolicy",
            env,
            # learning_rate=1e-3,
            seed=seed,
            tensorboard_log="./logs_riad",
            ent_coef=0.1,
        )

        # Create callback for this seed
        callback = TrainingCallback(
            video_folder=video_folder,
            eval_freq=eval_freq,
            total_timesteps=total_timesteps,
            seed=seed,
            mode=mode,
        )

        # Train model with callback
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save final evaluation video with integer index
        final_path = evaluate_and_save_video(
            model,
            env_id="Airplane2D",
            video_folder=video_folder,
            seed=seed,
            episode_index=callback.video_count,
            env_kwargs={"mode": mode},
        )
        # Rename to descriptive name
        final_name = f"seed{seed}_final"
        os.rename(final_path, os.path.join(video_folder, f"rl-video-{final_name}.mp4"))
        print(f"Saved final evaluation video for seed {seed}")


if __name__ == "__main__":
    main()
