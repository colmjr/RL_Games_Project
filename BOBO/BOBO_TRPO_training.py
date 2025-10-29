from BOBO_wrapper import SingleAgentEnv, RandomOpponent
def make_env():
    """Creates the single-agent BOBO environment with a random opponent."""
    return SingleAgentEnv(50, 20, RandomOpponent())


if __name__ == "__main__":
    """Train a PPO agent in the BOBO environment and save the model."""

    from sb3_contrib import TRPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([make_env])
    model = TRPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=1e-3
    )
    model.learn(1e6,progress_bar=True)
    model.save("BOBO_TRPO_results")
    print("Training complete. Model saved as 'BOBO_TRPO_results.zip'.")
    model = TRPO.load("BOBO_TRPO_results")
