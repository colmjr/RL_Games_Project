"""
This script trains a PPO agent to play as a player in the Gomoku environment against a random opponent.
The results are saved to "Gomoku_PPO_results.zip".
"""
from Gomoku_env import GRID_HEIGHT, GRID_WIDTH
from Gomoku_wrapper import SingleAgentEnv, RandomOpponent


def make_env():
    """Creates the single-agent Gomoku environment with a random opponent."""
    return SingleAgentEnv(GRID_HEIGHT * GRID_WIDTH, RandomOpponent())


if __name__ == "__main__":
    """Train a PPO agent in the Gomoku environment and save the model."""

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
    )
    model.learn(100_000)
    model.save("Gomoku_PPO_results")
    print("Training complete. Model saved as 'Gomoku_PPO_results.zip'.")
