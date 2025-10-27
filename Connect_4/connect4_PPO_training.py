"""
This script trains a PPO agent to play as a player in the connect 4 environment against a random opponent.
The results are saved to "connect4_PPO_results.zip".
"""
from connect4_env import ROWS, COLUMNS
from connect4_wrapper import SingleAgentEnv, RandomOpponent
def make_env():
    """Creates the single-agent connect 4 environment with a random opponent."""
    return SingleAgentEnv(ROWS * COLUMNS, RandomOpponent())


if __name__ == "__main__":
    """Train a PPO agent in the connect 4 environment and save the model."""

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
    model.learn(100_000,progress_bar=True)
    model.save("connect4_PPO_results")
    print("Training complete. Model saved as 'connect4_PPO_results.zip'.")
