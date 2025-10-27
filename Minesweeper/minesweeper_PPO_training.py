from  Minesweeper.minesweeper_env import MinesweeperEnv

def make_env():
    return MinesweeperEnv(10,20,6)


if __name__ == "__main__":
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
    model.save("Minesweeper_PPO_results")
    print("Training complete. Model saved as 'Minesweeper_PPO_results.zip'.")