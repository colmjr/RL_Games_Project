from minesweeper_env import  MinesweeperEnv
def make_env():
    return MinesweeperEnv(10,20,6)  # mine_multiple=6, gridheight=10, gridwidth=20
if __name__ == "__main__":
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
    model.save("connect4_TRPO_results")
    print("Training complete. Model saved as 'connect_TRPO_results.zip'.")
    model = TRPO.load("connect4_TRPO_results")
