import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from Gomoku.gomokuenv import CustomEnvironment, GRID_HEIGHT, GRID_WIDTH
from Gomoku.Gomoku_pettingzoo_sb3_wrapper import (
    RandomPolicy,
    make_sb3_env,
)

MAX_STEPS = 42

def _opponent_policy():
    probe_env = CustomEnvironment(MAX_STEPS)
    policy = RandomPolicy(probe_env.action_space("player_x"))
    probe_env.close()
    return policy

def make_env():
    opponent = _opponent_policy()

    def env_fn():
        return CustomEnvironment(MAX_STEPS)

    adapted_obs_space = spaces.Box(
        low=0.0,
        high=2.0,
        shape=(GRID_HEIGHT, GRID_WIDTH),
        dtype=np.float32,
    )

    return make_sb3_env(
        env_fn,
        controlled_agent="player_o",
        opponent_policies={"player_x": opponent},
        observation_adapter=lambda obs: np.asarray(obs, dtype=np.float32),
        override_observation_space=adapted_obs_space,
    )

if __name__ == "__main__":
    env_fn = make_env()
    vec_env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, n_steps=2048, batch_size=64, learning_rate=3e-4,
                ent_coef=0.01, gamma=0.99, verbose=1)
    model.learn(1_000_000)
    model.save("gomokuppo_results")
