"""
This script trains a PPO agent to play as a player in the Gomoku environment against a random opponent.
The results are saved to "Gomoku_PPO_results.zip".
"""
import random
import numpy as np
import gymnasium as gym
from collections import deque
from Gomoku.gomokuenv import CustomEnvironment, GRID_HEIGHT, GRID_WIDTH

class RandomOpponent:
    """Returns a random legal action (0-6)."""

    def __call__(self, obs=None):
        """Ignores observation and returns random action."""
        return CustomEnvironment.action_space(self, "player_x").sample()


class SingleAgentEnv(gym.Env):
    """
    Single-agent wrapper around Parallel CustomEnvironment.
    Agent is player1; opponent provided by opponent_policy.
    """

    def __init__(self, maxsteps, opponent_policy):
        """
        Transforms the two-player CustomEnvironment into a single-agent environment.
        Observation includes own points, opponent points, and opponent's last history_len moves.
        """
        super().__init__()
        self.env = CustomEnvironment(maxsteps)
        self.opponent = opponent_policy
        self.history_len = maxsteps
        obs_dim = 2 + maxsteps
        # low=0.0, high=1.0 for normalization, shape=(obs_dim, ) for saving observation dimensions
        self.observation_space = gym.  spaces.Box(low=0.0, high=1.0, shape=(obs_dim, ), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9) # Actions 0-8
        # Initialize history with default move (1)
        self.history = deque([1] * maxsteps, maxlen=maxsteps)
        self._last_obs = None

    def _encode_move(self, x, y):
        """Encodes move integer to float in [0,1]."""
        return float(x * GRID_WIDTH + y) / (GRID_HEIGHT * GRID_WIDTH)

    def _get_obs(self):
        """Constructs the observation array."""
        po = getattr(self.env, "point_o", 0) # Get player_o points
        px = getattr(self.env, "point_x", 0) # Get player_x points
        own = float(np.clip(po, 0, 20)) / 20.0 # Normalize own points
        opp = float(np.clip(px, 0, 20)) / 20.0 # Normalize opponent points
        hist = [self._encode_move(m) for m in list(self.history)] # Encode history
        return np.array([own, opp] + hist, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Resets the environment and initializes history."""
        try:
            _ = self.env.reset(seed=seed, options=options) # Reset with seed and options
        except TypeError: # For older gym versions
            try:
                _ = self.env.reset(seed) # Reset with seed only
            except Exception: # Fallback for very old versions
                pass
        # Initialize history with default move (1)
        self.history = deque([1] * self.history_len, maxlen=self.history_len)
        obs = self._get_obs() # Get initial observation
        self._last_obs = obs # Store last observation
        return obs, {}

    def step(self, action):
        """Takes a step in the environment using the agent's action and opponent's action."""
        opp_action = int(self.opponent(self._last_obs)) # Get opponent's action
        actions = {"player_o": int(action), "player_x": opp_action}  # Combine actions
        observations, rewards, terminations, truncations, infos = self.env.step(actions) # Get steps in env
        r = rewards.get("player_o", 0) # Reward for player_o
        terminated = bool(terminations.get("player_o", False)) # Termination status for player_o
        truncated = bool(truncations.get("player_o", False)) # Truncation status for player_o
        self.history.append(action)  # Update history with agent's action
        self.history.append(opp_action) # Update history with opponent's action
        obs = self._get_obs() # Update observation after step
        self._last_obs = obs # Store last observation
        info = infos.get("player_o", {}) if isinstance(infos, dict) else {} # Info for player_o
        return obs, float(r), terminated, truncated, info


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
    model.learn(1_000_000)
    model.save("Gomoku_PPO_results")
    print("Training complete. Model saved as 'Gomoku_PPO_results.zip'.")
