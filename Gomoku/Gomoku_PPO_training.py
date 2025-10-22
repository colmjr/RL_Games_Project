"""
This script trains a PPO agent to play as a player in the Gomoku environment against a random opponent.
The results are saved to "Gomoku_PPO_results.zip".
"""
import numpy as np
import gymnasium as gym
from Gomoku_env import CustomEnvironment, GRID_HEIGHT, GRID_WIDTH


class RandomOpponent:
    """Returns a random legal action."""

    def __call__(self, obs=None, env: CustomEnvironment = None):
        """Ignores observation and returns random legal action using env to avoid full columns."""
        try:
            # If env is available, return a tuple (x,y) for a random empty cell in env.grid.
            grid = env.grid  # type: ignore[attr-defined]
            empties = np.argwhere(grid == 0)
            if len(empties) == 0:
                return 0, 0
            idx = np.random.randint(0, len(empties))
            x, y = int(empties[idx, 0]), int(empties[idx, 1])
            return x, y
        except Exception:
            # Fallback to uniform sampling if env not provided or any error occurs
            x = int(np.random.randint(0, GRID_HEIGHT))
            y = int(np.random.randint(0, GRID_WIDTH))
            return x, y


class SingleAgentEnv(gym.Env):
    """
    Single-agent wrapper around Parallel CustomEnvironment.
    Agent is player_o; opponent provided by opponent_policy.
    """

    def __init__(self, maxsteps: int, opponent_policy):
        super().__init__()
        self.env = CustomEnvironment(maxsteps)
        self.opponent = opponent_policy
        # low=0.0, high=1.0 for normalization, shape=(GRID_HEIGHT * GRID_WIDTH,) for saving observation dimensions
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(GRID_HEIGHT * GRID_WIDTH,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(GRID_HEIGHT * GRID_WIDTH)
        self._last_obs = None

    def _get_obs(self, raw_obs=None):
        """Return normalized flattened board observation as float32 vector in [0,1]."""
        if raw_obs is None:
            raw_obs = self.env._board_to_obs()
        arr = np.asarray(raw_obs, dtype=np.float32).flatten()
        return arr / 2.0  # Normalize the value {0,1,2} to [0,1]

    def reset(self, *, seed=None, options=None):
        """Reset underlying parallel env and return the observation for player_o."""
        try:
            observations, infos = self.env.reset(seed=seed, options=options)
        except TypeError:
            observations, infos = self.env.reset(seed=seed)
        obs_raw = observations.get("player_o")
        obs = self._get_obs(obs_raw)
        self._last_obs = obs
        return obs, {}

    def step(self, action: int):
        """Take an integer action (0..H*W-1) for player_o and sample opponent action, call env.step."""
        # Decode action integer to (x,y) coordinates
        x = int(action) // GRID_WIDTH
        y = int(action) % GRID_WIDTH
        agent_move = (x, y)
        opp_move = self.opponent(self._last_obs, self.env)
        actions = {"player_o": np.array(agent_move, dtype=np.int64), "player_x": np.array(opp_move, dtype=np.int64)}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        r = float(rewards.get("player_o", 0.0))
        terminated = bool(terminations.get("player_o", False))
        truncated = bool(truncations.get("player_o", False))
        obs_raw = observations.get("player_o") # Get player_o observation
        obs = self._get_obs(obs_raw) # Process observation
        self._last_obs = obs
        info = infos.get("player_o", {}) if isinstance(infos, dict) else {}
        return obs, r, terminated, truncated, info


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
