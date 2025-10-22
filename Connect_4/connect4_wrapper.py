import numpy as np
import gymnasium as gym
from connect4_env import CustomEnvironment, ROWS, COLUMNS, column_is_full


class RandomOpponent:
    """Returns a random legal action."""

    def __call__(self, obs=None, env=None):
        """Ignores observation and returns random legal action using env to avoid full columns."""
        try:
            # If env available, prefer sampling from non-full columns to avoid immediate invalid-action terminations
            available = [c for c in range(COLUMNS) if not column_is_full(env.grid, c)]
            if not available:
                return 0
            else:
                return int(np.random.choice(available))
        except Exception:
            # Fallback to uniform sampling if env not provided or any error occurs
            return int(np.random.randint(0, COLUMNS))


class SingleAgentEnv(gym.Env):
    """
    Single-agent wrapper around Parallel CustomEnvironment.
    Agent is player_o; opponent provided by opponent_policy.
    """

    def __init__(self, maxsteps, opponent_policy):
        """
        Transforms the two-player CustomEnvironment into a single-agent environment.
        Observation exposes the flattened board for player_o.
        """
        super().__init__()
        self.env = CustomEnvironment(maxsteps)
        self.opponent = opponent_policy
        # low=0.0, high=1.0 for normalization, shape=(ROWS * COLUMNS,) for saving observation dimensions
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(ROWS * COLUMNS,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(COLUMNS)
        self._last_obs = None

    def _get_obs(self, raw_obs=None):
        """Return normalized flattened board observation as float32 vector in [0,1]."""
        if raw_obs is None:
            raw_obs = self.env._board_to_obs()
        arr = np.array(raw_obs, dtype=np.float32)
        return arr / 2.0  # Normalize the value {0,1,2} to [0,1]

    def reset(self, *, seed=None, options=None):
        """Resets the environment and returns player_o observation."""
        try:
            observations, infos = self.env.reset(seed=seed, options=options)
        except TypeError:
            observations, infos = self.env.reset(seed=seed)
        obs_raw = observations.get("player_o") # Get player_o observation
        obs = self._get_obs(obs_raw) # Process observation
        self._last_obs = obs # Store last observation
        return obs, {}

    def step(self, action):
        """Takes a step in the environment using the agent's action and opponent's action."""
        opp_action = int(self.opponent(self._last_obs, self.env)) # Get opponent action
        actions = {"player_o": int(action), "player_x": opp_action} # Combine actions
        observations, rewards, terminations, truncations, infos = self.env.step(actions) # Get steps in env
        r = float(rewards.get("player_o", 0.0)) # Reward for player_o
        terminated = bool(terminations.get("player_o", False)) # Termination status for player_o
        truncated = bool(truncations.get("player_o", False)) # Truncation status for player_o
        obs_raw = observations.get("player_o") # Get player_o observation
        obs = self._get_obs(obs_raw) # Process observation
        self._last_obs = obs # Store last observation
        info = infos.get("player_o", {}) if isinstance(infos, dict) else {} # Info for player_o
        return obs, r, terminated, truncated, info