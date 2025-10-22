import numpy as np
import gymnasium as gym
from collections import deque
from BOBO_env import CustomEnvironment


class RandomOpponent:
    """Returns a random legal action (0-8)."""

    def __call__(self, obs=None):
        """Ignores observation and returns random action."""
        # numpy.randint high is exclusive; use 9 to include action 8
        return int(np.random.randint(0, 9))


class SingleAgentEnv(gym.Env):
    """
    Single-agent wrapper around Parallel CustomEnvironment.
    Agent is player1; opponent provided by opponent_policy.
    """

    def __init__(self, maxsteps, history_len, opponent_policy):
        """
        Transforms the two-player CustomEnvironment into a single-agent environment.
        Observation includes own points, opponent points, and opponent's last history_len moves.
        """
        super().__init__()
        self.env = CustomEnvironment(maxsteps)
        self.opponent = opponent_policy
        self.history_len = history_len
        obs_dim = 2 + history_len
        # low=0.0, high=1.0 for normalization, shape=(obs_dim,) for saving observation dimensions
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9) # Actions 0-8
        # Initialize history with default move (1)
        self.history = deque([1] * history_len, maxlen=history_len)
        self._last_obs = None

    def _encode_move(self, m):
        """Encodes move integer to float in [0,1]."""
        return float(m) / 8.0

    def _get_obs(self):
        """Constructs the observation array."""
        p1 = getattr(self.env, "point1", 0) # Get player1 points
        p2 = getattr(self.env, "point2", 0) # Get player2 points
        own = float(np.clip(p1, 0, 20)) / 20.0 # Normalize own points
        opp = float(np.clip(p2, 0, 20)) / 20.0 # Normalize opponent points
        hist = [self._encode_move(m) for m in list(self.history)] # Encode history
        return np.array([own, opp] + hist, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Resets the environment and initializes history."""
        try:
            _ = self.env.reset(seed=seed, options=options)
        except TypeError:
            _ = self.env.reset(seed)
        # Initialize history with default move (1)
        self.history = deque([1] * self.history_len, maxlen=self.history_len)
        obs = self._get_obs() # Get initial observation
        self._last_obs = obs # Store last observation
        return obs, {}

    def step(self, action):
        """Takes a step in the environment using the agent's action and opponent's action."""
        opp_action = int(self.opponent(self._last_obs)) # Get opponent action
        actions = {"player1": int(action), "player2": opp_action} # Combine actions
        observations, rewards, terminations, truncations, infos = self.env.step(actions) # Get steps in env
        r = float(rewards.get("player1", 0)) # Reward for player1
        terminated = bool(terminations.get("player1", False)) # Termination status for player1
        truncated = bool(truncations.get("player1", False)) # Truncation status for player1
        self.history.append(opp_action) # Update history with opponent's action
        obs = self._get_obs() # Update observation after step
        self._last_obs = obs # Store last observation
        info = infos.get("player1", {}) if isinstance(infos, dict) else {} # Info for player1
        return obs, r, terminated, truncated, info