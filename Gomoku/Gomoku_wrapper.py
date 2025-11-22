import numpy as np
import gymnasium as gym
from Gomoku_env import CustomEnvironment, GRID_HEIGHT, GRID_WIDTH


class HeuristicOpponent:
    """Returns a heuristic action: Win > Block > Extend > Random."""
    def __init__(self, randomness: float = 0.0):
        self.randomness = randomness

    def __call__(self, obs=None, env: CustomEnvironment = None):
        if env is None:
            # Fallback to random if env is not provided
            return int(np.random.randint(0, GRID_HEIGHT)), int(np.random.randint(0, GRID_WIDTH))

        grid = env.grid
        empties = np.argwhere(grid == 0)
        if len(empties) == 0:
            return 0, 0

        # Chance to play randomly (mistake)
        if np.random.random() < self.randomness:
            idx = np.random.randint(0, len(empties))
            return int(empties[idx, 0]), int(empties[idx, 1])

        my_symbol = env._symbols["player_x"]
        opp_symbol = env._symbols["player_o"]

        # 1. Check for winning moves
        for x, y in empties:
            if env._is_winning_move(x, y, my_symbol):
                return x, y

        # 2. Check for blocking opponent's winning moves
        for x, y in empties:
            if env._is_winning_move(x, y, opp_symbol):
                return x, y

        # 3. Check for creating 4-in-a-row (Extend)
        for x, y in empties:
            if env._get_max_line_length(x, y, my_symbol) >= 4:
                return x, y

        # 4. Check for blocking opponent's 4-in-a-row
        for x, y in empties:
            if env._get_max_line_length(x, y, opp_symbol) >= 4:
                return x, y

        # 5. Random valid move
        idx = np.random.randint(0, len(empties))
        return int(empties[idx, 0]), int(empties[idx, 1])


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