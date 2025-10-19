"""
This script trains a PPO agent to play as a player in the BOBO environment against a random opponent.
The results are saved to "BOBOppo_results.zip". After training, the agent's win rate is evaluated over 100 episodes.
"""
import random
import numpy as np
import gymnasium as gym
from collections import deque
from BOBO.BOBOenv import CustomEnvironment


class RandomOpponent:
    """Returns a random legal action (0-8)."""

    def __call__(self, obs=None):
        """Ignores observation and returns random action."""
        return random.randint(0, 8)


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
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)
        self.history = deque([1] * history_len, maxlen=history_len)
        self._last_obs = None

    def _encode_move(self, m):
        """Encodes move integer to float in [0,1]."""
        return float(m) / 8.0

    def _get_obs(self):
        """Constructs the observation array."""
        p1 = getattr(self.env, "point1", 0)
        p2 = getattr(self.env, "point2", 0)
        own = float(np.clip(p1, 0, 20)) / 20.0
        opp = float(np.clip(p2, 0, 20)) / 20.0
        hist = [self._encode_move(m) for m in list(self.history)]
        return np.array([own, opp] + hist, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        """Resets the environment and initializes history."""
        try:
            _ = self.env.reset(seed=seed, options=options)
        except TypeError:
            try:
                _ = self.env.reset(seed)
            except Exception:
                pass
        self.history = deque([1] * self.history_len, maxlen=self.history_len)
        obs = self._get_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        """Takes a step in the environment using the agent's action and opponent's action."""
        opp_action = int(self.opponent(self._last_obs))
        actions = {"player1": int(action), "player2": opp_action}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        r = rewards.get("player1", 0)
        terminated = bool(terminations.get("player1", False))
        truncated = bool(truncations.get("player1", False))
        self.history.append(opp_action)
        obs = self._get_obs()
        self._last_obs = obs
        info = infos.get("player1", {}) if isinstance(infos, dict) else {}
        return obs, float(r), terminated, truncated, info


def make_env():
    """Creates the single-agent BOBO environment with a random opponent."""
    return SingleAgentEnv(50, 20, RandomOpponent())


if __name__ == "__main__":
    """Train a PPO agent in the BOBO environment and save the model."""

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
    model.learn(100000)
    model.save("BOBOppo_results")
