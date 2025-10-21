"""
Wrapper utilities to transform a PettingZoo ParallelEnv into a Gymnasium-compatible
single-agent environment that Stable Baselines3 (SB3) can consume.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class RandomPolicy:
    """Selects a random action from the provided Gymnasium space."""

    def __init__(self, action_space: spaces.Space):
        self._action_space = action_space

    def __call__(self, observation: Any | None = None) -> Any:
        del observation  # Opponent acts randomly, ignores observation.
        return self._action_space.sample()


class PettingZooToSB3Wrapper(gym.Env):
    """
    Single-agent Gymnasium view over a PettingZoo ``ParallelEnv``.

    The wrapper lets SB3 control one agent while opponent actions are produced
    by user-specified policies. Optional adapters translate between SB3-friendly
    observation/action formats and the underlying multi-agent environment.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env: ParallelEnv,
        controlled_agent: str,
        opponent_policies: Mapping[str, Callable[[Any], Any]],
        *,
        observation_adapter: Optional[Callable[[Any], Any]] = None,
        action_adapter: Optional[Callable[[Any], Any]] = None,
        info_adapter: Optional[Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]] = None,
        override_observation_space: Optional[spaces.Space] = None,
        override_action_space: Optional[spaces.Space] = None,
    ) -> None:
        """
        Args:
            env: The PettingZoo ``ParallelEnv`` instance to wrap. It must already
                be resettable and step-able in the usual PettingZoo fashion.
            controlled_agent: Agent id that SB3 will control.
            opponent_policies: Mapping from *other* agent ids to callables producing
                their actions (e.g. random or scripted policies).
            observation_adapter: Optional callable converting the controlled agent's
                observation from the environment into the format SB3 expects.
            action_adapter: Optional callable converting SB3's action into the
                format required by the PettingZoo environment.
            info_adapter: Optional callable to post-process the controlled agent's
                info dict before returning it to SB3.
            override_observation_space: Provide when the adapted observation no longer
                matches the environment's declared observation space.
            override_action_space: Provide when the adapted action no longer
                matches the environment's declared action space.
        """
        super().__init__()
        if not isinstance(env, ParallelEnv):
            raise TypeError("env must be an instance of pettingzoo.ParallelEnv")

        self.env = env
        self.controlled_agent = controlled_agent
        self._observation_adapter = observation_adapter
        self._action_adapter = action_adapter
        self._info_adapter = info_adapter
        self._last_observations: Dict[str, Any] = {}
        self._has_reset = False

        self.possible_agents = list(getattr(env, "possible_agents", []))
        if controlled_agent not in self.possible_agents:
            raise ValueError(f"{controlled_agent=} not present in env.possible_agents={self.possible_agents}")

        # Copy and validate opponent policies for agents other than the controlled one.
        self._opponent_policies: Dict[str, Callable[[Any], Any]] = {}
        for agent_id in self.possible_agents:
            if agent_id == controlled_agent:
                continue
            policy = opponent_policies.get(agent_id)
            if policy is None:
                raise ValueError(f"Missing opponent policy for agent '{agent_id}'")
            self._opponent_policies[agent_id] = policy

        # Determine Gymnasium spaces for SB3.
        env_action_space = env.action_space(controlled_agent)
        env_observation_space = env.observation_space(controlled_agent)

        self.action_space = override_action_space or env_action_space
        self.observation_space = override_observation_space or env_observation_space

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        observations, infos = self.env.reset(seed=seed, options=options)
        if not isinstance(observations, Mapping):
            raise TypeError("Expected env.reset() to return a mapping of agent observations.")

        self._last_observations = dict(observations)
        self._has_reset = True

        obs = observations[self.controlled_agent]
        info = infos.get(self.controlled_agent, {}) if isinstance(infos, Mapping) else {}

        if self._observation_adapter is not None:
            obs = self._observation_adapter(obs)
        if self._info_adapter is not None:
            info = self._info_adapter(info)

        return obs, info

    def step(self, action: Any):
        if not self._has_reset:
            raise RuntimeError("Call reset() before step().")

        # Build the joint action dictionary for all live agents.
        actions: Dict[str, Any] = {}
        current_agents = list(getattr(self.env, "agents", self.possible_agents))

        for agent_id in current_agents:
            if agent_id == self.controlled_agent:
                env_action = self._action_adapter(action) if self._action_adapter else action
            else:
                opp_policy = self._opponent_policies.get(agent_id)
                if opp_policy is None:
                    raise RuntimeError(f"No opponent policy provided for agent '{agent_id}'.")
                opp_obs = self._last_observations.get(agent_id)
                env_action = opp_policy(opp_obs)
            actions[agent_id] = env_action

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if not isinstance(observations, Mapping):
            raise TypeError("Expected env.step() to return a mapping of agent observations.")

        self._last_observations = dict(observations)

        obs = observations.get(self.controlled_agent)
        reward = rewards.get(self.controlled_agent, 0.0)
        terminated = bool(terminations.get(self.controlled_agent, False))
        truncated = bool(truncations.get(self.controlled_agent, False))
        info = infos.get(self.controlled_agent, {}) if isinstance(infos, Mapping) else {}

        if self._observation_adapter is not None:
            obs = self._observation_adapter(obs)
        if self._info_adapter is not None:
            info = self._info_adapter(info)

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_sb3_env(
    env_fn: Callable[[], ParallelEnv],
    *,
    controlled_agent: str,
    opponent_policies: Mapping[str, Callable[[Any], Any]],
    observation_adapter: Optional[Callable[[Any], Any]] = None,
    action_adapter: Optional[Callable[[Any], Any]] = None,
    info_adapter: Optional[Callable[[Optional[Dict[str, Any]]], Dict[str, Any]]] = None,
    override_observation_space: Optional[spaces.Space] = None,
    override_action_space: Optional[spaces.Space] = None,
) -> Callable[[], PettingZooToSB3Wrapper]:
    """
    Helper for Stable Baselines3 ``make_vec_env`` style factories.

    Example::

        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv([
            make_sb3_env(my_pettingzoo_env, controlled_agent=\"player_x\", opponent_policies={...})
        ])
        model = PPO(\"MlpPolicy\", vec_env, ...)
    """

    def _init() -> PettingZooToSB3Wrapper:
        env = env_fn()
        return PettingZooToSB3Wrapper(
            env,
            controlled_agent=controlled_agent,
            opponent_policies=opponent_policies,
            observation_adapter=observation_adapter,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
            override_observation_space=override_observation_space,
            override_action_space=override_action_space,
        )

    return _init


__all__ = [
    "PettingZooToSB3Wrapper",
    "RandomPolicy",
    "make_sb3_env",
]
