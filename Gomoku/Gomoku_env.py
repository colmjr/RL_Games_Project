from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

GRID_HEIGHT = 15
GRID_WIDTH = 15
WIN_LENGTH = 5

class CustomEnvironment(ParallelEnv):
    metadata = {"name": "gomoku_parallel_v0"}

    def __init__(self, maxsteps: int) -> None:
        if maxsteps <= 0:
            raise ValueError("maxsteps must be a positive integer")

        self.maxsteps = maxsteps
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.possible_agents = ["player_o", "player_x"]
        self.agents = list(self.possible_agents)

        # Map agents to numeric symbols stored on the grid.
        self._symbols: Dict[str, int] = {"player_o": 1, "player_x": 2}

        # Gymnasium spaces used by PettingZoo.
        self.action_spaces = {
            agent: spaces.MultiDiscrete([self.grid_height, self.grid_width])
            for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=len(self._symbols),
                shape=(self.grid_height, self.grid_width),
                dtype=np.int8,
            )
            for agent in self.possible_agents
        }
        self._rng = np.random.default_rng()
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        self.timestep = 0
        self._last_moves: Dict[str, Optional[Tuple[int, int]]] = {
            agent: None for agent in self.possible_agents
        }
    def reset(self, *, seed, options: Optional[Mapping[str, object]] = None): #options mapping just in case we want to add more later
        del options
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.grid.fill(0)
        self.timestep = 0
        self._last_moves = {agent: None for agent in self.possible_agents}

        observations = {agent: self._grid_observation() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Mapping[str, np.ndarray]):
        if not self.agents:
            raise RuntimeError("reset() must be called before step().")

        moves: Dict[str, Tuple[int, int]] = {}
        invalid_agent: Optional[str] = None

        # Validate and parse actions.
        for agent in self.possible_agents:
            if agent not in actions:
                raise KeyError(f"Missing action for agent '{agent}'")

            action = actions[agent]
            x, y = int(action[0]), int(action[1])
            if not self._is_within_bounds(x, y):
                invalid_agent = agent
                break
            moves[agent] = (x, y)

        # Prevent both agents from selecting the same cell simultaneously.
        if invalid_agent is None and moves["player_o"] == moves["player_x"]:
            invalid_agent = "player_x"

        # Check for collisions with existing stones.
        if invalid_agent is None:
            for agent, (x, y) in moves.items():
                if self.grid[x, y] != 0:
                    invalid_agent = agent
                    break

        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}

        winner: Optional[str] = None

        if invalid_agent is None:
            for agent, (x, y) in moves.items():
                symbol = self._symbols[agent]
                self.grid[x, y] = symbol
                self._last_moves[agent] = (x, y)

            for agent, (x, y) in moves.items():
                symbol = self._symbols[agent]
                if self._is_winning_move(x, y, symbol):
                    winner = agent
                    break

            self.timestep += 1
            board_full = not np.any(self.grid == 0)

            if winner is not None:
                rewards[winner] = 1.0
                rewards[self._other_agent(winner)] = -1.0
                print(f"'{winner}' wins!")
                terminations = {agent: True for agent in self.possible_agents}
                self.agents = []
            elif self.timestep >= self.maxsteps or board_full:
                truncations = {agent: True for agent in self.possible_agents}
                self.agents = []
        else:
            print(f"Agent '{invalid_agent}' made an invalid move.")
            losing_agent = invalid_agent
            rewards[losing_agent] = -1.0
            rewards[self._other_agent(losing_agent)] = 1.0
            truncations = {agent: True for agent in self.possible_agents}
            self.agents = []

        for agent in self.possible_agents:
            infos[agent] = {
                "last_move": self._last_moves.get(agent),
                "winner": winner,
            }

        observations = {agent: self._grid_observation() for agent in self.possible_agents}
        return observations, rewards, terminations, truncations, infos
    def render(self):
        grid_str = "\n".join(" ".join(str(cell) for cell in row) for row in self.grid)
        print(grid_str)
    def close(self):
        pass
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]
    def action_space(self, agent: str):
        return self.action_spaces[agent]
    

    def _grid_observation(self) -> np.ndarray:
        return self.grid.copy()
    def _is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_height and 0 <= y < self.grid_width and self.grid[x, y] == 0
    def _is_winning_move(self, x: int, y: int, symbol: int) -> bool:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):#each direction: horizontal, vertical, diagonal
            total = 1
            for step in range(1, WIN_LENGTH):
                nx = x + dx * step
                ny = y + dy * step
                if not self._is_within_bounds(nx, ny) or self.grid[nx, ny] != symbol:
                    break
                total += 1

            for step in range(1, WIN_LENGTH):
                nx = x - dx * step
                ny = y - dy * step
                if not self._is_within_bounds(nx, ny) or self.grid[nx, ny] != symbol:
                    break
                total += 1

            if total >= WIN_LENGTH:
                return True

        return False

    def _other_agent(self, agent: str) -> str:
        return "player_x" if agent == "player_o" else "player_o"

    def _board_to_obs(self):
        flat = [cell for row in self.grid for cell in row]
        return np.array(flat, dtype=np.int64)
