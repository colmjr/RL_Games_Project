from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

GRID_HEIGHT = 12
GRID_WIDTH = 12
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

        def _detect_conflicts() -> Optional[str]:
            if len(moves) != len(self.possible_agents):
                return None
            if moves["player_o"] == moves["player_x"]:
                return "player_x"
            for agent, (x, y) in moves.items():
                if self.grid[x, y] != 0:
                    return agent
            return None

        # Validate and parse actions.
        for agent in self.possible_agents:
            if agent not in actions:
                raise KeyError(f"Missing action for agent '{agent}'")

            action = actions[agent]
            x, y = int(action[0]), int(action[1])
            if not self._is_within_bounds(x, y):
                if invalid_agent is None:
                    invalid_agent = agent
                continue
            moves[agent] = (x, y)

        if invalid_agent is None:
            invalid_agent = _detect_conflicts()

        if invalid_agent == "player_o":
            fallback = self._random_empty_cell(exclude=moves.get("player_x"))
            if fallback is not None:
                moves["player_o"] = fallback
                invalid_agent = None
                invalid_agent = _detect_conflicts()

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
                
                # Step penalty
                rewards[agent] -= 0.01
                
                # Intermediate rewards for forming lines
                line_length = self._get_max_line_length(x, y, symbol)
                if line_length == 4:
                    rewards[agent] += 2.0
                    print(f"Reward: +2.0 for Line of 4 by {agent}")
                elif line_length == 3:
                    rewards[agent] += 0.5
                    print(f"Reward: +0.5 for Line of 3 by {agent}")

                # Intermediate rewards for blocking opponent
                opp_symbol = self._symbols[self._other_agent(agent)]
                # Check if this move blocked a winning move for the opponent
                # We temporarily set the cell to opponent's symbol and check line length
                self.grid[x, y] = opp_symbol
                opp_line_length = self._get_max_line_length(x, y, opp_symbol)
                self.grid[x, y] = symbol # Restore correct symbol

                if opp_line_length >= 5:
                    rewards[agent] += 5.0 # Critical block
                    print(f"Reward: +5.0 for Critical Block by {agent}")
                elif opp_line_length == 4:
                    rewards[agent] += 1.0 # Major block
                    print(f"Reward: +1.0 for Major Block by {agent}")
                elif opp_line_length == 3:
                    rewards[agent] += 0.5 # Minor block
                    print(f"Reward: +0.5 for Minor Block by {agent}")

            for agent, (x, y) in moves.items():
                symbol = self._symbols[agent]
                if self._is_winning_move(x, y, symbol):
                    winner = agent
                    break

            self.timestep += 1
            board_full = not np.any(self.grid == 0)

            if winner is not None:
                rewards[winner] += 10.0
                rewards[self._other_agent(winner)] -= 10.0
                print(f"'{winner}' wins!")
                terminations = {agent: True for agent in self.possible_agents}
                self.agents = []
            elif self.timestep >= self.maxsteps or board_full:
                truncations = {agent: True for agent in self.possible_agents}
                self.agents = []
        else:
            print(f"Agent '{invalid_agent}' made an invalid move.")
            losing_agent = invalid_agent
            rewards[losing_agent] = -50.0
            rewards[self._other_agent(losing_agent)] = 0.0 # Opponent doesn't get reward for enemy suicide to avoid learning to force it
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
    def _is_valid_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_height and 0 <= y < self.grid_width
    def _random_empty_cell(self, exclude: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
        empties = np.argwhere(self.grid == 0)
        if exclude is not None:
            ex_x, ex_y = exclude
            empties = [cell for cell in empties if not (cell[0] == ex_x and cell[1] == ex_y)]
        if not len(empties):
            return None
        idx = int(self._rng.integers(len(empties)))
        cell = empties[idx]
        return int(cell[0]), int(cell[1])
    def _is_winning_move(self, x: int, y: int, symbol: int) -> bool:
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):#each direction: horizontal, vertical, diagonal
            total = 1
            for step in range(1, WIN_LENGTH):
                nx = x + dx * step
                ny = y + dy * step
                if not self._is_valid_coord(nx, ny) or self.grid[nx, ny] != symbol:
                    break
                total += 1

            for step in range(1, WIN_LENGTH):
                nx = x - dx * step
                ny = y - dy * step
                if not self._is_valid_coord(nx, ny) or self.grid[nx, ny] != symbol:
                    break
                total += 1

            if total >= WIN_LENGTH:
                return True
        return False

    def _get_max_line_length(self, x: int, y: int, symbol: int) -> int:
        max_length = 0
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            current_length = 1
            # Check forward
            for step in range(1, WIN_LENGTH):
                nx = x + dx * step
                ny = y + dy * step
                if not self._is_valid_coord(nx, ny) or self.grid[nx, ny] != symbol:
                    break
                current_length += 1
            # Check backward
            for step in range(1, WIN_LENGTH):
                nx = x - dx * step
                ny = y - dy * step
                if not self._is_valid_coord(nx, ny) or self.grid[nx, ny] != symbol:
                    break
                current_length += 1
            max_length = max(max_length, current_length)
        return max_length

    def _other_agent(self, agent: str) -> str:
        return "player_x" if agent == "player_o" else "player_o"

    def _board_to_obs(self):
        flat = [cell for row in self.grid for cell in row]
        return np.array(flat, dtype=np.int64)
