"""Custom two-player environment for the game of connect 4 using PettingZoo."""
from copy import copy
from gymnasium.spaces import Discrete,MultiDiscrete
from pettingzoo import ParallelEnv
import numpy as np

ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER_O = 1
PLAYER_X = 2

def build_board():
    return [[EMPTY for _ in range(COLUMNS)] for _ in range(ROWS)]

def drop_token(board, col, token):#drops token
    for row in range(ROWS - 1, -1, -1):
        if board[row][col] == EMPTY:
            board[row][col] = token
            return row
    raise ValueError("Column is full")

def count_aligned(board, row, col, token, direction):
    total = 0
    step_row, step_col = direction
    r, c = row + step_row, col + step_col
    while 0 <= r < ROWS and 0 <= c < COLUMNS and board[r][c] == token:
        total += 1
        r += step_row
        c += step_col
    return total

def has_winner(board, row, col, token):
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for direction in directions:
        span = 1 + count_aligned(board, row, col, token, direction)
        span += count_aligned(board, row, col, token, (-direction[0], -direction[1]))
        if span >= 4:
            return True
    return False
def column_is_full(board, col):
    return board[0][col] != EMPTY

def board_is_full(board):
    return all(column_is_full(board, col) for col in range(COLUMNS))


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, maxsteps):
        self.maxsteps = maxsteps
        self.possible_agents = ["player_o", "player_x"]
        self.action_spaces = {
            agent: Discrete(COLUMNS) for agent in self.possible_agents
        }
        nvec = np.full(ROWS * COLUMNS, 3, dtype=np.int64)
        self.observation_spaces = {
            agent: MultiDiscrete(nvec, dtype=np.int64)
            for agent in self.possible_agents
        }
        self._reset_internal_state()

    def _reset_internal_state(self):
        self.grid = build_board()
        self.timestep = 0
        self.agents = copy(self.possible_agents)
        self.last_moves = {agent: None for agent in self.possible_agents}

    @staticmethod
    def _opponent(agent):
        return "player_x" if agent == "player_o" else "player_o"

    def _board_to_obs(self):
        flat = [cell for row in self.grid for cell in row]
        return np.array(flat, dtype=np.int64)

    def reset(self, seed=None, options=None):
        self._reset_internal_state()
        observations = {agent: self._board_to_obs() for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        winner = None

        for agent, token in (("player_o", PLAYER_O), ("player_x", PLAYER_X)):
            if agent not in actions:
                raise KeyError(f"Missing action for {agent}")

            col = int(actions[agent])
            opponent = self._opponent(agent)

            if col < 0 or col >= COLUMNS:
                rewards[agent] = -1.0
                rewards[opponent] = 1.0
                terminations = {a: True for a in self.agents}
                infos[agent]["invalid_action"] = "out_of_bounds"
                break

            if column_is_full(self.grid, col):
                rewards[agent] = -1.0
                rewards[opponent] = 1.0
                terminations = {a: True for a in self.agents}
                infos[agent]["invalid_action"] = "column_full"
                break

            row = drop_token(self.grid, col, token)
            self.last_moves[agent] = (row, col)

            if has_winner(self.grid, row, col, token):
                winner = agent
                rewards[agent] = 1.0
                rewards[opponent] = -1.0
                terminations = {a: True for a in self.agents}
                break
        else:
            self.timestep += 1
            if board_is_full(self.grid) or self.timestep >= self.maxsteps:
                truncations = {agent: True for agent in self.agents}

        if winner is not None:
            for agent in self.agents:
                infos[agent]["winner"] = agent == winner

        observations = {agent: self._board_to_obs() for agent in self.agents}
        for agent in self.agents:
            if self.last_moves[agent] is not None:
                infos[agent]["last_move"] = self.last_moves[agent]

        return observations, rewards, terminations, truncations, infos

    def render(self):
        for row in self.grid:
            print(" ".join(str(cell) for cell in row))

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
