from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Tetromino:
    name: str
    rotations: Tuple[np.ndarray, ...]


def _trim_matrix(matrix: np.ndarray) -> np.ndarray:
    rows = np.where(matrix.any(axis=1))[0]
    cols = np.where(matrix.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=matrix.dtype)
    return matrix[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _unique_rotations(base: np.ndarray) -> Tuple[np.ndarray, ...]:
    rotations: List[np.ndarray] = []
    current = base
    for _ in range(4):
        trimmed = _trim_matrix(current)
        if not any(np.array_equal(trimmed, existing) for existing in rotations):
            rotations.append(trimmed)
        current = np.rot90(current)
    return tuple(rotations)


BASE_TETROMINO_GRIDS: Dict[str, np.ndarray] = {
    "I": np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
    "O": np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
    "T": np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
    "S": np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
    "Z": np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
    "J": np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
    "L": np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    ),
}

TETROMINOES: Tuple[Tetromino, ...] = tuple(
    Tetromino(name, _unique_rotations(grid)) for name, grid in BASE_TETROMINO_GRIDS.items()
)
TETROMINO_LOOKUP: Dict[str, Tetromino] = {tetromino.name: tetromino for tetromino in TETROMINOES}
TETROMINO_NAMES: Tuple[str, ...] = tuple(BASE_TETROMINO_GRIDS.keys())


class TetrisEnv(gym.Env[np.ndarray, int]):
    """A Gymnasium environment for single-player Tetris."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        board_height: int = 20,
        board_width: int = 10,
        reward_mode: str = "score",
        sigmoid_scale: float = 0.02,
        soft_drop_reward: float = 1.0,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        if reward_mode not in {"score", "sigmoid"}:
            raise ValueError(f"Unknown reward_mode '{reward_mode}'. Expected 'score' or 'sigmoid'.")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if board_height <= 0 or board_width <= 0:
            raise ValueError("Board dimensions must be positive integers.")
        if sigmoid_scale <= 0:
            raise ValueError("sigmoid_scale must be positive.")

        self.board_height = board_height
        self.board_width = board_width
        self.reward_mode = reward_mode
        self.sigmoid_scale = float(sigmoid_scale)
        self.soft_drop_reward = float(soft_drop_reward)
        self.max_steps = max_steps
        self._render_mode = render_mode

        self.action_space = spaces.Discrete(3)  # 0: left, 1: right, 2: soft drop
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(self.board_height, self.board_width),
            dtype=np.int8,
        )

        self._line_clear_reward = {0: 0.0, 1: 100.0, 2: 300.0, 3: 500.0, 4: 800.0}

        self._board: np.ndarray = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._bag: List[str] = []
        self._rng: Optional[np.random.Generator] = None
        self._current_piece: Optional[Tetromino] = None
        self._rotation_index: int = 0
        self._current_row: int = 0
        self._current_col: int = 0
        self._score: float = 0.0
        self._reward_metric: float = 0.0
        self._lines_cleared: int = 0
        self._steps: int = 0
        self._terminated: bool = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._rng = np.random.default_rng(self.np_random.integers(1 << 32))
        self._board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self._bag.clear()
        self._current_piece = None
        self._rotation_index = 0
        self._current_row = 0
        self._current_col = 0
        self._score = 0.0
        self._lines_cleared = 0
        self._steps = 0
        self._terminated = False

        self._spawn_new_piece()
        self._reward_metric = self._current_reward_metric()

        observation = self._get_observation()
        info = {"score": self._score, "lines_cleared": self._lines_cleared}
        return observation, info

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("step() called after the episode terminated. Call reset().")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} outside of action space {self.action_space}")

        self._steps += 1
        metric_before = self._reward_metric
        soft_drop_cells = 0
        lines_cleared = 0

        # Horizontal movement
        if action == 0:
            self._try_shift(-1)
        elif action == 1:
            self._try_shift(1)

        # Voluntary soft drop
        if action == 2 and self._current_piece is not None:
            if self._can_place(self._current_matrix, self._current_row + 1, self._current_col):
                self._current_row += 1
                soft_drop_cells = 1
                self._score += self.soft_drop_reward

        # Gravity
        locked = False
        if self._current_piece is not None:
            if self._can_place(self._current_matrix, self._current_row + 1, self._current_col):
                self._current_row += 1
            else:
                self._lock_piece()
                locked = True

        if locked:
            lines_cleared = self._clear_complete_lines()
            self._score += self._line_clear_reward.get(lines_cleared, 0.0)
            self._lines_cleared += lines_cleared
            self._spawn_new_piece()

        current_metric = self._current_reward_metric()
        reward = current_metric - metric_before
        self._reward_metric = current_metric

        truncated = self.max_steps is not None and self._steps >= self.max_steps
        terminated = self._terminated

        observation = self._get_observation()
        info = {
            "score": self._score,
            "lines_cleared": self._lines_cleared,
            "last_lines_cleared": lines_cleared,
            "soft_drop_cells": soft_drop_cells,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self._render_mode != "rgb_array":
            raise ValueError("This environment supports only render_mode='rgb_array'.")

        grid = self._get_observation()
        rgb = np.zeros((self.board_height, self.board_width, 3), dtype=np.uint8)
        rgb[:, :] = (28, 28, 28)
        rgb[grid == 1] = (80, 200, 255)
        rgb[grid == 2] = (255, 120, 70)
        scale = 16
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        return rgb

    def close(self):
        pass

    # Helper methods

    def _spawn_new_piece(self) -> None:
        if self._rng is None:
            raise RuntimeError("Random generator not initialised. Call reset() first.")
        if not self._bag:
            bag = list(TETROMINO_NAMES)
            self._rng.shuffle(bag)
            self._bag.extend(bag)

        name = self._bag.pop()
        piece = TETROMINO_LOOKUP[name]
        self._current_piece = piece
        self._rotation_index = 0
        matrix = self._current_matrix
        self._current_row = -matrix.shape[0]
        self._current_col = (self.board_width - matrix.shape[1]) // 2

        if not self._can_place(matrix, self._current_row, self._current_col):
            self._terminated = True
            self._current_piece = None

    def _try_shift(self, delta_col: int) -> None:
        if self._current_piece is None:
            return
        new_col = self._current_col + delta_col
        if self._can_place(self._current_matrix, self._current_row, new_col):
            self._current_col = new_col

    def _lock_piece(self) -> None:
        if self._current_piece is None:
            return
        matrix = self._current_matrix
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if matrix[r, c]:
                    board_row = self._current_row + r
                    board_col = self._current_col + c
                    if 0 <= board_row < self.board_height:
                        self._board[board_row, board_col] = 1
        self._current_piece = None

    def _clear_complete_lines(self) -> int:
        full_rows = np.where(np.all(self._board == 1, axis=1))[0]
        if full_rows.size == 0:
            return 0
        self._board = np.delete(self._board, full_rows, axis=0)
        new_rows = np.zeros((full_rows.size, self.board_width), dtype=np.int8)
        self._board = np.vstack([new_rows, self._board])
        return int(full_rows.size)

    def _get_observation(self) -> np.ndarray:
        board = self._board.copy()
        if self._current_piece is not None and not self._terminated:
            matrix = self._current_matrix
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    if matrix[r, c]:
                        board_row = self._current_row + r
                        board_col = self._current_col + c
                        if 0 <= board_row < self.board_height and 0 <= board_col < self.board_width:
                            board[board_row, board_col] = 2
        return board

    def _current_reward_metric(self) -> float:
        if self.reward_mode == "sigmoid":
            return 1.0 / (1.0 + math.exp(-self.sigmoid_scale * self._score))
        return self._score

    def _can_place(self, matrix: np.ndarray, row: int, col: int) -> bool:
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                if matrix[r, c]:
                    board_row = row + r
                    board_col = col + c
                    if board_col < 0 or board_col >= self.board_width:
                        return False
                    if board_row >= self.board_height:
                        return False
                    if board_row >= 0 and self._board[board_row, board_col]:
                        return False
        return True

    @property
    def _current_matrix(self) -> np.ndarray:
        if self._current_piece is None:
            raise RuntimeError("Current piece is not set.")
        return self._current_piece.rotations[self._rotation_index]


def make_env(**kwargs) -> TetrisEnv:
    """Gymnasium-style helper for constructing the Tetris environment."""

    return TetrisEnv(**kwargs)

import numpy as np

env = make_env()
obs, info = env.reset(seed=0)
print('obs shape:', obs.shape)
print('info:', info)

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'step {step}: action={action}, reward={reward:.3f}, terminated={terminated}, truncated={truncated}')
    if terminated or truncated:
        break

env.close()
