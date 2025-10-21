# Chrome Dino Gymnasium Environment
#Implementation of a lightweight pixel-based Chrome Dino environment using Gymnasium.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class ObstacleTemplate:
    """Static parameters for an obstacle archetype."""

    width: int
    height: int
    min_gap: int
    max_gap: int
    min_altitude: int = 0
    max_altitude: int = 0


class ChromeDinoEnv(gym.Env[np.ndarray, np.ndarray]):
    """A lightweight Gymnasium recreation of the Chrome Dino game."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        screen_width: int = 160,
        screen_height: int = 90,
        max_duration_s: float = 120.0,
        sigmoid_scale: float = 0.02,
    ) -> None:
        super().__init__()

        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.ground_y = int(screen_height * 0.78)
        self.dino_x = int(screen_width * 0.12)

        self.dt = 1.0 / self.metadata["render_fps"]
        self.gravity = -2600.0
        self.jump_velocity = 880.0
        self.base_speed = 180.0
        self.max_speed = 540.0
        self.acceleration = 1.2 #acceleration of the background or conveyer belt that obstacles move
        self.crouch_transition_scale = 0.6 #escales the dino’s hitbox when you hold crouch
        self._render_mode = render_mode
        self._bg_value = 0
        self._ground_value = 70
        self._dino_value = 255
        self._obstacle_value = 180

        self.sigmoid_scale = sigmoid_scale
        if self.sigmoid_scale <= 0:
            raise ValueError("sigmoid_scale must be positive")

        self.max_duration_s = max_duration_s

        self.action_space = spaces.MultiBinary(2)  # [jump_pressed, crouch_pressed]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 1),
            dtype=np.uint8,
        )

        self.obstacle_templates: Tuple[ObstacleTemplate, ...] = (
            ObstacleTemplate(width=14, height=32, min_gap=90, max_gap=150),
            ObstacleTemplate(width=24, height=42, min_gap=110, max_gap=180),
            ObstacleTemplate(
                width=28,
                height=24,
                min_gap=130,
                max_gap=210,
                min_altitude=12,
                max_altitude=38,
            ),
        )

        self._rng: Optional[np.random.Generator] = None
        self._obstacles: List[Dict[str, float]] = []
        self._dino_y: float = 0.0
        self._dino_vel_y: float = 0.0
        self._is_crouching: bool = False
        self._time_elapsed: float = 0.0
        self._score: float = 0.0
        self._prev_score: float = 0.0
        self._terminated: bool = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._rng = np.random.default_rng(self.np_random.integers(1 << 32))
        self._obstacles = []
        self._dino_y = float(self.ground_y)
        self._dino_vel_y = 0.0
        self._is_crouching = False
        self._time_elapsed = 0.0
        self._score = 0.0
        self._prev_score = 0.0
        self._terminated = False

        self._spawn_obstacle(initial=True)

        observation = self._render_frame()
        info = {"score": self._score, "speed": self._current_speed}
        return observation, info

    def step(self, action: np.ndarray):
        if self._terminated:
            raise RuntimeError("step() called on terminated episode. Call reset().")

        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} outside of space {self.action_space}")

        jump_pressed = bool(action[0])
        crouch_pressed = bool(action[1])

        if crouch_pressed and self._on_ground:
            self._is_crouching = True
        elif not crouch_pressed:
            self._is_crouching = False

        if jump_pressed and self._on_ground:
            self._dino_vel_y = self.jump_velocity
            self._is_crouching = False

        self._dino_vel_y += self.gravity * self.dt
        self._dino_y += self._dino_vel_y * self.dt

        if self._dino_y <= self.ground_y:
            self._dino_y = float(self.ground_y)
            self._dino_vel_y = 0.0

        self._time_elapsed += self.dt
        distance_delta = self._current_speed * self.dt
        self._score += distance_delta

        self._advance_obstacles(distance_delta)

        self._terminated = self._check_collision()

        truncated = self._time_elapsed >= self.max_duration_s

        reward = self._sigmoid(self._score) - self._sigmoid(self._prev_score)
        self._prev_score = self._score

        observation = self._render_frame()
        info = {
            "score": self._score,
            "speed": self._current_speed,
            "crouching": self._is_crouching,
            "time_elapsed": self._time_elapsed,
        }

        return observation, reward, self._terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._render_mode == "rgb_array":
            frame = self._render_frame()
            return np.repeat(frame, 3, axis=2)
        raise NotImplementedError("Only render_mode='rgb_array' is supported.")

    def close(self) -> None:
        pass

    @property
    def _on_ground(self) -> bool:
        return math.isclose(self._dino_y, float(self.ground_y), rel_tol=1e-5)

    @property
    def _current_speed(self) -> float:
        return min(
            self.max_speed,
            self.base_speed + self.acceleration * (self._time_elapsed / self.dt) ** 0.5,
        )

    def _sigmoid(self, score: float) -> float:
        return 1.0 / (1.0 + math.exp(-score * self.sigmoid_scale))

    def _spawn_obstacle(self, *, initial: bool = False) -> None:
        assert self._rng is not None
        template = self._rng.choice(self.obstacle_templates)
        gap = self._rng.uniform(template.min_gap, template.max_gap)
        altitude = self._rng.uniform(template.min_altitude, template.max_altitude)

        if initial or not self._obstacles:
            x_position = self.screen_width + gap
        else:
            last = self._obstacles[-1]
            x_position = last["x"] + last["width"] + gap

        obstacle = {
            "x": float(x_position),
            "width": float(template.width),
            "height": float(template.height),
            "altitude": float(altitude),
        }
        self._obstacles.append(obstacle)

    def _advance_obstacles(self, distance_delta: float) -> None:
        if not self._obstacles:
            self._spawn_obstacle()
            return

        for obstacle in self._obstacles:
            obstacle["x"] -= distance_delta

        while self._obstacles and self._obstacles[0]["x"] + self._obstacles[0]["width"] < 0:
            self._obstacles.pop(0)

        if self._obstacles:
            last = self._obstacles[-1]
            if last["x"] + last["width"] < self.screen_width:
                self._spawn_obstacle()
        else:
            self._spawn_obstacle()

    def _check_collision(self) -> bool:
        if not self._obstacles:
            return False

        dino_height = self._crouched_height if self._is_crouching else self._running_height
        dino_width = self._crouched_width if self._is_crouching else self._running_width
        dino_top = self._dino_y - dino_height
        dino_left = self.dino_x
        dino_right = dino_left + dino_width
        dino_bottom = self._dino_y

        for obstacle in self._obstacles:
            obs_left = obstacle["x"]
            obs_right = obs_left + obstacle["width"]
            obs_bottom = self.ground_y - obstacle["altitude"]
            obs_top = obs_bottom - obstacle["height"]

            horizontally_overlapping = (dino_left < obs_right) and (dino_right > obs_left)
            vertically_overlapping = (dino_bottom > obs_top) and (dino_top < obs_bottom)

            if horizontally_overlapping and vertically_overlapping:
                return True
        return False

    @property
    def _running_height(self) -> float:
        return float(self.screen_height * 0.22)

    @property
    def _running_width(self) -> float:
        return float(self.screen_width * 0.075)

    @property
    def _crouched_height(self) -> float:
        return self._running_height * self.crouch_transition_scale

    @property
    def _crouched_width(self) -> float:
        return self._running_width * 1.3

    def _render_frame(self) -> np.ndarray:
        canvas = np.full((self.screen_height, self.screen_width), self._bg_value, dtype=np.uint8)

        canvas[self.ground_y : self.ground_y + 2, :] = self._ground_value

        for obstacle in self._obstacles:
            left = int(round(obstacle["x"]))
            right = int(round(obstacle["x"] + obstacle["width"]))
            bottom = int(round(self.ground_y - obstacle["altitude"]))
            top = max(0, bottom - int(round(obstacle["height"])) )
            if right <= 0 or left >= self.screen_width:
                continue
            left = max(0, left)
            right = min(self.screen_width, right)
            top = max(0, top)
            bottom = min(self.screen_height, bottom)
            canvas[top:bottom, left:right] = self._obstacle_value

        height = int(round(self._crouched_height if self._is_crouching else self._running_height))
        width = int(round(self._crouched_width if self._is_crouching else self._running_width))
        bottom = int(round(self._dino_y))
        top = max(0, bottom - height)
        left = self.dino_x
        right = min(self.screen_width, left + width)
        canvas[top:bottom, left:right] = self._dino_value

        return canvas[:, :, None]


def make_env(**kwargs) -> ChromeDinoEnv:
    """Convenience factory mirroring Gymnasium's registration pattern."""

    return ChromeDinoEnv(**kwargs)
