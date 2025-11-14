"""
FastAPI service exposing the Gomoku PPO agent so that a JS client can
send human moves and receive the agent's response plus the updated board.
"""
from __future__ import annotations

from typing import Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from stable_baselines3 import PPO

from Gomoku_env import GRID_HEIGHT, GRID_WIDTH
from Gomoku_wrapper import SingleAgentEnv

MODEL_PATH = "Gomoku_PPO_results"
MAX_STEPS = GRID_HEIGHT * GRID_WIDTH


def decode_action(action: int) -> Tuple[int, int]:
    """Convert a flattened action index back to (row, col)."""
    row = int(action) // GRID_WIDTH
    col = int(action) % GRID_WIDTH
    return row, col


class HumanPolicy:
    """Simple policy object that always returns the most recent human move."""

    def __init__(self) -> None:
        self._next_move: Tuple[int, int] = (0, 0)

    def set_move(self, move: Tuple[int, int]) -> None:
        self._next_move = (int(move[0]), int(move[1]))

    def __call__(self, *_args, **_kwargs) -> Tuple[int, int]:
        return self._next_move


class MoveRequest(BaseModel):
    row: int
    col: int
    reset: bool = False


class MoveResponse(BaseModel):
    agent_action: int
    agent_row: int
    agent_col: int
    reward: float
    done: bool
    board: Tuple[float, ...]
    info: Dict


class StateResponse(BaseModel):
    board: Tuple[float, ...]
    done: bool
    step: int


def _serialize_info(info: Dict) -> Dict:
    """Ensure tuples in info dict become JSON-friendly lists."""
    serialized: Dict = {}
    for key, value in info.items():
        if isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


class GomokuService:
    """Manages the PPO model, environment, and current observation."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self.human_policy = HumanPolicy()
        self.env = SingleAgentEnv(MAX_STEPS, self.human_policy)
        self.model = PPO.load(model_path)
        self.obs, _ = self.env.reset()
        self.done = False

    def reset(self) -> None:
        self.obs, _ = self.env.reset()
        self.done = False

    def play_move(self, row: int, col: int) -> MoveResponse:
        if self.done:
            self.reset()
        self.human_policy.set_move((row, col))
        action, _ = self.model.predict(self.obs, deterministic=True)
        agent_move = int(action)
        self.obs, reward, terminated, truncated, info = self.env.step(agent_move)
        self.done = bool(terminated or truncated)
        agent_row, agent_col = decode_action(agent_move)
        return MoveResponse(
            agent_action=agent_move,
            agent_row=agent_row,
            agent_col=agent_col,
            reward=float(reward),
            done=self.done,
            board=tuple(float(x) for x in self.obs),
            info=_serialize_info(info),
        )

    def get_state(self) -> StateResponse:
        env = self.env.env  # Underlying CustomEnvironment
        step = getattr(env, "timestep", 0)
        return StateResponse(board=tuple(float(x) for x in self.obs), done=self.done, step=int(step))


service = GomokuService()

app = FastAPI(title="Gomoku PPO Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/move", response_model=MoveResponse)
def play_move(req: MoveRequest) -> MoveResponse:
    if req.reset:
        service.reset()
    return service.play_move(req.row, req.col)


@app.post("/reset")
def reset_game() -> StateResponse:
    service.reset()
    return service.get_state()


@app.get("/state", response_model=StateResponse)
def get_state() -> StateResponse:
    return service.get_state()
