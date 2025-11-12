# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO
from Gomoku.Gomoku_wrapper import SingleAgentEnv
from Gomoku.Gomoku_env import GRID_HEIGHT, GRID_WIDTH

class HumanPolicy:
    def __init__(self): self._next = (0, 0)
    def set_move(self, move): self._next = move
    def __call__(self, *_): return self._next

class Move(BaseModel):
    row: int; col: int; reset: bool = False

app = FastAPI()
human = HumanPolicy()
env = SingleAgentEnv(GRID_HEIGHT * GRID_WIDTH, human)
model = PPO.load("Gomoku_PPO_results")
obs, _ = env.reset(); done = False

@app.post("/gomoku/move")
def play(move: Move):
    global obs, done
    if move.reset or done:
        obs, _ = env.reset(); done = False
    human.set_move((move.row, move.col))
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = bool(terminated or truncated)
    return {
        "agentMove": int(action),          # encode as single index
        "board": obs.tolist(),             # already normalized in SingleAgentEnv._get_obs
        "reward": reward,
        "done": done,
        "info": info,
    }
