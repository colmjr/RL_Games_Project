from typing import Dict, Tuple
import os
import gradio as gr
from sb3_contrib import TRPO

from BOBO_env import MOVES
from BOBO_wrapper import SingleAgentEnv

MODEL_PATH = "BOBO_TRPO_results"
MAX_STEPS = 50
HISTORY_LEN = 20
os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')
os.environ.setdefault('no_proxy', '127.0.0.1,localhost')

def _format_move(move_id: int) -> str:
    """Return a human-readable label for a move id."""
    move = MOVES[move_id]
    pretty = move["name"].replace("_", " ").title()
    return f"{pretty} ({move_id})"

MOVE_LABELS = [_format_move(mid) for mid in MOVES]
LABEL_TO_MOVE = {_format_move(mid): mid for mid in MOVES}

class HumanOpponent:
    """Callable policy that returns the human player's pending action."""

    def __init__(self, default_action: int = 1):
        self.next_action = default_action

    def set_next_action(self, action: int) -> None:
        self.next_action = int(action)

    def __call__(self, obs=None) -> int:
        return int(self.next_action)

class ModelSession:
    def __init__(self, model_path: str = (MODEL_PATH)):
        self.model = TRPO.load(model_path)
        self.human_policy = HumanOpponent()
        self.env = SingleAgentEnv(MAX_STEPS, HISTORY_LEN, self.human_policy)
        self.obs, _ = self.env.reset()
        self.done = False

    def reset(self) -> None:
        self.obs, _ = self.env.reset()
        self.done = False

    def step(self, human_move: int) -> Tuple[int, float, bool, Dict]:
        if self.done:
            self.reset()
        self.human_policy.set_next_action(human_move)
        agent_action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, reward, terminated, truncated, info = self.env.step(agent_action)
        self.done = bool(terminated or truncated)
        return int(agent_action), float(reward), self.done, info

def _format_status(session: ModelSession) -> str:
    # Create a markdown status block summarising game state.
    env = session.env.env  # Underlying CustomEnvironment instance
    max_steps = getattr(env, "maxsteps", MAX_STEPS)
    return (
        f"**Step**: {env.timestep} / {max_steps}  \n"
        f"**TRPO points**: {env.point1}  \n"
        f"**Your points**: {env.point2}"
    )

def _format_summary(agent_move: int, human_move: int, agent_reward: float, info: Dict, done: bool) -> str:
    """Compose markdown summary for the latest turn."""
    lines = [
        f"TRPO played **{_format_move(agent_move)}**.",
        f"You played **{_format_move(human_move)}**.",
        f"Rewards -> TRPO: {agent_reward:+}, You: {-agent_reward:+}.",
    ]
    winner = info.get("winner")
    if winner:
        winner_label = "TRPO agent" if winner == "player1" else "You"
        lines.append(f"Winner: **{winner_label}**")
    if done:
        lines.append("_Game finished. Select a move to auto-start a new match._")
    return "\n".join(lines)

def play_turn(choice: str, session: ModelSession) -> Tuple[str, str, ModelSession]:
    if session is None:
        session = ModelSession()
    if not choice:
        return "Pick a move to play a turn.", _format_status(session), session
    human_move = LABEL_TO_MOVE[choice]
    agent_move, agent_reward, done, info = session.step(human_move)
    summary = _format_summary(agent_move, human_move, agent_reward, info, done)
    status = _format_status(session)
    return summary, status, session

def reset_game(session: ModelSession) -> Tuple[str, str, ModelSession]:
    """Reset handler for the reset button."""
    if session is None:
        session = ModelSession()
    else:
        session.reset()
    return "New game started. Choose your move.", _format_status(session), session

def build_demo() -> gr.Blocks:
    initial_session = ModelSession()
    with gr.Blocks(title="Play BOBO vs TRPO Agent") as demo:
        gr.Markdown("# BOBO vs TRPO Agent")
        gr.Markdown(
            "Select a move each turn. The TRPO agent controls player 1. "
            "The first player to trigger a winning rule ends the round."
        )

        session_state = gr.State(initial_session)
        summary = gr.Markdown("New game started. Choose your move.")
        status = gr.Markdown(_format_status(initial_session))

        move_input = gr.Dropdown(choices=MOVE_LABELS, label="Your Move")
        play_button = gr.Button("Play Turn")
        reset_button = gr.Button("Reset Game")

        play_button.click(
            play_turn,
            inputs=[move_input, session_state],
            outputs=[summary, status, session_state],
        )
        reset_button.click(
            reset_game,
            inputs=session_state,
            outputs=[summary, status, session_state],
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)
