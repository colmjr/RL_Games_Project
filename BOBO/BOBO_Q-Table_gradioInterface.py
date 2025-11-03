import os
import numpy as np
import gradio as gr

# reuse helpers and constants from the PPO gradio file to avoid duplication
from BOBO.BOBO_PPO_gradioInterface import HumanOpponent, MOVE_LABELS, LABEL_TO_MOVE, _format_move, MAX_STEPS
from BOBO.BOBO_wrapper import SingleAgentEnv

class QTableSession:
    """Session wrapper that uses a saved Q-table (q_table.npy) as the agent policy."""
    def __init__(self, qpath: str = "q_table.npy"):
        qpath = os.path.expanduser(qpath)
        self.q_table = np.load(qpath)
        # bins inferred from q_table shape (all dims except last are bins)
        self.bins = list(self.q_table.shape[:-1])
        # history length inferred as bins length minus two (points dims)
        self.history_len = max(0, len(self.bins) - 2)
        self.human_policy = HumanOpponent()
        self.env = SingleAgentEnv(MAX_STEPS, self.history_len, self.human_policy)
        self.obs, _ = self.env.reset()
        self.done = False

    def _obs_to_idx(self, obs):
        idx = []
        for i, val in enumerate(obs):
            b = self.bins[i]
            raw = int(np.floor(float(val) * (b - 1) + 1e-8))
            raw = max(0, min(b - 1, raw))
            idx.append(raw)
        return tuple(idx)

    def reset(self):
        self.obs, _ = self.env.reset()
        self.done = False

    def step(self, human_move: int):
        if self.done:
            self.reset()
        self.human_policy.set_next_action(human_move)
        s_idx = self._obs_to_idx(self.obs)
        qvals = self.q_table[s_idx]
        best = np.flatnonzero(qvals == qvals.max())
        agent_action = int(np.random.choice(best))
        self.obs, reward, terminated, truncated, info = self.env.step(agent_action)
        self.done = bool(terminated or truncated)
        return int(agent_action), float(reward), self.done, info

def _format_status_for_session(session: QTableSession) -> str:
    env = session.env.env  # underlying custom env
    max_steps = getattr(env, "maxsteps", MAX_STEPS)
    return (
        f"**Step**: {env.timestep} / {max_steps}  \n"
        f"**Agent points**: {env.point1}  \n"
        f"**Your points**: {env.point2}"
    )

def _format_summary_q(agent_move: int, human_move: int, agent_reward: float, info: dict, done: bool) -> str:
    lines = [
        f"Agent played **{_format_move(agent_move)}**.",
        f"You played **{_format_move(human_move)}**.",
        f"Rewards → Agent: {agent_reward:+}, You: {-agent_reward:+}.",
    ]
    winner = info.get("winner")
    if winner:
        winner_label = "Agent" if winner == "player1" else "You"
        lines.append(f"Winner: **{winner_label}**")
    if done:
        lines.append("_Game finished. Select a move to auto-start a new match._")
    return "\n".join(lines)

def play_turn(choice: str, qtable_path: str, session) -> tuple:
    """Handle a human move triggered from the UI (Q-table only)."""
    # Instantiate or recreate session when path changed or missing
    if session is None or getattr(session, "q_table", None) is None:
        session = QTableSession(qtable_path)
    else:
        # if user changed the qtable path, recreate
        try:
            current_path = getattr(session, "_qpath", None)
        except Exception:
            current_path = None
        # simple recreation logic: if file path differs, recreate
        if qtable_path and os.path.expanduser(qtable_path) != current_path:
            session = QTableSession(qtable_path)

    # stash original path for simple detection
    session._qpath = os.path.expanduser(qtable_path)

    if not choice:
        return "Pick a move to play a turn.", _format_status_for_session(session), session

    human_move = LABEL_TO_MOVE[choice]
    agent_move, agent_reward, done, info = session.step(human_move)
    summary = _format_summary_q(agent_move, human_move, agent_reward, info, done)
    status = _format_status_for_session(session)
    return summary, status, session

def reset_game(qtable_path: str, session) -> tuple:
    if session is None:
        session = QTableSession(qtable_path)
    else:
        session.reset()
    return "New game started. Choose your move.", _format_status_for_session(session), session

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Play BOBO vs Q-Table Agent") as demo:
        gr.Markdown("# BOBO vs Q-Table Agent")
        gr.Markdown("Load a q_table.npy file and play against the learned Q-table policy.")

        session_state = gr.State(None)
        summary = gr.Markdown("New game started. Choose your move.")
        status = gr.Markdown("")

        qtable_input = gr.Textbox(value="q_table.npy", label="Q-table path (q_table.npy)")
        move_input = gr.Dropdown(choices=MOVE_LABELS, label="Your Move")
        play_button = gr.Button("Play Turn")
        reset_button = gr.Button("Reset Game")

        play_button.click(
            play_turn,
            inputs=[move_input, qtable_input, session_state],
            outputs=[summary, status, session_state],
        )
        reset_button.click(
            reset_game,
            inputs=[qtable_input, session_state],
            outputs=[summary, status, session_state],
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=True)

