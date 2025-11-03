import os
import numpy as np
from BOBO.BOBO_wrapper import SingleAgentEnv, RandomOpponent

class QLearningAgent:
    def __init__(self, env, bins, alpha=0.1, gamma=0.99, epsilon=1.0, min_epsilon=0.05, eps_decay=0.9995):
        self.env = env
        self.bins = bins  # list of ints, one per observation dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_decay = eps_decay

        # Build Q-table: shape = bins... x action_space.n
        self.q_shape = tuple(self.bins) + (self.env.action_space.n,)
        self.q_table = np.zeros(self.q_shape, dtype=np.float32)

    def _obs_to_idx(self, obs):
        """Discretize observation into integer indices for each dim."""
        idx = []
        for i, val in enumerate(obs):
            b = self.bins[i]
            # obs values normalized in [0,1] (per wrapper). Map to 0..(b-1)
            # Use floor with clipping for stability.
            raw = int(np.floor(float(val) * (b - 1) + 1e-8))
            raw = max(0, min(b - 1, raw))
            idx.append(raw)
        return tuple(idx)

    def select_action(self, obs_idx):
        """Epsilon-greedy selection from Q-table using discretized index tuple."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        qvals = self.q_table[obs_idx]
        # choose argmax (break ties randomly)
        best = np.flatnonzero(qvals == qvals.max())
        return int(np.random.choice(best))

    def update(self, s_idx, a, r, s2_idx, done):
        qsa = self.q_table[s_idx + (a,)]
        next_max = 0.0 if done else float(self.q_table[s2_idx].max())
        self.q_table[s_idx + (a,)] = qsa + self.alpha * (r + self.gamma * next_max - qsa)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_decay)


def train(
    max_episodes=20000,
    max_steps=50,
    history_len=4,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    min_epsilon=0.05,
    eps_decay=0.9995,
    save_path=None,
    eval_every=2000,
    eval_episodes=200
):
    opponent = RandomOpponent()
    env = SingleAgentEnv(max_steps, history_len, opponent)

    # bins: 21 for own points, 21 for opponent points, 9 bins for each history slot (moves 0-8)
    bins = [21, 21] + [9] * history_len

    agent = QLearningAgent(env, bins, alpha=alpha, gamma=gamma, epsilon=epsilon, min_epsilon=min_epsilon, eps_decay=eps_decay)

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        s_idx = agent._obs_to_idx(obs)
        done = False
        total_r = 0.0
        for step in range(max_steps + 5):
            a = agent.select_action(s_idx)
            obs2, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            s2_idx = agent._obs_to_idx(obs2)
            agent.update(s_idx, a, r, s2_idx, done)
            s_idx = s2_idx
            total_r += r
            if done:
                break
        agent.decay_epsilon()
        if ep % 500 == 0:
            print(f"Episode {ep}/{max_episodes}  total_reward={total_r:.3f}  epsilon={agent.epsilon:.4f}")
        if save_path and (ep % 2000 == 0 or ep == max_episodes):
            np.save(save_path, agent.q_table)

        if eval_every and (ep % eval_every == 0):
            avg = evaluate_policy(agent, env, eval_episodes)
            print(f"Eval at ep {ep}: avg_reward={avg:.3f}")

    if save_path:
        np.save(save_path, agent.q_table)
    return agent


def evaluate_policy(agent, env, episodes=200):
    """Run deterministic greedy policy (epsilon=0) for evaluation (no learning)."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        s_idx = agent._obs_to_idx(obs)
        ep_r = 0.0
        for _ in range(env.maxsteps + 5):
            a = agent.select_action(s_idx)
            obs2, r, terminated, truncated, info = env.step(a)
            ep_r += r
            s_idx = agent._obs_to_idx(obs2)
            if terminated or truncated:
                break
        total += ep_r
    agent.epsilon = old_epsilon
    return total / episodes


if __name__ == "__main__":
    # small default run; adjust parameters as needed
    out_file = os.path.join(os.path.dirname(__file__), "q_table.npy")
    agent = train(
        max_episodes=1000000,
        max_steps=50,
        history_len=4,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        min_epsilon=0.05,
        eps_decay=0.9995,
        save_path=out_file,
        eval_every=2000,
        eval_episodes=200
    )
    print("Training finished. Q-table saved to:", out_file)

