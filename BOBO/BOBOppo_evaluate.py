"""Evaluate a trained PPO model on the BOBO environment and print the win rate."""
import numpy as np
from stable_baselines3 import PPO
from BOBO.BOBOppo_training import make_env

def int_action_from_pred(action):
    """Convert the model's action prediction to an integer action."""
    if isinstance(action, np.ndarray): # Check if action is in a numpy array
        try:
            return int(action.item()) # Try to convert using item()
        except Exception: # Fallback to flattening
            return int(action.flatten()[0])
    elif isinstance(action, (list, tuple)): # Check if action is a list or tuple
        return int(action[0])
    else:
        return int(action)


def evaluate(model_path="BOBOppo_results", episodes=100):
    """Evaluate the trained PPO model and print the win rate."""
    model = PPO.load(model_path)
    env = make_env()
    wins = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        last_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act = int_action_from_pred(action)
            obs, reward, terminated, truncated, info = env.step(act)
            done = bool(terminated or truncated)
            last_reward = reward
        if last_reward == 1:
            wins += 1
    print(f"Win rate: {wins}/{episodes} = {wins / episodes:.2f}")


if __name__ == "__main__":
    """Run the evaluation of the trained PPO model."""
    evaluate()
