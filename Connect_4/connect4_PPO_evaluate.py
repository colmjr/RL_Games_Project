"""Evaluate a trained PPO model on the connect 4 environment and print the win rate."""
import numpy as np
from stable_baselines3 import PPO
from connect4_PPO_training import make_env


def int_action_from_pred(action):
    """Convert the model's action prediction to an integer action."""
    if isinstance(action, np.ndarray):  # Check if action is in a numpy array
        try:
            return int(action.item())  # Try to convert using item()
        except Exception:
            return int(action.flatten()[0]) # Flatten and take the first element
    elif isinstance(action, (list, tuple)):  # Check if action is a list or tuple
        return int(action[0])
    else:
        return int(action)


def evaluate(episodes, model_path):
    """Evaluate the trained PPO model and print the win rate."""
    model = PPO.load(model_path)  # Load the trained model
    env = make_env()  # Create the connect 4 environment
    wins = 0  # Initialize win counter
    for ep in range(episodes):
        # Reset the environment for a new episode
        obs, info = env.reset()
        done = False
        last_reward = 0.0
        while not done: # Loop until the episode is done
            action, _ = model.predict(obs, deterministic=True) # Get action from the model
            act = int_action_from_pred(action) # Convert action to integer
            obs, reward, terminated, truncated, info = env.step(act) # Take a step in the environment
            done = bool(terminated or truncated) # Update done flag
            last_reward = reward # Update last reward
        if last_reward == 1:
            wins += 1
    print(f"Win rate: {wins}/{episodes} = {wins / episodes:.2f}")


if __name__ == "__main__":
    """Run the evaluation of the trained PPO model."""
    evaluate(100, "connect4_PPO_results")
