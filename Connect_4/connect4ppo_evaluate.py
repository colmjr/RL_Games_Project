"""Evaluate a trained PPO model on the connect 4 environment and print the win rate."""
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from Connect_4.connect4ppo_training import make_env


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


def _resolve_model_path(model_path):
    """Resolve model_path to an actual .zip file path. Raise FileNotFoundError with hints if not found."""
    p = Path(model_path)
    # if exact file provided
    if p.exists() and p.is_file():
        return str(p)
    # try adding .zip
    p_zip = p.with_suffix(p.suffix + ".zip") if p.suffix == "" else p.with_suffix(".zip")
    if p_zip.exists():
        return str(p_zip)
    # if path is a directory, search for .zip files (prefer common names)
    if p.exists() and p.is_dir():
        candidates = list(p.glob("*.zip"))
        if not candidates:
            raise FileNotFoundError(f"No .zip model found in directory: {p}")
        # prefer 'best' or 'policy' in filename
        for name in ("best", "policy", "final", "model"):
            for c in candidates:
                if name in c.stem.lower():
                    return str(c)
        return str(candidates[0])
    # try model_path + ".zip"
    p2 = Path(str(model_path) + ".zip")
    if p2.exists():
        return str(p2)
    # fallback: list nearby files for helpful error
    raise FileNotFoundError(f"Model file not found: '{model_path}'. Tried: '{p}', '{p2}'. If you passed a directory, make sure it contains a .zip SB3 model.")


def evaluate(model_path="BOBOppo_results", episodes=100):
    """Evaluate the trained PPO model and print the win rate."""
    try:
        load_path = _resolve_model_path(model_path)
    except FileNotFoundError:
        # Fallback: search the project folder and current working dir for any .zip model
        project_root = Path(__file__).resolve().parents[1]  # RL_Games_Project
        candidates = list(project_root.rglob("*.zip"))
        chosen = None
        if not candidates:
            candidates = list(Path.cwd().rglob("*.zip"))
        if candidates:
            # prefer filenames containing common keywords
            for name in ("best", "policy", "final", "model"):
                for c in candidates:
                    if name in c.stem.lower():
                        chosen = c
                        break
                if chosen:
                    break
            if not chosen:
                chosen = candidates[0]
            print(f"[INFO] Model '{model_path}' not found; using discovered model: {chosen}")
            load_path = str(chosen)
        else:
            print(f"[ERROR] Model file not found: '{model_path}'. Searched {project_root} and {Path.cwd()}.")
            return

    model = PPO.load(load_path)
    env = make_env()
    wins = 0
    for ep in range(episodes):
        # support both reset() -> obs or (obs, info)
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs, info = reset_ret
        else:
            obs, info = reset_ret, {}
        done = False
        last_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act = int_action_from_pred(action)
            step_ret = env.step(act)
            # support both 5-tuple (obs, reward, terminated, truncated, info) and 4-tuple (obs, reward, done, info)
            if isinstance(step_ret, tuple) and len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                obs, reward, done, info = step_ret
                done = bool(done)
            else:
                # unexpected signature: try mapping as dict-like
                try:
                    obs = step_ret["obs"]
                    reward = step_ret.get("reward", 0)
                    done = bool(step_ret.get("done", False))
                    info = step_ret.get("info", {})
                except Exception:
                    raise RuntimeError("Unsupported env.step() return format.")
            last_reward = reward
        if last_reward == 1:
            wins += 1
    print(f"Win rate: {wins}/{episodes} = {wins / episodes:.2f}")


if __name__ == "__main__":
    """Run the evaluation of the trained PPO model."""
    evaluate()
