import cv2
from pacman_env import PacmanEnv
import numpy as np

def save_obs(layout):
    env = PacmanEnv(layout)
    obs, _ = env.reset()
    
    print(f"Layout: {layout}")
    print(f"Obs Shape: {obs.shape}")
    print(f"Min Val: {obs.min()}, Max Val: {obs.max()}")
    
    # Check if it's all zeros
    if np.all(obs == 0):
        print("WARNING: Observation is all black!")
    
    # Save to file
    filename = f"debug_obs_{layout}.png"
    # Convert RGB to BGR for OpenCV
    bgr_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, bgr_obs)
    print(f"Saved to {filename}\n")

if __name__ == "__main__":
    save_obs("empty")
    save_obs("classic")
