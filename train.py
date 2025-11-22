#!/usr/bin/env python
"""
train.py – train a separate DQN for every available board layout
           ("classic", "empty", "spiral", "spiral_harder").

Saved weight files:
    pacman_dqn_<layout>.pt
"""

from __future__ import annotations
import argparse, torch, torch.optim as optim
from pathlib import Path
from pacman_env import PacmanEnv
import random
from dqn_agent import get_arch, ReplayMemory, select_action, optimise, DEVICE

# ───────── hyper‑parameters ─────────
NUM_EPISODES      = 1000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 200
BATCH_SIZE        = 128
MEMORY_CAP        = 20_000
GAMMA             = 0.99
LR                = 1e-3
EPS               = (1.0, 0.05, 8_000)   # ε‑greedy schedule (start, end, decay)

# ───────── single‑layout trainer ─────────
def train_mixed(layouts: list[str], episodes: int, model_name: str, arch: str) -> Path:
    # Create a dummy env just to get observation shape and action count
    tmp_env = PacmanEnv(layouts[0])
    obs_shape = tmp_env.observation_space.shape
    n_actions = tmp_env.action_space.n
    tmp_env.close()
    
    print(f"Initializing Multi-Task Training on: {layouts}")
    print(f"Architecture: {arch}")
    print(f"Using device: {DEVICE}")
    
    # Use factory to create networks
    policy = get_arch(arch, obs_shape, n_actions).to(DEVICE)
    target = get_arch(arch, obs_shape, n_actions).to(DEVICE)
    target.load_state_dict(policy.state_dict())
    
    print("Created policy and target networks")
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory    = ReplayMemory(MEMORY_CAP)
    print("Created optimizer and memory")
    
    step = 0
    
    for ep in range(1, episodes + 1):
        # 1. Randomly select a layout for this episode
        current_layout = random.choice(layouts)
        env = PacmanEnv(current_layout)
        
        state, _ = env.reset()
        done, ep_reward = False, 0.0
        
        print(f"[Ep {ep}/{episodes}] Layout: {current_layout}") # Optional: noisy
        
        while not done:
            # Capture state for reward shaping
            prev_pos = env.pac_pos
            # Find distance to closest pellet
            prev_min_dist = 0
            if env.pellets:
                prev_min_dist = min(abs(p[0] - prev_pos[0]) + abs(p[1] - prev_pos[1]) for p in env.pellets)

            action = select_action(state, policy, step, *EPS)
            step += 1

            next_state, reward, done, _, _ = env.step(action)

            # REWARD SHAPING: Encourage moving closer to pellets
            if not done and env.pellets:
                curr_pos = env.pac_pos
                curr_min_dist = min(abs(p[0] - curr_pos[0]) + abs(p[1] - curr_pos[1]) for p in env.pellets)
                
                if curr_min_dist < prev_min_dist:
                    reward += 0.3  # Bonus for moving closer
                elif curr_min_dist >= prev_min_dist:
                    reward -= 0.1  # Penalty for moving away

            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())
        
        env.close()

        if ep % 100 == 0 or ep == episodes:
            print(f"[Ep {ep:4d}] Last Layout: {current_layout:15s} | reward = {ep_reward:6.1f}")

    # Save the final "Generalist" model
    if len(layouts) > 1:
        weight_path = Path(f"pacman_dqn_mixed_{model_name}.pt")
    else:
        weight_path = Path(f"pacman_dqn_{layouts[0]}_{model_name}.pt")
    
    checkpoint = {
        'arch': arch,
        'state_dict': policy.state_dict()
    }
    torch.save(checkpoint, weight_path)
    print(f"Training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on all Pac‑Man layouts")
    parser.add_argument(
        "--layout", type=str, default=None,
        choices=["classic", "empty", "spiral", "spiral_harder", "mixed"],
        help="Specific layout to train on, or 'mixed' for all (default: mixed)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="quick 200-episode run per layout instead of full 4000"
    )
    parser.add_argument(
        "--name", type=str, default="default",
        help="A name for the model version, used for the saved file"
    )
    parser.add_argument(
        "--arch", type=str, default="original", choices=["original", "deep_v1"],
        help="Network architecture to use (default: original)"
    )
    args = parser.parse_args()
    episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES

    # Define the list of layouts to train on
    if args.layout == "mixed" or args.layout is None:
        # Train on ALL layouts randomly
        layouts_to_train = ["classic", "empty", "spiral", "spiral_harder"]
        train_mixed(layouts_to_train, episodes, args.name, args.arch)
    else:
        # Train on a single specific layout (legacy mode)
        # We can re-use the mixed trainer with a list of length 1
        train_mixed([args.layout], episodes, args.name, args.arch)
