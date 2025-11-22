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
def train_layout(layout: str, episodes: int, model_name: str, arch: str) -> Path:
    env = PacmanEnv(layout)
    obs_shape = env.observation_space.shape        # (H, W, C)
    n_actions = env.action_space.n
    print(f"Created environment: {layout} (arch={arch})")
    
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
        state, _ = env.reset()
        done, ep_reward = False, 0.0
        print(f"[{layout}, episode {ep}].")
        while not done:
            action = select_action(state, policy, step, *EPS)
            step += 1

            next_state, reward, done, _, _ = env.step(action)
            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        if ep % 100 == 0 or ep == episodes:
            print(f"[{layout}] Episode {ep:4d} | reward = {ep_reward:6.1f}")

    env.close()
    weight_path = Path(f"pacman_dqn_{layout}_{model_name}.pt")
    
    # Save dictionary with metadata
    checkpoint = {
        'arch': arch,
        'state_dict': policy.state_dict()
    }
    torch.save(checkpoint, weight_path)
    print(f"[{layout}] training finished → {weight_path.resolve()}")
    return weight_path

# ───────── CLI ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on all Pac‑Man layouts")
    parser.add_argument(
        "--layout", type=str, default=None,
        choices=["classic", "empty", "spiral", "spiral_harder"],
        help="Specific layout to train on (default: all/hardcoded list)"
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

    # If user provides a layout, use only that one. Otherwise use the hardcoded list.
    if args.layout:
        layouts_to_train = [args.layout]
    else:
        # Default list if no layout specified
        layouts_to_train = ["spiral_harder"] 

    for layout in layouts_to_train:
        train_layout(layout, episodes, args.name, args.arch)
