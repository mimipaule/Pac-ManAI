#!/usr/bin/env python
"""
train_parallel.py – Train a single DQN using multiple parallel environments.

Key Features:
- Uses gymnasium.vector.AsyncVectorEnv (or SyncVectorEnv) for parallelism
- Collects batches of experiences
- Updates the same shared model
- Significantly faster wall-clock time for data collection
"""

from __future__ import annotations
import argparse, torch, torch.optim as optim
import numpy as np
from pathlib import Path
import gymnasium as gym
from pacman_env import PacmanEnv
import random
import math
from dqn_agent import get_arch, ReplayMemory, optimise, DEVICE

# ───────── hyper‑parameters ─────────
NUM_EPISODES_TOTAL = 2000      # Total episodes across all workers
NUM_ENVS = 10                  # Number of parallel environments
BATCH_SIZE = 256       # Increased for GPU efficiency
MEMORY_CAP = 50_000      # Increased buffer size
GAMMA = 0.99
LR = 1e-3
# ε‑greedy schedule (start, end, decay_steps)
# Note: decay is based on global steps, which accumulate faster with N envs
EPS_PARAMS = (1.0, 0.05, 20_000)
TARGET_FREQ = 200              # Target network update frequency (global steps)
UPDATES_PER_STEP = 2           # Perform multiple updates per step to keep up with data collection

def make_env(layout_list: list[str]):
    """Factory function to create an environment with a random layout from the list."""
    def _init():
        # We pick a random layout each time the env is created/reset
        # Note: For VectorEnv, the env is created once.
        # To support random layouts per episode in a VectorEnv,
        # we might need a wrapper or just pick one layout per worker.
        # For simplicity here, we pick one random layout per worker at start.
        # A better approach for "mixed" training is to use a wrapper that
        # reseeds/changes layout on reset, but PacmanEnv takes layout in __init__.
        layout = random.choice(layout_list)
        return PacmanEnv(layout)
    return _init

def select_action_batch(states: np.ndarray, net: torch.nn.Module, step: int,
                       eps_start: float, eps_end: float, eps_decay: int, n_envs: int) -> list[int]:
    """
    Select actions for a batch of states using epsilon-greedy.
    """
    eps = eps_end + (eps_start - eps_end) * math.exp(-step / eps_decay)

    actions = []

    # We need to decide for each env whether to explore or exploit
    # It's more efficient to do one big forward pass for the exploiters

    # 1. Identify which envs will explore (random) vs exploit (policy)
    explore_mask = [random.random() < eps for _ in range(n_envs)]

    # 2. Get Q-values for ALL states (batched inference is cheap)
    with torch.no_grad():
        # states shape: (n_envs, H, W, C) -> need (n_envs, C, H, W) float
        # Check if states are already torch tensor or numpy
        if isinstance(states, np.ndarray):
            states_t = torch.as_tensor(states, device=DEVICE)
        else:
            states_t = states

        # If the network expects standard batch processing
        q_values = net(states_t) # Shape: (n_envs, n_actions)
        best_actions = q_values.argmax(dim=1).cpu().numpy()

    # 3. Assemble final actions list
    for i in range(n_envs):
        if explore_mask[i]:
            # Random action
            actions.append(random.randrange(q_values.shape[1]))
        else:
            actions.append(best_actions[i])

    return actions

def train_parallel(layouts: list[str], total_episodes: int, model_name: str, arch: str, load_path: str = None):
    # 1. Create Vector Environment
    # We create N environments. Each can potentially have a different layout if we randomize in make_env
    # However, standard AsyncVectorEnv keeps the process alive.
    # If we want TRULY mixed training where layouts change *between episodes* within the same worker,
    # we'd need to modify PacmanEnv to allow layout switching on reset().
    # For now, we'll just assign random layouts to the N workers initially.

    env_fns = [make_env(layouts) for _ in range(NUM_ENVS)]

    # 'async' puts each env in a separate process (good for CPU bound)
    # 'sync' runs them in serial (good for debugging)
    envs = gym.vector.AsyncVectorEnv(env_fns)

    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    print(f"Initialized {NUM_ENVS} parallel environments.")
    print(f"Layouts available: {layouts}")
    print(f"Architecture: {arch} | Device: {DEVICE}")

    # 2. Setup Network & Optimiser
    policy = get_arch(arch, obs_shape, n_actions).to(DEVICE)
    target = get_arch(arch, obs_shape, n_actions).to(DEVICE)

    if load_path:
        print(f"Loading pre-trained weights from: {load_path}")
        checkpoint = torch.load(load_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['state_dict'])
        else:
            policy.load_state_dict(checkpoint)
        print("Weights loaded successfully!")

    target.load_state_dict(policy.state_dict())

    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_CAP)

    # 3. Training Loop State
    # Reset all envs initially
    states, _ = envs.reset()

    # Warmup phase: fill memory with random actions before training
    print(f"Warming up replay memory with {BATCH_SIZE} transitions...")
    warmup_steps = 0
    while len(memory) < BATCH_SIZE:
        # Use random actions for warmup
        actions = [random.randrange(n_actions) for _ in range(NUM_ENVS)]
        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        for i in range(NUM_ENVS):
            real_next_state = next_states[i]
            if dones[i] and "final_observation" in infos and infos["_final_observation"][i]:
                real_next_state = infos["final_observation"][i]
            memory.push(states[i], actions[i], rewards[i], real_next_state, float(dones[i]))
        states = next_states

    global_step = 0
    episodes_finished = 0

    # Stats tracking
    # We can't easily track "per layout" stats perfectly because we don't strictly know
    # which worker has which layout if we randomized them inside the lambda.
    # But we CAN track global win rate.
    # If we want per-layout stats, we need to know what layout each worker has.
    # Let's approximate by just tracking global stats for now or inferring if possible.
    # For simplicity -> Global Win Rate & Total Reward

    recent_rewards = []
    recent_wins = 0
    stats = {'episodes': 0, 'wins': 0}

    print(f"Starting training for ~{total_episodes} episodes...")

    while episodes_finished < total_episodes:
        # A. Select actions for all envs
        actions = select_action_batch(states, policy, global_step, *EPS_PARAMS, NUM_ENVS)

        # B. Step all envs
        next_states, rewards, terminations, truncations, infos = envs.step(actions)

        # 'dones' in vector envs is typically terminations | truncations
        dones = terminations | truncations

        # C. Store transitions & Handle Episode Ends
        for i in range(NUM_ENVS):
            # Store transition
            # Note: vector env auto-resets when done.
            # 'next_states[i]' is actually the INITIAL state of the NEW episode if done[i] is True.
            # The 'final_observation' is usually in infos['final_observation'][i] or similar.

            real_next_state = next_states[i]
            is_done = dones[i]

            if is_done:
                # If done, the transition should be (s, a, r, terminal_state, done)
                # But 'next_states[i]' is already reset. We need the terminal state.
                # gym.vector usually provides this in info
                # BUT for simple DQN, storing the reset state as 'next_state' with done=True
                # is often "good enough" or we treat it carefully.
                # The standard practice: use 'final_observation' if available.
                if "final_observation" in infos:
                     # Some vector env implementations handle this differently
                     # Let's check if it exists and is valid
                     if infos["_final_observation"][i]:
                         real_next_state = infos["final_observation"][i]

                episodes_finished += 1

                stats['episodes'] += 1

                # Check win condition (approximate via reward or info if available)
                # In our PacmanEnv, winning gives a large reward (e.g. +50 at end)
                # or we can check "if len(pellets) == 0" but we don't have direct access to env object easily.
                # We'll use reward threshold or rely on the fact that we print periodically.

                # Win is roughly > 40 reward
                if rewards[i] > 40:
                     stats['wins'] += 1

                # Tracking stats
                ep_reward = 0 # We don't easily track cumulative reward per episode in parallel
                              # without a separate tracker array. Let's add one.

            memory.push(states[i], actions[i], rewards[i], real_next_state, float(is_done))

        # Update current state
        states = next_states
        global_step += NUM_ENVS # We took N steps total

        # D. Optimization Step
        # Since we collect NUM_ENVS steps of data per loop, we should ideally do
        # multiple gradient updates to keep the 'replay ratio' healthy.
        for _ in range(UPDATES_PER_STEP):
            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)

        # E. Target Network Update
        if global_step % TARGET_FREQ < NUM_ENVS:
            # Rough check to update periodically
            target.load_state_dict(policy.state_dict())

        # F. Logging
        if episodes_finished % 10 == 0 and episodes_finished > 0:
            # We can't easily print "Episode X finished" continuously because they finish randomly.
            # Just print status every 50 episodes finished.
            print(f"Progress: {episodes_finished}/{total_episodes} eps | Global Step: {global_step}", end='\r')

    print(f"\nTraining finished after {episodes_finished} episodes.")

    import json

    # Print Summary
    print("\n" + "="*40)
    print("       PARALLEL TRAINING STATISTICS")
    print("="*40)
    n = stats['episodes']
    w = stats['wins']
    rate = (w / n * 100) if n > 0 else 0.0
    print(f"Total Episodes: {n}")
    print(f"Total Wins:     {w}")
    print(f"Win Rate:       {rate:.1f}%")
    print("="*40 + "\n")

    # Save to JSON
    with open("training_stats.json", "w") as f:
        json.dump(stats, f, indent=4)
    print("Saved statistics to training_stats.json")

    envs.close()

    # Save
    if len(layouts) > 1:
        weight_path = Path(f"pacman_dqn_mixed_parallel_{model_name}.pt")
    else:
        weight_path = Path(f"pacman_dqn_{layouts[0]}_parallel_{model_name}.pt")

    torch.save({
        'arch': arch,
        'state_dict': policy.state_dict()
    }, weight_path)
    print(f"Saved model to {weight_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN using Parallel Environments")
    parser.add_argument("--layout", type=str, default="mixed",
                        choices=["classic", "empty", "spiral", "spiral_harder", "mixed"])
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES_TOTAL)
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--arch", type=str, default="original", choices=["original", "deep_v1"])
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel envs")
    parser.add_argument("--load", type=str, default=None, help="Path to .pt file to load weights from")

    args = parser.parse_args()

    # Set global NUM_ENVS
    NUM_ENVS = args.workers

    if args.layout == "mixed":
        layouts = ["classic", "empty", "spiral", "spiral_harder"]
    else:
        layouts = [args.layout]

    train_parallel(layouts, args.episodes, args.name, args.arch, args.load)
