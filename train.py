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
import matplotlib.pyplot as plt

# ───────── hyper‑parameters ─────────
NUM_EPISODES      = 1000
NUM_EPISODES_FAST = 200
TARGET_FREQ       = 200
BATCH_SIZE        = 256
MEMORY_CAP        = 20_000
GAMMA             = 0.95
LR                = 1e-3
EPS               = (1.0, 0.05, 20_000)   # ε‑greedy schedule (start, end, decay)

# ───────── single‑layout trainer ─────────
def train_mixed(layouts: list[str], episodes: int, model_name: str, arch: str, load_path: str = None) -> Path:
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

    if load_path:
        print(f"Loading pre-trained weights from: {load_path}")
        checkpoint = torch.load(load_path, map_location=DEVICE)
        # Handle both new dictionary format and legacy state_dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['state_dict'])
        else:
            policy.load_state_dict(checkpoint)
        print("Weights loaded successfully!")

    target.load_state_dict(policy.state_dict())

    print("Created policy and target networks")
    optimiser = optim.Adam(policy.parameters(), lr=LR)
    memory    = ReplayMemory(MEMORY_CAP)
    print("Created optimizer and memory")

    # Track stats per layout
    layout_stats = {l: {'wins': 0, 'episodes': 0} for l in layouts}

    # Track rewards for plotting
    episode_rewards = []
    episode_numbers = []
    episode_wins = []  # Track wins (1) and losses (0)

    # Setup matplotlib for live plotting with two subplots
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Training Progress', fontsize=14, fontweight='bold')

    # Initialize step counter
    # For transfer learning: start with low epsilon (high step count)
    # For training from scratch: start with high epsilon (step = 0)
    if load_path:
        step = 100_000  # This gives epsilon ≈ 0.05 (minimal exploration for fine-tuning)
        print(f"Transfer learning mode: Starting with low exploration (epsilon ≈ 0.05)")
    else:
        step = 0
        print(f"Training from scratch: Starting with high exploration (epsilon = 1.0)")

    for ep in range(1, episodes + 1):
        # 1. Randomly select a layout for this episode
        current_layout = random.choice(layouts)
        env = PacmanEnv(current_layout)

        state, _ = env.reset()
        done, ep_reward = False, 0.0

        # print(f"[Ep {ep}/{episodes}] Layout: {current_layout}") # Optional: noisy

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

            # Layout-specific reward normalization to ensure fair comparison
            if done:
                if len(env.pellets) == 0:  # Win on any layout
                    reward = 100.0  # Standardized win reward
                else:  # Loss - use progress-based penalty
                    if current_layout == "classic":
                        # Classic: Map [0, 2000] to [-100, -10]
                        # This preserves "almost winning" vs "dying early" signal
                        # while ensuring losses are always negative
                        progress = min(reward / 2000.0, 0.95)
                        reward = -100.0 + (90.0 * progress)
                    else:
                        # Small layouts already give negative rewards on loss
                        # Keep as-is (typically -30 to -60)
                        pass
            else:
                # Intermediate steps: scale down classic rewards
                if current_layout == "classic":
                    reward = reward / 10.0

            # REWARD SHAPING: Encourage moving closer to pellets
            if not done and env.pellets:
                curr_pos = env.pac_pos
                curr_min_dist = min(abs(p[0] - curr_pos[0]) + abs(p[1] - curr_pos[1]) for p in env.pellets)

                # Ghost avoidance (Manhattan distance)
                prev_ghost_dist = min(abs(g[0] - prev_pos[0]) + abs(g[1] - prev_pos[1]) for g in env.ghost_pos)
                curr_ghost_dist = min(abs(g[0] - curr_pos[0]) + abs(g[1] - curr_pos[1]) for g in env.ghost_pos)

                if curr_pos == prev_pos:
                    reward -= 2.0  # Big penalty for hitting wall/staying still

                # Prioritize ghost avoidance if close
                elif curr_ghost_dist < 5:
                    if curr_ghost_dist < prev_ghost_dist:
                        reward -= 1.0  # Penalty for moving closer to danger
                    elif curr_ghost_dist > prev_ghost_dist:
                        reward += 0.5  # Bonus for escaping
                    else:
                        reward -= 0.3  # Penalty for staying in the same place

                # Otherwise focus on food
                elif curr_min_dist < prev_min_dist:
                    reward += 0.5  # Bonus for moving closer
                elif curr_min_dist >= prev_min_dist:
                    reward -= 0.1  # Penalty for moving away

            memory.push(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward

            optimise(memory, policy, target, optimiser, BATCH_SIZE, GAMMA)
            if step % TARGET_FREQ == 0:
                target.load_state_dict(policy.state_dict())

        won = (len(env.pellets) == 0)
        env.close()

        # Track rewards and wins
        episode_rewards.append(ep_reward)
        episode_numbers.append(ep)
        episode_wins.append(1 if won else 0)

        if True:  # Print every episode
            result = "WIN " if won else "LOSS"
            print(f"[Ep {ep:4d}] {result} | Layout: {current_layout:15s} | reward = {ep_reward:7.1f}")

            # Update stats
            layout_stats[current_layout]['episodes'] += 1
            if won:
                layout_stats[current_layout]['wins'] += 1

        # Update plot every 10 episodes
        if ep % 10 == 0:
            # Plot 1: Win Rate (more stable metric)
            ax1.clear()
            if len(episode_wins) >= 10:
                window = min(50, len(episode_wins))
                win_rates = [sum(episode_wins[max(0, i-window+1):i+1]) / window * 100
                            for i in range(len(episode_wins))]
                ax1.plot(episode_numbers, win_rates, 'g-', linewidth=2, label=f'{window}-Episode Win Rate')
                ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Target')

            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Win Rate (%)')
            ax1.set_title(f'Win Rate Over Time (Ep {ep}/{episodes})')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(-5, 105)

            # Plot 2: Rewards (with moving average)
            ax2.clear()
            ax2.plot(episode_numbers, episode_rewards, alpha=0.3, linewidth=1, color='blue', label='Episode Reward')

            if len(episode_rewards) >= 10:
                window = min(50, len(episode_rewards))
                moving_avg = [sum(episode_rewards[max(0, i-window+1):i+1]) / window
                              for i in range(len(episode_rewards))]
                ax2.plot(episode_numbers, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Avg Reward')

            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Episode Rewards')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.pause(0.01)

        # Save checkpoint every 1000 episodes
        if ep % 500 == 0:
            # Create directory: checkpoints/<model_name>/
            checkpoint_dir = Path("checkpoints") / model_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Create filename based on layout(s)
            if len(layouts) > 1:
                layout_tag = "+".join(sorted(layouts))
                checkpoint_filename = f"pacman_dqn_{layout_tag}_ep{ep}.pt"
            else:
                checkpoint_filename = f"pacman_dqn_{layouts[0]}_ep{ep}.pt"

            checkpoint_path = checkpoint_dir / checkpoint_filename

            checkpoint_data = {
                'arch': arch,
                'state_dict': policy.state_dict(),
                'episode': ep,
                'step': step
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\n[Checkpoint] Saved to {checkpoint_path}")

    # Save the final model with appropriate naming
    if len(layouts) > 1:
        # For multiple layouts, create a descriptive name
        layout_tag = "+".join(sorted(layouts))
        weight_path = Path(f"pacman_dqn_{layout_tag}_{model_name}.pt")
    else:
        weight_path = Path(f"pacman_dqn_{layouts[0]}_{model_name}.pt")

    import json

    print("\n" + "="*40)
    print("       TRAINING STATISTICS")
    print("="*40)
    for l in layouts:
        stats = layout_stats[l]
        n = stats['episodes']
        w = stats['wins']
        rate = (w / n * 100) if n > 0 else 0.0
        print(f"{l:15s} | {n:4d} eps | {w:4d} wins | {rate:5.1f}%")
    print("="*40 + "\n")

    # Save to JSON
    with open("training_stats.json", "w") as f:
        json.dump(layout_stats, f, indent=4)
    print("Saved statistics to training_stats.json")

    # Save final plot
    plt.ioff()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Training Progress: {layouts} - {model_name}', fontsize=14, fontweight='bold')

    # Plot 1: Win Rate
    if len(episode_wins) >= 10:
        window = min(50, len(episode_wins))
        win_rates = [sum(episode_wins[max(0, i-window+1):i+1]) / window * 100
                    for i in range(len(episode_wins))]
        ax1.plot(episode_numbers, win_rates, 'g-', linewidth=2, label=f'{window}-Episode Win Rate')
        ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Target')

    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Win Rate (%)', fontsize=11)
    ax1.set_title('Win Rate Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-5, 105)

    # Plot 2: Rewards
    ax2.plot(episode_numbers, episode_rewards, alpha=0.3, linewidth=1, color='blue', label='Episode Reward')

    if len(episode_rewards) >= 10:
        window = min(50, len(episode_rewards))
        moving_avg = [sum(episode_rewards[max(0, i-window+1):i+1]) / window
                      for i in range(len(episode_rewards))]
        ax2.plot(episode_numbers, moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Avg Reward')

    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Episode Rewards', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save plot
    plot_path = Path(f"training_plot_{model_name}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot to {plot_path}")
    plt.close()

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
        help="Single layout to train on, or 'mixed' for all (for backward compatibility)"
    )
    parser.add_argument(
        "--layouts", type=str, nargs='+', default=None,
        choices=["classic", "empty", "spiral", "spiral_harder"],
        help="Multiple layouts for cumulative training (e.g., --layouts empty spiral)"
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
        "--arch", type=str, default="original", choices=["original", "deep_v1", "multiscale"],
        help="Network architecture to use (default: original)"
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a .pt file to load weights from (transfer learning)"
    )
    args = parser.parse_args()
    episodes = NUM_EPISODES_FAST if args.fast else NUM_EPISODES

    # Define the list of layouts to train on
    if args.layouts:
        # Custom layout combination (cumulative transfer learning)
        layouts_to_train = args.layouts
        print(f"Cumulative training mode: {' + '.join(layouts_to_train)}")
        train_mixed(layouts_to_train, episodes, args.name, args.arch, args.load)
    elif args.layout == "mixed" or args.layout is None:
        # Train on ALL layouts randomly
        layouts_to_train = ["classic", "empty", "spiral", "spiral_harder"]
        train_mixed(layouts_to_train, episodes, args.name, args.arch, args.load)
    else:
        # Train on a single specific layout
        train_mixed([args.layout], episodes, args.name, args.arch, args.load)