#!/usr/bin/env python
"""
evaluate.py - Evaluate a trained model on a specific layout
"""
import argparse
import torch
from pacman_env import PacmanEnv
from dqn_agent import get_arch, DEVICE

def evaluate_model(model_path: str, layout: str, num_episodes: int = 100, render: bool = False):
    """Evaluate a trained model"""

    # Load checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        arch = checkpoint.get('arch', 'original')
        state_dict = checkpoint['state_dict']
        print(f"Architecture: {arch}")
    else:
        arch = 'original'
        state_dict = checkpoint
        print("Architecture: Legacy (assuming original)")

    # Create environment
    env = PacmanEnv(layout)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    # Create network
    policy = get_arch(arch, obs_shape, n_actions).to(DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"\nEvaluating on layout: {layout}")
    print(f"Number of episodes: {num_episodes}")
    print("-" * 50)

    wins = 0
    total_reward = 0
    episode_lengths = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            # Greedy action selection (no exploration)
            with torch.no_grad():
                state_t = torch.as_tensor(state, device=DEVICE).unsqueeze(0)
                q_values = policy(state_t)
                action = int(q_values.argmax())

            state, reward, done, _, _ = env.step(action)
            ep_reward += reward
            steps += 1

            if steps > 1000:  # Timeout
                break

        won = (len(env.pellets) == 0)
        if won:
            wins += 1

        total_reward += ep_reward
        episode_lengths.append(steps)

        if ep % 10 == 0:
            print(f"Episode {ep}/{num_episodes}: {wins}/{ep} wins ({wins/ep*100:.1f}%)")

    env.close()

    # Print results
    print("-" * 50)
    print(f"\n📊 EVALUATION RESULTS")
    print(f"Win Rate: {wins}/{num_episodes} = {wins/num_episodes*100:.1f}%")
    print(f"Average Reward: {total_reward/num_episodes:.1f}")
    print(f"Average Episode Length: {sum(episode_lengths)/len(episode_lengths):.1f} steps")
    print(f"Min/Max Episode Length: {min(episode_lengths)}/{max(episode_lengths)} steps")

    return wins / num_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model")
    parser.add_argument("model", type=str, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--layout", type=str, default="empty",
                       choices=["classic", "empty", "spiral", "spiral_harder"],
                       help="Layout to evaluate on")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to evaluate")

    args = parser.parse_args()
    evaluate_model(args.model, args.layout, args.episodes)
