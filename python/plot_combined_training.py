#!/usr/bin/env python3
"""Plot combined learning curves from multiple TensorBoard event files."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_all_tensorboard_data(logdir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load and combine data from all TensorBoard event files."""
    # Find run directories (SAC_*, PPO_*, DQN_*)
    run_dirs = []
    for pattern in ["SAC_*", "PPO_*", "DQN_*"]:
        run_dirs.extend(sorted(logdir.glob(pattern)))

    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard logs found in {logdir}")

    latest_run = run_dirs[-1]
    print(f"Loading data from: {latest_run}")

    # Find all event files
    event_files = sorted(latest_run.glob("events.out.tfevents.*"))
    print(f"Found {len(event_files)} event file(s)")

    # Combine data from all event files
    combined_data: Dict[str, List[Tuple[int, float]]] = {}

    for event_file in event_files:
        print(f"  Loading: {event_file.name}")
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()

        for tag in ea.Tags().get('scalars', []):
            events = ea.Scalars(tag)
            if tag not in combined_data:
                combined_data[tag] = []
            combined_data[tag].extend([(e.step, e.value) for e in events])

    # Sort by step and remove duplicates
    for tag in combined_data:
        combined_data[tag] = sorted(set(combined_data[tag]), key=lambda x: x[0])

    return combined_data


def plot_sac_learning_curves(data: Dict[str, List[Tuple[int, float]]], output_path: Path, title: str = "SAC Training"):
    """Plot SAC-specific learning curves."""
    plt.style.use('seaborn-v0_8-darkgrid')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Eval Reward (top-left) - Most important
    if 'eval/mean_reward' in data:
        steps, rewards = zip(*data['eval/mean_reward'])
        axes[0, 0].plot(steps, rewards, 'o-', linewidth=2, markersize=4, color='blue', alpha=0.7)
        axes[0, 0].axhline(y=1000, color='green', linestyle='--', alpha=0.5, label='Perfect (1000)')
        axes[0, 0].set_title('Evaluation Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Annotate best
        max_idx = np.argmax(rewards)
        axes[0, 0].annotate(f'Best: {rewards[max_idx]:.0f}',
                           xy=(steps[max_idx], rewards[max_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, color='darkblue')

    # 2. Eval Episode Length (top-right)
    if 'eval/mean_ep_length' in data:
        steps, lengths = zip(*data['eval/mean_ep_length'])
        axes[0, 1].plot(steps, lengths, 'o-', linewidth=2, markersize=4, color='green', alpha=0.7)
        axes[0, 1].axhline(y=1000, color='green', linestyle='--', alpha=0.5, label='Max (1000)')
        axes[0, 1].set_title('Evaluation Episode Length', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Mean Length (steps)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Training Reward (bottom-left)
    if 'rollout/ep_rew_mean' in data:
        steps, rewards = zip(*data['rollout/ep_rew_mean'])
        axes[1, 0].plot(steps, rewards, linewidth=1, alpha=0.5, color='blue')

        # Rolling average
        window = min(50, len(rewards) // 10)
        if window > 1:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(steps[window-1:], rolling_mean, linewidth=2.5, color='darkblue',
                           label=f'Rolling Avg (w={window})')

        axes[1, 0].set_title('Training Episode Reward', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Entropy Coefficient (bottom-right) - SAC specific
    if 'train/ent_coef' in data:
        steps, ent_coef = zip(*data['train/ent_coef'])
        axes[1, 1].plot(steps, ent_coef, linewidth=1.5, color='purple', alpha=0.7)
        axes[1, 1].set_title('Entropy Coefficient (auto-tuned)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('ent_coef')
        axes[1, 1].grid(True, alpha=0.3)
    elif 'train/actor_loss' in data:
        # Fallback: show actor loss
        steps, losses = zip(*data['train/actor_loss'])
        axes[1, 1].plot(steps, losses, linewidth=1, alpha=0.5, color='red')
        window = min(100, len(losses) // 10)
        if window > 1:
            rolling_mean = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(steps[window-1:], rolling_mean, linewidth=2, color='darkred')
        axes[1, 1].set_title('Actor Loss', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_stats(data: Dict[str, List[Tuple[int, float]]]):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)

    if 'eval/mean_reward' in data:
        rewards = [v for _, v in data['eval/mean_reward']]
        steps = [s for s, _ in data['eval/mean_reward']]
        best_idx = np.argmax(rewards)
        print(f"Best Eval Reward:    {rewards[best_idx]:.2f} (at {steps[best_idx]:,} steps)")
        print(f"Final Eval Reward:   {rewards[-1]:.2f} (at {steps[-1]:,} steps)")

    if 'eval/mean_ep_length' in data:
        lengths = [v for _, v in data['eval/mean_ep_length']]
        print(f"Best Eval Length:    {max(lengths):.0f} steps")
        print(f"Final Eval Length:   {lengths[-1]:.0f} steps")

    if 'rollout/ep_rew_mean' in data:
        rewards = [v for _, v in data['rollout/ep_rew_mean']]
        print(f"Final Train Reward:  {rewards[-1]:.2f}")

    if 'train/ent_coef' in data:
        ent_coefs = [v for _, v in data['train/ent_coef']]
        print(f"Final Entropy Coef:  {ent_coefs[-1]:.4f}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Plot combined SAC learning curves")
    parser.add_argument("--logdir", type=str, required=True, help="Log directory path")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--title", type=str, default="SAC Training - Thrower Indicator Mode", help="Plot title")

    args = parser.parse_args()

    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: {logdir} not found")
        return

    print("Loading TensorBoard data...")
    data = load_all_tensorboard_data(logdir)
    print(f"Loaded {len(data)} metrics")

    output_path = Path(args.output) if args.output else logdir.parent / "plots" / "combined_learning_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plot...")
    plot_sac_learning_curves(data, output_path, title=args.title)

    print_stats(data)


if __name__ == "__main__":
    main()
