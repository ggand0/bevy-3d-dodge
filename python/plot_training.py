#!/usr/bin/env python3
"""Plot training curves from TensorBoard logs.

Usage:
    python plot_training.py [--logdir logs] [--output plots/]
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(logdir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load data from TensorBoard event files.

    Args:
        logdir: Directory containing TensorBoard logs

    Returns:
        Dictionary mapping metric names to list of (step, value) tuples
    """
    # Find the latest run directory (DQN_* or PPO_*)
    run_dirs = sorted(logdir.glob("DQN_*")) + sorted(logdir.glob("PPO_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No TensorBoard logs found in {logdir}")

    latest_run = run_dirs[-1]
    print(f"Loading data from: {latest_run}")

    # Load TensorBoard data
    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()

    # Extract scalars
    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def plot_training_curves(data: Dict[str, List[Tuple[int, float]]], output_dir: Path):
    """Plot training curves for RL metrics.

    Args:
        data: Dictionary of metric name -> [(step, value)] from TensorBoard
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    # 1. Episode Reward (most important metric)
    if 'rollout/ep_rew_mean' in data:
        fig, ax = plt.subplots(figsize=(12, 6))
        steps, rewards = zip(*data['rollout/ep_rew_mean'])

        ax.plot(steps, rewards, linewidth=2, label='Episode Reward (mean)')

        # Add rolling average
        window = min(50, len(rewards) // 10)
        if window > 1:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], rolling_mean, linewidth=3,
                   label=f'Rolling Average (window={window})', alpha=0.7)

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Episode Reward')
        ax.set_title('Episode Reward Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'episode_reward.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'episode_reward.png'}")
        plt.close()

    # 2. Episode Length
    if 'rollout/ep_len_mean' in data:
        fig, ax = plt.subplots(figsize=(12, 6))
        steps, lengths = zip(*data['rollout/ep_len_mean'])

        ax.plot(steps, lengths, linewidth=2, label='Episode Length (mean)')

        # Add rolling average
        window = min(50, len(lengths) // 10)
        if window > 1:
            rolling_mean = np.convolve(lengths, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], rolling_mean, linewidth=3,
                   label=f'Rolling Average (window={window})', alpha=0.7)

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Episode Length (steps)')
        ax.set_title('Episode Length Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'episode_length.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'episode_length.png'}")
        plt.close()

    # 3. Training Loss
    if 'train/loss' in data:
        fig, ax = plt.subplots(figsize=(12, 6))
        steps, losses = zip(*data['train/loss'])

        # Plot raw loss (often noisy)
        ax.plot(steps, losses, linewidth=1, alpha=0.3, label='Loss (raw)')

        # Add rolling average
        window = min(100, len(losses) // 10)
        if window > 1:
            rolling_mean = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(steps[window-1:], rolling_mean, linewidth=2,
                   label=f'Loss (smoothed, window={window})')

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('TD Loss')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for loss

        plt.tight_layout()
        plt.savefig(output_dir / 'training_loss.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'training_loss.png'}")
        plt.close()

    # 4. Exploration Rate
    if 'rollout/exploration_rate' in data:
        fig, ax = plt.subplots(figsize=(12, 6))
        steps, eps = zip(*data['rollout/exploration_rate'])

        ax.plot(steps, eps, linewidth=2)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Exploration Rate (ε)')
        ax.set_title('ε-Greedy Exploration Decay')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'exploration_rate.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'exploration_rate.png'}")
        plt.close()

    # 5. Evaluation Metrics (if available)
    eval_metrics = [k for k in data.keys() if k.startswith('eval/')]
    if eval_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Eval reward
        if 'eval/mean_reward' in data:
            steps, rewards = zip(*data['eval/mean_reward'])
            axes[0].plot(steps, rewards, 'o-', linewidth=2, markersize=8)
            axes[0].set_xlabel('Timesteps')
            axes[0].set_ylabel('Mean Reward (evaluation)')
            axes[0].set_title('Evaluation Reward (every 5k steps)')
            axes[0].grid(True, alpha=0.3)

        # Eval episode length
        if 'eval/mean_ep_length' in data:
            steps, lengths = zip(*data['eval/mean_ep_length'])
            axes[1].plot(steps, lengths, 'o-', linewidth=2, markersize=8, color='orange')
            axes[1].set_xlabel('Timesteps')
            axes[1].set_ylabel('Mean Episode Length (evaluation)')
            axes[1].set_title('Evaluation Episode Length (every 5k steps)')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_metrics.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'evaluation_metrics.png'}")
        plt.close()

    # 6. Combined Overview (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Reward
    if 'rollout/ep_rew_mean' in data:
        steps, rewards = zip(*data['rollout/ep_rew_mean'])
        axes[0, 0].plot(steps, rewards, linewidth=1.5, alpha=0.7)
        window = min(50, len(rewards) // 10)
        if window > 1:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(steps[window-1:], rolling_mean, linewidth=2.5, color='red')
        axes[0, 0].set_title('Episode Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)

    # Episode Length
    if 'rollout/ep_len_mean' in data:
        steps, lengths = zip(*data['rollout/ep_len_mean'])
        axes[0, 1].plot(steps, lengths, linewidth=1.5, alpha=0.7)
        window = min(50, len(lengths) // 10)
        if window > 1:
            rolling_mean = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(steps[window-1:], rolling_mean, linewidth=2.5, color='green')
        axes[0, 1].set_title('Episode Length', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Mean Length (steps)')
        axes[0, 1].grid(True, alpha=0.3)

    # Training Loss
    if 'train/loss' in data:
        steps, losses = zip(*data['train/loss'])
        axes[1, 0].plot(steps, losses, linewidth=1, alpha=0.3)
        window = min(100, len(losses) // 10)
        if window > 1:
            rolling_mean = np.convolve(losses, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(steps[window-1:], rolling_mean, linewidth=2, color='purple')
        axes[1, 0].set_title('Training Loss (TD Error)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Loss (log scale)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

    # Exploration Rate
    if 'rollout/exploration_rate' in data:
        steps, eps = zip(*data['rollout/exploration_rate'])
        axes[1, 1].plot(steps, eps, linewidth=2, color='orange')
        axes[1, 1].set_title('Exploration Rate (ε-greedy)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('ε')
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('DQN Training Overview - Bevy 3D Dodge', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_overview.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'training_overview.png'}")
    plt.close()

    print(f"\nAll plots saved to: {output_dir}/")


def print_summary_statistics(data: Dict[str, List[Tuple[int, float]]]):
    """Print summary statistics from training."""
    print("\n" + "="*70)
    print("Training Summary Statistics")
    print("="*70)

    # Final values
    if 'rollout/ep_rew_mean' in data:
        final_reward = data['rollout/ep_rew_mean'][-1][1]
        max_reward = max(v for _, v in data['rollout/ep_rew_mean'])
        print(f"Final episode reward:     {final_reward:.2f}")
        print(f"Peak episode reward:      {max_reward:.2f}")

    if 'rollout/ep_len_mean' in data:
        final_length = data['rollout/ep_len_mean'][-1][1]
        max_length = max(v for _, v in data['rollout/ep_len_mean'])
        print(f"Final episode length:     {final_length:.0f} steps")
        print(f"Peak episode length:      {max_length:.0f} steps")

    if 'train/loss' in data:
        final_loss = data['train/loss'][-1][1]
        print(f"Final training loss:      {final_loss:.4f}")

    if 'rollout/exploration_rate' in data:
        final_eps = data['rollout/exploration_rate'][-1][1]
        print(f"Final exploration rate:   {final_eps:.3f}")

    # Evaluation stats
    if 'eval/mean_reward' in data:
        eval_rewards = [v for _, v in data['eval/mean_reward']]
        print(f"\nEvaluation (every 5k steps):")
        print(f"  Best eval reward:       {max(eval_rewards):.2f}")
        print(f"  Final eval reward:      {eval_rewards[-1]:.2f}")

    print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard logs")
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Directory containing TensorBoard logs (default: logs/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots",
        help="Directory to save plots (default: plots/)",
    )

    args = parser.parse_args()

    logdir = Path(args.logdir)
    output_dir = Path(args.output)

    if not logdir.exists():
        print(f"Error: Log directory not found: {logdir}")
        return

    print("Loading TensorBoard data...")
    data = load_tensorboard_data(logdir)

    print(f"Found {len(data)} metrics")
    print("Available metrics:", list(data.keys())[:10], "...")

    print("\nGenerating plots...")
    plot_training_curves(data, output_dir)

    print_summary_statistics(data)


if __name__ == "__main__":
    main()
