#!/usr/bin/env python3
"""Evaluate trained DQN agent on Bevy 3D dodge game.

Usage:
    python eval.py models/best/best_model.zip [--episodes 10] [--port 8000]
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
from stable_baselines3 import DQN

from bevy_dodge_env import BevyDodgeEnv


def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    port: int = 8000,
    max_steps_per_episode: int = 1000,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate trained agent for n_episodes.

    Args:
        model_path: Path to saved model (.zip file)
        n_episodes: Number of episodes to run
        port: Port of Bevy API server
        max_steps_per_episode: Maximum steps per episode
        deterministic: Use deterministic actions (no exploration)
        verbose: Whether to print progress

    Returns:
        Dictionary with statistics (avg_reward, avg_steps, success_rate, etc.)
    """
    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    # Create environment
    env = BevyDodgeEnv(port=port)

    if verbose:
        print(f"✓ Model loaded")
        print(f"✓ Environment created (port {port})")
        print()

    episode_rewards = []
    episode_lengths = []
    successes = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0
        success = False

        if verbose:
            print(f"=== Episode {episode + 1}/{n_episodes} ===")

        while steps < max_steps_per_episode:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if verbose and steps % 50 == 0:
                print(f"  Step {steps}: reward={reward:.2f}, "
                      f"projectiles={info.get('projectile_count', 'N/A')}")

            if terminated or truncated:
                # Check if reached max steps without collision (success)
                if truncated and not terminated:
                    success = True

                if verbose:
                    reason = "collision" if terminated else "max steps"
                    status = "✓ SUCCESS" if success else "✗ FAILED"
                    print(f"  {status} - Episode ended ({reason}) after {steps} steps")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        successes.append(success)

        if verbose:
            print(f"  Total reward: {episode_reward:.2f}")
            print()

    # Calculate statistics
    stats = {
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_steps": np.mean(episode_lengths),
        "std_steps": np.std(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "success_rate": np.mean(successes),
        "total_episodes": n_episodes,
        "total_steps": np.sum(episode_lengths),
    }

    env.close()
    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to saved model (.zip file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of Bevy API server (default: 8000)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (exploration enabled)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Check model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    print("=" * 70)
    print("DQN Agent Evaluation - Bevy 3D Dodge Game")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print()

    try:
        start_time = time.time()
        stats = evaluate_agent(
            model_path=str(model_path),
            n_episodes=args.episodes,
            port=args.port,
            max_steps_per_episode=args.max_steps,
            deterministic=not args.stochastic,
            verbose=not args.quiet,
        )
        elapsed_time = time.time() - start_time

        # Print summary
        print("=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Total episodes:     {stats['total_episodes']}")
        print(f"Total steps:        {stats['total_steps']:.0f}")
        print(f"Total time:         {elapsed_time:.2f}s")
        print(f"Steps per second:   {stats['total_steps'] / elapsed_time:.1f}")
        print()
        print(f"Average reward:     {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Average steps:      {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")
        print(f"Reward range:       [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print(f"Success rate:       {stats['success_rate']:.1%}")
        print()

        if stats['success_rate'] > 0.8:
            print("🎉 Excellent performance!")
        elif stats['success_rate'] > 0.5:
            print("👍 Good performance!")
        elif stats['success_rate'] > 0.2:
            print("📈 Moderate performance - needs more training")
        else:
            print("⚠️  Poor performance - needs more training")

    except ConnectionError as e:
        print(f"✗ Failed to connect to Bevy server: {e}")
        print("\nMake sure the Bevy game is running:")
        print(f"  cargo run -- --port {args.port}")
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
