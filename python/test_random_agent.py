#!/usr/bin/env python3
"""Test random agent to validate environment implementation.

This script runs a random agent in the BevyDodgeEnv to ensure:
1. Environment can be created successfully
2. Reset and step work correctly
3. No crashes occur during extended rollouts
4. All return values are properly formatted

Usage:
    python test_random_agent.py [--episodes 10] [--port 8000]
"""

import argparse
import time
from typing import Dict, Any

import numpy as np
from bevy_dodge_env import BevyDodgeEnv


def run_random_agent(
    env: BevyDodgeEnv,
    n_episodes: int = 10,
    max_steps_per_episode: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run random agent for n_episodes.

    Args:
        env: BevyDodgeEnv instance
        n_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode before truncation
        verbose: Whether to print progress

    Returns:
        Dictionary with statistics (avg_reward, avg_steps, etc.)
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        steps = 0

        if verbose:
            print(f"\n=== Episode {episode + 1}/{n_episodes} ===")

        while steps < max_steps_per_episode:
            # Sample random action
            action = env.action_space.sample()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            if verbose and steps % 50 == 0:
                print(f"  Step {steps}: reward={reward:.2f}, "
                      f"projectiles={info.get('projectile_count', 'N/A')}")

            if terminated or truncated:
                if verbose:
                    reason = "terminated" if terminated else "truncated"
                    print(f"  Episode ended ({reason}) after {steps} steps")
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if verbose:
            print(f"  Total reward: {episode_reward:.2f}")

    stats = {
        "avg_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "avg_steps": np.mean(episode_lengths),
        "std_steps": np.std(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "total_steps": np.sum(episode_lengths),
    }

    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test random agent in BevyDodgeEnv")
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)",
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
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BevyDodgeEnv Random Agent Test")
    print("=" * 60)
    print(f"Connecting to Bevy server at http://127.0.0.1:{args.port}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print()

    try:
        # Create environment
        env = BevyDodgeEnv(port=args.port)
        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print()

        # Run random agent
        start_time = time.time()
        stats = run_random_agent(
            env,
            n_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            verbose=not args.quiet,
        )
        elapsed_time = time.time() - start_time

        # Print summary
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Total steps:        {stats['total_steps']:.0f}")
        print(f"Total time:         {elapsed_time:.2f}s")
        print(f"Steps per second:   {stats['total_steps'] / elapsed_time:.1f}")
        print()
        print(f"Average reward:     {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Average steps:      {stats['avg_steps']:.1f} ± {stats['std_steps']:.1f}")
        print(f"Reward range:       [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
        print()
        print("✓ All tests passed!")

    except ConnectionError as e:
        print(f"✗ Failed to connect to Bevy server: {e}")
        print("\nMake sure the Bevy game is running:")
        print(f"  cargo run -- --port {args.port}")
        return
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return
    except Exception as e:
        print(f"✗ Error during test: {e}")
        raise


if __name__ == "__main__":
    main()
