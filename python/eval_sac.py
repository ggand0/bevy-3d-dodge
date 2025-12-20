#!/usr/bin/env python3
"""Evaluate trained SAC model on Bevy dodge game.

Usage:
    python eval_sac.py <model_path> [--episodes 20] [--render]
"""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from bevy_dodge_env import BevyDodgeEnv


def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    max_steps: int = 1000,
    port: int = 8000,
    level: int = None,
    deterministic: bool = True,
    render: bool = False,
    sprint_multiplier: float = None,
    spawn_angle_degrees: float = None,
):
    """Evaluate trained SAC agent.

    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        port: Bevy server port
        level: Game difficulty level (1 or 2), auto-detected from path if not provided
        deterministic: Use deterministic policy (no exploration)
        render: Whether to render (not used, kept for compatibility)

    Returns:
        Dictionary with evaluation statistics
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)
    print(f"✓ Model loaded (policy: {model.policy.__class__.__name__})")

    # Detect action space configuration from model
    action_dim = model.action_space.shape[0]
    action_space_map = {
        3: "basic_3d",
        4: "basic_4d_jump",
        5: "tilt_5d",
        6: "full_6d",
    }
    action_space_type = action_space_map.get(action_dim, "unknown")
    print(f"  Detected action space: {action_space_type} ({action_dim}D)")

    # Auto-detect level from model path if not provided
    if level is None:
        if "level1" in model_path.lower():
            level = 1
        elif "level2" in model_path.lower():
            level = 2
        else:
            level = 2  # Default to level 2 (hard)
            print(f"  ⚠ Could not detect level from path, defaulting to level {level}")

    print(f"  Detected level: {level}")
    print()

    # Create temporary environment to configure server
    config_parts = [f"Level {level}", action_space_type]
    if sprint_multiplier is not None:
        config_parts.append(f"sprint: {sprint_multiplier} ({1+sprint_multiplier}x)")
    if spawn_angle_degrees is not None:
        config_parts.append(f"spawn angle: ±{spawn_angle_degrees}°")
    print(f"Configuring Bevy server ({', '.join(config_parts)})...")

    temp_env = BevyDodgeEnv(port=port)

    # Configure the server with detected settings
    if action_space_type != "unknown":
        temp_env.configure(
            level=level,
            action_space_type=action_space_type,
            sprint_multiplier=sprint_multiplier,
            spawn_angle_degrees=spawn_angle_degrees,
        )
        temp_env.reset()  # Sync state
    else:
        print(f"⚠ Warning: Unknown action space dimension {action_dim}, using server default")
        temp_env.configure(
            level=level,
            sprint_multiplier=sprint_multiplier,
            spawn_angle_degrees=spawn_angle_degrees,
        )
        temp_env.reset()

    del temp_env

    # Create real environment (queries updated action space)
    env = BevyDodgeEnv(port=port)
    print(f"✓ Connected to Bevy server at http://127.0.0.1:{port}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Enable training mode to hide controls and prevent keyboard interruptions
    env.start_training()
    print("✓ Training mode enabled - controls hidden, R key disabled")
    print()

    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    episode_info = []

    print(f"Running {n_episodes} evaluation episodes...")
    print(f"Mode: {'deterministic' if deterministic else 'stochastic'}")
    print()

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False

        start_time = time.time()

        while not (done or truncated) and episode_length < max_steps:
            # Get action from model
            action, _states = model.predict(obs, deterministic=deterministic)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

        elapsed = time.time() - start_time

        # Store results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_info.append({
            "reward": episode_reward,
            "length": episode_length,
            "success": episode_length >= max_steps,
            "elapsed": elapsed,
        })

        # Print episode summary
        success_marker = "✓" if episode_length >= max_steps else "✗"
        print(f"Episode {episode + 1:2d}: {success_marker} "
              f"Reward: {episode_reward:7.2f}, "
              f"Steps: {episode_length:4d}, "
              f"Time: {elapsed:5.1f}s")

    env.close()

    # Calculate statistics
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    success_count = sum(1 for info in episode_info if info["success"])

    stats = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(rewards_array),
        "std_reward": np.std(rewards_array),
        "min_reward": np.min(rewards_array),
        "max_reward": np.max(rewards_array),
        "mean_length": np.mean(lengths_array),
        "std_length": np.std(lengths_array),
        "min_length": np.min(lengths_array),
        "max_length": np.max(lengths_array),
        "success_rate": success_count / n_episodes * 100,
        "success_count": success_count,
        "episode_info": episode_info,
    }

    return stats


def print_summary(stats: dict):
    """Print evaluation summary statistics."""
    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Total episodes:        {stats['n_episodes']}")
    print()
    print(f"Mean reward:           {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward range:          [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
    print()
    print(f"Mean episode length:   {stats['mean_length']:.1f} ± {stats['std_length']:.1f} steps")
    print(f"Length range:          [{stats['min_length']}, {stats['max_length']}] steps")
    print()
    print(f"Success rate:          {stats['success_rate']:.1f}% "
          f"({stats['success_count']}/{stats['n_episodes']} episodes)")
    print("=" * 70)
    print()

    # Episode breakdown
    if stats['n_episodes'] <= 50:
        print("Episode Breakdown:")
        print("-" * 70)
        success_episodes = [i for i, info in enumerate(stats['episode_info']) if info['success']]
        failed_episodes = [i for i, info in enumerate(stats['episode_info']) if not info['success']]

        if success_episodes:
            print(f"✓ Success ({len(success_episodes)}): Episodes {', '.join(map(str, [i+1 for i in success_episodes]))}")
        if failed_episodes:
            print(f"✗ Failed  ({len(failed_episodes)}): Episodes {', '.join(map(str, [i+1 for i in failed_episodes]))}")
        print("-" * 70)
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAC model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate best model with 20 episodes:
    python eval_sac.py results/sac_baseline/20251117_213229/models/best/best_model.zip --episodes 20

    # Quick test with 5 episodes:
    python eval_sac.py results/sac_baseline/20251117_213229/models/final_model.zip --episodes 5
        """
    )

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
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bevy server port (default: 8000)",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2],
        default=None,
        help="Game difficulty level (1=Baseline, 2=Hard). Auto-detected from path if not provided",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy instead of deterministic",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (Bevy handles rendering)",
    )
    parser.add_argument(
        "--sprint-multiplier",
        type=float,
        default=None,
        help="Sprint speed multiplier (e.g., 1.0=2x, 2.0=3x). Uses level default if not specified.",
    )
    parser.add_argument(
        "--spawn-angle",
        type=float,
        default=None,
        help="Half-angle for spawn fan in degrees (e.g., 30=±30°). Uses level default if not specified.",
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    # Print header
    print("=" * 70)
    print("SAC Agent Evaluation - Bevy 3D Dodge Game")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Mode: {'stochastic' if args.stochastic else 'deterministic'}")
    print()

    try:
        # Run evaluation
        stats = evaluate_agent(
            model_path=str(model_path),
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            port=args.port,
            level=args.level,
            deterministic=not args.stochastic,
            render=args.render,
            sprint_multiplier=args.sprint_multiplier,
            spawn_angle_degrees=args.spawn_angle,
        )

        # Print summary
        print_summary(stats)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
