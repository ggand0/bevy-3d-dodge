#!/usr/bin/env python3
"""Train DQN agent on Bevy 3D dodge game.

Usage:
    # Using YAML config (recommended):
    python train.py --config python/configs/default.yaml
    python train.py --config python/configs/improved_baseline.yaml

    # Using CLI arguments (legacy):
    python train.py --steps 100000 --port 8000 --save-dir models/
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from bevy_dodge_env import BevyDodgeEnv
from config import DQNConfig


def make_env(port: int) -> gym.Env:
    """Create and wrap environment."""
    env = BevyDodgeEnv(port=port)
    env = Monitor(env)  # Wrap for logging
    return env


def train(config: DQNConfig, verbose: int = 1) -> None:
    """Train DQN agent on Bevy dodge game.

    Args:
        config: DQNConfig instance with all hyperparameters
        verbose: Verbosity level
    """
    # Create directories
    save_path = Path(config.save_dir)
    log_path = Path(config.log_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DQN Training - Bevy 3D Dodge Game")
    print("=" * 70)
    print(f"Total timesteps:     {config.total_timesteps:,}")
    print(f"Learning rate:       {config.learning_rate}")
    print(f"Buffer size:         {config.buffer_size:,}")
    print(f"Batch size:          {config.batch_size}")
    print(f"Gamma:               {config.gamma}")
    print(f"Exploration:         {config.exploration_initial_eps} → {config.exploration_final_eps}")
    print(f"Network arch:        {config.net_arch if config.net_arch else '[64, 64] (default)'}")
    print(f"Save directory:      {save_path}")
    print(f"Log directory:       {log_path}")
    print()

    # Create environment
    print(f"Connecting to Bevy server at http://127.0.0.1:{config.port}")
    env = DummyVecEnv([lambda: make_env(config.port)])
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(config.port)])

    # Create DQN agent
    print("Creating DQN agent...")

    # Build policy kwargs if custom network architecture is specified
    policy_kwargs: Optional[Dict[str, Any]] = None
    if config.net_arch is not None:
        policy_kwargs = {"net_arch": config.net_arch}

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(log_path),
        verbose=verbose,
        device="auto",  # Will use CUDA/ROCm if available, otherwise CPU
    )

    # Print model device
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ Using GPU: {device_name}")
    else:
        print("⚠ Using CPU (no GPU detected)")
    print()

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.save_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix="dqn_dodge",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(log_path / "eval"),
        eval_freq=config.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=config.n_eval_episodes,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Train
    print("Starting training...")
    print(f"Monitor with: tensorboard --logdir {log_path}")
    print()

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    finally:
        # Save final model
        final_path = save_path / "final_model"
        model.save(final_path)
        print(f"\n✓ Final model saved to {final_path}")

    # Close environments
    env.close()
    eval_env.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent on Bevy dodge game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using YAML config (recommended):
    python train.py --config python/configs/default.yaml
    python train.py --config python/configs/improved_baseline.yaml

    # Using CLI arguments (legacy):
    python train.py --steps 100000 --lr 0.0001

    # Override specific parameters from config:
    python train.py --config python/configs/default.yaml --steps 200000
        """
    )

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (e.g., python/configs/default.yaml)",
    )

    # Legacy CLI arguments (for backward compatibility and overrides)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port of Bevy API server (overrides config)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save models (overrides config)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Replay buffer size (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = DQNConfig.from_yaml(args.config)
    else:
        print("Using default configuration (no config file specified)")
        config = DQNConfig()

    # Override config with CLI arguments if provided
    if args.steps is not None:
        config.total_timesteps = args.steps
    if args.port is not None:
        config.port = args.port
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    train(config)


if __name__ == "__main__":
    main()
