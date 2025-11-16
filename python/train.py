#!/usr/bin/env python3
"""Train DQN agent on Bevy 3D dodge game.

Usage:
    python train.py [--steps 100000] [--port 8000] [--save-dir models/]
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from bevy_dodge_env import BevyDodgeEnv


def make_env(port: int) -> gym.Env:
    """Create and wrap environment."""
    env = BevyDodgeEnv(port=port)
    env = Monitor(env)  # Wrap for logging
    return env


def train(
    total_timesteps: int = 100_000,
    port: int = 8000,
    save_dir: str = "models",
    log_dir: str = "logs",
    eval_freq: int = 5000,
    save_freq: int = 10000,
    learning_rate: float = 1e-4,
    buffer_size: int = 50_000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    gamma: float = 0.99,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.3,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    verbose: int = 1,
) -> None:
    """Train DQN agent on Bevy dodge game.

    Args:
        total_timesteps: Total number of training steps
        port: Port of Bevy API server
        save_dir: Directory to save model checkpoints
        log_dir: Directory for TensorBoard logs
        eval_freq: Evaluate every N steps
        save_freq: Save checkpoint every N steps
        learning_rate: Learning rate for Adam optimizer
        buffer_size: Replay buffer size
        learning_starts: Start training after N steps
        batch_size: Minibatch size for training
        gamma: Discount factor
        target_update_interval: Update target network every N steps
        exploration_fraction: Fraction of training for exploration decay
        exploration_initial_eps: Initial exploration rate
        exploration_final_eps: Final exploration rate
        verbose: Verbosity level
    """
    # Create directories
    save_path = Path(save_dir)
    log_path = Path(log_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DQN Training - Bevy 3D Dodge Game")
    print("=" * 70)
    print(f"Total timesteps:     {total_timesteps:,}")
    print(f"Learning rate:       {learning_rate}")
    print(f"Buffer size:         {buffer_size:,}")
    print(f"Batch size:          {batch_size}")
    print(f"Gamma:               {gamma}")
    print(f"Exploration:         {exploration_initial_eps} → {exploration_final_eps}")
    print(f"Save directory:      {save_path}")
    print(f"Log directory:       {log_path}")
    print()

    # Create environment
    print(f"Connecting to Bevy server at http://127.0.0.1:{port}")
    env = DummyVecEnv([lambda: make_env(port)])
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(port)])

    # Create DQN agent
    print("Creating DQN agent...")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
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
        save_freq=save_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix="dqn_dodge",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(log_path / "eval"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Train
    print("Starting training...")
    print(f"Monitor with: tensorboard --logdir {log_path}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
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
    parser = argparse.ArgumentParser(description="Train DQN agent on Bevy dodge game")
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100,000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of Bevy API server (default: 8000)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models/)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for TensorBoard logs (default: logs/)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50_000,
        help="Replay buffer size (default: 50,000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )

    args = parser.parse_args()

    train(
        total_timesteps=args.steps,
        port=args.port,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
