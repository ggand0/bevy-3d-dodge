#!/usr/bin/env python3
"""Train PPO agent on Bevy 3D dodge game.

Usage:
    python train_ppo.py --config python/configs/ppo_baseline.yaml
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from bevy_dodge_env import BevyDodgeEnv
from config import TrainingConfig


def make_env(port: int) -> gym.Env:
    """Create and wrap environment."""
    env = BevyDodgeEnv(port=port)
    env = Monitor(env)  # Wrap for logging
    return env


def train(config: TrainingConfig, config_name: Optional[str] = None, verbose: int = 1) -> None:
    """Train PPO agent on Bevy dodge game.

    Args:
        config: Config instance with all hyperparameters
        config_name: Name of config file (used for organizing results)
        verbose: Verbosity level
    """
    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine config name for directory structure
    if config_name:
        config_basename = Path(config_name).stem
        run_dir = Path("results") / config_basename / timestamp
    else:
        run_dir = Path("results") / "ppo_cli" / timestamp

    # Create subdirectories for models and logs
    save_path = run_dir / "models"
    log_path = run_dir / "logs"
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PPO Training - Bevy 3D Dodge Game")
    print("=" * 70)
    print(f"Run directory:       {run_dir}")
    print(f"Config:              {config_name if config_name else 'CLI arguments'}")
    print(f"Timestamp:           {timestamp}")
    print()
    print(f"Total timesteps:     {config.total_timesteps:,}")
    print(f"Learning rate:       {config.learning_rate}")
    print(f"Batch size:          {config.batch_size}")
    print(f"N steps:             {config.n_steps}")
    print(f"N epochs:            {config.n_epochs}")
    print(f"Gamma:               {config.gamma}")
    print(f"GAE lambda:          {config.gae_lambda}")
    print(f"Clip range:          {config.clip_range}")
    print(f"Network arch:        {config.net_arch if config.net_arch else '[64, 64] (default)'}")
    print(f"Difficulty level:    {config.level} ({'Baseline' if config.level == 1 else 'Hard'})")
    print()
    print(f"Models saved to:     {save_path}")
    print(f"Logs saved to:       {log_path}")
    print()

    # First, create a temporary environment to configure the game
    print(f"Connecting to Bevy server at http://127.0.0.1:{config.port}")
    temp_env = BevyDodgeEnv(port=config.port)

    # Configure game settings (level, action space, and optional params)
    level_name = "Level 1 (Baseline)" if config.level == 1 else "Level 2 (Hard)"
    action_space_type = getattr(config, 'action_space_type', 'discrete')
    sprint_multiplier = getattr(config, 'sprint_multiplier', None)
    spawn_angle_degrees = getattr(config, 'spawn_angle_degrees', None)

    config_parts = [f"{level_name}", f"action space: {action_space_type}"]
    if sprint_multiplier is not None:
        config_parts.append(f"sprint: {sprint_multiplier} ({1+sprint_multiplier}x)")
    if spawn_angle_degrees is not None:
        config_parts.append(f"spawn angle: ±{spawn_angle_degrees}°")
    print(f"Configuring game: {', '.join(config_parts)}...")

    temp_env.configure(
        level=config.level,
        action_space_type=action_space_type,
        sprint_multiplier=sprint_multiplier,
        spawn_angle_degrees=spawn_angle_degrees,
    )
    print(f"✓ Game configured: {', '.join(config_parts)}")

    # Reset to ensure config is fully applied and synced to API server's shared state
    # This ensures the next environment creation will query the correct action space
    temp_env.reset()
    del temp_env  # Close temporary environment
    print()

    # Now create the actual training environment (will query the updated action space)
    print("Creating training environment with configured action space...")
    env = DummyVecEnv([lambda: make_env(config.port)])
    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()

    # Enable training mode to prevent accidental keyboard interruptions
    print("Enabling training mode...")
    env.envs[0].unwrapped.start_training()
    print("✓ Training mode enabled - R key disabled, camera controls still available")
    print()

    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(config.port)])

    # Create PPO agent
    print("Creating PPO agent...")

    # Build policy kwargs if custom network architecture is specified
    policy_kwargs: Optional[Dict[str, Any]] = None
    if config.net_arch is not None:
        policy_kwargs = {"net_arch": config.net_arch}

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
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
        name_prefix="ppo_dodge",
        save_replay_buffer=False,  # PPO doesn't use replay buffer
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

    # Save config to run directory for reproducibility
    config.to_yaml(str(run_dir / "config.yaml"))
    print(f"✓ Config saved to {run_dir / 'config.yaml'}")

    # Train
    print("\nStarting training...")
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

        # Disable training mode
        print("\nDisabling training mode...")
        try:
            env.envs[0].unwrapped.end_training()
            print("✓ Training mode disabled - returning to human control")
        except Exception as e:
            print(f"⚠ Failed to disable training mode: {e}")

    # Close environments
    env.close()
    eval_env.close()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on Bevy dodge game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using YAML config (recommended):
    python train_ppo.py --config python/configs/ppo_baseline.yaml

    # Override specific parameters:
    python train_ppo.py --config python/configs/ppo_baseline.yaml --steps 500000
        """
    )

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # CLI argument overrides
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

    args = parser.parse_args()

    # Load configuration
    config_name = None
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = TrainingConfig.from_yaml(args.config)
        config_name = args.config
    else:
        print("Error: --config is required")
        print("Example: python train_ppo.py --config python/configs/ppo_baseline.yaml")
        return

    # Override config with CLI arguments if provided
    if args.steps is not None:
        config.total_timesteps = args.steps
    if args.port is not None:
        config.port = args.port

    train(config, config_name=config_name)


if __name__ == "__main__":
    main()
