#!/usr/bin/env python3
"""Train SAC agent on Bevy 3D dodge game.

SAC (Soft Actor-Critic) is an off-policy algorithm that:
- Uses a replay buffer for sample efficiency
- Has automatic entropy tuning for exploration
- Works well with continuous action spaces

Usage:
    python train_sac.py --config python/configs/sac_level2_basic3d.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import psutil
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from tensorboard.backend.event_processing import event_accumulator

from bevy_dodge_env import BevyDodgeEnv
from bevy_dodge_env.vec_env import make_vec_env
from config import TrainingConfig


def estimate_memory_usage(
    image_width: int = 256,
    image_height: int = 256,
    grayscale: bool = True,
    frame_stack: int = 4,
    buffer_size: int = 100000,
    n_envs: int = 1,
) -> dict:
    """Estimate memory usage for CNN training with image observations.

    Returns dict with memory estimates in bytes and formatted strings.
    """
    channels = 1 if grayscale else 3

    # Single observation size (after frame stacking)
    obs_size_bytes = image_width * image_height * channels * frame_stack

    # Replay buffer stores observations, next_observations, actions, rewards, dones
    # Each transition: obs + next_obs + action(3) + reward(1) + done(1) = ~2*obs + 20 bytes
    transition_size = 2 * obs_size_bytes + 20
    buffer_mem = buffer_size * transition_size

    # Parallel env overhead (each subprocess holds recent observations)
    env_mem = n_envs * obs_size_bytes * 10  # ~10 recent frames per env

    # Model overhead (CNN + MLP, rough estimate)
    # NatureCNN: ~2M params, plus actor/critic networks
    model_mem = 50 * 1024 * 1024  # ~50MB for model weights + gradients

    # Batch processing overhead
    batch_mem = 128 * obs_size_bytes * 2  # batch_size * obs * 2

    total_mem = buffer_mem + env_mem + model_mem + batch_mem

    def fmt(b: int) -> str:
        if b >= 1024**3:
            return f"{b / 1024**3:.1f} GB"
        elif b >= 1024**2:
            return f"{b / 1024**2:.1f} MB"
        else:
            return f"{b / 1024:.1f} KB"

    return {
        "obs_size_bytes": obs_size_bytes,
        "buffer_mem": buffer_mem,
        "env_mem": env_mem,
        "model_mem": model_mem,
        "total_mem": total_mem,
        "obs_size_str": fmt(obs_size_bytes),
        "buffer_mem_str": fmt(buffer_mem),
        "env_mem_str": fmt(env_mem),
        "model_mem_str": fmt(model_mem),
        "total_mem_str": fmt(total_mem),
    }


def check_memory_requirements(
    image_width: int = 256,
    image_height: int = 256,
    grayscale: bool = True,
    frame_stack: int = 4,
    buffer_size: int = 100000,
    n_envs: int = 1,
) -> bool:
    """Check if system has enough memory for training. Returns True if OK."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    est = estimate_memory_usage(
        image_width=image_width,
        image_height=image_height,
        grayscale=grayscale,
        frame_stack=frame_stack,
        buffer_size=buffer_size,
        n_envs=n_envs,
    )

    channels = 1 if grayscale else 3

    print("=" * 60)
    print("MEMORY ESTIMATION")
    print("=" * 60)
    print(f"System memory:       {available_gb:.1f} GB available / {total_gb:.1f} GB total")
    print()
    print(f"Image config:        {image_width}x{image_height}x{channels} ({'grayscale' if grayscale else 'RGB'})")
    print(f"Frame stack:         {frame_stack} frames")
    print(f"Observation size:    {est['obs_size_str']} per stacked observation")
    print()
    print(f"Replay buffer:       {buffer_size:,} transitions × {est['obs_size_str']} × 2 = {est['buffer_mem_str']}")
    print(f"Parallel envs:       {n_envs} × overhead = {est['env_mem_str']}")
    print(f"Model + gradients:   ~{est['model_mem_str']}")
    print()
    print(f"ESTIMATED TOTAL:     {est['total_mem_str']}")
    print("=" * 60)

    estimated_gb = est['total_mem'] / (1024**3)

    if estimated_gb > available_gb * 0.8:
        print(f"⚠️  WARNING: Estimated usage ({estimated_gb:.1f} GB) exceeds 80% of available memory!")
        print(f"   Consider reducing buffer_size or n_envs to prevent OOM.")
        print("=" * 60)
        return False
    else:
        print(f"✓ Memory check passed ({estimated_gb:.1f} GB estimated, {available_gb:.1f} GB available)")
        print("=" * 60)
        return True


class CleanupReplayBufferCallback(BaseCallback):
    """Callback that deletes old replay buffer files after checkpoints.

    Keeps only the latest replay buffer to prevent disk space exhaustion,
    especially important for CNN training with large image observations.
    """

    def __init__(self, checkpoint_path: Path, name_prefix: str = "sac_dodge", verbose: int = 0):
        super().__init__(verbose)
        self.checkpoint_path = checkpoint_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        # Check for replay buffer files
        buffer_files = sorted(
            self.checkpoint_path.glob(f"{self.name_prefix}_replay_buffer_*.pkl"),
            key=lambda p: self._get_steps_from_filename(p)
        )

        # Delete all but the latest replay buffer
        if len(buffer_files) > 1:
            for old_buffer in buffer_files[:-1]:
                try:
                    old_buffer.unlink()
                    if self.verbose > 0:
                        print(f"🗑 Deleted old replay buffer: {old_buffer.name}")
                except OSError as e:
                    if self.verbose > 0:
                        print(f"⚠ Failed to delete {old_buffer.name}: {e}")

        return True

    @staticmethod
    def _get_steps_from_filename(path: Path) -> int:
        """Extract step number from filename like 'sac_dodge_replay_buffer_50000_steps.pkl'."""
        try:
            # sac_dodge_replay_buffer_50000_steps.pkl -> 50000
            parts = path.stem.split("_")
            # Find the part that's a number before "steps"
            for i, part in enumerate(parts):
                if part == "steps" and i > 0:
                    return int(parts[i - 1])
            return 0
        except (IndexError, ValueError):
            return 0


def make_env(port: int = 8000, socket_path: str = "/tmp/bevy_rl.sock", transport: str = "grpc") -> gym.Env:
    """Create and wrap environment."""
    if transport == "grpc":
        env = BevyDodgeEnv(socket_path=socket_path, transport="grpc")
    else:
        env = BevyDodgeEnv(port=port, transport="http")
    env = Monitor(env)  # Wrap for logging
    return env


def load_tensorboard_data(logdir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Load data from TensorBoard event files."""
    run_dirs = sorted(logdir.glob("SAC_*"))
    if not run_dirs:
        return {}

    latest_run = run_dirs[-1]
    ea = event_accumulator.EventAccumulator(str(latest_run))
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        data[tag] = [(e.step, e.value) for e in events]

    return data


def plot_learning_curves(log_path: Path, output_dir: Path) -> None:
    """Plot learning curves after training completes."""
    print("\nGenerating learning curves...")

    data = load_tensorboard_data(log_path)
    if not data:
        print("⚠ No TensorBoard data found, skipping plots")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    # Combined learning curves plot (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Episode Reward
    if 'rollout/ep_rew_mean' in data:
        steps, rewards = zip(*data['rollout/ep_rew_mean'])
        axes[0, 0].plot(steps, rewards, linewidth=1.5, alpha=0.6, label='Raw')
        window = min(50, max(1, len(rewards) // 10))
        if window > 1 and len(rewards) >= window:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(steps[window-1:], rolling_mean, linewidth=2.5, color='red', label=f'Smoothed (w={window})')
        axes[0, 0].set_title('Episode Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode Length
    if 'rollout/ep_len_mean' in data:
        steps, lengths = zip(*data['rollout/ep_len_mean'])
        axes[0, 1].plot(steps, lengths, linewidth=1.5, alpha=0.6, label='Raw')
        window = min(50, max(1, len(lengths) // 10))
        if window > 1 and len(lengths) >= window:
            rolling_mean = np.convolve(lengths, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(steps[window-1:], rolling_mean, linewidth=2.5, color='green', label=f'Smoothed (w={window})')
        axes[0, 1].set_title('Episode Length', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Mean Length (steps)')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Eval Reward (from EvalCallback)
    if 'eval/mean_reward' in data:
        steps, rewards = zip(*data['eval/mean_reward'])
        axes[1, 0].plot(steps, rewards, 'o-', linewidth=2, markersize=6, color='blue')
        axes[1, 0].set_title('Evaluation Reward', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Mean Eval Reward')
        axes[1, 0].grid(True, alpha=0.3)
        # Mark best
        best_idx = np.argmax(rewards)
        axes[1, 0].scatter([steps[best_idx]], [rewards[best_idx]], color='gold', s=150, zorder=5, marker='*', label=f'Best: {rewards[best_idx]:.1f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No eval data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Evaluation Reward', fontsize=12, fontweight='bold')

    # 4. Eval Episode Length
    if 'eval/mean_ep_length' in data:
        steps, lengths = zip(*data['eval/mean_ep_length'])
        axes[1, 1].plot(steps, lengths, 'o-', linewidth=2, markersize=6, color='orange')
        axes[1, 1].set_title('Evaluation Episode Length', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('Mean Eval Length')
        axes[1, 1].grid(True, alpha=0.3)
        # Mark best
        best_idx = np.argmax(lengths)
        axes[1, 1].scatter([steps[best_idx]], [lengths[best_idx]], color='gold', s=150, zorder=5, marker='*', label=f'Best: {lengths[best_idx]:.0f}')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'No eval data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Evaluation Episode Length', fontsize=12, fontweight='bold')

    plt.suptitle('SAC Training - Learning Curves', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / 'learning_curves.png', dpi=150)
    print(f"✓ Saved: {plots_dir / 'learning_curves.png'}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)

    if 'rollout/ep_rew_mean' in data:
        rewards = [v for _, v in data['rollout/ep_rew_mean']]
        print(f"Final reward:    {rewards[-1]:.2f}")
        print(f"Peak reward:     {max(rewards):.2f}")

    if 'rollout/ep_len_mean' in data:
        lengths = [v for _, v in data['rollout/ep_len_mean']]
        print(f"Final ep length: {lengths[-1]:.0f}")
        print(f"Peak ep length:  {max(lengths):.0f}")

    if 'eval/mean_reward' in data:
        eval_rewards = [v for _, v in data['eval/mean_reward']]
        print(f"Best eval reward: {max(eval_rewards):.2f}")
        print(f"Final eval reward: {eval_rewards[-1]:.2f}")

    print("=" * 50)


def train(
    config: TrainingConfig,
    config_name: Optional[str] = None,
    verbose: int = 1,
    resume_path: Optional[str] = None,
) -> None:
    """Train SAC agent on Bevy dodge game.

    Args:
        config: Config instance with all hyperparameters
        config_name: Name of config file (used for organizing results)
        verbose: Verbosity level
        resume_path: Path to existing run directory to resume training from
    """
    # Handle resume vs new run
    if resume_path:
        run_dir = Path(resume_path)
        if not run_dir.exists():
            print(f"Error: Resume path not found: {resume_path}")
            return
        save_path = run_dir / "models"
        log_path = run_dir / "logs"
        print(f"Resuming training from: {run_dir}")
    else:
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine config name for directory structure
        if config_name:
            config_basename = Path(config_name).stem
            run_dir = Path("results") / config_basename / timestamp
        else:
            run_dir = Path("results") / "sac_cli" / timestamp

        # Create subdirectories for models and logs
        save_path = run_dir / "models"
        log_path = run_dir / "logs"
        save_path.mkdir(parents=True, exist_ok=True)
        log_path.mkdir(parents=True, exist_ok=True)

    # Get SAC-specific parameters with defaults
    buffer_size = getattr(config, 'buffer_size', 1_000_000)
    learning_starts = getattr(config, 'learning_starts', 10000)
    batch_size = getattr(config, 'batch_size', 256)
    tau = getattr(config, 'tau', 0.005)
    train_freq = getattr(config, 'train_freq', 1)
    gradient_steps = getattr(config, 'gradient_steps', 1)
    ent_coef = getattr(config, 'ent_coef', 'auto')  # SAC's automatic entropy tuning

    print("=" * 70)
    print("SAC Training - Bevy 3D Dodge Game")
    print("=" * 70)
    print(f"Run directory:       {run_dir}")
    print(f"Config:              {config_name if config_name else 'CLI arguments'}")
    if not resume_path:
        print(f"Timestamp:           {timestamp}")
    print()
    print(f"Total timesteps:     {config.total_timesteps:,}")
    print(f"Learning rate:       {config.learning_rate}")
    print(f"Buffer size:         {buffer_size:,}")
    print(f"Learning starts:     {learning_starts:,}")
    print(f"Batch size:          {batch_size}")
    print(f"Tau (soft update):   {tau}")
    print(f"Train freq:          {train_freq}")
    print(f"Gradient steps:      {gradient_steps}")
    print(f"Entropy coef:        {ent_coef}")
    print(f"Gamma:               {config.gamma}")
    print(f"Network arch:        {config.net_arch if config.net_arch else '[256, 256] (default)'}")
    print(f"Difficulty level:    {config.level} ({'Baseline' if config.level == 1 else 'Hard'})")
    print()
    print(f"Models saved to:     {save_path}")
    print(f"Logs saved to:       {log_path}")
    print()

    # Check memory requirements for image observations
    observation_mode = getattr(config, 'observation_mode', None)
    frame_stack = getattr(config, 'frame_stack', 1) or 1
    image_grayscale = getattr(config, 'image_grayscale', False) or False
    n_envs = getattr(config, 'n_envs', 1)

    if observation_mode == "topdown":
        check_memory_requirements(
            image_width=256,  # Default, matches config.rs IMAGE_OBS_WIDTH
            image_height=256,  # Default, matches config.rs IMAGE_OBS_HEIGHT
            grayscale=image_grayscale,
            frame_stack=frame_stack,
            buffer_size=buffer_size,
            n_envs=n_envs,
        )
        print()

    # First, create a temporary environment to configure the game
    transport = getattr(config, 'transport', 'grpc')
    socket_path = getattr(config, 'socket_path', '/tmp/bevy_rl.sock')

    # For parallel envs, connect to first socket (_0.sock)
    if n_envs > 1 and transport == "grpc":
        first_socket = f"{socket_path.replace('.sock', '')}_0.sock"
        print(f"Connecting to Bevy server via gRPC at {first_socket}")
        temp_env = BevyDodgeEnv(socket_path=first_socket, transport="grpc")
    elif transport == "grpc":
        print(f"Connecting to Bevy server via gRPC at {socket_path}")
        temp_env = BevyDodgeEnv(socket_path=socket_path, transport="grpc")
    else:
        print(f"Connecting to Bevy server via HTTP at http://127.0.0.1:{config.port}")
        temp_env = BevyDodgeEnv(port=config.port, transport="http")

    # Configure game settings (level, action space, and optional params)
    level_name = "Level 1 (Baseline)" if config.level == 1 else "Level 2 (Hard)"
    action_space_type = getattr(config, 'action_space_type', 'basic_3d')  # SAC needs continuous
    sprint_multiplier = getattr(config, 'sprint_multiplier', None)
    spawn_angle_degrees = getattr(config, 'spawn_angle_degrees', None)
    observation_mode = getattr(config, 'observation_mode', None)
    thrower_delay_seconds = getattr(config, 'thrower_delay_seconds', None)
    image_grayscale = getattr(config, 'image_grayscale', None)

    config_parts = [f"{level_name}", f"action space: {action_space_type}"]
    if sprint_multiplier is not None:
        config_parts.append(f"sprint: {sprint_multiplier} ({1+sprint_multiplier}x)")
    if spawn_angle_degrees is not None:
        config_parts.append(f"spawn angle: ±{spawn_angle_degrees}°")
    if observation_mode is not None:
        config_parts.append(f"obs: {observation_mode}")
    if thrower_delay_seconds is not None:
        config_parts.append(f"thrower delay: {thrower_delay_seconds}s")
    if image_grayscale is not None:
        config_parts.append(f"grayscale: {image_grayscale}")
    print(f"Configuring game: {', '.join(config_parts)}...")

    temp_env.configure(
        level=config.level,
        action_space_type=action_space_type,
        sprint_multiplier=sprint_multiplier,
        spawn_angle_degrees=spawn_angle_degrees,
        observation_mode=observation_mode,
        thrower_delay_seconds=thrower_delay_seconds,
        image_grayscale=image_grayscale,
    )
    print(f"✓ Game configured: {', '.join(config_parts)}")

    # Reset to ensure config is fully applied and synced to API server's shared state
    temp_env.reset()
    del temp_env  # Close temporary environment
    print()

    # Now create the actual training environment
    n_envs = getattr(config, 'n_envs', 1)

    if n_envs > 1:
        # Build config kwargs for parallel environments
        config_kwargs = {
            'level': config.level,
            'action_space_type': action_space_type,
            'observation_mode': observation_mode,
        }
        if sprint_multiplier is not None:
            config_kwargs['sprint_multiplier'] = sprint_multiplier
        if spawn_angle_degrees is not None:
            config_kwargs['spawn_angle_degrees'] = spawn_angle_degrees
        if thrower_delay_seconds is not None:
            config_kwargs['thrower_delay_seconds'] = thrower_delay_seconds
        if image_grayscale is not None:
            config_kwargs['image_grayscale'] = image_grayscale

        # Pre-configure ALL game servers before creating SubprocVecEnv
        # This ensures all servers have the same observation/action space
        if transport == "grpc":
            print(f"Configuring {n_envs} game servers on sockets {socket_path.replace('.sock', '')}_0.sock to _{n_envs-1}.sock...")
            for i in range(n_envs):
                sock = f"{socket_path.replace('.sock', '')}_{i}.sock"
                pre_env = BevyDodgeEnv(socket_path=sock, transport="grpc")
                pre_env.configure(**config_kwargs)
                pre_env.reset()
                pre_env.close()
                print(f"  ✓ {sock} configured")
        else:
            print(f"Configuring {n_envs} game servers on ports {config.port}-{config.port + n_envs - 1}...")
            for i in range(n_envs):
                port = config.port + i
                pre_env = BevyDodgeEnv(port=port, transport="http")
                pre_env.configure(**config_kwargs)
                pre_env.reset()
                pre_env.close()
                print(f"  ✓ Port {port} configured")

        print(f"Creating {n_envs} parallel environments...")
        env = make_vec_env(
            n_envs=n_envs,
            start_port=config.port,
            socket_base=socket_path.replace('.sock', ''),
            transport=transport,
            config_kwargs=config_kwargs,
        )
    else:
        print("Creating training environment with configured action space...")
        env = DummyVecEnv([lambda: make_env(config.port, socket_path, transport)])

    # Detect if using image observations
    is_image_obs = observation_mode == "topdown"
    policy_type = "CnnPolicy" if is_image_obs else "MlpPolicy"

    # Apply frame stacking for image observations (helps CNN infer velocity)
    frame_stack = getattr(config, 'frame_stack', None)
    if frame_stack and frame_stack > 1 and is_image_obs:
        print(f"Applying frame stacking: {frame_stack} frames")
        env = VecFrameStack(env, n_stack=frame_stack)

    print(f"✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Policy type: {policy_type}")
    if frame_stack and frame_stack > 1:
        print(f"  Frame stack: {frame_stack}")
    print()

    # Enable training mode to prevent accidental keyboard interruptions
    # (For parallel envs, start_training is called in vec_env.py during init)
    if n_envs == 1:
        print("Enabling training mode...")
        env.envs[0].unwrapped.start_training()
        print("✓ Training mode enabled - R key disabled, camera controls still available")
    else:
        print(f"✓ Training mode enabled for {n_envs} parallel environments")
    print()

    # Create evaluation environment (use first socket for parallel envs)
    eval_socket = f"{socket_path.replace('.sock', '')}_0.sock" if n_envs > 1 else socket_path
    eval_env = DummyVecEnv([lambda: make_env(config.port, eval_socket, transport)])
    # Wrap eval env with same settings as training env
    if frame_stack and frame_stack > 1 and is_image_obs:
        eval_env = VecFrameStack(eval_env, n_stack=frame_stack)
    if is_image_obs:
        eval_env = VecTransposeImage(eval_env)

    # Create or load SAC agent
    if resume_path:
        resume_path_obj = Path(resume_path)

        # Check if resume_path is a direct path to a .zip file
        if resume_path_obj.suffix == ".zip" and resume_path_obj.exists():
            model_to_load = resume_path_obj
        else:
            # It's a run directory - find model to load
            checkpoint_dir = save_path / "checkpoints"
            final_model = save_path / "final_model.zip"

            model_to_load = None
            # Prioritize final_model (represents completed training), fall back to checkpoints
            if final_model.exists():
                model_to_load = final_model
            elif checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("sac_dodge_*.zip"))
                if checkpoints:
                    # Sort by step number numerically (extract from filename)
                    def get_steps(p: Path) -> int:
                        # sac_dodge_125379_steps.zip -> 125379
                        try:
                            return int(p.stem.split("_")[2])
                        except (IndexError, ValueError):
                            return 0
                    checkpoints.sort(key=get_steps)
                    model_to_load = checkpoints[-1]

        if model_to_load is None:
            print("Error: No model found to resume from")
            return

        print(f"Loading model from: {model_to_load}")
        model = SAC.load(model_to_load, env=env, tensorboard_log=str(log_path))
        print(f"✓ Model loaded successfully")
    else:
        print("Creating SAC agent...")

        # Build policy kwargs if custom network architecture is specified
        policy_kwargs: Optional[Dict[str, Any]] = None
        if config.net_arch is not None:
            policy_kwargs = {"net_arch": config.net_arch}

        model = SAC(
            policy=policy_type,
            env=env,
            learning_rate=config.learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=config.gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_path),
            verbose=verbose,
            device="auto",
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
        name_prefix="sac_dodge",
        save_replay_buffer=True,  # SAC uses replay buffer - save it for resume
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

    # Cleanup callback to delete old replay buffers (saves disk space for CNN training)
    cleanup_callback = CleanupReplayBufferCallback(
        checkpoint_path=save_path / "checkpoints",
        name_prefix="sac_dodge",
        verbose=1,
    )

    callbacks = [checkpoint_callback, eval_callback, cleanup_callback]

    # Save config to run directory for reproducibility (only on new runs)
    if not resume_path:
        config.to_yaml(str(run_dir / "config.yaml"))
        print(f"✓ Config saved to {run_dir / 'config.yaml'}")

    # Train
    print("\nStarting training...")
    print(f"Monitor with: tensorboard --logdir {log_path}")
    if resume_path:
        print("Note: Resuming from checkpoint, total_timesteps is ADDITIONAL steps to train")
    print()

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(resume_path),
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

    # Plot learning curves
    plot_learning_curves(log_path, run_dir)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train SAC agent on Bevy dodge game",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using YAML config (recommended):
    python train_sac.py --config python/configs/sac_level2_basic3d.yaml

    # Override specific parameters:
    python train_sac.py --config python/configs/sac_level2_basic3d.yaml --steps 500000

    # Resume training from a previous run:
    python train_sac.py --config python/configs/sac_level2_basic3d.yaml --resume results/sac_level2_basic3d/20251209_120000 --steps 500000
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing run directory to resume training from",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
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
        print("Example: python train_sac.py --config python/configs/sac_level2_basic3d.yaml")
        return

    # Override config with CLI arguments if provided
    if args.steps is not None:
        config.total_timesteps = args.steps
    if args.port is not None:
        config.port = args.port
    if args.n_envs is not None:
        config.n_envs = args.n_envs

    train(config, config_name=config_name, resume_path=args.resume)


if __name__ == "__main__":
    main()
