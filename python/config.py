"""Training configuration and hyperparameters."""

from dataclasses import dataclass


@dataclass
class DQNConfig:
    """DQN training hyperparameters."""

    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    learning_starts: int = 1000
    batch_size: int = 32
    gamma: float = 0.99
    target_update_interval: int = 1000

    # Exploration
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    # Network architecture (MlpPolicy default: [64, 64])
    # Can override with policy_kwargs=dict(net_arch=[256, 256])

    # Logging and checkpointing
    eval_freq: int = 5000
    save_freq: int = 10000
    n_eval_episodes: int = 5

    # Environment
    port: int = 8000
    max_episode_steps: int = 1000

    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"


# Quick training config for testing
@dataclass
class QuickTestConfig(DQNConfig):
    """Quick test configuration with reduced timesteps."""

    total_timesteps: int = 10_000
    buffer_size: int = 10_000
    eval_freq: int = 2000
    save_freq: int = 5000


# Long training config for serious runs
@dataclass
class LongTrainingConfig(DQNConfig):
    """Long training configuration."""

    total_timesteps: int = 500_000
    buffer_size: int = 100_000
    learning_rate: float = 5e-5
    eval_freq: int = 10_000
    save_freq: int = 25_000
