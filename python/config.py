"""Training configuration and hyperparameters."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List
import yaml


@dataclass
class TrainingConfig:
    """Training hyperparameters for RL algorithms (DQN, PPO, etc.)."""

    # Algorithm
    algorithm: str = "DQN"  # "DQN" or "PPO"

    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 1e-4
    batch_size: int = 32
    gamma: float = 0.99

    # DQN-specific
    buffer_size: Optional[int] = 50_000
    learning_starts: Optional[int] = 1000
    target_update_interval: Optional[int] = 1000
    exploration_fraction: Optional[float] = 0.3
    exploration_initial_eps: Optional[float] = 1.0
    exploration_final_eps: Optional[float] = 0.05

    # PPO-specific
    n_steps: Optional[int] = 2048
    n_epochs: Optional[int] = 10
    gae_lambda: Optional[float] = 0.95
    clip_range: Optional[float] = 0.2
    ent_coef: Optional[float] = 0.01
    vf_coef: Optional[float] = 0.5
    max_grad_norm: Optional[float] = 0.5

    # Network architecture (MlpPolicy default: [64, 64])
    # Can override with policy_kwargs=dict(net_arch=[256, 256])
    net_arch: Optional[List[int]] = None

    # Logging and checkpointing
    eval_freq: int = 5000
    save_freq: int = 10000
    n_eval_episodes: int = 5

    # Environment
    port: int = 8000
    max_episode_steps: int = 1000
    level: int = 1  # Difficulty level (1 for baseline, 2 for hard)
    action_space_type: str = "discrete"  # Action space type ("discrete", "basic_3d", etc.)
    sprint_multiplier: Optional[float] = None  # Sprint speed multiplier (e.g., 2.0 = 3x speed). None = use level default
    spawn_angle_degrees: Optional[float] = None  # Half-angle for spawn fan (e.g., 30 = ±30°). None = use level default
    observation_mode: Optional[str] = None  # "standard" (65-dim) or "with_thrower" (69-dim). None = use level default
    thrower_delay_seconds: Optional[float] = None  # Delay before thrower spawns projectile. None = use level default

    # Frame stacking for image observations (helps CNN infer velocity)
    frame_stack: Optional[int] = None  # Number of frames to stack (e.g., 4). None = no stacking

    # Paths
    save_dir: str = "models"
    log_dir: str = "logs"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            TrainingConfig instance with values from YAML
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Create config with YAML values, using defaults for missing keys
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where YAML file will be saved
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


# Quick training config for testing
@dataclass
class QuickTestConfig(TrainingConfig):
    """Quick test configuration with reduced timesteps."""

    total_timesteps: int = 10_000
    buffer_size: int = 10_000
    eval_freq: int = 2000
    save_freq: int = 5000


# Long training config for serious runs
@dataclass
class LongTrainingConfig(TrainingConfig):
    """Long training configuration."""

    total_timesteps: int = 500_000
    buffer_size: int = 100_000
    learning_rate: float = 5e-5
    eval_freq: int = 10_000
    save_freq: int = 25_000
