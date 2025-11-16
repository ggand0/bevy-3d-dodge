"""Vectorized environment utilities for parallel training.

Note: This module requires stable-baselines3 to be installed.
Install with: uv sync --extra train
"""

from typing import Callable, List

import gymnasium as gym

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
except ImportError as e:
    raise ImportError(
        "stable-baselines3 is required for vectorized environments. "
        "Install with: uv sync --extra train"
    ) from e

from bevy_dodge_env.environment import BevyDodgeEnv


def make_env(port: int, host: str = "127.0.0.1") -> Callable[[], gym.Env]:
    """Create a function that returns a BevyDodgeEnv instance.

    This is the format required by SubprocVecEnv.

    Args:
        port: Port number for this environment instance
        host: Host address (default: "127.0.0.1")

    Returns:
        Function that creates and returns a BevyDodgeEnv instance
    """
    def _init() -> gym.Env:
        return BevyDodgeEnv(host=host, port=port)
    return _init


def make_vec_env(
    n_envs: int,
    start_port: int = 8000,
    host: str = "127.0.0.1",
) -> SubprocVecEnv:
    """Create a vectorized environment with multiple parallel Bevy instances.

    Each environment connects to a different Bevy instance on a different port.
    You must start n_envs Bevy instances on ports [start_port, start_port+n_envs).

    Example:
        # Terminal 1: cargo run -- --port 8000
        # Terminal 2: cargo run -- --port 8001
        # Terminal 3: cargo run -- --port 8002
        # Terminal 4: cargo run -- --port 8003

        # Python:
        vec_env = make_vec_env(n_envs=4, start_port=8000)

    Args:
        n_envs: Number of parallel environments
        start_port: Starting port number (default: 8000)
        host: Host address for all instances (default: "127.0.0.1")

    Returns:
        SubprocVecEnv with n_envs parallel environments
    """
    env_fns: List[Callable[[], gym.Env]] = [
        make_env(port=start_port + i, host=host)
        for i in range(n_envs)
    ]

    return SubprocVecEnv(env_fns)
