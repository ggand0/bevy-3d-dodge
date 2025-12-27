"""Vectorized environment utilities for parallel training.

Note: This module requires stable-baselines3 to be installed.
Install with: uv sync --extra train
"""

from typing import Callable, List, Optional, Dict, Any

import gymnasium as gym

try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    raise ImportError(
        "stable-baselines3 is required for vectorized environments. "
        "Install with: uv sync --extra train"
    ) from e

from bevy_dodge_env.environment import BevyDodgeEnv


def make_env(
    port: int,
    host: str = "127.0.0.1",
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[], gym.Env]:
    """Create a function that returns a configured BevyDodgeEnv instance.

    This is the format required by SubprocVecEnv.

    Args:
        port: Port number for this environment instance
        host: Host address (default: "127.0.0.1")
        config_kwargs: Optional dict of configuration to pass to env.configure()
            Example: {'level': 2, 'observation_mode': 'topdown', ...}

    Returns:
        Function that creates and returns a configured BevyDodgeEnv instance
    """
    def _init() -> gym.Env:
        env = BevyDodgeEnv(host=host, port=port)
        if config_kwargs:
            env.configure(**config_kwargs)
            env.refresh_spaces()  # Update observation/action spaces after config change
            env.reset()  # Apply configuration
            env.start_training()  # Disable R key reset in game
        return Monitor(env)
    return _init


def make_vec_env(
    n_envs: int,
    start_port: int = 8000,
    host: str = "127.0.0.1",
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> SubprocVecEnv:
    """Create a vectorized environment with multiple parallel Bevy instances.

    Each environment connects to a different Bevy instance on a different port.
    You must start n_envs Bevy instances on ports [start_port, start_port+n_envs).

    Example:
        # Start 4 game servers:
        # ./start_parallel_servers.sh 4

        # Python:
        vec_env = make_vec_env(
            n_envs=4,
            start_port=8000,
            config_kwargs={
                'level': 2,
                'observation_mode': 'topdown',
                'action_space_type': 'basic_3d',
            }
        )

    Args:
        n_envs: Number of parallel environments
        start_port: Starting port number (default: 8000)
        host: Host address for all instances (default: "127.0.0.1")
        config_kwargs: Optional dict of configuration to pass to each env

    Returns:
        SubprocVecEnv with n_envs parallel environments
    """
    env_fns: List[Callable[[], gym.Env]] = [
        make_env(port=start_port + i, host=host, config_kwargs=config_kwargs)
        for i in range(n_envs)
    ]

    return SubprocVecEnv(env_fns)
