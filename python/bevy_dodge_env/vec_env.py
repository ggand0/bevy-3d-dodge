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
    port: Optional[int] = None,
    socket_path: Optional[str] = None,
    host: str = "127.0.0.1",
    transport: str = "grpc",
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[], gym.Env]:
    """Create a function that returns a configured BevyDodgeEnv instance.

    This is the format required by SubprocVecEnv.

    Args:
        port: Port number for HTTP transport (ignored if transport="grpc")
        socket_path: Unix socket path for gRPC transport (required if transport="grpc")
        host: Host address for HTTP transport (default: "127.0.0.1")
        transport: Transport type - "grpc" (default) or "http"
        config_kwargs: Optional dict of configuration to pass to env.configure()
            Example: {'level': 2, 'observation_mode': 'topdown', ...}

    Returns:
        Function that creates and returns a configured BevyDodgeEnv instance
    """
    def _init() -> gym.Env:
        if transport == "grpc":
            env = BevyDodgeEnv(socket_path=socket_path, transport="grpc")
        else:
            env = BevyDodgeEnv(host=host, port=port, transport="http")
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
    socket_base: str = "/tmp/bevy_rl",
    host: str = "127.0.0.1",
    transport: str = "grpc",
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> SubprocVecEnv:
    """Create a vectorized environment with multiple parallel Bevy instances.

    For gRPC (default): Each env connects to a different socket file.
    For HTTP: Each env connects to a different port.

    Example (gRPC):
        # Start 4 game servers with different sockets:
        # cargo run --release -- --headless --socket-path /tmp/bevy_rl_0.sock
        # cargo run --release -- --headless --socket-path /tmp/bevy_rl_1.sock
        # ...

        vec_env = make_vec_env(
            n_envs=4,
            socket_base="/tmp/bevy_rl",  # Creates /tmp/bevy_rl_0.sock, etc.
            config_kwargs={'level': 2, 'observation_mode': 'topdown'}
        )

    Example (HTTP):
        # Start 4 game servers on different ports:
        # ./start_parallel_servers.sh 4

        vec_env = make_vec_env(
            n_envs=4,
            start_port=8000,
            transport="http",
            config_kwargs={'level': 2, 'observation_mode': 'topdown'}
        )

    Args:
        n_envs: Number of parallel environments
        start_port: Starting port number for HTTP (default: 8000)
        socket_base: Base path for gRPC sockets (default: "/tmp/bevy_rl")
            Socket files will be named {socket_base}_{i}.sock
        host: Host address for HTTP (default: "127.0.0.1")
        transport: Transport type - "grpc" (default) or "http"
        config_kwargs: Optional dict of configuration to pass to each env

    Returns:
        SubprocVecEnv with n_envs parallel environments
    """
    if transport == "grpc":
        env_fns: List[Callable[[], gym.Env]] = [
            make_env(
                socket_path=f"{socket_base}_{i}.sock",
                transport="grpc",
                config_kwargs=config_kwargs,
            )
            for i in range(n_envs)
        ]
    else:
        env_fns = [
            make_env(
                port=start_port + i,
                host=host,
                transport="http",
                config_kwargs=config_kwargs,
            )
            for i in range(n_envs)
        ]

    return SubprocVecEnv(env_fns)
