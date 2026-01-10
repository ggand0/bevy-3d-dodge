"""gRPC client for Bevy RL environment.

This module provides a low-level gRPC client for communicating with the
Bevy game server using Unix domain sockets for maximum performance.
"""

import json
from typing import Any, Dict, Optional, Tuple

import grpc
import numpy as np

from . import rl_env_pb2
from . import rl_env_pb2_grpc


class GrpcEnvClient:
    """Low-level gRPC client for communicating with Bevy game server.

    This client connects via Unix domain socket or TCP for gRPC communication.
    Image observations are sent as raw bytes (no Base64 encoding).

    Args:
        socket_path: Path to the Unix domain socket (default: /tmp/bevy_rl.sock)
        host: Host for TCP connection (if using TCP instead of Unix socket)
        port: Port for TCP connection (if using TCP instead of Unix socket)
        timeout: Request timeout in seconds (default: 30.0)
    """

    def __init__(
        self,
        socket_path: Optional[str] = "/tmp/bevy_rl.sock",
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
    ) -> None:
        self.timeout = timeout

        # Create channel based on connection type
        if host is not None and port is not None:
            # TCP connection
            self.channel = grpc.insecure_channel(f"{host}:{port}")
            self._connection_str = f"{host}:{port}"
        else:
            # Unix domain socket
            # Note: grpc.default_authority is needed to work around percent-encoding
            # issues in grpc-python 1.57+ with Unix sockets
            self.socket_path = socket_path
            self.channel = grpc.insecure_channel(
                f"unix://{socket_path}",
                options=[("grpc.default_authority", "localhost")],
            )
            self._connection_str = f"unix://{socket_path}"

        self.stub = rl_env_pb2_grpc.RlEnvironmentStub(self.channel)

        # Cache for observation/action space info
        self._obs_space: Optional[Dict[str, Any]] = None
        self._action_space: Optional[Dict[str, Any]] = None
        self._is_image_obs: Optional[bool] = None

    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space metadata.

        Returns:
            Dictionary with shape, dtype, low, high keys.
        """
        if self._obs_space is None:
            response = self.stub.GetObservationSpace(
                rl_env_pb2.ObservationSpaceRequest(),
                timeout=self.timeout,
            )
            self._obs_space = {
                "shape": list(response.shape),
                "dtype": response.dtype,
                "low": response.low,
                "high": response.high,
            }
            self._is_image_obs = response.dtype == "uint8"
        return self._obs_space

    def get_action_space(self) -> Dict[str, Any]:
        """Get action space metadata.

        Returns:
            Dictionary with type and either n (discrete) or shape/low/high (box).
        """
        if self._action_space is None:
            response = self.stub.GetActionSpace(
                rl_env_pb2.ActionSpaceRequest(),
                timeout=self.timeout,
            )
            if response.HasField("discrete"):
                self._action_space = {
                    "type": "Discrete",
                    "n": response.discrete.n,
                }
            elif response.HasField("box"):
                self._action_space = {
                    "type": "Box",
                    "shape": list(response.box.shape),
                    "low": response.box.low,
                    "high": response.box.high,
                }
            else:
                raise ValueError("Unknown action space type")
        return self._action_space

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation.

        Returns:
            Tuple of (observation, info).
        """
        response = self.stub.Reset(
            rl_env_pb2.ResetRequest(),
            timeout=self.timeout,
        )

        # Ensure we have observation space info
        self.get_observation_space()

        if self._is_image_obs:
            # Direct bytes - no Base64 decoding needed!
            obs_shape = tuple(self._obs_space["shape"])
            observation = np.frombuffer(
                response.image_observation, dtype=np.uint8
            ).reshape(obs_shape)
        else:
            observation = np.array(response.observation, dtype=np.float32)

        # Parse info (values are JSON strings)
        info = {k: json.loads(v) for k, v in response.info.items()}

        return observation, info

    def step(
        self, action: Any
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step.

        Args:
            action: Action to take (int for discrete, array for continuous).

        Returns:
            Tuple of (observation, reward, done, truncated, info).
        """
        # Build request based on action type
        request = rl_env_pb2.StepRequest()

        action_space = self.get_action_space()
        if action_space["type"] == "Discrete":
            request.discrete_action = int(action)
        else:
            values = action.tolist() if hasattr(action, "tolist") else list(action)
            request.continuous_action.values.extend(values)

        response = self.stub.Step(request, timeout=self.timeout)

        if self._is_image_obs:
            obs_shape = tuple(self._obs_space["shape"])
            observation = np.frombuffer(
                response.image_observation, dtype=np.uint8
            ).reshape(obs_shape)
        else:
            observation = np.array(response.observation, dtype=np.float32)

        # Parse info (values are JSON strings)
        info = {k: json.loads(v) for k, v in response.info.items()}

        return observation, response.reward, response.done, response.truncated, info

    def configure(
        self,
        level: Optional[int] = None,
        action_space_type: Optional[str] = None,
        sprint_multiplier: Optional[float] = None,
        spawn_angle_degrees: Optional[float] = None,
        observation_mode: Optional[str] = None,
        thrower_delay_seconds: Optional[float] = None,
        image_obs_width: Optional[int] = None,
        image_obs_height: Optional[int] = None,
        image_grayscale: Optional[bool] = None,
        # Reward parameters
        collision_penalty: Optional[float] = None,
        survival_reward: Optional[float] = None,
        dodge_bonus_threshold: Optional[float] = None,
        dodge_bonus_multiplier: Optional[float] = None,
        # Level parameters
        projectile_speed: Optional[float] = None,
        projectile_spawn_interval: Optional[float] = None,
        max_projectiles: Optional[int] = None,
        player_speed: Optional[float] = None,
    ) -> None:
        """Configure environment settings.

        Args:
            level: Difficulty level (1 or 2)
            action_space_type: Action space type (discrete, basic_3d, etc.)
            sprint_multiplier: Sprint speed multiplier
            spawn_angle_degrees: Spawn angle in degrees
            observation_mode: Observation mode (standard, with_thrower, topdown)
            thrower_delay_seconds: Delay before thrower spawns projectile
            image_obs_width: Image observation width
            image_obs_height: Image observation height
            image_grayscale: Whether to use grayscale images
            collision_penalty: Death penalty (default: -100.0)
            survival_reward: Per-step survival reward (default: 1.0)
            dodge_bonus_threshold: Distance threshold for dodge bonus (default: 2.0)
            dodge_bonus_multiplier: Multiplier for dodge bonus (default: 0.5)
            projectile_speed: Speed of projectiles
            projectile_spawn_interval: Time between projectile spawns
            max_projectiles: Maximum number of projectiles
            player_speed: Player movement speed
        """
        request = rl_env_pb2.ConfigureRequest()

        if level is not None:
            request.level = level
        if action_space_type is not None:
            request.action_space_type = action_space_type
        if sprint_multiplier is not None:
            request.sprint_multiplier = sprint_multiplier
        if spawn_angle_degrees is not None:
            request.spawn_angle_degrees = spawn_angle_degrees
        if observation_mode is not None:
            request.observation_mode = observation_mode
        if thrower_delay_seconds is not None:
            request.thrower_delay_seconds = thrower_delay_seconds
        if image_obs_width is not None:
            request.image_obs_width = image_obs_width
        if image_obs_height is not None:
            request.image_obs_height = image_obs_height
        if image_grayscale is not None:
            request.image_grayscale = image_grayscale
        # Reward parameters
        if collision_penalty is not None:
            request.collision_penalty = collision_penalty
        if survival_reward is not None:
            request.survival_reward = survival_reward
        if dodge_bonus_threshold is not None:
            request.dodge_bonus_threshold = dodge_bonus_threshold
        if dodge_bonus_multiplier is not None:
            request.dodge_bonus_multiplier = dodge_bonus_multiplier
        # Level parameters
        if projectile_speed is not None:
            request.projectile_speed = projectile_speed
        if projectile_spawn_interval is not None:
            request.projectile_spawn_interval = projectile_spawn_interval
        if max_projectiles is not None:
            request.max_projectiles = max_projectiles
        if player_speed is not None:
            request.player_speed = player_speed

        self.stub.Configure(request, timeout=self.timeout)

        # Clear cached spaces - they may have changed
        self._obs_space = None
        self._action_space = None
        self._is_image_obs = None

    def set_level(self, level: int) -> None:
        """Set difficulty level.

        Args:
            level: Level number (1 or 2)
        """
        self.stub.SetLevel(
            rl_env_pb2.SetLevelRequest(level=level),
            timeout=self.timeout,
        )

    def start_training(self) -> None:
        """Enable training mode (disables keyboard controls)."""
        self.stub.StartTraining(
            rl_env_pb2.StartTrainingRequest(),
            timeout=self.timeout,
        )

    def end_training(self) -> None:
        """Disable training mode."""
        self.stub.EndTraining(
            rl_env_pb2.EndTrainingRequest(),
            timeout=self.timeout,
        )

    def close(self) -> None:
        """Close the gRPC channel."""
        try:
            self.end_training()
        except Exception:
            pass
        self.channel.close()
