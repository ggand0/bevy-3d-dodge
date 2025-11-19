"""Gymnasium environment for Bevy 3D dodge game."""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces


class BevyDodgeEnv(gym.Env):
    """Gymnasium environment wrapper for Bevy 3D dodge game.

    Connects to a running Bevy game instance via HTTP REST API.

    Args:
        host: Host address of the Bevy API server (default: "127.0.0.1")
        port: Port of the Bevy API server (default: 8000)
        timeout: Request timeout in seconds (default: 5.0)
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        timeout: float = 5.0,
    ) -> None:
        super().__init__()

        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

        # Fetch space specifications from Bevy API
        try:
            obs_space_info = self._get(f"{self.base_url}/observation_space")
            action_space_info = self._get(f"{self.base_url}/action_space")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Bevy server at {self.base_url}. "
                f"Make sure the game is running. Error: {e}"
            ) from e

        # Define observation space
        self.observation_space = spaces.Box(
            low=obs_space_info["low"],
            high=obs_space_info["high"],
            shape=tuple(obs_space_info["shape"]),
            dtype=np.float32,
        )

        # Define action space
        if action_space_info["type"] == "Discrete":
            self.action_space = spaces.Discrete(action_space_info["n"])
        elif action_space_info["type"] == "Box":
            self.action_space = spaces.Box(
                low=action_space_info["low"],
                high=action_space_info["high"],
                shape=tuple(action_space_info["shape"]),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported action space type: {action_space_info['type']}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed (currently unused by Bevy backend)
            options: Additional options (currently unused)

        Returns:
            observation: Initial observation as numpy array
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        try:
            response = self._post(f"{self.base_url}/reset", {})
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to reset environment: {e}") from e

        observation = np.array(response["observation"], dtype=np.float32)
        info = response["info"]

        return observation, info

    def step(
        self,
        action,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with the given action.

        Args:
            action: Action (int for discrete action space, np.ndarray for continuous)

        Returns:
            observation: New observation as numpy array
            reward: Reward for this step
            terminated: Whether episode ended due to terminal condition (collision)
            truncated: Whether episode ended due to max steps
            info: Additional information dictionary
        """
        # Convert action based on action space type
        if isinstance(self.action_space, spaces.Discrete):
            action_payload = int(action)
        elif isinstance(self.action_space, spaces.Box):
            action_payload = action.tolist() if isinstance(action, np.ndarray) else list(action)
        else:
            raise ValueError(f"Unsupported action space type: {type(self.action_space)}")

        try:
            response = self._post(
                f"{self.base_url}/step",
                {"action": action_payload},
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to execute step: {e}") from e

        observation = np.array(response["observation"], dtype=np.float32)
        reward = float(response["reward"])
        terminated = bool(response["done"])
        truncated = bool(response.get("truncated", False))
        info = response["info"]

        return observation, reward, terminated, truncated, info

    def start_training(self) -> None:
        """Enable training mode - disables R key reset to prevent accidental interruptions.

        Call this at the beginning of training to ensure the game won't be accidentally
        reset via keyboard during RL training. Camera controls remain enabled for observation.
        """
        try:
            response = requests.post(f"{self.base_url}/start_training", timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to start training mode: {e}") from e

    def end_training(self) -> None:
        """Disable training mode - re-enables R key reset and returns to human control.

        Call this at the end of training to restore normal keyboard controls.
        """
        try:
            response = requests.post(f"{self.base_url}/end_training", timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to end training mode: {e}") from e

    def set_level(self, level: int) -> None:
        """Set the game difficulty level.

        Args:
            level: Level number (1 for baseline, 2 for hard)

        Note:
            - Level 1 is the original baseline difficulty used for previous models
            - Level 2 has faster projectiles, more frequent spawning, and more projectiles
            - Calling this will reset the game environment
        """
        if level not in (1, 2):
            raise ValueError(f"Invalid level: {level}. Must be 1 or 2")

        try:
            response = requests.post(
                f"{self.base_url}/set_level",
                json={"level": level},
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to set level: {e}") from e

    def configure(
        self,
        level: Optional[int] = None,
        action_space_type: Optional[str] = None,
    ) -> None:
        """Configure game settings.

        Args:
            level: Optional level number (1 for baseline, 2 for hard)
            action_space_type: Optional action space type ("discrete" or "continuous")

        Note:
            - This is the preferred way to configure the game before training
            - Calling this will reset the game environment
            - At least one parameter must be provided
        """
        if level is None and action_space_type is None:
            raise ValueError("At least one configuration parameter must be provided")

        if level is not None and level not in (1, 2):
            raise ValueError(f"Invalid level: {level}. Must be 1 or 2")

        if action_space_type is not None and action_space_type.lower() not in ("discrete", "continuous"):
            raise ValueError(f"Invalid action_space_type: {action_space_type}. Must be 'discrete' or 'continuous'")

        config_data = {}
        if level is not None:
            config_data["level"] = level
        if action_space_type is not None:
            config_data["action_space_type"] = action_space_type

        try:
            response = requests.post(
                f"{self.base_url}/configure",
                json=config_data,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to configure game: {e}") from e

    def close(self) -> None:
        """Close the environment and disable training mode if enabled."""
        try:
            # Ensure training mode is disabled when environment is closed
            self.end_training()
        except Exception:
            # Ignore errors during cleanup
            pass

    def _post(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send POST request to API.

        Args:
            url: Full URL to send request to
            data: JSON data to send

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: On network/HTTP errors
        """
        response = requests.post(url, json=data, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _get(self, url: str) -> Dict[str, Any]:
        """Send GET request to API.

        Args:
            url: Full URL to send request to

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: On network/HTTP errors
        """
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
