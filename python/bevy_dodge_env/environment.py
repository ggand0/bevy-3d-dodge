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
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step with the given action.

        Args:
            action: Action index (0-4 for discrete action space)

        Returns:
            observation: New observation as numpy array
            reward: Reward for this step
            terminated: Whether episode ended due to terminal condition (collision)
            truncated: Whether episode ended due to max steps
            info: Additional information dictionary
        """
        try:
            response = self._post(
                f"{self.base_url}/step",
                {"action": int(action)},
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to execute step: {e}") from e

        observation = np.array(response["observation"], dtype=np.float32)
        reward = float(response["reward"])
        terminated = bool(response["done"])
        truncated = bool(response.get("truncated", False))
        info = response["info"]

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the environment (no cleanup needed for HTTP client)."""
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
