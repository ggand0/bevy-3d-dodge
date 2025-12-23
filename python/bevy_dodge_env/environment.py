"""Gymnasium environment for Bevy 3D dodge game."""

import base64
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
        timeout: float = 30.0,  # Increased from 5.0 for image observations
    ) -> None:
        super().__init__()

        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

        # Use a Session for connection pooling and keep-alive
        self._session = requests.Session()
        # Configure connection adapter with retry capability
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # Fetch space specifications from Bevy API
        try:
            obs_space_info = self._get(f"{self.base_url}/observation_space")
            action_space_info = self._get(f"{self.base_url}/action_space")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Failed to connect to Bevy server at {self.base_url}. "
                f"Make sure the game is running. Error: {e}"
            ) from e

        # Define observation space - check dtype for image vs vector observations
        obs_dtype = obs_space_info.get("dtype", "float32")
        self._is_image_obs = obs_dtype == "uint8"

        if self._is_image_obs:
            # Image observation: (height, width, channels) with uint8
            self.observation_space = spaces.Box(
                low=int(obs_space_info["low"]),
                high=int(obs_space_info["high"]),
                shape=tuple(obs_space_info["shape"]),
                dtype=np.uint8,
            )
        else:
            # Vector observation: (obs_dim,) with float32
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

        # Decode observation based on mode
        if self._is_image_obs:
            # Decode base64 image and reshape to (H, W, C)
            image_bytes = base64.b64decode(response["image_observation"])
            observation = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                self.observation_space.shape
            )
        else:
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

        # Decode observation based on mode
        if self._is_image_obs:
            # Decode base64 image and reshape to (H, W, C)
            image_bytes = base64.b64decode(response["image_observation"])
            observation = np.frombuffer(image_bytes, dtype=np.uint8).reshape(
                self.observation_space.shape
            )
        else:
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
        sprint_multiplier: Optional[float] = None,
        spawn_angle_degrees: Optional[float] = None,
        observation_mode: Optional[str] = None,
        thrower_delay_seconds: Optional[float] = None,
        image_grayscale: Optional[bool] = None,
    ) -> None:
        """Configure game settings.

        Args:
            level: Optional level number (1 for baseline, 2 for hard)
            action_space_type: Optional action space type ("discrete", "basic_3d", etc.)
            sprint_multiplier: Optional sprint speed multiplier (e.g., 2.0 = 3x speed at full sprint)
            spawn_angle_degrees: Optional half-angle for spawn fan (e.g., 30 = ±30° = 60° total)
            observation_mode: Optional observation mode ("standard" for 65-dim, "with_thrower" for 69-dim, "topdown" for 84x84 image)
            thrower_delay_seconds: Optional delay before thrower indicator spawns projectile
            image_grayscale: Optional grayscale mode (True for 1 channel, False for 3 RGB channels)

        Note:
            - This is the preferred way to configure the game before training
            - Calling this will reset the game environment internally
            - After calling configure(), you must call reset() or create a new environment
              instance to ensure the updated action space is queried correctly
            - At least one parameter must be provided

        Example:
            >>> env = BevyDodgeEnv()
            >>> env.configure(action_space_type="basic_3d", sprint_multiplier=2.0, spawn_angle_degrees=30)
            >>> env.reset()  # Ensures config is synced
        """
        if all(p is None for p in [level, action_space_type, sprint_multiplier, spawn_angle_degrees, observation_mode, thrower_delay_seconds, image_grayscale]):
            raise ValueError("At least one configuration parameter must be provided")

        if level is not None and level not in (1, 2):
            raise ValueError(f"Invalid level: {level}. Must be 1 or 2")

        if sprint_multiplier is not None and (sprint_multiplier < 0 or sprint_multiplier > 10):
            raise ValueError(f"Invalid sprint_multiplier: {sprint_multiplier}. Must be between 0 and 10")

        if spawn_angle_degrees is not None and (spawn_angle_degrees <= 0 or spawn_angle_degrees > 180):
            raise ValueError(f"Invalid spawn_angle_degrees: {spawn_angle_degrees}. Must be between 0 and 180")

        if observation_mode is not None and observation_mode not in ("standard", "with_thrower", "topdown"):
            raise ValueError(f"Invalid observation_mode: {observation_mode}. Must be 'standard', 'with_thrower', or 'topdown'")

        if thrower_delay_seconds is not None and (thrower_delay_seconds <= 0 or thrower_delay_seconds > 10):
            raise ValueError(f"Invalid thrower_delay_seconds: {thrower_delay_seconds}. Must be between 0 and 10")

        # Note: action_space_type validation is handled server-side
        # Valid values: "discrete", "basic_3d", "basic_4d_jump", "tilt_5d", "full_6d"

        config_data = {}
        if level is not None:
            config_data["level"] = level
        if action_space_type is not None:
            config_data["action_space_type"] = action_space_type
        if sprint_multiplier is not None:
            config_data["sprint_multiplier"] = sprint_multiplier
        if spawn_angle_degrees is not None:
            config_data["spawn_angle_degrees"] = spawn_angle_degrees
        if observation_mode is not None:
            config_data["observation_mode"] = observation_mode
        if thrower_delay_seconds is not None:
            config_data["thrower_delay_seconds"] = thrower_delay_seconds
        if image_grayscale is not None:
            config_data["image_grayscale"] = image_grayscale

        try:
            response = requests.post(
                f"{self.base_url}/configure",
                json=config_data,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to configure game: {e}") from e

    @property
    def is_image_observation(self) -> bool:
        """Return True if the environment uses image observations (for CNN policies)."""
        return self._is_image_obs

    def close(self) -> None:
        """Close the environment and disable training mode if enabled."""
        try:
            # Ensure training mode is disabled when environment is closed
            self.end_training()
        except Exception:
            # Ignore errors during cleanup
            pass
        finally:
            # Close the HTTP session
            try:
                self._session.close()
            except Exception:
                pass

    def _post(self, url: str, data: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """Send POST request to API with retry logic.

        Args:
            url: Full URL to send request to
            data: JSON data to send
            max_retries: Maximum number of retry attempts for transient errors

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: On network/HTTP errors after all retries
        """
        import time
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = self._session.post(url, json=data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Wait briefly before retry (exponential backoff)
                    wait_time = 0.1 * (2 ** attempt)
                    print(f"⚠ Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise
            except requests.exceptions.RequestException:
                # Don't retry other types of errors
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    def _get(self, url: str) -> Dict[str, Any]:
        """Send GET request to API.

        Args:
            url: Full URL to send request to

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: On network/HTTP errors
        """
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
