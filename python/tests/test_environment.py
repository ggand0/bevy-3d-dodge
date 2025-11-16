"""Tests for BevyDodgeEnv."""

import pytest
import numpy as np
from bevy_dodge_env import BevyDodgeEnv


@pytest.fixture
def env() -> BevyDodgeEnv:
    """Create environment fixture."""
    return BevyDodgeEnv(port=8000)


def test_env_creation(env: BevyDodgeEnv) -> None:
    """Test that environment can be created and connects to API."""
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None


def test_observation_space(env: BevyDodgeEnv) -> None:
    """Test observation space has correct shape and dtype."""
    assert env.observation_space.shape == (65,)
    assert env.observation_space.dtype == np.float32
    assert env.observation_space.low == -100.0
    assert env.observation_space.high == 100.0


def test_action_space(env: BevyDodgeEnv) -> None:
    """Test action space has correct size."""
    from gymnasium.spaces import Discrete
    assert isinstance(env.action_space, Discrete)
    assert env.action_space.n == 5


def test_reset(env: BevyDodgeEnv) -> None:
    """Test reset returns valid observation."""
    obs, info = env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (65,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)

    # Check observation is in valid range
    assert np.all(obs >= -100.0)
    assert np.all(obs <= 100.0)


def test_step(env: BevyDodgeEnv) -> None:
    """Test step returns valid transition."""
    env.reset()

    # Take a step with valid action
    action = 1  # UP
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (65,)
    assert obs.dtype == np.float32

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Check observation is in valid range
    assert np.all(obs >= -100.0)
    assert np.all(obs <= 100.0)


def test_multiple_steps(env: BevyDodgeEnv) -> None:
    """Test multiple consecutive steps work correctly."""
    env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.reset()
            break


def test_invalid_action(env: BevyDodgeEnv) -> None:
    """Test that invalid actions raise appropriate errors."""
    env.reset()

    # Action outside valid range should be caught by API
    with pytest.raises(Exception):
        env.step(999)


def test_episode_info(env: BevyDodgeEnv) -> None:
    """Test that info dict contains expected keys."""
    env.reset()
    _, _, _, _, info = env.step(0)

    assert "episode_steps" in info
    assert "projectile_count" in info
    assert isinstance(info["episode_steps"], int)
    assert isinstance(info["projectile_count"], int)
