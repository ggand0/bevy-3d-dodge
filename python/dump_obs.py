#!/usr/bin/env python3
"""Dump observation images from the game to inspect what the CNN sees."""

import argparse
import numpy as np
from PIL import Image
from pathlib import Path

from bevy_dodge_env import BevyDodgeEnv


def main():
    parser = argparse.ArgumentParser(description="Dump observation images")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", type=str, default="obs_dump")
    parser.add_argument("--n-frames", type=int, default=10)
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Connect and configure
    env = BevyDodgeEnv(port=args.port)
    env.configure(
        level=2,
        observation_mode="topdown",
        action_space_type="basic_3d",
        image_grayscale=args.grayscale,
    )
    env.refresh_spaces()  # Re-query observation space after config change

    print(f"Observation space: {env.observation_space}")
    print(f"Saving {args.n_frames} frames to {output_dir}/")

    obs, _ = env.reset()

    import time

    # Wait for first projectile to spawn (0.5s at Level 2)
    print("  Waiting 1s for projectiles to spawn...")
    for _ in range(30):
        action = env.action_space.sample()
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
        time.sleep(0.03)

    for i in range(args.n_frames):
        # Save observation
        if args.grayscale:
            # Grayscale: (H, W, 1) -> (H, W)
            img_array = obs[:, :, 0] if obs.ndim == 3 else obs
            img = Image.fromarray(img_array, mode='L')
        else:
            # RGB: (H, W, 3)
            img = Image.fromarray(obs)

        img.save(output_dir / f"obs_{i:03d}.png")
        print(f"  Saved obs_{i:03d}.png - shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min()}, {obs.max()}]")

        # Take random action and wait a bit for game to progress
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        time.sleep(0.05)  # 50ms between frames

        if done or truncated:
            obs, _ = env.reset()
            # Wait for projectiles after reset
            time.sleep(0.6)

    env.close()
    print(f"\nDone! Check {output_dir}/ for images")


if __name__ == "__main__":
    main()
