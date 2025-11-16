# Bevy 3D Dodge - RL Training Game

A 3D projectile dodging game built with Bevy (Rust) designed for training reinforcement learning agents with Python.

## Overview

This project combines:
- Bevy Game Engine (Rust) for a performant 3D game
- HTTP/REST API for Python communication (coming in Phase 2)
- Stable-Baselines3 (Python) for RL training
- PyTorch with ROCm/CUDA support for GPU training

The goal is to train RL agents that learn to dodge incoming projectiles through trial and error.

## Quick Start

### Prerequisites

**Rust:**
- Rust 1.70+ (install via [rustup](https://rustup.rs/))

**Python (for RL training, Phase 2):**
- Python 3.10+
- uv package manager
- PyTorch with ROCm (AMD GPU) or CUDA (NVIDIA GPU)

### Building and Running

```bash
# Clone and navigate to project
cd bevy_3d_dodge

# Build the project
cargo build

# Run on AMD GPU (with Vulkan settings)
./run.sh

# Run on NVIDIA GPU or CPU
cargo run
```

**Note:** Tested on AMD 7900XTX with ROCm 6.4.3, but should work on NVIDIA GPUs and CPU-only systems.

### Controls

- WASD or Arrow Keys: Move player
- R: Reset game after game over
- F1: Toggle camera debug mode
- ESC: Quit

**Camera Debug Mode (F1):**
- Middle Mouse + Drag: Pan camera
- Right Mouse + Drag: Rotate view
- Scroll Wheel: Zoom in/out
- U/O Keys: Move camera up/down
- Arrow Keys: Rotate camera

## Environment Mechanics

- Agent: Blue capsule that moves in the XY plane
- Obstacles: Red spheres that spawn from +Y direction and move toward -Y
- Objective: Maximize survival time by avoiding collisions
- Episode Termination: Collision with any projectile (agent turns red)

## Project Structure

```
bevy_3d_dodge/
├── src/
│   ├── main.rs              # Entry point, scene setup
│   ├── config.rs            # Game configuration
│   └── game/
│       ├── player.rs        # Player entity and movement
│       ├── projectile.rs    # Projectile spawning
│       ├── camera.rs        # Camera setup
│       └── collision.rs     # Collision detection
├── python/                  # Python RL code (coming soon)
├── devlogs/                 # Development documentation
└── Cargo.toml              # Rust dependencies
```

## Configuration

Edit game parameters in `src/config.rs`:

```rust
GameConfig {
    player_speed: 5.0,                    // Player movement speed
    player_start_height: 1.0,             // Player Z position
    projectile_speed: 3.0,                // Projectile velocity
    projectile_spawn_interval: 2.0,       // Seconds between spawns
    projectile_spawn_distance: 20.0,      // Spawn X position
    max_projectiles: 10,                  // Max simultaneous projectiles
}
```

## Current Features

- 3D game with player movement and projectile dodging mechanics
- Isaac Sim-style visual environment with coordinate axes
- Camera debug mode for scene inspection
- Game state management and reset functionality

## Roadmap

- HTTP REST API for Python communication
- Gymnasium environment wrapper
- Stable-Baselines3 training scripts
- Vectorized environment support

## License

MIT

## Acknowledgments

- [Bevy Engine](https://bevyengine.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Inspired by various Bevy + RL projects (bevy_rl, entity-gym-rs)
