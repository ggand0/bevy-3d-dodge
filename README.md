# Bevy 3D Dodge Game with RL Training

A 3D projectile dodging game built with Bevy (Rust) with reinforcement learning capabilities. Train AI agents using SAC, PPO, or DQN via a Gymnasium-compatible Python interface.

**Fully open-source stack**: Bevy (MIT/Apache) + Stable-Baselines3 (MIT) + PyTorch (BSD)

## Quick Start

```bash
# Terminal 1: Start game server (gRPC)
# NVIDIA/CUDA:
./target/release/bevy_3d_dodge --headless --fps 240

# AMD (requires Vulkan ICD override):
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
  ./target/release/bevy_3d_dodge --headless --fps 240

# Terminal 2: Train SAC agent
uv run --directory python python train_sac.py --config configs/sac_mlp_grpc.yaml
```
For parallel training (4 instances, ~250 FPS):


```bash
# Terminal 1: Start 4 parallel servers
./start_grpc_servers.sh 4  # Edit script to remove AMD env vars for NVIDIA

# Terminal 2: Train with parallel envs
uv run --directory python python train_sac.py --config configs/sac_mlp_grpc.yaml
```

## Features

- **gRPC + Unix sockets**: High-throughput training (~250 FPS with 4 parallel envs)
- **Multiple algorithms**: SAC (continuous), PPO, DQN (discrete)
- **Flexible observations**: 65-69 dim vectors or 256×256 images
- **Headless mode**: Fast training without rendering
- **TensorBoard**: Real-time training visualization

## Installation

```bash
# Build Rust game
cargo build --release

# Install Python dependencies
uv sync --extra train
```

**Prerequisites**: Rust 1.70+, Python 3.10+, [uv](https://astral.sh/uv)

## Usage

### Training

| Algorithm | Command |
|-----------|---------|
| SAC (recommended) | `uv run --directory python python train_sac.py --config configs/sac_mlp_grpc.yaml` |
| PPO | `uv run --directory python python train_ppo.py --config configs/ppo_baseline.yaml` |

### Evaluation

```bash
# Visual evaluation (start game with rendering)
cargo run --release -- --fps 60  # NVIDIA
# AMD: VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run --release -- --fps 60

# Run eval script
uv run --directory python python eval_sac.py results/<run>/models/final_model.zip --episodes 10
```

### Server Options

| Flag | Description | Default |
|------|-------------|---------|
| `--headless` | No window (faster training) | off |
| `--fps <N>` | Tick rate cap | 60 |
| `--socket-path <PATH>` | Unix socket for gRPC | /tmp/bevy_rl.sock |
| `--http` | Use HTTP instead of gRPC | off |
| `--port <N>` | HTTP port | 8000 |

## Configuration

Training configs in `python/configs/`:

- `sac_mlp_grpc.yaml` - SAC with vector obs + gRPC + parallel envs
- `sac_topdown_cnn.yaml` - SAC with image obs (CNN)
- `ppo_baseline.yaml` - PPO baseline

## Performance

| Setup | Training FPS |
|-------|--------------|
| MLP + gRPC + 4 parallel envs | ~250 FPS |
| MLP + gRPC + 1 env | ~100 FPS |
| CNN + gRPC | ~6 FPS (GPU-bound) |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture and benchmarks.

## Playing Manually

```bash
cargo run --release  # NVIDIA
# AMD: VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run --release
```

**Controls**: WASD (move), Space (jump), Mouse (camera), R (reset)

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
