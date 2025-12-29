# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Python Training (RL)                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ SAC/PPO/DQN  │◄────►│ Gymnasium    │                     │
│  │ Agent (SB3)  │      │ Wrapper      │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                               │ gRPC (Unix socket)           │
│                               │ or HTTP REST API             │
└───────────────────────────────┼──────────────────────────────┘
                                │
┌───────────────────────────────┼──────────────────────────────┐
│                  Bevy Game Engine (Rust)                     │
│                               │                              │
│  ┌────────────────────────────▼─────────────────────────────┐│
│  │   gRPC Server (tonic) / HTTP Server (axum)               ││
│  │   Unix socket: /tmp/bevy_rl.sock                         ││
│  │   HTTP: port 8000                                        ││
│  └────────────────────┬─────────────────────────────────────┘│
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────────┐│
│  │         RL Environment Manager                           ││
│  │  - Observation: 65-69 dim vector or 256x256 image        ││
│  │  - Actions: Discrete (5) or Continuous (3-6 dim)         ││
│  │  - Rewards: +1 survival, -100 collision                  ││
│  └────────────────────┬─────────────────────────────────────┘│
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────────┐│
│  │            Game Core (ECS)                               ││
│  │  - Player movement & physics                             ││
│  │  - Projectile spawning & physics                         ││
│  │  - Collision detection                                   ││
│  │  - 3D rendering with PBR materials                       ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Transport Options

### gRPC (Default, Recommended)

- **Unix domain sockets**: Low-latency IPC (~0.1ms per step)
- **Protocol Buffers**: Efficient binary serialization
- **Throughput**: ~250 FPS with parallel environments (during training)

```bash
# Single server
./target/release/bevy_3d_dodge --headless --socket-path /tmp/bevy_rl.sock

# Parallel servers (4 instances)
./start_grpc_servers.sh 4
```

### HTTP (Legacy)

- **REST API**: JSON-based communication
- **Throughput**: ~50 FPS
- **Use case**: Debugging, browser-based tools

```bash
./target/release/bevy_3d_dodge --headless --http --port 8000
```

## Parallel Environment Setup

For high-throughput training, run multiple headless game instances:

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Training                           │
│                                                              │
│              ┌──────────────────────┐                       │
│              │    SubprocVecEnv     │                       │
│              │    (n_envs=4)        │                       │
│              └──────────┬───────────┘                       │
│                         │                                    │
│         ┌───────────────┼───────────────┐                   │
│         │               │               │                    │
│         ▼               ▼               ▼                    │
│    _0.sock         _1.sock         _2.sock         _3.sock  │
└─────────┼───────────────┼───────────────┼───────────────┼───┘
          │               │               │               │
          ▼               ▼               ▼               ▼
     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
     │  Bevy   │     │  Bevy   │     │  Bevy   │     │  Bevy   │
     │Instance │     │Instance │     │Instance │     │Instance │
     │   #0    │     │   #1    │     │   #2    │     │   #3    │
     └─────────┘     └─────────┘     └─────────┘     └─────────┘
```

## Observation Modes

### Vector Observations (MLP Policy)

| Mode | Dimensions | Description |
|------|------------|-------------|
| `standard` | 65 | Player + 10 projectiles |
| `with_thrower` | 69 | + thrower position/direction |

### Image Observations (CNN Policy)

| Mode | Shape | Description |
|------|-------|-------------|
| `topdown` | 256×256×1 | Grayscale top-down view |

## Action Spaces

### Discrete (DQN, PPO)

5 actions: Noop, Up, Down, Left, Right

### Continuous (SAC)

| Type | Dimensions | Description |
|------|------------|-------------|
| `basic_3d` | 3 | [vx, vy, sprint] |
| `basic_4d_jump` | 4 | + jump |
| `full_6d` | 6 | + camera tilt |

## Project Structure

```
bevy-3d-dodge/
├── src/
│   ├── main.rs              # Entry point, Bevy app setup
│   ├── game/                # Game logic modules
│   │   ├── player.rs        # Player movement & physics
│   │   ├── projectile.rs    # Projectile spawning & physics
│   │   └── collision.rs     # Collision detection
│   └── rl/
│       ├── grpc_api.rs      # gRPC server (tonic)
│       ├── api.rs           # HTTP REST API (axum)
│       └── environment.rs   # Reward calculation
│
├── python/
│   ├── bevy_dodge_env/      # Gymnasium wrapper
│   │   ├── environment.py   # BevyDodgeEnv class
│   │   └── grpc_client.py   # gRPC client
│   ├── train_sac.py         # SAC training
│   ├── train_ppo.py         # PPO training
│   └── configs/             # YAML configurations
│
├── proto/
│   └── rl_env.proto         # gRPC service definition
│
└── docs/
    └── ARCHITECTURE.md      # This file
```

## Performance Benchmarks

| Setup | Data Collection | During Training |
|-------|-----------------|-----------------|
| CNN + gRPC (1 env) | 225 FPS | ~6 FPS |
| MLP + HTTP (1 env) | ~50 FPS | ~40 FPS |
| MLP + gRPC (1 env) | 225 FPS | ~100 FPS |
| MLP + gRPC (4 envs) | ~850 FPS | ~250 FPS |
