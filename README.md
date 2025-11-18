# Bevy 3D Dodge Game with RL Training

A 3D projectile dodging game built with Bevy game engine (Rust) with reinforcement learning capabilities. Train AI agents to master dodging using DQN and PPO algorithms via a Gymnasium-compatible Python interface.

## Quick Start

```bash
# Terminal 1: Start the game
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run

# Terminal 2: Train PPO agent
uv run python python/train_ppo.py --config python/configs/ppo_baseline.yaml

# Terminal 3: Monitor training
uv run tensorboard --logdir results/ppo_baseline/<timestamp>/logs
```

## Features

### Game Features
- **3D Environment**: Realistic lighting with HDR skybox and image-based lighting
- **Physics-Based Gameplay**: Projectiles with arc trajectories and gravity
- **Player Controls**: WASD movement, space to jump, orbit camera with mouse
- **Collision Detection**: Real-time collision system with game-over mechanics
- **Visual Feedback**: Score tracking, game state display

### RL Training Features
- **HTTP REST API**: Expose game as Gymnasium environment
- **GPU Acceleration**: PyTorch with AMD ROCm 6.4 support
- **Multiple Algorithms**: DQN and PPO support
- **65-Dimensional Observations**: Player + projectile positions and velocities
- **5 Discrete Actions**: Noop, Up, Down, Left, Right
- **TensorBoard Integration**: Real-time training visualization
- **YAML Configurations**: Easy hyperparameter management

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Python Training (RL)                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ PPO/DQN      │◄────►│ Gymnasium    │                     │
│  │ Agent (SB3)  │      │ Wrapper      │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                               │ HTTP REST API                │
└───────────────────────────────┼──────────────────────────────┘
                                │
┌───────────────────────────────┼──────────────────────────────┐
│                  Bevy Game Engine (Rust)                     │
│                               │                              │
│  ┌────────────────────────────▼─────────────────────┐       │
│  │      Axum HTTP Server (port 8000)                │       │
│  │  /reset  /step  /observation_space  /action_space│       │
│  └────────────────────┬─────────────────────────────┘       │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────┐       │
│  │         RL Environment Manager                   │       │
│  │  - Observation: 65-dim state vector              │       │
│  │  - Actions: 5 discrete movements                 │       │
│  │  - Rewards: +1 survival, -100 collision          │       │
│  └────────────────────┬─────────────────────────────┘       │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────┐       │
│  │            Game Core (ECS)                       │       │
│  │  - Player movement & physics                     │       │
│  │  - Projectile spawning & physics                 │       │
│  │  - Collision detection                           │       │
│  │  - 3D rendering with PBR materials               │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- **Rust** (1.70+): [Install Rust](https://rustup.rs/)
- **Python** (3.10+): Required for RL training
- **uv**: Fast Python package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **AMD GPU** (optional but recommended): For GPU-accelerated training with ROCm

### Setup

```bash
# Clone repository
git clone <repository-url>
cd bevy_3d_dodge

# Build Rust game
cargo build --release

# Install Python dependencies (includes PyTorch ROCm)
uv sync --extra train
```

## Usage

### Playing Manually

Run the game with keyboard controls:

```bash
# For AMD GPUs (recommended)
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run --release

# Standard
cargo run --release
```

**Controls:**
- **WASD**: Move player
- **Space**: Jump
- **Mouse**: Orbit camera
- **R**: Reset after game over

### Training RL Agents

#### 1. Start the Game (API Server)

The game automatically starts an HTTP API server on port 8000:

```bash
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run
```

You should see:
```
RL API server listening on http://127.0.0.1:8000
```

#### 2. Test Environment Connection

Test the Python environment wrapper:

```bash
uv run python python/test_random_agent.py --episodes 5
```

This runs a random agent to verify the environment is working correctly.

#### 3. Train Agents

**PPO (Proximal Policy Optimization):**
```bash
uv run python python/train_ppo.py --config python/configs/ppo_baseline.yaml
```

**DQN (Deep Q-Network):**
```bash
# Baseline configuration
uv run python python/train.py --config python/configs/dqn_baseline.yaml

# Improved configuration
uv run python python/train.py --config python/configs/dqn_improved_baseline.yaml

# Quick test
uv run python python/train.py --config python/configs/dqn_quick_test.yaml
```

**Training outputs:**
```
results/<algorithm>_<config>/<timestamp>/
├── models/
│   ├── best/best_model.zip      # Best performing model
│   ├── checkpoints/             # Periodic checkpoints
│   └── final_model.zip          # Final model
├── logs/                         # TensorBoard logs
└── config.yaml                   # Saved configuration
```

#### 4. Monitor Training with TensorBoard

In a separate terminal:

```bash
uv run tensorboard --logdir results/<algorithm>_<config>/<timestamp>/logs
```

Open http://localhost:6006 to view:
- Episode reward over time
- Episode length over time
- Training loss
- Algorithm-specific metrics

#### 5. Evaluate Trained Models

**PPO models:**
```bash
uv run python python/eval_ppo.py \
  results/ppo_baseline/<timestamp>/models/best/best_model.zip \
  --episodes 20
```

**DQN models:**
```bash
uv run python python/eval.py \
  results/dqn_baseline/<timestamp>/models/best/best_model.zip \
  --episodes 20
```

**Evaluation metrics:**
- Average reward ± standard deviation
- Average episode length
- Success rate (episodes reaching max steps without collision)
- Reward range (min/max)

#### 6. Plot Training Curves

```bash
uv run python python/plot_training.py \
  --logdir results/<algorithm>_<config>/<timestamp>/logs \
  --output results/<algorithm>_<config>/<timestamp>/plots
```

Generates:
- Episode reward progression
- Episode length progression
- Training loss curves
- Evaluation metrics
- Combined overview plots

## Configuration

All training configurations are specified in YAML format under `python/configs/`:

- **ppo_baseline.yaml**: PPO with standard hyperparameters
- **dqn_baseline.yaml**: DQN baseline configuration
- **dqn_improved_baseline.yaml**: DQN with improved hyperparameters
- **dqn_quick_test.yaml**: Quick test configuration (10k steps)

See individual YAML files for detailed hyperparameter settings. You can create custom configurations or override parameters via command line:

```bash
# Override specific parameters
uv run python python/train_ppo.py --config python/configs/ppo_baseline.yaml --steps 500000
```

## RL Environment Specification

### Observation Space

**Type:** `Box(65,)` float32, range [-100, 100]

**Structure:**
```
[0-2]:   Player position (x, y, z)
[3-4]:   Player velocity (vx, vy)
[5-64]:  Up to 10 projectiles × 6 values each:
         - Position (x, y, z)
         - Velocity (vx, vy, vz)
         Zero-padded if fewer than 10 projectiles exist
```

### Action Space

**Type:** `Discrete(5)`

```
0: NOOP   - No movement
1: UP     - Move in +Y direction
2: DOWN   - Move in -Y direction
3: LEFT   - Move in -X direction
4: RIGHT  - Move in +X direction
```

### Reward Function

```python
+1.0   per timestep (survival reward)
-100.0 on collision (terminal penalty)
+0.5   close dodge bonus (distance < 2.0 units, scaled by proximity)
```

### Episode Termination

- **Done (terminated=True)**: Player collides with projectile
- **Truncated (truncated=True)**: Maximum steps reached (default: 1000)

## Project Structure

```
bevy-3d-dodge/
├── src/
│   ├── main.rs              # Entry point, Bevy app setup
│   ├── game/                # Game logic modules
│   │   ├── player.rs        # Player movement & physics
│   │   ├── projectile.rs    # Projectile spawning & physics
│   │   ├── camera.rs        # Orbit camera system
│   │   └── collision.rs     # Collision detection
│   └── rl/                  # RL integration modules
│       ├── api.rs           # HTTP REST API (Axum)
│       ├── observation.rs   # State extraction (65-dim vector)
│       ├── action.rs        # Action parsing & application
│       └── environment.rs   # Reward calculation & episode management
│
├── python/
│   ├── bevy_dodge_env/      # Gymnasium wrapper package
│   │   ├── environment.py   # BevyDodgeEnv class
│   │   └── vec_env.py       # Vectorized environment utilities
│   ├── train.py             # DQN training script
│   ├── train_ppo.py         # PPO training script
│   ├── eval.py              # DQN model evaluation
│   ├── eval_ppo.py          # PPO model evaluation
│   ├── plot_training.py     # Training curve plotting
│   ├── config.py            # Unified configuration class
│   ├── test_random_agent.py # Environment testing
│   └── configs/
│       ├── ppo_baseline.yaml
│       ├── dqn_baseline.yaml
│       ├── dqn_improved_baseline.yaml
│       └── dqn_quick_test.yaml
│
├── results/                 # Training artifacts (gitignored)
├── assets/                  # Game assets (HDR skybox, etc.)
├── Cargo.toml               # Rust dependencies
└── pyproject.toml           # Python dependencies + ROCm config
```

## API Endpoints

The game exposes these HTTP endpoints on `http://127.0.0.1:8000`:

- **POST /reset**: Reset environment
  - Response: `{observation: float[], info: {}}`

- **POST /step**: Execute action
  - Request: `{action: int}`
  - Response: `{observation: float[], reward: float, done: bool, truncated: bool, info: {}}`

- **GET /observation_space**: Query observation space
  - Response: `{shape: [65], dtype: "float32", low: -100, high: 100}`

- **GET /action_space**: Query action space
  - Response: `{type: "Discrete", n: 5}`

## Development

### Running Tests

**Rust tests:**
```bash
cargo test
```

**Python tests:**
```bash
uv run pytest python/tests/
```

### Code Formatting

**Rust:**
```bash
cargo fmt
cargo clippy
```

**Python:**
```bash
uv run ruff check python/
uv run ruff format python/
```

## Credits

Built with:
- [Bevy](https://bevyengine.org/) - Rust game engine
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface

## License

MIT License - see LICENSE file for details
