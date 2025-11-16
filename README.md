# Bevy 3D Dodge Game with RL Training

A 3D projectile dodging game built with Bevy game engine (Rust) with reinforcement learning capabilities. Train AI agents to master dodging using DQN and other RL algorithms via a Gymnasium-compatible Python interface.

## Features

### Game Features
- **3D Environment**: Realistic lighting with HDR skybox and image-based lighting
- **Physics-Based Gameplay**: Projectiles with arc trajectories and gravity
- **Player Controls**: WASD movement, space to jump, orbit camera with mouse
- **Collision Detection**: Real-time collision system with game-over mechanics
- **Visual Feedback**: Score tracking, game state display

### RL Training Features
- **HTTP REST API**: Expose game as OpenAI Gym environment
- **Gymnasium Wrapper**: Standard RL interface for Python
- **GPU Acceleration**: PyTorch with AMD ROCm 6.4 support
- **DQN Training**: Stable-Baselines3 integration with TensorBoard monitoring
- **65-Dimensional Observations**: Player + projectile positions and velocities
- **5 Discrete Actions**: Noop, Up, Down, Left, Right

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Python Training (RL)                        │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │ DQN Agent    │◄────►│ Gymnasium    │                     │
│  │ (SB3)        │      │ Wrapper      │                     │
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

### Clone Repository

```bash
git clone <repository-url>
cd bevy-3d-dodge
```

### Install Rust Dependencies

```bash
cargo build --release
```

### Install Python Dependencies

**Core dependencies** (for environment only):
```bash
uv sync
```

**Training dependencies** (includes PyTorch ROCm):
```bash
uv sync --extra train
```

This will install:
- PyTorch 2.9.1+rocm6.4 (AMD GPU support)
- Stable-Baselines3 2.7.0 (DQN and other RL algorithms)
- TensorBoard (training visualization)
- Gymnasium (RL environment interface)

## Usage

### Playing the Game Manually

Run the game with keyboard controls:

```bash
# For AMD GPUs (recommended)
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run --release

# Or simply
cargo run --release
```

**Controls:**
- **WASD**: Move player horizontally
- **Space**: Jump
- **Mouse**: Orbit camera around player
- **R**: Reset game after game over

### Training RL Agents

#### 1. Start the Game (API Server)

The game automatically starts an HTTP API server on port 8000:

```bash
VK_LOADER_DEBUG=error VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json cargo run --release
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

#### 3. Train DQN Agent

Start training with default hyperparameters (100k steps):

```bash
uv run python python/train.py --steps 100000
```

**Training options:**
```bash
# Quick test run (10k steps)
uv run python python/train.py --steps 10000

# Long training run (500k steps)
uv run python python/train.py --steps 500000 --buffer-size 100000

# Custom hyperparameters
uv run python python/train.py \
  --steps 200000 \
  --lr 0.00005 \
  --batch-size 64 \
  --buffer-size 100000
```

**What happens during training:**
- Model checkpoints saved to `models/checkpoints/` every 10k steps
- Best model saved to `models/best/` based on evaluation
- TensorBoard logs written to `logs/`
- Real-time progress bar shows reward, episode length, loss
- GPU automatically detected and used (AMD ROCm or NVIDIA CUDA)

#### 4. Monitor Training with TensorBoard

In a separate terminal:

```bash
uv run tensorboard --logdir logs
```

Open http://localhost:6006 to view:
- Episode reward over time
- Episode length over time
- Training loss
- Exploration rate decay
- GPU utilization

#### 5. Evaluate Trained Model

After training completes:

```bash
# Evaluate best model
uv run python python/eval.py models/best/best_model.zip --episodes 20

# Evaluate final model
uv run python python/eval.py models/final_model.zip --episodes 10

# Stochastic evaluation (with exploration)
uv run python python/eval.py models/best/best_model.zip --episodes 10 --stochastic
```

**Evaluation metrics:**
- Average reward ± standard deviation
- Average episode length
- Success rate (episodes reaching max steps without collision)
- Reward range (min/max)

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
│   ├── eval.py              # Model evaluation script
│   ├── config.py            # Hyperparameter configurations
│   └── test_random_agent.py # Environment testing
│
├── models/                  # Trained models (gitignored)
├── logs/                    # TensorBoard logs (gitignored)
├── assets/                  # Game assets (HDR skybox, etc.)
├── Cargo.toml               # Rust dependencies
└── pyproject.toml           # Python dependencies + ROCm config
```

## RL Environment Specification

### Observation Space

**Type:** `Box(shape=(65,), dtype=float32, low=-100, high=100)`

**Layout:**
- Indices 0-2: Player position (x, y, z)
- Indices 3-4: Player velocity (vx, vy)
- Indices 5-64: Up to 10 projectiles × 6 values each
  - Position (x, y, z)
  - Velocity (vx, vy, vz)
  - Zero-padded if fewer than 10 projectiles exist

### Action Space

**Type:** `Discrete(5)`

- **0**: NOOP (no movement)
- **1**: UP (move in +Y direction)
- **2**: DOWN (move in -Y direction)
- **3**: LEFT (move in -X direction)
- **4**: RIGHT (move in +X direction)

### Reward Function

- **+1.0**: Base survival reward per timestep
- **-100.0**: Collision penalty (terminal state)
- **+0.5**: Bonus for close dodges (distance < 2.0 units, scaled by distance)

### Episode Termination

- **Done (terminated=True)**: Player collides with projectile
- **Truncated (truncated=True)**: Maximum steps reached (default: 1000)

## Training Results

### Initial Baseline (100k steps)

**Setup:**
- Hardware: AMD Radeon RX 7900 XTX
- Algorithm: DQN with MLP policy [64, 64]
- Training time: ~35 minutes
- Throughput: ~47-50 FPS

**Performance:**
- Early episodes: Random exploration, ~40-50 reward
- Learning observed: Loss decreases from 0.4 → 0.1
- Episode length improves as agent learns to survive longer

## Hyperparameters

### Default DQN Configuration

```python
learning_rate: 1e-4
buffer_size: 50,000
batch_size: 32
gamma: 0.99
exploration_fraction: 0.3  # First 30% of training
exploration_initial_eps: 1.0
exploration_final_eps: 0.05
target_update_interval: 1000 steps
```

### Network Architecture

- **Policy**: MLP (Multi-Layer Perceptron)
- **Hidden layers**: [64, 64] (default)
- **Activation**: ReLU
- **Optimizer**: Adam

To customize:
```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    # ... other hyperparameters
)
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

## Troubleshooting

### GPU Not Detected

**Check PyTorch installation:**
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**For AMD GPUs:**
- Ensure ROCm 6.4 is installed on your system
- Verify PyTorch was installed from the ROCm index (check `pyproject.toml`)
- ROCm uses the CUDA API compatibility layer (`torch.cuda.*`)

### Connection Errors

If Python can't connect to the game:
1. Verify the game is running and shows "RL API server listening..."
2. Check firewall settings allow localhost connections on port 8000
3. Try the test script: `uv run python python/test_random_agent.py`

### Training Too Slow

**CPU-only fallback:**
If GPU isn't detected, training will use CPU (slower but functional).

**Reduce observation frequency:**
The environment runs at ~50 FPS, which is reasonable for this task.

**Use smaller networks:**
Modify `policy_kwargs=dict(net_arch=[32, 32])` for faster training.

## Future Enhancements

- [ ] Headless mode for faster training (disable rendering)
- [ ] Additional RL algorithms (PPO, SAC, Rainbow DQN)
- [ ] Pixel-based observations (CNN policies)
- [ ] Curriculum learning (progressive difficulty)
- [ ] Multi-agent support
- [ ] Web deployment (WASM build)

## Credits

Built with:
- [Bevy](https://bevyengine.org/) - Rust game engine
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface

## License

MIT License - see LICENSE file for details
