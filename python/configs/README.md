# Training Configurations

This directory contains YAML configuration files for different training experiments.

## Available Configurations

### `default.yaml`
Standard DQN configuration matching the original 100k training run.

**Usage:**
```bash
uv run python python/train.py --config python/configs/default.yaml
```

**Key parameters:**
- Total timesteps: 100,000
- Learning rate: 1e-4
- Buffer size: 50,000
- Network: [64, 64] (default)

### `quick_test.yaml`
Fast configuration for rapid iteration and testing.

**Usage:**
```bash
uv run python python/train.py --config python/configs/quick_test.yaml
```

**Key parameters:**
- Total timesteps: 10,000
- Learning rate: 1e-4
- Buffer size: 10,000
- More frequent checkpoints for quick feedback

## Creating Custom Configurations

### 1. Copy an existing config
```bash
cp python/configs/default.yaml python/configs/my_experiment.yaml
```

### 2. Edit parameters
```yaml
# My custom experiment
total_timesteps: 200000
learning_rate: 0.00003
buffer_size: 75000
net_arch: [128, 128, 64]  # Deeper network
```

### 3. Run training
```bash
uv run python python/train.py --config python/configs/my_experiment.yaml
```

## Parameter Reference

### Training Hyperparameters
- `total_timesteps`: Total training steps (e.g., 100000)
- `learning_rate`: Adam optimizer learning rate (e.g., 0.0001)
- `buffer_size`: Replay buffer capacity (e.g., 50000)
- `learning_starts`: Steps before training begins (e.g., 1000)
- `batch_size`: Minibatch size for gradient updates (e.g., 32)
- `gamma`: Discount factor for future rewards (e.g., 0.99)
- `target_update_interval`: Steps between target network updates (e.g., 1000)

### Exploration (ε-greedy)
- `exploration_fraction`: Fraction of training for ε decay (e.g., 0.3 = first 30%)
- `exploration_initial_eps`: Starting exploration rate (e.g., 1.0 = 100% random)
- `exploration_final_eps`: Final exploration rate (e.g., 0.05 = 5% random)

### Network Architecture
- `net_arch`: Hidden layer sizes (e.g., `[256, 256]` or `null` for default [64, 64])

### Logging and Checkpointing
- `eval_freq`: Steps between evaluation runs (e.g., 5000)
- `save_freq`: Steps between checkpoints (e.g., 10000)
- `n_eval_episodes`: Episodes per evaluation (e.g., 5)

### Environment
- `port`: Bevy API server port (default: 8000)
- `max_episode_steps`: Maximum steps per episode (default: 1000)

### Paths
- `save_dir`: Directory for model checkpoints (default: "models")
- `log_dir`: Directory for TensorBoard logs (default: "logs")

## Overriding Parameters via CLI

You can override specific parameters from a config file:

```bash
# Use improved_baseline.yaml but train for 400k steps
uv run python python/train.py \
  --config python/configs/improved_baseline.yaml \
  --steps 400000

# Use default.yaml but with lower learning rate
uv run python python/train.py \
  --config python/configs/default.yaml \
  --lr 0.00005 \
  --buffer-size 100000
```

## Experiment Tracking Best Practices

1. **Name configs descriptively**: `experiment_name_lr1e5_buffer100k.yaml`
2. **Document changes**: Add comments explaining why you changed parameters
3. **Version control**: Commit config files alongside code changes
4. **Track results**: Reference config files in training logs and devlogs

## Example: Hyperparameter Search

Create multiple configs for grid search:

```bash
# python/configs/hp_search/lr_1e4.yaml
learning_rate: 0.0001
buffer_size: 100000

# python/configs/hp_search/lr_5e5.yaml
learning_rate: 0.00005
buffer_size: 100000

# python/configs/hp_search/lr_1e5.yaml
learning_rate: 0.00001
buffer_size: 100000
```

Run all experiments:
```bash
for config in python/configs/hp_search/*.yaml; do
  uv run python python/train.py --config "$config"
done
```
