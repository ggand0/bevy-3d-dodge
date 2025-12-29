# Plan: Master Level 2 Before Wider Angles

## Goal
The SAC agent hasn't solved Level 2 (±60° = 120° fan) yet. Focus on improving performance there through systematic experiments before trying wider spawn angles or multi-agent.

## Current State
- **Reward**: +1.0 survival, -100 collision, +0.5*(2-dist) dodge bonus for dist<2.0
- **Level 2**: ±60° = 120° fan, 0.5s interval, 0.8s flight time
- **1M steps result**: Best eval 874.66, final avg 660.48 ± 307 (high variance)
- **Episode length**: ~715 steps avg (max is 1000)

---

## Experiment Plan (Sequential)

### Experiment 1: Longer Training (2M steps)
**Goal**: See if current arch just needs more time.

**Config**: `python/configs/sac_2m_baseline.yaml`
- `total_timesteps: 2000000`
- Everything else same as current sac_mlp_grpc.yaml

**Success criteria**: Eval reward > 900, episode length > 900

---

### Experiment 2: Architecture Improvements
**Goal**: Try wider networks and/or temporal memory.

**Options** (test one at a time):
1. **Wider MLP**: `net_arch: [512, 512]` or `[256, 256, 256]`
2. **LSTM policy**: Add recurrence to remember projectile patterns

**Files to modify**:
- `python/configs/sac_wide_arch.yaml` - wider MLP config
- For LSTM: may need custom policy class in SB3

---

### Experiment 3: Reward Tuning
**Goal**: Encourage riskier/more active behavior.

**File**: [src/rl/environment.rs](src/rl/environment.rs)

**Options**:
1. **Bigger dodge window**: threshold 2.0 → 3.0 (more situations trigger bonus)
2. **Higher multiplier**: 0.5 → 1.0 (double the close-call reward)
3. **Both**: threshold 3.0 AND multiplier 1.0

**Current code** (line 64-67):
```rust
if min_distance < 2.0 {
    reward += (2.0 - min_distance) * 0.5;
}
```

**Tuned version**:
```rust
if min_distance < 3.0 {
    reward += (3.0 - min_distance) * 1.0;
}
```

---

## Phase 2: Wider Angles (After Level 2 Mastery)

Once Level 2 is solved (>90% survival), add Level 3:
- `spawn_angle_degrees: 90.0` (±90° = 180° fan)
- Then 120° (240° total), then 180° (360° full)

---

## Phase 3: Multi-Agent (Future)

After single-agent mastery:
1. Scripted predictive thrower
2. Learned thrower agent

---

## Files to Modify

| File | Changes |
|------|---------|
| `python/configs/sac_2m_baseline.yaml` | New config for 2M training |
| `python/configs/sac_wide_arch.yaml` | Wider network experiment |
| [src/rl/environment.rs](src/rl/environment.rs) | Tune dodge bonus (Exp 3) |

---

## Execution Order

1. **Now**: Create 2M baseline config and start training
2. **While training**: Prepare wide arch config
3. **After 2M baseline**: Compare results, decide next experiment
4. **Reward tuning**: Test after arch experiments (requires Rust rebuild)
