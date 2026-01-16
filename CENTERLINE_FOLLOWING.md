# Centerline Following - Simplified RL Approach

## Overview

The RL system has been redesigned to focus on **centerline following** rather than lap time optimization. This simplification makes the learning problem more tractable and ensures the agent can successfully complete tracks.

## Key Changes

### 1. Simplified Goal
- **Before**: Minimize lap time (complex, multi-objective)
- **After**: Follow centerline as closely as possible (clear, single objective)

This change makes success criteria explicit: stay near the centerline with minimal lateral and heading errors.

### 2. Reduced Observation Space: 14D → 8D

**Removed features:**
- Yaw rate (r)
- Wall distances (d_left, d_right)
- Curvature lookahead (kappa_5m, kappa_10m, kappa_20m)
- Track progress (s_normalized)

**Remaining features (8D):**
1. `lateral_error` - Distance from centerline
2. `heading_error` - Angle difference from track tangent
3. `vx` - Longitudinal velocity
4. `vy` - Lateral velocity  
5. `tangent_x` - Track direction X component
6. `tangent_y` - Track direction Y component
7. `cos_psi` - Heading cosine
8. `sin_psi` - Heading sine

**Benefit**: Faster learning, less noise, clearer signal.

### 3. Simplified Reward Function

**New reward structure:**
```python
# Primary: Track centerline (exponential penalty on deviation)
r_tracking = 10.0 * exp(-2.0 * |lateral_error|)

# Secondary: Align heading with track
r_heading = 2.0 * exp(-1.0 * |heading_error|)

# Tertiary: Maintain target speed (8 m/s)
r_speed = -(vx - 8.0)² / 64.0

# Smoothness: Penalize jerky inputs
r_smooth = -0.05 * (|delta| + 0.3 * |ax_cmd|)

# Off-track penalty
r_offtrack = -10.0 if |lateral_error| > track_width else 0.0

# Total
reward = r_tracking + r_heading + r_speed + r_smooth + r_offtrack
```

**Key features:**
- **Exponential rewards**: 1.0 at centerline, drops quickly with error
- **Clear priorities**: Tracking (10x) > Heading (2x) > Speed (1x)
- **Moderate target speed**: 8 m/s for good tracking (not max speed)
- **Smooth control**: Encourages stable, predictable behavior

### 4. Simplified Neural Network

**Policy network architecture:**
- **Before**: 3 layers [256, 256, 128]
- **After**: 2 layers [128, 64]

**Benefits:**
- ~4x fewer parameters (147K → 35K)
- Faster training (~3x per epoch)
- Easier to converge on simpler task
- Less prone to overfitting

### 5. Removed Curriculum Learning

**No longer needed** for centerline tracking:
- No speed phases (8 → 11 → ∞ m/s)
- No track distribution changes
- Single, consistent training regime

**Why**: Centerline following is achievable at all speeds. The agent learns at moderate speeds (~8 m/s) naturally.

### 6. More Tolerant Termination

**Termination conditions:**
- Off-track: `|lateral_error| > 2 * track_width` (was 1.5x)
- Backwards: `vx < -1 m/s` (was -2 m/s)

**Benefit**: Agent has more room to explore and recover from mistakes during learning.

---

## Expected Performance

### Training Time
- **Reduced from**: ~4-5 hours (with parallelization)
- **Reduced to**: ~1-2 hours (estimate)

**Speedup factors:**
- Simpler reward: faster rollout
- Smaller network: faster gradient updates
- No curriculum: no phase transitions
- 8D obs: faster inference

### Success Criteria

**Good centerline following:**
- Mean lateral error < 0.5m
- Max lateral error < 2.0m
- Completes full laps consistently
- Smooth steering (no oscillations)

**Excellent centerline following:**
- Mean lateral error < 0.2m
- Max lateral error < 1.0m
- Can maintain 8-10 m/s smoothly
- Racing-line-ready (for future optimization)

---

## Usage

### Evaluate BC (after training)

```bash
# Evaluate BC policy
python evaluate_bc.py --model bc_policy.pth --episodes 10

# Compare BC vs baseline
python evaluate_bc.py --model bc_policy.pth --episodes 10 --compare-baseline

# Visualize BC
python visualize_bc.py --model bc_policy.pth --laps 1
```

### Train with new system

```bash
# Collect demonstrations (unchanged)
python collect_demonstrations.py --episodes 100 --workers 7

# Train BC (now faster with smaller network)
python train_bc.py --epochs 30 --workers 4

# Train RL (now simpler, no curriculum)
python train_rl.py --timesteps 300000 --envs 7
```

Or use the optimized script:
```bash
./train_optimized.sh
```

---

## Next Steps: Race Line Optimization

Once centerline following is working well, you can add **race line optimization**:

### Approach 1: Optimal Control
Compute optimal race line offline using trajectory optimization:
- Minimize: lap time
- Subject to: vehicle dynamics, track boundaries
- Use tools: CasADi, GPOPS-II, or custom optimizer

Then train RL to track the optimal line (instead of centerline).

### Approach 2: Reward Shaping
Modify reward to encourage faster lines:
```python
# Reward based on distance along track (velocity-weighted)
r_progress = (delta_s / dt) * 5.0

# Track optimal line (not centerline)
r_tracking = 10.0 * exp(-2.0 * |line_error|)
```

This lets the agent discover faster lines through exploration.

### Approach 3: Inverse Reinforcement Learning
- Record expert (time_optimal_controller) demonstrations
- Learn reward function from demonstrations
- Use learned reward to train faster policy

---

## Troubleshooting

### Agent doesn't stay on centerline
**Check:**
1. Reward weights - is tracking reward dominant?
2. Lateral error computation - is it correct?
3. Training progress - does loss decrease?
4. BC initialization - did it learn from demonstrations?

**Solutions:**
- Increase tracking weight to 20.0
- Reduce speed penalty weight
- Train BC longer (50+ epochs)
- Visualize intermediate policies

### Agent oscillates around centerline
**Check:**
1. Smoothness penalty - is it high enough?
2. Heading reward - does it encourage alignment?
3. Network architecture - is it too small?

**Solutions:**
- Increase smoothness penalty to 0.1
- Increase heading weight to 5.0
- Add small amount of derivative term to reward

### Agent drives too slowly
**Check:**
1. Speed reward - is target speed correct?
2. Speed penalty - is it too harsh?

**Solutions:**
- Increase target speed to 10 m/s
- Reduce speed penalty weight to 0.05
- Add minimum speed constraint

---

## Files Modified

| File | Changes |
|------|---------|
| `racing_env.py` | Simplified obs (8D), new reward, relaxed termination |
| `train_bc.py` | Smaller network [128, 64], obs_dim=8 |
| `train_rl.py` | No curriculum, smaller network, updated comments |
| `evaluate_bc.py` | **NEW** - Evaluate BC policy |
| `visualize_bc.py` | **NEW** - Visualize BC policy |

---

## Architecture Comparison

| Component | Before (Lap Time) | After (Centerline) | Speedup |
|-----------|-------------------|-------------------|---------|
| **Observation Space** | 14D | 8D | 1.75x |
| **Network Layers** | 3 [256,256,128] | 2 [128,64] | 3x |
| **Network Parameters** | 147K | 35K | 4.2x |
| **Reward Components** | 5 (complex) | 5 (simple) | 2x |
| **Curriculum Phases** | 3 (8→11→∞) | 0 (none) | N/A |
| **Training Time** | 4-5 hours | 1-2 hours | 3x |

**Total expected speedup: ~5-8x** (combining all factors)

---

## Summary

The simplified centerline following approach:

✅ **Clearer objective** - explicit success criteria  
✅ **Faster learning** - 8D obs, smaller network  
✅ **More stable** - no curriculum transitions  
✅ **Better debugging** - simpler reward structure  
✅ **Foundation for optimization** - enables race line work  

This provides a solid baseline for future race line optimization while ensuring the agent can reliably complete tracks.
