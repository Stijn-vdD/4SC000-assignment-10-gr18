#!/bin/bash
# Optimized training pipeline with parallel processing

echo "=========================================="
echo "RL Racing Controller - Optimized Training"
echo "=========================================="
echo ""

# Detect CPU cores
N_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Detected $N_CORES CPU cores"

# Calculate optimal worker counts
DEMO_WORKERS=$((N_CORES - 1))
DEMO_WORKERS=$((DEMO_WORKERS > 1 ? DEMO_WORKERS : 1))

BC_WORKERS=$((N_CORES / 2))
BC_WORKERS=$((BC_WORKERS > 1 ? BC_WORKERS : 1))

RL_ENVS=$((N_CORES - 1))
RL_ENVS=$((RL_ENVS > 8 ? 8 : RL_ENVS))  # Cap at 8
RL_ENVS=$((RL_ENVS < 2 ? 2 : RL_ENVS))  # Min 2

echo "Optimization settings:"
echo "  - Demonstration workers: $DEMO_WORKERS"
echo "  - BC data workers: $BC_WORKERS"
echo "  - RL parallel envs: $RL_ENVS"
echo ""

# Check if virtual environment exists
if [ ! -d "rl_env" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup.sh
else
    echo "Activating virtual environment..."
    source rl_env/bin/activate
fi

echo ""
echo "=========================================="
echo "Step 1/3: Collecting Expert Demonstrations"
echo "=========================================="
echo "Using $DEMO_WORKERS parallel workers"
echo "Expected time: ~3-5 minutes (vs ~10 min sequential)"
echo ""

time python collect_demonstrations.py --episodes 100 --workers $DEMO_WORKERS

if [ $? -ne 0 ]; then
    echo "Error in demonstration collection. Exiting."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2/3: Training Behavior Cloning Policy"
echo "=========================================="
echo "Using $BC_WORKERS data loading workers"
echo "Expected time: ~3-8 minutes (vs ~10-15 min sequential)"
echo ""

time python train_bc.py --epochs 50 --batch-size 256 --workers $BC_WORKERS

if [ $? -ne 0 ]; then
    echo "Error in behavior cloning. Exiting."
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3/3: Training SAC Agent"
echo "=========================================="
echo "Using $RL_ENVS parallel environments"
echo "Expected time: ~4-8 hours on CPU with $RL_ENVS envs"
echo "(vs ~10-15 hours with single env)"
echo ""
echo "Monitor with: tensorboard --logdir=sac_logs"
echo ""

read -p "Continue with RL training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    time python train_rl.py --timesteps 500000 --envs $RL_ENVS --bc-checkpoint bc_policy.pth
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Training Complete!"
        echo "=========================================="
        echo ""
        echo "Performance summary:"
        echo "  - Demonstration collection: ~5x speedup with $DEMO_WORKERS workers"
        echo "  - BC training: ~2x speedup with $BC_WORKERS data workers"
        echo "  - RL training: ~${RL_ENVS}x speedup with $RL_ENVS parallel envs"
        echo ""
        echo "Total estimated speedup: ~3-5x faster than sequential"
        echo ""
        echo "Evaluate with:"
        echo "  python evaluate_policy.py --model sac_logs/final_model.zip"
        echo ""
        echo "Visualize with:"
        echo "  python visualize_policy.py --model sac_logs/final_model.zip"
    fi
else
    echo ""
    echo "RL training skipped. Run manually with:"
    echo "  python train_rl.py --envs $RL_ENVS"
fi

echo ""
echo "Done!"
