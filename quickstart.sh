#!/bin/bash
# Quick start script to run the complete training pipeline

echo "=========================================="
echo "RL Racing Controller - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "rl_env" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup.sh
else
    echo "Virtual environment found. Activating..."
    source rl_env/bin/activate
fi

echo ""
echo "Starting training pipeline..."
echo ""

# Step 1: Collect demonstrations
echo "Step 1/3: Collecting expert demonstrations..."
echo "This will take ~5-10 minutes"
python collect_demonstrations.py

if [ $? -ne 0 ]; then
    echo "Error in demonstration collection. Exiting."
    exit 1
fi

echo ""
echo "=========================================="

# Step 2: Behavior cloning
echo "Step 2/3: Training behavior cloning policy..."
echo "This will take ~5-15 minutes"
python train_bc.py

if [ $? -ne 0 ]; then
    echo "Error in behavior cloning. Exiting."
    exit 1
fi

echo ""
echo "=========================================="

# Step 3: RL training
echo "Step 3/3: Training SAC agent..."
echo "This will take ~6-12 hours on CPU, ~2-4 hours on GPU"
echo "You can monitor progress with: tensorboard --logdir=sac_logs"
echo ""
read -p "Continue with RL training? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python train_rl.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Training complete!"
        echo ""
        echo "To evaluate the trained policy:"
        echo "  python evaluate_policy.py --model sac_logs/final_model.zip"
        echo ""
        echo "To visualize the policy:"
        echo "  python visualize_policy.py --model sac_logs/final_model.zip"
    fi
else
    echo ""
    echo "RL training skipped. You can run it later with:"
    echo "  python train_rl.py"
fi

echo ""
echo "Done!"
