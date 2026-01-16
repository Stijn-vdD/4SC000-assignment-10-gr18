#!/bin/bash
# Setup script for RL racing controller training

echo "Setting up RL training environment..."

# Activate virtual environment
source rl_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source rl_env/bin/activate"
echo ""
echo "Then follow the training pipeline in README_RL.md"
