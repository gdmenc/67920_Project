#!/bin/bash
# Full training run for Hierarchical Meta-Policy
# Trains meta-policy to select among pre-trained sub-policies
# 
# Hardware requirements: 128 cores, 2 A100 GPUs, 320GB RAM
# Expected runtime: Several hours to days depending on convergence
# Target: ~600M environment steps (2000 iterations × 100 workers × 3001 steps)

set -e  # Exit on error

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Load environment variables from .env if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
fi

# Login to wandb if API key is available
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging into wandb..."
    wandb login "$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY not found in environment. Wandb logging may fail."
    echo "Set WANDB_API_KEY in .env file or run 'wandb login' manually."
fi

# Stop any existing Ray cluster
echo "Checking for existing Ray cluster..."
if ray status &> /dev/null; then
    echo "Stopping existing Ray cluster..."
    ray stop
fi

# Start Ray cluster with all available resources
echo "Starting Ray cluster..."
ray start --head

echo ""
echo "=========================================="
echo "  Hierarchical Meta-Policy Training"
echo "  Config: hierarchical_meta.yaml"
echo "  Wandb project: gr_football-hierarchical_meta_mlp_ppo_11_vs_11_hard"
echo "=========================================="
echo ""

# Run training
python3 light_malib/main_pbt.py --config expr_configs/hierarchical/11_vs_11_hard/hierarchical_meta.yaml

# Cleanup
echo "Stopping Ray cluster..."
ray stop

echo ""
echo "Training completed!"
echo "Check wandb for results: https://wandb.ai"
echo "Logs saved to: ./logs/"

