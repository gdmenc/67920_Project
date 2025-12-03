#!/bin/bash
# Integration test for Hierarchical Meta-Policy
# Runs 3 iterations with 4 workers to verify pipeline works end-to-end

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

# Start Ray cluster
echo "Starting Ray cluster..."
ray start --head

echo ""
echo "=========================================="
echo "  Running Hierarchical Integration Test"
echo "  Config: hierarchical_meta_test.yaml"
echo "  wandb project: gr_football-hierarchical_meta_test_5_vs_5_hard"
echo "=========================================="
echo ""

# Run integration test
python3 light_malib/main_pbt.py --config expr_configs/hierarchical/5_vs_5_hard/hierarchical_meta_test.yaml

# Cleanup
echo "Stopping Ray cluster..."
ray stop

echo ""
echo "Integration test completed!"

