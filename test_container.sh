#!/bin/bash
# Container validation script
# Usage: ./test_container.sh [container.sif] [--gpu]

set -e

CONTAINER="${1:-containers/gfootball.sif}"
EXTRA_ARGS="${@:2}"

if [ ! -f "$CONTAINER" ]; then
    echo "❌ Container not found: $CONTAINER"
    exit 1
fi

echo "=============================================="
echo "Testing container: $CONTAINER"
echo "=============================================="

# Determine if we should use --nv for GPU passthrough
NV_FLAG=""
if [[ "$EXTRA_ARGS" == *"--gpu"* ]]; then
    NV_FLAG="--nv"
    echo "GPU mode enabled (--nv)"
fi

# Run the test suite
apptainer exec $NV_FLAG "$CONTAINER" python /home/maxt114/67920_Project/GRF_MARL/test_grf_marl.py $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Container validation PASSED"
else
    echo ""
    echo "❌ Container validation FAILED"
fi

exit $EXIT_CODE

