# 67920_Project

MARL/gfootball implementation for 6.7920 class project, compatible with MIT Engaging cluster.

## Prerequisites

- Access to MIT Engaging cluster (or similar SLURM-based HPC)
- Apptainer/Singularity available (`module load apptainer` on Engaging)

## Quick Start

### 1. Build the Container

First, set up temporary build directories (required for large container builds):

```bash
# Create temporary build and output directories
mkdir -p $(pwd)/apptainer_tmp_build
mkdir -p $(pwd)/containers

# Set environment variables for Apptainer
export APPTAINER_TMPDIR=$(pwd)/apptainer_tmp_build
export SINGULARITY_TMPDIR=$(pwd)/apptainer_tmp_build
```

Then build the container:

```bash
module load apptainer
apptainer build --fakeroot containers/gfootball.sif gfootball.def
```

> **Note:** The build takes ~15-30 minutes. The container includes PyTorch 2.0.1+cu118, Google Research Football, and all MARL dependencies.

### 2. Run Compatibility Tests

Test your container setup before training:

```bash
# Quick test (no GPU required)
apptainer exec containers/gfootball.sif python GRF_MARL/test_grf_marl.py --quick

# Full test with GPU (requires GPU allocation, see below)
apptainer exec --nv containers/gfootball.sif python GRF_MARL/test_grf_marl.py --gpu
```

### 3. Start an Interactive Shell

For development and debugging:

```bash
# CPU-only shell
apptainer shell containers/gfootball.sif

# GPU-enabled shell (requires GPU allocation)
apptainer shell --nv containers/gfootball.sif
```

### 4. Run Training

Training requires GPU resources. START A TMUX SESSION FIRST. Then allocate a GPU node:

```bash
# Request GPU allocation (adjust partition/time as needed; example: 1 GPU)
salloc -p mit_normal_gpu --gpus=1 -t 360 --mem=64G -c 16

# Request specific GPU type (e.g., A100)
salloc -p mit_preemptable --gres=gpu:a100:1 -t 720 --mem=320G -c 128

```

Then run training:

```bash
apptainer exec --nv containers/gfootball.sif bash GRF_MARL/train_light_malib.sh
```

> **Important:** The `--nv` flag is required for GPU passthrough to the container.

## SLURM Examples

### Interactive GPU Session

```bash
# 1 GPU, 6 hours, 64GB RAM, 16 CPUs
salloc -p mit_normal_gpu --gpus=1 -t 360 --mem=64G -c 16

# 2 GPUs, 6 hours, 128GB RAM, 32 CPUs
salloc -p mit_normal_gpu --gpus=2 -t 360 --mem=128G -c 32
```

### Run Tests on GPU Node

```bash
salloc -p mit_normal_gpu --gpus=1 -t 30 --mem=32G -c 8
apptainer exec --nv containers/gfootball.sif python GRF_MARL/test_grf_marl.py --gpu
```

### Hierarchical Meta-Policy (5v5 Hard)

The project includes a hierarchical meta-policy for 5v5 GRF, configured in `GRF_MARL/expr_configs/hierarchical/5_vs_5_hard/hierarchical_meta.yaml`:

- **Meta-policy**: MLP + PPO selecting among fixed, pretrained 5v5 policies in `light_malib/trained_models/gr_football/5_vs_5/` (e.g., `defense_v3`, `PassingMain_v2`, `3-1_formation`, `3-1_LongPass`).
- **Execution**: Rollout workers run low-level GRF control; a top-level policy picks which pretrained sub-policy to use over time using PPO.
- **Training entrypoint**: `GRF_MARL/run_hierarchical_train.sh` (full run) and `GRF_MARL/run_hierarchical_test.sh` (short integration test).

To train the hierarchical meta-policy:

```bash
apptainer exec --nv containers/gfootball.sif bash GRF_MARL/run_hierarchical_train.sh
```

## Project Structure

```
67920_Project/
├── gfootball.def           # Container definition file
├── containers/
│   └── gfootball.sif       # Built container image
├── football/               # Google Research Football (local copy)
├── GRF_MARL/               # Multi-Agent RL framework
│   ├── train_light_malib.sh    # Training script
│   ├── test_grf_marl.py        # Compatibility test suite
│   ├── requirements.txt        # Python dependencies
│   └── expr_configs/           # Experiment configurations
└── test_container.sh       # Container validation wrapper
```

## Troubleshooting

### Container build fails with space errors
Make sure you've set the `APPTAINER_TMPDIR` and `SINGULARITY_TMPDIR` environment variables to a location with sufficient space.

### CUDA not available inside container
Ensure you're using the `--nv` flag when running `apptainer exec` or `apptainer shell`.

### Training stops immediately
Check that `eval_only: False` in your config file (`expr_configs/.../ippo.yaml`). If set to `True`, it only runs evaluation.

## Configuration

Training configs are in `GRF_MARL/expr_configs/`. Key settings in `ippo.yaml`:

```yaml
eval_only: False          # Set to False for actual training
framework:
  max_rounds: 100         # Number of PSRO rounds
  stopper:
    kwargs:
      max_steps: 5000     # Training steps per round
```
