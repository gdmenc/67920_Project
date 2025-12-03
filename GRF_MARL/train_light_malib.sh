#!/bin/bash
# Works with Apptainer's default home directory binding
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"

ray start --head

python3 light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/ippo.yaml

ray stop
