#!/usr/bin/env python3
"""
Convert pre-trained models from CUDA to CPU storage.

This fixes Ray serialization issues where models saved on CUDA fail to 
deserialize on CPU-only Ray workers (like PolicyServer).

Run this ONCE before training:
    python utils/convert_models_to_cpu.py
"""

import torch
import os
import sys

# Add project root to path (go up one level from utils/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIRS = [
    "light_malib/trained_models/gr_football/5_vs_5/defense_v3",
    "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2",
    "light_malib/trained_models/gr_football/5_vs_5/3-1_formation",
    "light_malib/trained_models/gr_football/5_vs_5/3-1_LongPass",
]

MODEL_FILES = ["actor.pt", "critic.pt", "backbone.pt"]


def convert_model_to_cpu(path):
    """Load a model and re-save it with CPU tensors."""
    print(f"  Loading {path}...")
    
    # Load with CPU mapping
    model = torch.load(path, map_location='cpu')
    
    # Check if it's a nn.Module or state_dict
    if hasattr(model, 'state_dict'):
        # It's a full model - move to CPU and get fresh state
        model = model.cpu()
        # Create backup
        backup_path = path + '.cuda_backup'
        if not os.path.exists(backup_path):
            os.rename(path, backup_path)
            print(f"  Backup saved to {backup_path}")
        # Save the CPU model
        torch.save(model, path)
    else:
        # It's already a state_dict - ensure all tensors are CPU
        cpu_state = {k: v.cpu() for k, v in model.items()}
        backup_path = path + '.cuda_backup'
        if not os.path.exists(backup_path):
            os.rename(path, backup_path)
            print(f"  Backup saved to {backup_path}")
        torch.save(cpu_state, path)
    
    print(f"  Saved CPU version to {path}")
    return True


def main():
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print("=" * 60)
    print("Converting pre-trained models from CUDA to CPU storage")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    print()
    
    converted = 0
    skipped = 0
    errors = 0
    
    for model_dir in MODEL_DIRS:
        if not os.path.exists(model_dir):
            print(f"[SKIP] Directory not found: {model_dir}")
            skipped += 1
            continue
            
        print(f"\n[DIR] {model_dir}")
        
        for fname in MODEL_FILES:
            path = os.path.join(model_dir, fname)
            
            if not os.path.exists(path):
                continue
                
            try:
                convert_model_to_cpu(path)
                converted += 1
            except Exception as e:
                print(f"  [ERROR] Failed to convert {path}: {e}")
                errors += 1
    
    print()
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print("=" * 60)
    
    if errors > 0:
        print("\nSome conversions failed. Check the errors above.")
        return 1
    
    print("\nYou can now run the training script!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

