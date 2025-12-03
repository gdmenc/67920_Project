#!/usr/bin/env python3
"""
Script to inspect pretrained model architectures and parameter shapes.
Run with: python3 inspect_models.py
"""

import torch
import os
import pickle

def inspect_state_dict(sd, name="Model"):
    """Print state dict layer names and shapes"""
    print(f"\n{name} Layers:")
    total_params = 0
    for layer_name, param in sd.items():
        if hasattr(param, 'shape'):
            num_params = param.numel()
            total_params += num_params
            print(f"  {layer_name}: {list(param.shape)} ({num_params:,} params)")
        else:
            print(f"  {layer_name}: {type(param).__name__}")
    print(f"  Total parameters: {total_params:,}")
    return total_params

def inspect_model_file(model_path):
    """Load and inspect a .pt model file"""
    try:
        model = torch.load(model_path, map_location='cpu')
        
        # Handle different save formats
        if hasattr(model, 'state_dict'):
            # It's a nn.Module
            return model.state_dict(), type(model).__name__
        elif isinstance(model, dict):
            # It's already a state dict
            return model, "state_dict"
        else:
            return None, f"Unknown format: {type(model)}"
    except Exception as e:
        return None, f"Error: {e}"

def main():
    models_dir = "light_malib/trained_models/gr_football/5_vs_5"
    
    print("=" * 70)
    print("PRETRAINED MODEL INSPECTION")
    print("=" * 70)

    
    for model_name in sorted(os.listdir(models_dir)):
        model_path = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        actor_path = os.path.join(model_path, "actor.pt")
        critic_path = os.path.join(model_path, "critic.pt")
        desc_path = os.path.join(model_path, "desc.pkl")
        
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")
        
        # Check what files exist
        files = []
        for f in ['actor.pt', 'critic.pt', 'desc.pkl']:
            if os.path.exists(os.path.join(model_path, f)):
                files.append(f)
        print(f"Files present: {files}")
        
        # Inspect actor
        if os.path.exists(actor_path):
            sd, model_type = inspect_model_file(actor_path)
            if sd:
                print(f"\nACTOR ({model_type}):")
                inspect_state_dict(sd, "Actor")
                
                # Try to infer architecture from layer names
                if any('rnn' in k.lower() for k in sd.keys()):
                    print("  [Has RNN layers]")
                if any('transformer' in k.lower() or 'attention' in k.lower() for k in sd.keys()):
                    print("  [Has Transformer/Attention layers]")
            else:
                print(f"\nACTOR: {model_type}")
        
        # Inspect critic
        if os.path.exists(critic_path):
            sd, model_type = inspect_model_file(critic_path)
            if sd:
                print(f"\nCRITIC ({model_type}):")
                inspect_state_dict(sd, "Critic")
            else:
                print(f"\nCRITIC: {model_type}")
        
        # Try to read desc.pkl (may fail due to dependencies)
        if os.path.exists(desc_path):
            try:
                # Try with restricted unpickler to avoid import errors
                import io
                with open(desc_path, 'rb') as f:
                    content = f.read()
                print(f"\nDESC.PKL: {len(content)} bytes")
            except Exception as e:
                print(f"\nDESC.PKL: Could not read ({e})")
    
    # Also check 11v11 models
    models_dir_11 = "light_malib/trained_models/gr_football/11_vs_11"
    if os.path.exists(models_dir_11):
        print(f"\n\n{'=' * 70}")
        print("11 VS 11 MODELS")
        print("=" * 70)
        
        for model_name in sorted(os.listdir(models_dir_11)):
            model_path = os.path.join(models_dir_11, model_name)
            if not os.path.isdir(model_path):
                continue
                
            actor_path = os.path.join(model_path, "actor.pt")
            
            print(f"\n{'=' * 70}")
            print(f"MODEL: {model_name}")
            print(f"{'=' * 70}")
            
            files = [f for f in os.listdir(model_path)]
            print(f"Files present: {files}")
            
            if os.path.exists(actor_path):
                sd, model_type = inspect_model_file(actor_path)
                if sd:
                    print(f"\nACTOR ({model_type}):")
                    inspect_state_dict(sd, "Actor")
                else:
                    print(f"\nACTOR: {model_type}")

if __name__ == "__main__":
    main()

