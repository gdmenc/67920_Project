
import torch
import torch.nn as nn
import numpy as np

class MockMetaActor(nn.Module):
    def __init__(self, num_players=10, rnn_state_size=128):
        super().__init__()
        self.num_players = num_players
        self.rnn_layer_num = 1
        self.rnn_state_size = rnn_state_size
        
    def forward(self, observations, actor_rnn_states):
        # This mirrors the logic in MetaActor.forward that causes the error
        batch_size = observations.shape[0] // self.num_players
        print(f"Batch size derived: {batch_size}")
        print(f"RNN state shape: {actor_rnn_states.shape}")
        
        # This line causes the error
        rnn_states_reshaped = actor_rnn_states.view(batch_size, self.num_players, self.rnn_layer_num, self.rnn_state_size)
        return rnn_states_reshaped

def reproduce_error():
    num_players = 10
    rnn_size = 128
    obs_dim = 133
    
    # Simulate what happens in rollout_func
    # 1. We initialize RNN state for ONE agent (batch_size=1)
    # This is what main_policy.get_initial_state(batch_size=1) returns
    meta_actor_rnn_state = torch.zeros(1, 1, rnn_size) 
    
    # 2. But policy_inputs[agent_id][CUR_OBS] apparently has 10 observations (for the whole team)
    # because MetaActor calculates batch_size = obs.shape[0] // num_players
    # If batch_size is 1, then obs must have 10 items.
    observations = torch.zeros(10, obs_dim)
    
    actor = MockMetaActor(num_players, rnn_size)
    
    print("Attempting forward pass with REPLICATED shapes (The Fix)...")
    try:
        # The Fix: Replicate state for all players
        replicated_state = meta_actor_rnn_state.repeat(num_players, 1, 1)
        print(f"Replicated state shape: {replicated_state.shape}")
        
        output = actor(observations, replicated_state)
        print("Success! Forward pass completed.")
        print(f"Output shape: {output.shape}")
        
    except RuntimeError as e:
        print(f"Caught unexpected error: {e}")

if __name__ == "__main__":
    reproduce_error()
