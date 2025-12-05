
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from light_malib.algorithm.hierarchical.policy import HierarchicalMAPPO
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.utils.episode import EpisodeKey
from unittest.mock import MagicMock, patch
import gym
import numpy as np

def debug_shapes():
    print("Debugging RNN Shapes...")
    
    # Mock MAPPO.load to return a dummy policy
    mock_policy = MagicMock()
    mock_policy.feature_encoder = MagicMock()
    mock_policy.feature_encoder.observation_space = gym.spaces.Box(low=-1, high=1, shape=(133,), dtype=np.float32)
    mock_policy.feature_encoder.action_space = gym.spaces.Discrete(19)
    # Mock actor/critic for sub-policy
    mock_policy.actor = MagicMock()
    mock_policy.actor.rnn_layer_num = 1
    mock_policy.actor.rnn_state_size = 128
    
    with patch('light_malib.algorithm.mappo.policy.MAPPO.load', return_value=mock_policy):
    
        # Mock config
        model_config = {
            "model": "gr_football.hierarchical",
            "initialization": {
                "use_orthogonal": True,
                "gain": 1.0
            },
            "actor": {
                "network": "mlp",
                "layers": [
                    {"units": 256, "activation": "ReLU"},
                    {"units": 128, "activation": "ReLU"}
                ],
                "output": {"activation": False}
            },
            "critic": {
                "network": "mlp",
                "layers": [
                    {"units": 256, "activation": "ReLU"},
                    {"units": 128, "activation": "ReLU"}
                ],
                "output": {"activation": False}
            }
        }
        custom_config = {
            "num_agents": 4,
            "FE_cfg": {
                "num_players": 4, # Using 4 for this test
                "encoder_type": "encoder_global_enhanced"
            },
            "sub_policies": [
                {"name": "sub1", "path": "mock_path_1", "model_config": model_config, "custom_config": {}},
                {"name": "sub2", "path": "mock_path_2", "model_config": model_config, "custom_config": {}}
            ],
            "commitment_config": {"mode": "both", "min_steps": 5},
            
            "use_rnn": False,
            "rnn_layer_num": 1,
            "rnn_data_chunk_length": 16,
            "use_feature_normalization": True,
            "use_popart": True,
            "popart_beta": 0.99999,
            "entropy_coef": 0.05,
            "clip_param": 0.2,
            "use_cuda": False
        }
        
        # Mock observation space
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(105,), dtype=np.float32)
        act_space = gym.spaces.Discrete(19)
        
        # Initialize policy
        try:
            policy = HierarchicalMAPPO(
                "gr_football",
                model_config,
                act_space,
                model_config,
                custom_config
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Failed to init policy: {e}")
            return

        # Check initial state shape
        initial_state = policy.get_initial_state(batch_size=1)
        actor_rnn = initial_state[EpisodeKey.ACTOR_RNN_STATE]
        print(f"Initial Actor RNN State (batch=1): {actor_rnn.shape}")
        
        if actor_rnn.shape != (1, 1, 128):
            print("WARNING: Initial state shape is NOT (1, 1, 128)!")
            
        # Simulate rollout loop logic
        meta_actor_rnn_state = actor_rnn.copy()
        num_players = 4
        
        # Create inputs
        obs = np.random.randn(num_players, 105).astype(np.float32)
        
        # Replicate state (The Fix)
        replicated_state = np.repeat(meta_actor_rnn_state, num_players, axis=0)
        print(f"Replicated State: {replicated_state.shape}")
        
        # Compute meta action
        meta_inputs = {
            EpisodeKey.CUR_OBS: obs,
            EpisodeKey.ACTOR_RNN_STATE: replicated_state,
            EpisodeKey.CRITIC_RNN_STATE: replicated_state, # Mocking critic same as actor
            EpisodeKey.ACTION_MASK: np.ones((num_players, 19)),
            EpisodeKey.DONE: np.zeros((num_players, 1))
        }
        
        print("Computing meta action...")
        meta_output = policy.compute_meta_action(
            inference=True,
            explore=True,
            to_numpy=True,
            **meta_inputs
        )
        
        out_rnn = meta_output[EpisodeKey.ACTOR_RNN_STATE]
        print(f"Output RNN State: {out_rnn.shape}")
        
        # Check slicing
        sliced_rnn = out_rnn[0:1].copy()
        print(f"Sliced RNN State: {sliced_rnn.shape}")
        
        if sliced_rnn.shape != (1, 1, 128):
            print("ERROR: Sliced RNN state is not (1, 1, 128)!")

if __name__ == "__main__":
    debug_shapes()

