
import gym
import numpy as np
import torch
import sys
from unittest.mock import MagicMock

import types
# Mock gfootball module before it's imported
sys.modules['gfootball'] = MagicMock()
sys.modules['gfootball.env'] = MagicMock()
sys.modules['gfootball.env.football_env'] = MagicMock()

# Mock light_malib.envs.gr_football as a package
gr_football_pkg = types.ModuleType('light_malib.envs.gr_football')
sys.modules['light_malib.envs.gr_football'] = gr_football_pkg

# Mock submodules
sys.modules['light_malib.envs.gr_football.env'] = MagicMock()
sys.modules['light_malib.envs.gr_football.tools'] = MagicMock()
# Mock encoders as a package
encoders_pkg = types.ModuleType('light_malib.envs.gr_football.encoders')
sys.modules['light_malib.envs.gr_football.encoders'] = encoders_pkg

# Mock dummy encoder
dummy_encoder = types.ModuleType('light_malib.envs.gr_football.encoders.encoder_dummy')
dummy_encoder.FeatureEncoder = MagicMock()
sys.modules['light_malib.envs.gr_football.encoders.encoder_dummy'] = dummy_encoder

# Mock basic encoder
basic_encoder = types.ModuleType('light_malib.envs.gr_football.encoders.encoder_basic')
basic_encoder.FeatureEncoder = MagicMock()
sys.modules['light_malib.envs.gr_football.encoders.encoder_basic'] = basic_encoder

# Mock simple115 encoder
simple_encoder = types.ModuleType('light_malib.envs.gr_football.encoders.encoder_simple115')
simple_encoder.FeatureEncoder = MagicMock()
sys.modules['light_malib.envs.gr_football.encoders.encoder_simple115'] = simple_encoder

from light_malib.algorithm.hierarchical.policy import HierarchicalMAPPO
from light_malib.model.gr_football.hierarchical.meta_actor import MetaActor
from light_malib.model.gr_football.hierarchical.meta_critic import MetaCritic

def test_observation_aggregation():
    print("Testing Hierarchical Model Observation Aggregation...")
    
    # Configuration
    num_players = 4
    obs_dim = 10
    num_sub_policies = 3
    
    # Mock observation space (single agent)
    obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,))
    action_space = gym.spaces.Discrete(19)
    
    # Mock config
    model_config = {
        "model": "gr_football.hierarchical",
        "initialization": {
             "use_orthogonal": True,
             "gain": 1.0
        },
        "actor": {
            "network": "mlp",
            "layers": [{"units": 64, "activation": "ReLU"}],
            "output": {"activation": False}
        },
        "critic": {
            "network": "mlp",
            "layers": [{"units": 64, "activation": "ReLU"}],
            "output": {"activation": False}
        }
    }
    
    custom_config = {
        "num_agents": num_players,
        "sub_policies": [ # Dummy sub-policies
             {"name": "p1", "path": "dummy_path"},
             {"name": "p2", "path": "dummy_path"},
             {"name": "p3", "path": "dummy_path"}
        ],
        "FE_cfg": {
             "num_players": num_players
        }
    }

    # Patch importlib to avoid loading actual sub-policies from disk
    import unittest.mock as mock
    with mock.patch('light_malib.algorithm.hierarchical.policy.MAPPO.load') as mock_load:
        # Mock sub-policy loading
        mock_policy = mock.MagicMock()
        mock_policy.actor = torch.nn.Linear(obs_dim, 19)
        mock_policy.observation_space = obs_space
        mock_policy.feature_encoder = mock.MagicMock()
        mock_load.return_value = mock_policy
        
        # Initialize Policy
        print("Initializing HierarchicalMAPPO...")
        try:
            policy = HierarchicalMAPPO(
                "test_policy",
                obs_space,
                action_space,
                model_config,
                custom_config
            )
            print("Successfully initialized HierarchicalMAPPO.")
        except Exception as e:
            print(f"FAILED to initialize policy: {e}")
            return

        # Check Meta-Observation Space
        expected_meta_dim = num_players * obs_dim
        actual_meta_dim = policy.observation_space.shape[0]
        print(f"Meta Observation Dim: Expected={expected_meta_dim}, Actual={actual_meta_dim}")
        
        if expected_meta_dim == actual_meta_dim:
             print("PASS: Observation space dimension is correct.")
        else:
             print("FAIL: Observation space dimension is incorrect.")
             return

        # Test Forward Pass
        batch_size = 2
        total_obs_rows = batch_size * num_players
        
        # Create dummy observations [batch * num_players, obs_dim]
        observations = torch.randn(total_obs_rows, obs_dim)
        
        # Create dummy RNN states
        actor_rnn_states = torch.zeros(total_obs_rows, 1, 64) # Assuming default RNN size
        critic_rnn_states = torch.zeros(total_obs_rows, 1, 64)
        rnn_masks = torch.ones(total_obs_rows, 1)
        
        print("\nTesting MetaActor forward pass...")
        try:
            # We need to manually invoke the inner actor because policy.compute_meta_action handles data differently
            meta_actions, _, _, _ = policy.actor(
                observations, actor_rnn_states, rnn_masks, None, True, None
            )
            print(f"MetaActor Output Shape: {meta_actions.shape}")
            if meta_actions.shape[0] == total_obs_rows:
                 print("PASS: MetaActor forward pass successful.")
            else:
                 print(f"FAIL: MetaActor output shape mismatch. Expected {total_obs_rows}")
        except Exception as e:
            print(f"FAIL: MetaActor forward pass raised exception: {e}")
            import traceback
            traceback.print_exc()

        print("\nTesting MetaCritic forward pass...")
        try:
            values, _ = policy.critic(
                observations, critic_rnn_states, rnn_masks
            )
            print(f"MetaCritic Output Shape: {values.shape}")
            if values.shape[0] == total_obs_rows:
                 print("PASS: MetaCritic forward pass successful.")
            else:
                 print(f"FAIL: MetaCritic output shape mismatch. Expected {total_obs_rows}")
        except Exception as e:
            print(f"FAIL: MetaCritic forward pass raised exception: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_observation_aggregation()
