import numpy as np
import torch
import sys
import os
import gym

# Add project root to path
# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from light_malib.envs.gr_football.encoders import encoder_global_basic, encoder_global_enhanced
from light_malib.model.gr_football.hierarchical.meta_actor import MetaActor
from light_malib.model.gr_football.hierarchical.meta_critic import MetaCritic
from light_malib.utils.episode import EpisodeKey

class MockState:
    def __init__(self, obs):
        self.obs = obs
        self.action_list = []
        
    def get_offside(self, obs):
        # Mock offside return (left_team, right_team)
        return np.zeros(len(obs["left_team"])), np.zeros(len(obs["right_team"]))

def create_dummy_obs(num_players=10):
    num_left = num_players // 2
    num_right = num_players - num_left
    
    obs = {
        "ball": [0.0, 0.0, 0.0],
        "ball_direction": [0.0, 0.0, 0.0],
        "ball_owned_team": -1,
        "game_mode": 0,
        "steps_left": 3000,
        "score": [0, 0],
        
        "left_team": np.zeros((num_left, 2)),
        "left_team_direction": np.zeros((num_left, 2)),
        "left_team_roles": np.zeros(num_left, dtype=int),
        "left_team_tired_factor": np.zeros(num_left),
        "left_team_yellow_card": np.zeros(num_left),
        "left_team_active": np.ones(num_left),
        
        "right_team": np.zeros((num_right, 2)),
        "right_team_direction": np.zeros((num_right, 2)),
        "right_team_tired_factor": np.zeros(num_right),
        "right_team_yellow_card": np.zeros(num_right),
        "right_team_active": np.ones(num_right),
        
        "sticky_actions": np.zeros(10), # For active player
    }
    return obs

def test_encoder(encoder_cls, name):
    print(f"\n--- Testing {name} ---")
    num_players = 10
    encoder = encoder_cls.FeatureEncoder(num_players=num_players)
    
    # Create dummy states for 5 players (left team)
    states = []
    base_obs = create_dummy_obs(num_players)
    for i in range(5):
        # Each player has slightly different sticky actions to test aggregation
        p_obs = base_obs.copy()
        p_obs["sticky_actions"] = np.zeros(10)
        p_obs["sticky_actions"][i] = 1.0 # Set one sticky bit
        states.append(MockState(p_obs))
        
    encoded = encoder.encode(states)
    
    print(f"Encoded count: {len(encoded)}")
    print(f"Feature dim: {encoded[0].shape}")
    print(f"Observation Space: {encoder.observation_space}")
    
    # Verify consistency
    first = encoded[0]
    for i, feat in enumerate(encoded):
        if not np.array_equal(feat, first):
            print(f"ERROR: Feature mismatch at index {i}")
            return None
    print("Consistency Check: PASS (All players get identical global state)")
    
    return encoded[0].shape[0]

def test_models(obs_dim):
    print("\n--- Testing Meta Models ---")
    num_players = 5 # Controlled players
    batch_size = 2 # Batches
    
    # Mock Configs
    model_config = {
        "actor": {
            "network": "mlp", 
            "rnn_layer_num": 1, 
            "rnn_state_size": 64, 
            "fc_layer_dim": [64], 
            "output": {"activation": False},
            "layers": [{"units": 64, "activation": "ReLU"}]
        },
        "critic": {
            "network": "mlp", 
            "rnn_layer_num": 1, 
            "rnn_state_size": 64, 
            "fc_layer_dim": [64], 
            "output": {"activation": False},
            "layers": [{"units": 64, "activation": "ReLU"}]
        },
        "initialization": {"use_orthogonal": True, "gain": 1.0}
    }
    custom_config = {
        "use_rnn": True,
        "rnn_layer_num": 1,
        "num_agents": num_players, # Important for reshaping
        "use_feature_normalization": True,
        "use_cuda": False
    }
    
    # Inputs
    # Batch of observations: [batch * num_players, obs_dim]
    total_batch = batch_size * num_players
    obs = torch.randn(total_batch, obs_dim)
    # Ensure identical global states per team
    obs_reshaped = obs.view(batch_size, num_players, obs_dim)
    obs_reshaped[:, :, :] = obs_reshaped[:, 0:1, :] # Broadcast first to all
    obs = obs_reshaped.view(total_batch, obs_dim)
    
    rnn_states = torch.zeros(total_batch, 1, 64)
    masks = torch.ones(total_batch, 1)
    
    # Create proper mock observation space
    mock_obs_space = gym.spaces.Box(low=-1000, high=1000, shape=(obs_dim,))
    
    # --- Actor ---
    actor = MetaActor(
        model_config["actor"],
        mock_obs_space,
        num_sub_policies=4,
        custom_config=custom_config,
        initialization=model_config["initialization"]
    )
    
    actions, _, _, _ = actor(obs, rnn_states, masks, None, explore=True, actions=None)
    
    print(f"Actor Output Shape: {actions.shape}")
    if actions.shape[0] != total_batch:
        print("ERROR: Actor output shape mismatch")
        
    # Verify broadcasting (all players in a batch should have same action)
    actions_reshaped = actions.view(batch_size, num_players)
    for b in range(batch_size):
        team_actions = actions_reshaped[b]
        if not torch.all(team_actions == team_actions[0]):
             print(f"ERROR: Actor broadcasting failed for batch {b}. Actions: {team_actions}")
        else:
             print(f"Batch {b} Actions: {team_actions} (Consistent)")

    # --- Critic ---
    critic = MetaCritic(
        model_config["critic"],
        mock_obs_space,
        custom_config=custom_config,
        initialization=model_config["initialization"]
    )
    
    values, _ = critic(obs, rnn_states, masks)
    
    print(f"Critic Output Shape: {values.shape}")
    
    values_reshaped = values.view(batch_size, num_players)
    for b in range(batch_size):
        team_values = values_reshaped[b]
        if not torch.all(team_values == team_values[0]):
             print(f"ERROR: Critic broadcasting failed for batch {b}")
        else:
             print(f"Batch {b} Values Consistent")

if __name__ == "__main__":
    dim_basic = test_encoder(encoder_global_basic, "Global Basic")
    dim_enhanced = test_encoder(encoder_global_enhanced, "Global Enhanced")
    
    if dim_enhanced:
        test_models(dim_enhanced)

    # --- Integration Test with Policy Class ---
    print("\n--- Testing HierarchicalMAPPO Integration ---")
    from light_malib.algorithm.hierarchical.policy import HierarchicalMAPPO
    
    # Mock full config based on hierarchical_meta_test.yaml
    policy_config = {
        "registered_name": "HierarchicalMAPPO",
        "observation_space": gym.spaces.Box(low=-1000, high=1000, shape=(100,)), # Dummy, will be overwritten by encoder
        "action_space": gym.spaces.Discrete(4), # 4 sub-policies
        "model_config": {
            "model": "gr_football.hierarchical",
            "initialization": {"use_orthogonal": True, "gain": 1.0},
            "actor": {"network": "mlp", "rnn_layer_num": 1, "rnn_state_size": 64, "fc_layer_dim": [64], "output": {"activation": False}, "layers": [{"units": 64, "activation": "ReLU"}]},
            "critic": {"network": "mlp", "rnn_layer_num": 1, "rnn_state_size": 64, "fc_layer_dim": [64], "output": {"activation": False}, "layers": [{"units": 64, "activation": "ReLU"}]}
        },
        "custom_config": {
            "FE_cfg": {
                "num_players": 10,
                "encoder_type": "encoder_global_enhanced" # Explicitly test dynamic loading
            },
            "sub_policies": [
                {"name": "mock_p1", "path": "/tmp/mock_p1"}, # Paths won't be loaded in this mock test unless we mock load
                {"name": "mock_p2", "path": "/tmp/mock_p2"}
            ],
            "commitment": {"mode": "steps", "min_steps": 10},
            "use_rnn": True,
            "rnn_layer_num": 1,
            "use_feature_normalization": True,
            "use_cuda": False
        }
    }
    
    # We need to mock _load_sub_policies to avoid actual file loading
    original_load = HierarchicalMAPPO._load_sub_policies
    HierarchicalMAPPO._load_sub_policies = lambda self: None # Skip loading
    
    try:
        policy = HierarchicalMAPPO(**policy_config)
        print(f"Policy initialized with encoder: {type(policy.feature_encoder).__name__}")
        print(f"Policy observation space: {policy.observation_space}")
        
        if "GlobalEnhanced" not in type(policy.feature_encoder).__name__ and "FeatureEncoder" not in str(type(policy.feature_encoder)):
             # The class name is FeatureEncoder, but module should be encoder_global_enhanced
             pass 
        
        # Verify it's the right module
        if policy.feature_encoder.__module__.endswith("encoder_global_enhanced"):
            print("SUCCESS: Loaded encoder_global_enhanced")
        else:
            print(f"ERROR: Loaded wrong encoder module: {policy.feature_encoder.__module__}")

        # Test compute_action (training mode)
        batch_size = 2
        num_players = 10
        obs_dim = policy.observation_space.shape[0]
        
        obs = np.random.randn(batch_size * num_players, obs_dim).astype(np.float32)
        # Broadcast for consistency
        obs = obs.reshape(batch_size, num_players, obs_dim)
        obs[:, :, :] = obs[:, 0:1, :]
        obs = obs.reshape(batch_size * num_players, obs_dim)
        
        rnn_states = np.zeros((batch_size * num_players, 1, 64), dtype=np.float32)
        masks = np.ones((batch_size * num_players, 1), dtype=np.float32)
        
        output = policy.compute_action(
            inference=False,
            **{
                EpisodeKey.CUR_OBS: obs,
                EpisodeKey.ACTOR_RNN_STATE: rnn_states,
                EpisodeKey.CRITIC_RNN_STATE: rnn_states,
                EpisodeKey.DONE: masks,
                "explore": True
            }
        )
        
        actions = output[EpisodeKey.ACTION]
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        print(f"Policy output actions shape: {actions.shape}")
        
        # Verify broadcasting
        actions_reshaped = actions.reshape(batch_size, num_players)
        if np.all(actions_reshaped[:, 0:1] == actions_reshaped):
            print("SUCCESS: Policy actions are consistent across team")
        else:
            print("ERROR: Policy actions are NOT consistent")
            
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        HierarchicalMAPPO._load_sub_policies = original_load # Restore
