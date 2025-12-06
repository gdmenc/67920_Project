# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Meta-policy actor for hierarchical RL.
Outputs a distribution over sub-policies instead of low-level actions.
"""

import torch
import torch.nn as nn
from gym.spaces import Discrete

from light_malib.algorithm.common.rnn_net import RNNNet


class MetaActor(nn.Module):
    """
    Meta-policy actor that selects which sub-policy to use.
    
    Unlike the standard Actor that outputs 19 actions, this outputs
    logits over K sub-policies.
    """
    
    def __init__(
        self,
        model_config,
        observation_space,
        num_sub_policies: int,
        custom_config,
        initialization,
    ):
        super().__init__()
        
        self.num_sub_policies = num_sub_policies
        
        # Extract num_players for reshaping
        # Try to find it in custom_config, fallback to FE_cfg if needed
        if "num_agents" in custom_config:
            self.num_players = custom_config["num_agents"]
        elif "FE_cfg" in custom_config and "num_players" in custom_config["FE_cfg"]:
            self.num_players = custom_config["FE_cfg"]["num_players"]
        else:
            # Fallback/Default (e.g. for 5v5)
            self.num_players = 10 
            
        # Create a modified action space for the meta-policy
        meta_action_space = Discrete(num_sub_policies)
        
        # Use the same RNNNet base as regular actors
        # But with action_space = num_sub_policies instead of 19
        self._base_net = RNNNet(
            model_config,
            observation_space,
            meta_action_space,
            custom_config,
            initialization,
        )
        
        # Copy RNN properties
        self.rnn_layer_num = self._base_net.rnn_layer_num
        self.rnn_state_size = self._base_net.rnn_state_size
        
    def forward(self, observations, actor_rnn_states, rnn_masks, action_masks, explore, actions):
        """
        Forward pass for meta-policy with Global State.
        
        Args:
            observations: [batch_size * num_players, global_obs_dim]
                          (Each player has identical global state)
            ...
            
        Returns:
            actions: [batch_size * num_players] (Identical for players in same team)
            ...
        """
        # Reshape to [batch, num_players, obs_dim]
        # We assume batch_size is divisible by num_players
        batch_size = observations.shape[0] // self.num_players
        obs_dim = observations.shape[1]
        
        obs_reshaped = observations.view(batch_size, self.num_players, obs_dim)
        
        # Flatten observations from all players to form the global state
        # Shape: [batch, num_players * obs_dim]
        # This gives the meta-policy access to everything any agent sees
        global_obs = obs_reshaped.view(batch_size, -1)
        
        # Handle RNN states
        # actor_rnn_states: [batch * num_players, layer, hidden]
        # We only need one RNN state per team (since they act as one unit)
        # But the framework maintains one per player.
        # We'll take the first player's RNN state, update it, and broadcast back.
        rnn_states_reshaped = actor_rnn_states.view(batch_size, self.num_players, self.rnn_layer_num, self.rnn_state_size)
        global_rnn_states = rnn_states_reshaped[:, 0, :, :]
        
        # Handle Masks
        # rnn_masks: [batch * num_players, 1]
        masks_reshaped = rnn_masks.view(batch_size, self.num_players, 1)
        global_masks = masks_reshaped[:, 0, :]
        
        # Forward pass with single global state per team
        # logits: [batch, num_sub_policies]
        # new_rnn_states: [batch, layer, hidden]
        logits, new_rnn_states = self._base_net.forward(
            global_obs, global_rnn_states, global_masks
        )
        
        # Sample actions
        dist = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            # Sample once per team
            team_actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            # If actions provided (training), they should be [batch * num_players]
            # We reshape and take first to check log_prob
            actions_reshaped = actions.view(batch_size, self.num_players)
            team_actions = actions_reshaped[:, 0]
            dist_entropy = dist.entropy()
            
        # Compute log probs
        # action_log_probs: [batch]
        team_action_log_probs = dist.log_prob(team_actions)
        
        # Broadcast results back to all players
        # actions: [batch] -> [batch, num_players] -> [batch * num_players]
        actions_broadcast = team_actions.unsqueeze(1).repeat(1, self.num_players).view(-1)
        
        # log_probs: [batch] -> [batch * num_players]
        log_probs_broadcast = team_action_log_probs.unsqueeze(1).repeat(1, self.num_players).view(-1, 1)
        
        # rnn_states: [batch, layer, hidden] -> [batch * num_players, layer, hidden]
        rnn_states_broadcast = new_rnn_states.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, self.rnn_layer_num, self.rnn_state_size)
        
        # entropy: [batch] -> [batch * num_players]
        if dist_entropy is not None:
            entropy_broadcast = dist_entropy.unsqueeze(1).repeat(1, self.num_players).view(-1, 1)
        else:
            entropy_broadcast = None
            
        return actions_broadcast, rnn_states_broadcast, log_probs_broadcast, entropy_broadcast
    
    def logits(self, obs, rnn_states, masks):
        """Get raw logits without sampling."""
        # Similar reshaping logic
        batch_size = obs.shape[0] // self.num_players
        obs_reshaped = obs.view(batch_size, self.num_players, -1)
        # Global obs is concatenation of all
        global_obs = obs_reshaped.view(batch_size, -1)
        
        rnn_states_reshaped = rnn_states.view(batch_size, self.num_players, self.rnn_layer_num, self.rnn_state_size)
        global_rnn_states = rnn_states_reshaped[:, 0, :, :]
        
        masks_reshaped = masks.view(batch_size, self.num_players, 1)
        global_masks = masks_reshaped[:, 0, :]
        
        logits, _ = self._base_net.forward(global_obs, global_rnn_states, global_masks)
        
        # Broadcast logits
        logits_broadcast = logits.unsqueeze(1).repeat(1, self.num_players, 1).view(-1, self.num_sub_policies)
        
        return logits_broadcast

