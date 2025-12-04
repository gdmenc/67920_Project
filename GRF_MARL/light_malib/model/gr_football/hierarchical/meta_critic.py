# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Meta-policy critic for hierarchical RL.
Standard value function V(s) for the meta-policy.
"""

import torch
import torch.nn as nn
from gym.spaces import Discrete

from light_malib.algorithm.common.rnn_net import RNNNet


class MetaCritic(nn.Module):
    """
    Meta-policy critic that estimates state value for meta-level decisions.
    
    This is essentially the same as the standard critic, outputting V(s).
    """
    
    def __init__(
        self,
        model_config,
        observation_space,
        custom_config,
        initialization,
    ):
        super().__init__()
        
        # Extract num_players for reshaping
        if "num_agents" in custom_config:
            self.num_players = custom_config["num_agents"]
        elif "FE_cfg" in custom_config and "num_players" in custom_config["FE_cfg"]:
            self.num_players = custom_config["FE_cfg"]["num_players"]
        else:
            self.num_players = 10 
        
        # Critic outputs a single value V(s)
        value_space = Discrete(1)
        
        self._base_net = RNNNet(
            model_config,
            observation_space,
            value_space,
            custom_config,
            initialization,
        )
        
        # Copy RNN properties
        self.rnn_layer_num = self._base_net.rnn_layer_num
        self.rnn_state_size = self._base_net.rnn_state_size
        
    def forward(self, observations, critic_rnn_states, rnn_masks):
        """
        Forward pass for meta-critic with Global State.
        
        Args:
            observations: [batch_size * num_players, global_obs_dim]
            critic_rnn_states: RNN hidden states
            rnn_masks: Masks for RNN
            
        Returns:
            values: State values [batch_size * num_players, 1]
            critic_rnn_states: Updated RNN states
        """
        # Reshape to [batch, num_players, obs_dim]
        batch_size = observations.shape[0] // self.num_players
        obs_dim = observations.shape[1]
        
        obs_reshaped = observations.view(batch_size, self.num_players, obs_dim)
        global_obs = obs_reshaped[:, 0, :]
        
        # Handle RNN states
        rnn_states_reshaped = critic_rnn_states.view(batch_size, self.num_players, self.rnn_layer_num, self.rnn_state_size)
        global_rnn_states = rnn_states_reshaped[:, 0, :, :]
        
        # Handle Masks
        masks_reshaped = rnn_masks.view(batch_size, self.num_players, 1)
        global_masks = masks_reshaped[:, 0, :]
        
        # Forward pass
        # values: [batch, 1]
        values, new_rnn_states = self._base_net.forward(
            global_obs, global_rnn_states, global_masks
        )
        
        # Broadcast results back
        # values: [batch, 1] -> [batch, num_players, 1] -> [batch * num_players, 1]
        values_broadcast = values.unsqueeze(1).repeat(1, self.num_players, 1).view(-1, 1)
        
        # rnn_states: [batch, layer, hidden] -> [batch * num_players, layer, hidden]
        rnn_states_broadcast = new_rnn_states.unsqueeze(1).repeat(1, self.num_players, 1, 1).view(-1, self.rnn_layer_num, self.rnn_state_size)
        
        return values_broadcast, rnn_states_broadcast

