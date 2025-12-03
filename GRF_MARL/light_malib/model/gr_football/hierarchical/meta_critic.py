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
        Forward pass for meta-critic.
        
        Args:
            observations: Current observations [batch_size, obs_dim]
            critic_rnn_states: RNN hidden states (if using RNN)
            rnn_masks: Masks for RNN (done flags)
            
        Returns:
            values: State values [batch_size, 1]
            critic_rnn_states: Updated RNN states
        """
        values, critic_rnn_states = self._base_net.forward(
            observations, critic_rnn_states, rnn_masks
        )
        
        return values, critic_rnn_states

