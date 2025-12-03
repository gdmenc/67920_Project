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
        Forward pass for meta-policy.
        
        Args:
            observations: Current observations [batch_size, obs_dim]
            actor_rnn_states: RNN hidden states (if using RNN)
            rnn_masks: Masks for RNN (done flags)
            action_masks: Not used for meta-policy (all sub-policies always available)
            explore: Whether to sample or take argmax
            actions: If provided, compute log_prob of these actions
            
        Returns:
            actions: Selected sub-policy indices [batch_size]
            actor_rnn_states: Updated RNN states
            action_log_probs: Log probabilities of selected actions
            dist_entropy: Entropy of the distribution (if actions provided)
        """
        # Get logits from base network
        logits, actor_rnn_states = self._base_net.forward(
            observations, actor_rnn_states, rnn_masks
        )
        
        # For meta-policy, all sub-policies are always available
        # No action masking needed
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = None
        else:
            dist_entropy = dist.entropy()
            
        action_log_probs = dist.log_prob(actions)
        
        return actions, actor_rnn_states, action_log_probs, dist_entropy
    
    def logits(self, obs, rnn_states, masks):
        """Get raw logits without sampling."""
        return self._base_net.forward(obs, rnn_states, masks)

