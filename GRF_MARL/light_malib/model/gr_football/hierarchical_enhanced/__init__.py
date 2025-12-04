# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Hierarchical model for meta-policy that selects among pre-trained sub-policies.

This module provides:
- MetaActor: Outputs distribution over sub-policies
- MetaCritic: Standard value function for meta-level decisions  
- FeatureEncoder: Reuses the basic encoder from gr_football
"""

from light_malib.model.gr_football.hierarchical.meta_actor import MetaActor as Actor
from light_malib.model.gr_football.hierarchical.meta_critic import MetaCritic as Critic
from light_malib.envs.gr_football.encoders.encoder_enhanced import FeatureEncoder

__all__ = ['Actor', 'Critic', 'FeatureEncoder']

