# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Hierarchical MAPPO Loss.

The meta-policy uses standard PPO loss since it's just selecting
among discrete sub-policy options. We inherit from MAPPOLoss
with minimal modifications.
"""

from light_malib.algorithm.mappo.loss import MAPPOLoss
from light_malib.registry import registry


@registry.registered(registry.LOSS)
class HierarchicalMAPPOLoss(MAPPOLoss):
    """
    Loss function for Hierarchical MAPPO.
    
    Since the meta-policy is just a standard policy that outputs
    discrete actions (sub-policy indices), we can use the exact
    same PPO loss as MAPPO.
    
    The key difference is that the "actions" being evaluated are
    sub-policy indices (0-3) rather than low-level game actions (0-18).
    """
    
    def __init__(self):
        super(HierarchicalMAPPOLoss, self).__init__()
        
    def reset(self, policy, config):
        """
        Reset loss function for new training task.
        
        Override to ensure we're using the correct algorithm name.
        """
        # Call parent reset
        super().reset(policy, config)
        
        # Force MAPPO behavior (no sequential updates for hierarchical)
        # The meta-policy is a single policy making team-wide decisions
        self._use_seq = False
        self._use_two_stage = False
        self._use_co_ma_ratio = False

