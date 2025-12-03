# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Hierarchical MAPPO Trainer.

Trainer for the hierarchical meta-policy. Uses standard PPO training
since the meta-policy is just selecting among discrete sub-policy options.
"""

from light_malib.algorithm.mappo.trainer import MAPPOTrainer
from .loss import HierarchicalMAPPOLoss
from light_malib.registry import registry


@registry.registered(registry.TRAINER)
class HierarchicalMAPPOTrainer(MAPPOTrainer):
    """
    Trainer for Hierarchical MAPPO.
    
    Uses the same training procedure as MAPPO since the meta-policy
    is trained with standard PPO on (meta_obs, meta_action, meta_reward) tuples.
    """
    
    def __init__(self, tid):
        super().__init__(tid)
        self.id = tid
        # Use the hierarchical loss instead of standard MAPPO loss
        self._loss = HierarchicalMAPPOLoss()
    
    def optimize(self, batch, **kwargs):
        """
        Optimize the meta-policy.
        
        The batch contains meta-level data:
        - CUR_OBS: Observations when meta-decisions were made
        - ACTION: Sub-policy indices selected
        - REWARD: Accumulated rewards during sub-policy execution
        - DONE: Episode termination flags
        """
        return super().optimize(batch, **kwargs)

