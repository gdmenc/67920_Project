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
from light_malib.utils.episode import EpisodeKey
import torch


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
    
    def preprocess(self, batch, **kwargs):
        """
        Convert padded segment_rewards into an undiscounted summed reward per meta step.
        Discounting/bootstrapping still happens in return_compute (with gamma^segment_length).
        
        Expects:
            batch["segment_rewards"]: [B, T, N, L, 1] (padded)
            batch["segment_length"]:  [B, T, N, 1]
        Produces:
            Overwrites EpisodeKey.REWARD with the masked sum over the actual segment length.
        """
        if "segment_rewards" in batch and "segment_length" in batch:
            seg_rewards = batch["segment_rewards"]
            seg_lengths = batch["segment_length"]
            
            # Convert to torch
            if not isinstance(seg_rewards, torch.Tensor):
                seg_rewards = torch.as_tensor(seg_rewards, dtype=torch.float32)
            if not isinstance(seg_lengths, torch.Tensor):
                seg_lengths = torch.as_tensor(seg_lengths, dtype=torch.float32)
            
            device = self.loss.policy.device
            seg_rewards = seg_rewards.to(device)
            seg_lengths = seg_lengths.to(device)
            
            # seg_rewards: [B, T, N, L, 1]
            L = seg_rewards.shape[3]
            idx = torch.arange(L, device=device).view(1, 1, 1, L, 1)
            mask = idx < seg_lengths.unsqueeze(3)
            summed = (seg_rewards * mask).sum(dim=3, keepdim=False)  # [B, T, N, 1]
            batch[EpisodeKey.REWARD] = summed
        
        return batch
    
    def optimize(self, batch, **kwargs):
        """
        Optimize the meta-policy.
        
        The batch contains meta-level data:
        - CUR_OBS: Observations when meta-decisions were made
        - ACTION: Sub-policy indices selected
        - REWARD: Accumulated rewards during sub-policy execution
        - DONE: Episode termination flags
        """
        batch = self.preprocess(batch, **kwargs)
        return super().optimize(batch, **kwargs)

