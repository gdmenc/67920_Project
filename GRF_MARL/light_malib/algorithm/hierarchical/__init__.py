# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Hierarchical RL algorithm module.

This module provides a hierarchical MAPPO implementation where a meta-policy
learns to select among pre-trained sub-policies based on game state.
"""

from .policy import HierarchicalMAPPO
from .loss import HierarchicalMAPPOLoss
from .trainer import HierarchicalMAPPOTrainer

__all__ = ['HierarchicalMAPPO', 'HierarchicalMAPPOLoss', 'HierarchicalMAPPOTrainer']

