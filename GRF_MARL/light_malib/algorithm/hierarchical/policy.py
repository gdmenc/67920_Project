# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Hierarchical MAPPO Policy that selects among pre-trained sub-policies.

The meta-policy learns to choose which sub-policy to execute based on
the current game state. Sub-policies are frozen and not updated.

Multi-Encoder Support:
- Sub-policies may use different feature encoders than the meta-policy
- When executing a sub-policy with a different encoder, raw observations
  are re-encoded using that sub-policy's encoder
- This enables mixing policies trained with different encoders
"""

import copy
import os
import pickle
import gym
import torch
import numpy as np

from torch import nn
from light_malib.utils.logger import Logger
from light_malib.utils.typing import DataTransferType, Tuple, Any, Dict, List
from light_malib.utils.episode import EpisodeKey

from light_malib.algorithm.common.policy import Policy
from light_malib.algorithm.mappo.policy import MAPPO, hard_update, shape_adjusting
from light_malib.algorithm.utils import PopArt

import wrapt
import tree
import importlib
from light_malib.registry import registry


@registry.registered(registry.POLICY)
class HierarchicalMAPPO(Policy):
    """
    Hierarchical MAPPO policy that learns to select among pre-trained sub-policies.
    
    The meta-policy outputs a distribution over K sub-policies.
    The selected sub-policy then executes low-level actions.
    
    Multi-Encoder Support:
    - Stores feature encoders for each sub-policy
    - Automatically re-encodes raw observations when sub-policy uses different encoder
    - Tracks encoder compatibility for efficient execution
    
    Commitment can be enforced via:
    - Step-based: Minimum number of steps before switching
    - Event-based: Switch only on game events (possession change, game mode change)
    - Both: Combination of step and event triggers
    """
    
    def __init__(
        self,
        registered_name: str,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        model_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,
        **kwargs,
    ):
        self.random_exploration = False  # Can be set for epsilon-greedy exploration
        
        # Load the model module for meta-policy
        model_type = model_config.get("model", "gr_football.hierarchical")
        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("light_malib.model.{}".format(model_type))
        
        # Feature encoder from the model
        FE_cfg = custom_config.get('FE_cfg', None)
        if FE_cfg is not None:
            # Check for custom encoder type
            encoder_type = FE_cfg.get('encoder_type', None)
            if encoder_type:
                Logger.info(f"Loading custom feature encoder: {encoder_type}")
                try:
                    encoder_module = importlib.import_module(f"light_malib.envs.gr_football.encoders.{encoder_type}")
                    self.feature_encoder = encoder_module.FeatureEncoder(**FE_cfg)
                except ImportError as e:
                    Logger.error(f"Failed to load encoder {encoder_type}: {e}")
                    raise
            else:
                self.feature_encoder = model.FeatureEncoder(**FE_cfg)
        else:
            self.feature_encoder = model.FeatureEncoder()
            
        # Get observation/action spaces from feature encoder
        observation_space = self.feature_encoder.observation_space
        low_level_action_space = self.feature_encoder.action_space  # 19 actions
        
        # Store meta-policy observation shape for compatibility checking
        self.meta_obs_shape = observation_space.shape
        
        # Sub-policy configuration
        self.sub_policy_configs = custom_config.get('sub_policies', [])
        self.num_sub_policies = len(self.sub_policy_configs)
        
        if self.num_sub_policies == 0:
            raise ValueError("No sub-policies configured! Please specify sub_policies in custom_config.")
        
        Logger.info(f"Hierarchical policy with {self.num_sub_policies} sub-policies")
        
        # Meta-policy action space = number of sub-policies
        meta_action_space = gym.spaces.Discrete(self.num_sub_policies)
        
        # Meta-policy observation space = Concatenation of all player observations
        # This fixes the issue where only the first player's observation was being used
        # If using global encoder (where all obs are same), this adds redundancy but is robust
        # If using local encoder, this essentially creates a global view from local parts
        
        # Extract num_players for meta-policy observation space calculation
        if "num_agents" in custom_config:
            self.num_players = custom_config["num_agents"]
        elif "FE_cfg" in custom_config and "num_players" in custom_config["FE_cfg"]:
            self.num_players = custom_config["FE_cfg"]["num_players"]
        else:
            self.num_players = 4 # Default fallback for this specific scenario
            Logger.warning(f"Could not determine num_players from config, defaulting to {self.num_players}")

        single_obs_shape = observation_space.shape
        meta_obs_dim = single_obs_shape[0] * self.num_players
        meta_observation_space = gym.spaces.Box(
            low=observation_space.low[0], # Assuming same bounds
            high=observation_space.high[0],
            shape=(meta_obs_dim,),
            dtype=observation_space.dtype
        )
        
        # Disable MAPPO's default behavior of ignoring the passed observation_space
        custom_config["use_feature_encoder_obs"] = False
        
        super(HierarchicalMAPPO, self).__init__(
            registered_name=registered_name,
            observation_space=meta_observation_space,  # Use concatenated space
            action_space=meta_action_space,
            model_config=model_config,
            custom_config=custom_config,
        )
        
        # Always initialize on CPU to avoid Ray serialization issues with CUDA tensors
        # Policy will be moved to GPU via to_device() when needed for training
        self.device = torch.device("cpu")
        self._use_cuda = custom_config.get("use_cuda", False)  # Store for later use
        
        # Create meta-policy actor and critic
        # Use meta_observation_space (concatenated) instead of single-agent observation_space
        self.actor = model.Actor(
            self.model_config["actor"],
            meta_observation_space,
            self.num_sub_policies,
            self.custom_config,
            self.model_config["initialization"],
        )
        
        self.critic = model.Critic(
            self.model_config["critic"],
            meta_observation_space,
            self.custom_config,
            self.model_config["initialization"],
        )
        
        # Store the low-level action space for sub-policies
        self.low_level_action_space = low_level_action_space
        
        # Initialize sub-policy storage
        self.sub_policies = []           # Actor nn.Modules
        self.sub_policy_names = []       # Names for logging
        self.sub_policy_encoders = []    # Feature encoders (for re-encoding)
        self.sub_policy_obs_shapes = []  # Observation shapes
        self.sub_policy_needs_reencoding = []  # Whether re-encoding is needed
        
        # Load frozen sub-policies with their encoders
        self._load_sub_policies()
        
        # Commitment configuration
        self.commitment_config = custom_config.get('commitment', {})
        self.commitment_mode = self.commitment_config.get('mode', 'both')
        self.min_commitment_steps = self.commitment_config.get('min_steps', 50)
        self.commitment_events = self.commitment_config.get('events', ['possession_change', 'game_mode_change'])
        
        # Commitment state (per-batch tracking, will be reset in rollout)
        self._current_sub_policy_idx = None
        self._steps_since_switch = 0
        self._last_ball_owned_team = None
        self._last_game_mode = None
        
        # PopArt for value normalization
        if custom_config.get("use_popart", False):
            self.value_normalizer = PopArt(
                1, device=self.device, beta=custom_config.get("popart_beta", 0.99999)
            )
        
        self.share_backbone = False  # No backbone sharing for hierarchical
        
    def _load_sub_policies(self):
        """Load all pre-trained sub-policies with their encoders and freeze them.
        
        Multi-Encoder Support:
        - Loads both actor and feature encoder for each sub-policy
        - Checks encoder compatibility with meta-policy
        - Stores information needed for re-encoding at execution time
        """
        for policy_cfg in self.sub_policy_configs:
            name = policy_cfg['name']
            path = policy_cfg['path']
            
            Logger.info(f"Loading sub-policy '{name}' from {path}")
            
            try:
                # Load full policy to get both actor and encoder
                sub_policy = MAPPO.load(path)
                
                # Extract actor and force all tensors to CPU with new storage
                # This is critical for Ray serialization - CUDA tensors fail to deserialize on CPU workers
                # Using state_dict copy ensures tensors don't retain CUDA storage references
                sub_actor = sub_policy.actor
                cpu_state_dict = {k: v.cpu().clone() for k, v in sub_actor.state_dict().items()}
                sub_actor.load_state_dict(cpu_state_dict)
                sub_actor = sub_actor.cpu()
                
                # Freeze all parameters
                for param in sub_actor.parameters():
                    param.requires_grad = False
                    
                # Set to eval mode
                sub_actor.eval()
                
                # Extract feature encoder
                sub_encoder = sub_policy.feature_encoder
                sub_obs_shape = sub_policy.observation_space.shape
                
                # Check if re-encoding is needed
                needs_reencoding = (sub_obs_shape != self.meta_obs_shape)
                
                # Store everything
                self.sub_policies.append(sub_actor)
                self.sub_policy_names.append(name)
                self.sub_policy_encoders.append(sub_encoder)
                self.sub_policy_obs_shapes.append(sub_obs_shape)
                self.sub_policy_needs_reencoding.append(needs_reencoding)
                
                # Log encoder compatibility
                if needs_reencoding:
                    Logger.warning(
                        f"Sub-policy '{name}' uses different encoder: "
                        f"obs_shape={sub_obs_shape} vs meta={self.meta_obs_shape}. "
                        f"Will re-encode raw observations when executing."
                    )
                else:
                    Logger.info(
                        f"Sub-policy '{name}' uses compatible encoder: "
                        f"obs_shape={sub_obs_shape}"
                    )
                
                # Explicitly delete the rest to free memory
                del sub_policy
                
            except Exception as e:
                Logger.error(f"Failed to load sub-policy '{name}' from {path}: {e}")
                raise
                
        # Summary
        num_compatible = sum(1 for x in self.sub_policy_needs_reencoding if not x)
        num_reencoding = sum(1 for x in self.sub_policy_needs_reencoding if x)
        Logger.info(
            f"Successfully loaded {len(self.sub_policies)} sub-policy actors: "
            f"{num_compatible} compatible, {num_reencoding} need re-encoding"
        )
        Logger.info(f"Sub-policy names: {self.sub_policy_names}")
    
    def get_sub_policy_encoder(self, sub_policy_idx: int):
        """Get the feature encoder for a specific sub-policy.
        
        Args:
            sub_policy_idx: Index of the sub-policy
            
        Returns:
            The feature encoder for that sub-policy
        """
        return self.sub_policy_encoders[sub_policy_idx]
    
    def needs_reencoding(self, sub_policy_idx: int) -> bool:
        """Check if a sub-policy needs raw observations to be re-encoded.
        
        Args:
            sub_policy_idx: Index of the sub-policy
            
        Returns:
            True if the sub-policy uses a different encoder than meta-policy
        """
        return self.sub_policy_needs_reencoding[sub_policy_idx]
    
    def get_encoder_info(self) -> Dict:
        """Get information about all sub-policy encoders.
        
        Returns:
            Dict with encoder compatibility information for each sub-policy
        """
        return {
            name: {
                'obs_shape': shape,
                'needs_reencoding': needs,
                'compatible': not needs,
            }
            for name, shape, needs in zip(
                self.sub_policy_names,
                self.sub_policy_obs_shapes,
                self.sub_policy_needs_reencoding
            )
        }
    
    def reset_commitment_state(self):
        """Reset commitment tracking state. Called at episode start."""
        self._current_sub_policy_idx = None
        self._steps_since_switch = 0
        self._last_ball_owned_team = None
        self._last_game_mode = None
        
    def _detect_events(self, observations: np.ndarray) -> bool:
        """
        Detect if any commitment-breaking events occurred.
        
        Events are detected from the encoded observation features.
        The basic encoder includes ball_owned_by_us and game state info.
        
        Args:
            observations: Encoded observations from feature encoder
            
        Returns:
            True if an event occurred that should trigger a policy switch decision
        """
        if 'possession_change' not in self.commitment_events and 'game_mode_change' not in self.commitment_events:
            return False
            
        # For now, we'll need the raw observation to detect events
        # This will be handled in the rollout function which has access to raw obs
        # Here we just return False and let rollout handle event detection
        return False
    
    def should_switch_policy(self, steps_since_switch: int, event_occurred: bool) -> bool:
        """
        Determine if the meta-policy should make a new decision.
        
        Args:
            steps_since_switch: Number of steps since last policy switch
            event_occurred: Whether a commitment-breaking event occurred
            
        Returns:
            True if meta-policy should select a new sub-policy
        """
        if self._current_sub_policy_idx is None:
            # First step, must select a policy
            return True
            
        # Enforce minimum commitment steps as a hard constraint
        if steps_since_switch < self.min_commitment_steps:
            return False
            
        if self.commitment_mode == 'steps':
            return steps_since_switch >= self.min_commitment_steps
        elif self.commitment_mode == 'events':
            return event_occurred
        elif self.commitment_mode == 'both':
            # Switch ONLY if BOTH conditions are met
            # This enforces strict commitment: must wait min_steps AND wait for an event
            return steps_since_switch >= self.min_commitment_steps and event_occurred
        else:
            # Default: always allow switching
            return True
    
    def get_initial_state(self, batch_size) -> Dict:
        """Get initial RNN states for meta-policy."""
        return {
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(
                (batch_size, self.actor.rnn_layer_num, self.actor.rnn_state_size)
            ),
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(
                (batch_size, self.critic.rnn_layer_num, self.critic.rnn_state_size)
            ),
        }
    
    def to_device(self, device):
        """Move policy to specified device.
        
        Creates a deep copy and moves to the specified device,
        following the same pattern as MAPPO.to_device().
        
        Note: Feature encoders are CPU-only (pure NumPy), so they are not moved.
        """
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy.actor = self_copy.actor.to(device)
        self_copy.critic = self_copy.critic.to(device)
        
        # Move sub-policy actors to device (they are just nn.Modules)
        for i, sub_actor in enumerate(self_copy.sub_policies):
            self_copy.sub_policies[i] = sub_actor.to(device)
            
        if self.custom_config.get("use_popart", False):
            self_copy.value_normalizer = self_copy.value_normalizer.to(device)
            self_copy.value_normalizer.tpdv = dict(dtype=torch.float32, device=device)
            
        return self_copy
    
    @shape_adjusting
    def compute_meta_action(self, **kwargs):
        """
        Compute meta-policy action (which sub-policy to use).
        
        This is the meta-level decision. Returns the index of the selected sub-policy.
        """
        # Convert numpy to tensor
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v, device=self.device, requires_grad=False)
                kwargs[k] = v
                
        actions = kwargs.get(EpisodeKey.ACTION, None)
        explore = kwargs.get("explore", True)
        inference = kwargs.get("inference", True)
        no_critic = kwargs.get("no_critic", False)
        to_numpy = kwargs.get("to_numpy", False)
        
        if not inference:
            explore = False
            
        with torch.set_grad_enabled(not inference):
            observations = kwargs[EpisodeKey.CUR_OBS]
            actor_rnn_states = kwargs[EpisodeKey.ACTOR_RNN_STATE]
            critic_rnn_states = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            action_masks = kwargs.get(EpisodeKey.ACTION_MASK, None)  # Not used for meta
            rnn_masks = kwargs[EpisodeKey.DONE]
            
            # Handle global state (for consistency with MAPPO)
            if EpisodeKey.CUR_STATE not in kwargs:
                states = observations
            else:
                states = kwargs[EpisodeKey.CUR_STATE]
            
            # Meta-actor forward pass
            meta_actions, actor_rnn_states, action_log_probs, dist_entropy = self.actor(
                observations, actor_rnn_states, rnn_masks, action_masks, explore, actions
            )
            
            if to_numpy:
                actor_rnn_states = actor_rnn_states.detach().cpu().numpy()
                meta_actions = meta_actions.detach().cpu().numpy()
                
                # Random exploration for meta-policy (epsilon-greedy over sub-policies)
                if hasattr(self, 'random_exploration') and self.random_exploration:
                    import random
                    exploration_actions = np.zeros(meta_actions.shape, dtype=int)
                    for i in range(len(meta_actions)):
                        if random.uniform(0, 1) < self.random_exploration:
                            exploration_actions[i] = int(random.choice(range(self.num_sub_policies)))
                        else:
                            exploration_actions[i] = int(meta_actions[i])
                    meta_actions = exploration_actions
                
                action_log_probs = action_log_probs.detach().cpu().numpy()
                
            ret = {
                EpisodeKey.ACTION_LOG_PROB: action_log_probs,
                EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states,
            }
            
            if kwargs.get(EpisodeKey.ACTION, None) is None:
                ret[EpisodeKey.ACTION] = meta_actions
            else:
                ret[EpisodeKey.ACTION_ENTROPY] = dist_entropy
                
            if not no_critic:
                values, critic_rnn_states = self.critic(
                    states, critic_rnn_states, rnn_masks
                )
                
                if to_numpy:
                    values = values.detach().cpu().numpy()
                    critic_rnn_states = critic_rnn_states.detach().cpu().numpy()
                    
                ret[EpisodeKey.STATE_VALUE] = values
                ret[EpisodeKey.CRITIC_RNN_STATE] = critic_rnn_states
                
            return ret
    
    def compute_action(self, **kwargs):
        """
        Full hierarchical action computation.
        
        For inference during rollout:
        1. Meta-policy selects sub-policy index
        2. Selected sub-policy computes low-level action
        
        For training, this just computes meta-level outputs.
        """
        # During training (inference=False), we only need meta-level outputs
        inference = kwargs.get("inference", True)
        
        if not inference:
            # Training mode: compute meta-action outputs for PPO update
            return self.compute_meta_action(**kwargs)
        
        # Inference mode: compute both meta and low-level actions
        # First get meta-policy decision
        meta_output = self.compute_meta_action(**kwargs)
        
        # The meta_action is the sub-policy index
        meta_action = meta_output[EpisodeKey.ACTION]
        
        # Store additional info for rollout
        meta_output['meta_action'] = meta_action
        meta_output['sub_policy_names'] = self.sub_policy_names
        
        return meta_output
    
    def execute_sub_policy(self, sub_policy_idx: int, raw_states=None, **kwargs) -> Dict:
        """
        Execute the selected sub-policy's actor to get low-level actions.
        
        Multi-Encoder Support:
        - If raw_states is provided AND sub-policy needs re-encoding,
          the raw observations are re-encoded using the sub-policy's encoder
        - Otherwise, uses the observations from kwargs directly
        
        Args:
            sub_policy_idx: Index of the sub-policy to execute
            raw_states: Optional list of raw observation states for re-encoding.
                        Required if sub-policy uses different encoder.
            **kwargs: Observation and state inputs including:
                - CUR_OBS: Encoded observations (used if compatible encoder)
                - ACTOR_RNN_STATE: RNN states
                - ACTION_MASK: Action masks
                - DONE: Done flags
                - explore: Whether to explore (default False)
                - to_numpy: Whether to return numpy arrays (default True)
            
        Returns:
            Low-level action outputs from the sub-policy actor
        """
        if isinstance(sub_policy_idx, np.ndarray):
            # If batch, assume all same for now (team-wide decision)
            sub_policy_idx = int(sub_policy_idx[0])
        elif isinstance(sub_policy_idx, torch.Tensor):
            sub_policy_idx = int(sub_policy_idx.item())
            
        sub_actor = self.sub_policies[sub_policy_idx]
        needs_reencoding = self.sub_policy_needs_reencoding[sub_policy_idx]
        
        # Handle observations - re-encode if necessary
        if needs_reencoding and raw_states is not None:
            # Re-encode raw observations using sub-policy's encoder
            sub_encoder = self.sub_policy_encoders[sub_policy_idx]
            encoded_obs = sub_encoder.encode(raw_states)
            observations = np.array(encoded_obs, dtype=np.float32)
        elif needs_reencoding and raw_states is None:
            # Need re-encoding but no raw states provided
            raise ValueError(
                f"Sub-policy '{self.sub_policy_names[sub_policy_idx]}' requires re-encoding "
                f"(obs_shape={self.sub_policy_obs_shapes[sub_policy_idx]} vs meta={self.meta_obs_shape}), "
                f"but raw_states was not provided. Pass raw_states to execute_sub_policy()."
            )
        else:
            # Compatible encoder, use provided observations
            observations = kwargs[EpisodeKey.CUR_OBS]
        
        # Convert to tensor
        if isinstance(observations, np.ndarray):
            observations = torch.tensor(observations, device=self.device, dtype=torch.float32)
            
        actor_rnn_states = kwargs.get(EpisodeKey.ACTOR_RNN_STATE)
        if actor_rnn_states is not None and isinstance(actor_rnn_states, np.ndarray):
            actor_rnn_states = torch.tensor(actor_rnn_states, device=self.device, dtype=torch.float32)
            
        rnn_masks = kwargs.get(EpisodeKey.DONE)
        if rnn_masks is not None and isinstance(rnn_masks, np.ndarray):
            rnn_masks = torch.tensor(rnn_masks, device=self.device, dtype=torch.float32)
            
        action_masks = kwargs.get(EpisodeKey.ACTION_MASK)
        if action_masks is not None and isinstance(action_masks, np.ndarray):
            action_masks = torch.tensor(action_masks, device=self.device, dtype=torch.float32)
        
        explore = kwargs.get("explore", False)  # Usually False for sub-policy execution
        to_numpy = kwargs.get("to_numpy", True)
        
        # Execute sub-policy actor (frozen, no grad)
        with torch.no_grad():
            actions, actor_rnn_states, action_log_probs, _ = sub_actor(
                observations, actor_rnn_states, rnn_masks, action_masks, explore, None
            )
            
        if to_numpy:
            actions = actions.detach().cpu().numpy()
            if actor_rnn_states is not None:
                actor_rnn_states = actor_rnn_states.detach().cpu().numpy()
            action_log_probs = action_log_probs.detach().cpu().numpy()
            
        return {
            EpisodeKey.ACTION: actions,
            EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
            EpisodeKey.ACTION_LOG_PROB: action_log_probs,
        }
    
    @shape_adjusting
    def value_function(self, **kwargs):
        """Compute value function for meta-policy."""
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v, device=self.device, requires_grad=False)
                kwargs[k] = v
                
        inference = kwargs.get("inference", True)
        to_numpy = kwargs.get("to_numpy", False)
        
        with torch.set_grad_enabled(not inference):
            observations = kwargs[EpisodeKey.CUR_OBS]
            critic_rnn_state = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            rnn_mask = kwargs[EpisodeKey.DONE]
            
            value, _ = self.critic(observations, critic_rnn_state, rnn_mask)
            
            if to_numpy:
                value = value.cpu().numpy()
                
            return {EpisodeKey.STATE_VALUE: value}
    
    def train(self):
        """Set meta-policy to training mode. Sub-policies stay in eval."""
        self.actor.train()
        self.critic.train()
        if self.custom_config.get("use_popart", False):
            self.value_normalizer.train()
    
    def eval(self):
        """Set all to eval mode."""
        self.actor.eval()
        self.critic.eval()
        if self.custom_config.get("use_popart", False):
            self.value_normalizer.eval()
        # Sub-policy actors are always in eval mode
        for sub_actor in self.sub_policies:
            sub_actor.eval()
    
    def dump(self, dump_dir):
        """Save the meta-policy (sub-policies are not saved as they're pre-trained)."""
        os.makedirs(dump_dir, exist_ok=True)
        torch.save(self.actor, os.path.join(dump_dir, "meta_actor.pt"))
        torch.save(self.critic, os.path.join(dump_dir, "meta_critic.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))
        
        # Save sub-policy config and encoder info for reference
        sub_policy_info = {
            'names': self.sub_policy_names,
            'configs': self.sub_policy_configs,
            'obs_shapes': [tuple(s) for s in self.sub_policy_obs_shapes],
            'needs_reencoding': self.sub_policy_needs_reencoding,
            'meta_obs_shape': tuple(self.meta_obs_shape),
        }
        pickle.dump(sub_policy_info, open(os.path.join(dump_dir, "sub_policies.pkl"), "wb"))
    
    @staticmethod
    def load(dump_dir, **kwargs):
        """Load a saved hierarchical policy."""
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)
            
        res = HierarchicalMAPPO(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )
        
        # Load meta-policy weights (always on CPU to avoid Ray serialization issues)
        actor_path = os.path.join(dump_dir, "meta_actor.pt")
        if os.path.exists(actor_path):
            actor = torch.load(actor_path, map_location='cpu')
            hard_update(res.actor, actor)
            
        critic_path = os.path.join(dump_dir, "meta_critic.pt")
        if os.path.exists(critic_path):
            critic = torch.load(critic_path, map_location='cpu')
            hard_update(res.critic, critic)
            
        return res
