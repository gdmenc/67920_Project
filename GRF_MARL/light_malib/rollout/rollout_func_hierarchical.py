# Copyright 2024 Hierarchical RL Extension
# Licensed under the Apache License, Version 2.0

"""
Hierarchical rollout function for meta-policy training.

Key differences from standard rollout:
- Meta-policy decides sub-policy at decision points (not every step)
- Accumulates rewards during sub-policy commitment periods
- Detects events (possession change, game mode change) for switching
- Collects meta-level training data (meta_obs, meta_action, meta_reward)

Multi-Encoder Support:
- Maintains access to raw observations for sub-policies with different encoders
- Re-encodes raw observations when executing incompatible sub-policies
"""

from typing import OrderedDict
import numpy as np
from light_malib.utils.logger import Logger
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.base_env import BaseEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_table_name

from .rollout_func import (
    rename_fields,
    select_fields,
    update_fields,
    stack_step_data,
    env_reset,
    pull_policies,
    submit_traj,
)


class EventDetector:
    """Detects game events for hierarchical policy switching."""
    
    def __init__(self, events_to_detect: list):
        self.events_to_detect = events_to_detect
        self.last_ball_owned_team = None
        self.last_game_mode = None
        
    def reset(self):
        """Reset event tracking state."""
        self.last_ball_owned_team = None
        self.last_game_mode = None
        
    def detect(self, raw_obs: dict) -> bool:
        """
        Detect if any tracked event occurred.
        
        Args:
            raw_obs: Raw observation dict from gfootball containing
                     'ball_owned_team', 'game_mode', etc.
                     
        Returns:
            True if an event occurred that should trigger policy switch
        """
        event_occurred = False
        
        # Check possession change
        if 'possession_change' in self.events_to_detect:
            current_ball_owned = raw_obs.get('ball_owned_team', -1)
            if self.last_ball_owned_team is not None:
                if current_ball_owned != self.last_ball_owned_team:
                    # Possession changed (including to/from nobody)
                    event_occurred = True
            self.last_ball_owned_team = current_ball_owned
            
        # Check game mode change
        if 'game_mode_change' in self.events_to_detect:
            current_game_mode = raw_obs.get('game_mode', 0)
            if self.last_game_mode is not None:
                if current_game_mode != self.last_game_mode:
                    event_occurred = True
            self.last_game_mode = current_game_mode
            
        return event_occurred


def extract_event_info_from_state(env, agent_id: str) -> dict:
    """
    Extract raw observation info needed for event detection from environment state.
    
    Args:
        env: The GRFootball environment
        agent_id: The agent to get state for
        
    Returns:
        Dict with ball_owned_team, game_mode, etc.
    """
    try:
        # Get raw state from first player of the agent
        states = env.states
        if agent_id == "agent_0":
            state = states[0]
        else:
            state = states[-1]
            
        if state.obs is not None:
            return {
                'ball_owned_team': state.obs.get('ball_owned_team', -1),
                'game_mode': state.obs.get('game_mode', 0),
            }
    except Exception:
        pass
        
    return {'ball_owned_team': -1, 'game_mode': 0}


def get_raw_states_for_agent(env, agent_id: str) -> list:
    """
    Extract raw observation states for an agent from the environment.
    
    This is needed for multi-encoder support when sub-policies use
    different feature encoders than the meta-policy.
    
    Args:
        env: The GRFootball environment
        agent_id: The agent to get states for
        
    Returns:
        List of raw state objects (with .obs attribute) for each player
    """
    try:
        states = env.states
        # Determine which states belong to this agent
        # agent_0 controls left team (first N players)
        # agent_1 controls right team (last M players)
        if agent_id == "agent_0":
            num_players = env.num_players.get(agent_id, 4)
            return states[:num_players]
        else:
            num_players = env.num_players.get(agent_id, 1)
            return states[-num_players:]
    except Exception as e:
        Logger.warning(f"Failed to get raw states for agent {agent_id}: {e}")
        return None


def rollout_func(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: BaseEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs
):
    """
    Hierarchical rollout function with multi-encoder support.
    
    The meta-policy selects which sub-policy to use, and the sub-policy
    executes low-level actions. Meta-level data is collected for training.
    
    Multi-Encoder Support:
    - Extracts raw observations from environment for sub-policies
    - Passes raw states to execute_sub_policy for re-encoding when needed
    
    Args:
        eval: Whether this is an evaluation rollout
        rollout_worker: The rollout worker instance
        rollout_desc: Description of the rollout task
        env: The environment instance
        behavior_policies: Dict of agent_id -> (policy_id, policy)
        data_server: Server for storing training data
        rollout_length: Maximum rollout length
        **kwargs: Additional arguments including:
            - sample_length: Length of trajectory chunks
            - episode_mode: 'traj' or 'time-step'
            - credit_reassign_cfg: Credit reassignment config
    """
    sample_length = kwargs.get("sample_length", rollout_length)
    render = kwargs.get("render", False)
    if render:
        env.render()
        
    episode_mode = kwargs.get('episode_mode', 'traj')
    
    # Setup policies and encoders
    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id
        policy.eval()
        
    custom_reset_config = {
        "feature_encoders": feature_encoders,
        "main_agent_id": rollout_desc.agent_id,
        "rollout_length": rollout_length,
    }
    
    # Initialize environment
    step_data = env_reset(env, behavior_policies, custom_reset_config)
    
    # Get the main hierarchical policy for commitment tracking
    main_agent_id = rollout_desc.agent_id
    main_policy_id, main_policy = behavior_policies[main_agent_id]
    
    # Check if this is a hierarchical policy
    is_hierarchical = hasattr(main_policy, 'execute_sub_policy')
    
    # Check if any sub-policy needs re-encoding (multi-encoder support)
    any_needs_reencoding = False
    if is_hierarchical and hasattr(main_policy, 'sub_policy_needs_reencoding'):
        any_needs_reencoding = any(main_policy.sub_policy_needs_reencoding)
        if any_needs_reencoding:
            Logger.info(
                f"Multi-encoder mode enabled: "
                f"{sum(main_policy.sub_policy_needs_reencoding)} sub-policies need re-encoding"
            )
    
        # Initialize commitment tracking
        main_policy.reset_commitment_state()
        
        # Get commitment config
        commitment_config = main_policy.commitment_config
        commitment_mode = commitment_config.get('mode', 'both')
        min_commitment_steps = commitment_config.get('min_steps', 50)
        commitment_events = commitment_config.get('events', ['possession_change', 'game_mode_change'])
        
        # Initialize event detector
        event_detector = EventDetector(commitment_events)
        event_detector.reset()
        
        # Tracking state
        current_sub_policy_idx = None
        steps_since_switch = 0
        accumulated_reward = 0.0
        meta_decision_obs = None
        meta_decision_state = None
        episode_switch_count = 0  # Track number of policy switches per episode
        
        # Meta-policy RNN states (initialized to zeros)
        # We need to maintain these across steps for the meta-policy
        meta_actor_rnn_state = main_policy.get_initial_state(batch_size=1)[EpisodeKey.ACTOR_RNN_STATE]
        meta_critic_rnn_state = main_policy.get_initial_state(batch_size=1)[EpisodeKey.CRITIC_RNN_STATE]
        
        # Sub-policy RNN states (one per sub-policy, per agent)
        num_players = env.num_players[main_agent_id]
        sub_policy_rnn_states = {}
        for i, sub_actor in enumerate(main_policy.sub_policies):
            sub_policy_rnn_states[i] = np.zeros(
                (num_players, sub_actor.rnn_layer_num, sub_actor.rnn_state_size),
                dtype=np.float32
            )
    
    step = 0
    step_data_list = []
    meta_step_data_list = []  # For hierarchical: meta-level transitions
    results = []
    
    while step <= rollout_length:
        # Prepare policy inputs
        policy_inputs = rename_fields(
            step_data, 
            [EpisodeKey.NEXT_OBS, EpisodeKey.NEXT_STATE], 
            [EpisodeKey.CUR_OBS, EpisodeKey.CUR_OBS]
        )
        
        policy_outputs = {}
        global_timer.record("inference_start")
        
        for agent_id, (policy_id, policy) in behavior_policies.items():
            if is_hierarchical and agent_id == main_agent_id:
                # Hierarchical policy: meta-decision + sub-policy execution
                
                # Detect events
                event_info = extract_event_info_from_state(env, agent_id)
                event_occurred = event_detector.detect(event_info)
                
                # Check if we should make a new meta-decision
                should_switch = (
                    current_sub_policy_idx is None or
                    main_policy.should_switch_policy(steps_since_switch, event_occurred)
                )
                
                if should_switch:
                    episode_switch_count += 1  # Increment switch counter
                    
                    # Save previous meta-transition if we had one
                    if current_sub_policy_idx is not None and meta_decision_obs is not None:
                        # Create meta-level transition with all required fields for training
                        meta_transition = {
                            EpisodeKey.CUR_OBS: meta_decision_obs,
                            EpisodeKey.ACTION: np.array([current_sub_policy_idx] * num_players),
                            EpisodeKey.REWARD: np.array([[accumulated_reward]] * num_players),
                            EpisodeKey.DONE: policy_inputs[agent_id][EpisodeKey.DONE],
                            EpisodeKey.ACTION_MASK: meta_decision_state.get(EpisodeKey.ACTION_MASK),
                            # Use the SAVED input RNN state from when the decision was made
                            # Broadcast to num_players to match other fields
                            EpisodeKey.ACTOR_RNN_STATE: np.repeat(
                                meta_decision_state.get(EpisodeKey.ACTOR_RNN_STATE), 
                                num_players, axis=0
                            ),
                            EpisodeKey.CRITIC_RNN_STATE: np.repeat(
                                meta_decision_state.get(EpisodeKey.CRITIC_RNN_STATE), 
                                num_players, axis=0
                            ),
                            # Required for PPO loss computation
                            EpisodeKey.ACTION_LOG_PROB: meta_decision_state.get(EpisodeKey.ACTION_LOG_PROB),
                            EpisodeKey.STATE_VALUE: meta_decision_state.get(EpisodeKey.STATE_VALUE),
                        }
                        meta_step_data_list.append(meta_transition)
                    
                    # Prepare inputs for meta-decision
                    # We must use the current meta-policy RNN state (input state)
                    meta_inputs = policy_inputs[agent_id].copy()
                    
                    # Replicate RNN states for all players (MetaActor expects batch * num_players)
                    # meta_actor_rnn_state is [1, layer, hidden], we need [num_players, layer, hidden]
                    meta_inputs[EpisodeKey.ACTOR_RNN_STATE] = np.repeat(
                        meta_actor_rnn_state, num_players, axis=0
                    )
                    meta_inputs[EpisodeKey.CRITIC_RNN_STATE] = np.repeat(
                        meta_critic_rnn_state, num_players, axis=0
                    )
                    
                    # Make new meta-decision
                    meta_output = main_policy.compute_meta_action(
                        inference=True,
                        explore=not eval,
                        to_numpy=True,
                        **meta_inputs
                    )
                    
                    # Get selected sub-policy (team-wide decision)
                    meta_actions = meta_output[EpisodeKey.ACTION]
                    new_sub_policy_idx = int(meta_actions[0])  # Same for all agents
                    
                    prev_sub_policy_idx = main_policy._current_sub_policy_idx
                    
                    # Update policy's internal state
                    main_policy._current_sub_policy_idx = new_sub_policy_idx
                    
                    # Only reset RNN state and increment counter if policy ACTUALLY changed
                    if new_sub_policy_idx != prev_sub_policy_idx:
                         episode_switch_count += 1
                         
                         # Reset sub-policy RNN state for new policy
                         sub_actor = main_policy.sub_policies[new_sub_policy_idx]
                         sub_policy_rnn_states[new_sub_policy_idx] = np.zeros(
                             (num_players, sub_actor.rnn_layer_num, sub_actor.rnn_state_size),
                             dtype=np.float32
                         )
                    else:
                         # Keep existing RNN state (continuity)
                         pass
                    
                    # Save meta-decision state for later (including log_prob and value for training)
                    # IMPORTANT: Save the INPUT RNN state (single copy), not the output state
                    meta_decision_obs = policy_inputs[agent_id][EpisodeKey.CUR_OBS].copy()
                    meta_decision_state = {
                        EpisodeKey.ACTION_MASK: policy_inputs[agent_id].get(EpisodeKey.ACTION_MASK),
                        EpisodeKey.ACTOR_RNN_STATE: meta_actor_rnn_state.copy(),  # Save INPUT state (single)
                        EpisodeKey.CRITIC_RNN_STATE: meta_critic_rnn_state.copy(), # Save INPUT state (single)
                        EpisodeKey.ACTION_LOG_PROB: meta_output.get(EpisodeKey.ACTION_LOG_PROB),
                        EpisodeKey.STATE_VALUE: meta_output.get(EpisodeKey.STATE_VALUE),
                    }
                    
                    
                    # DEBUG LOGGING
                    # print(f"DEBUG: Saved meta_decision_state CRITIC shape: {meta_decision_state[EpisodeKey.CRITIC_RNN_STATE].shape}")
                    
                    # Update meta-policy RNN state for next time
                    # Meta output returns states for all players, but they are identical for the team
                    # We just take the first one to maintain our single state
                    if meta_output.get(EpisodeKey.ACTOR_RNN_STATE) is not None:
                        # print(f"DEBUG: meta_output ACTOR shape: {meta_output[EpisodeKey.ACTOR_RNN_STATE].shape}")
                        meta_actor_rnn_state = meta_output[EpisodeKey.ACTOR_RNN_STATE][0:1].copy()
                        # print(f"DEBUG: Updated meta_actor_rnn_state shape: {meta_actor_rnn_state.shape}")
                        
                    if meta_output.get(EpisodeKey.CRITIC_RNN_STATE) is not None:
                        # print(f"DEBUG: meta_output CRITIC shape: {meta_output[EpisodeKey.CRITIC_RNN_STATE].shape}")
                        meta_critic_rnn_state = meta_output[EpisodeKey.CRITIC_RNN_STATE][0:1].copy()
                        # print(f"DEBUG: Updated meta_critic_rnn_state shape: {meta_critic_rnn_state.shape}")
                        # print(f"DEBUG: Updated meta_critic_rnn_state shape: {meta_critic_rnn_state.shape}")
                        
                        if meta_critic_rnn_state.shape[0] != 1:
                             Logger.error(f"CRITICAL: meta_critic_rnn_state became shape {meta_critic_rnn_state.shape}!")
                    
                    # Reset tracking
                    steps_since_switch = 0
                    accumulated_reward = 0.0
                
                # Prepare sub-policy inputs
                sub_policy_input = {
                    EpisodeKey.CUR_OBS: policy_inputs[agent_id][EpisodeKey.CUR_OBS],
                    EpisodeKey.ACTOR_RNN_STATE: sub_policy_rnn_states[current_sub_policy_idx],
                    EpisodeKey.ACTION_MASK: policy_inputs[agent_id].get(EpisodeKey.ACTION_MASK),
                    EpisodeKey.DONE: policy_inputs[agent_id][EpisodeKey.DONE],
                }
                
                # Get raw states for re-encoding if needed
                raw_states = None
                if any_needs_reencoding and main_policy.needs_reencoding(current_sub_policy_idx):
                    raw_states = get_raw_states_for_agent(env, agent_id)
                    if raw_states is None:
                        Logger.warning(
                            f"Could not get raw states for re-encoding sub-policy "
                            f"'{main_policy.sub_policy_names[current_sub_policy_idx]}'. "
                            f"Using encoded observations (may cause errors)."
                        )
                
                # Execute selected sub-policy
                sub_output = main_policy.execute_sub_policy(
                    current_sub_policy_idx,
                    raw_states=raw_states,
                    **sub_policy_input,
                    explore=not eval,
                    to_numpy=True,
                )
                
                # Update sub-policy RNN state
                if sub_output.get(EpisodeKey.ACTOR_RNN_STATE) is not None:
                    sub_policy_rnn_states[current_sub_policy_idx] = sub_output[EpisodeKey.ACTOR_RNN_STATE]
                
                # Policy output for environment step
                policy_outputs[agent_id] = {
                    EpisodeKey.ACTION: sub_output[EpisodeKey.ACTION],
                    EpisodeKey.ACTION_LOG_PROB: sub_output.get(EpisodeKey.ACTION_LOG_PROB),
                    EpisodeKey.ACTOR_RNN_STATE: policy_inputs[agent_id].get(EpisodeKey.ACTOR_RNN_STATE),
                    EpisodeKey.CRITIC_RNN_STATE: policy_inputs[agent_id].get(EpisodeKey.CRITIC_RNN_STATE),
                }
                
                # For training data, we store the meta-action info
                policy_outputs[agent_id]['meta_action'] = current_sub_policy_idx
                policy_outputs[agent_id]['steps_since_switch'] = steps_since_switch
                
            else:
                # Standard policy (opponent or non-hierarchical)
                policy_outputs[agent_id] = policy.compute_action(
                    inference=True,
                    explore=not eval,
                    to_numpy=True,
                    step=kwargs.get('rollout_epoch', 0),
                    **policy_inputs[agent_id]
                )
        
        global_timer.time("inference_start", "inference_end", "inference")
        
        # Step environment
        actions = select_fields(policy_outputs, [EpisodeKey.ACTION])
        
        global_timer.record("env_step_start")
        env_rets = env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")
        
        # Update hierarchical tracking
        if is_hierarchical:
            # Accumulate reward for meta-policy
            reward = env_rets[main_agent_id].get(EpisodeKey.REWARD, np.zeros((num_players, 1)))
            accumulated_reward += float(np.sum(reward))
            steps_since_switch += 1
        
        # Record data after env step
        step_data = update_fields(
            step_data, 
            select_fields(env_rets, [EpisodeKey.REWARD, EpisodeKey.DONE])
        )
        step_data = update_fields(
            step_data,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTION, EpisodeKey.ACTION_LOG_PROB, EpisodeKey.STATE_VALUE],
            ),
        )
        
        # Save data of trained agent for training
        step_data_list.append(step_data[rollout_desc.agent_id])
        
        # Record data for next step
        step_data = update_fields(
            env_rets,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTOR_RNN_STATE, EpisodeKey.CRITIC_RNN_STATE],
            ),
        )
        
        step += 1
        
        # Submit samples to server (for long episodes)
        if not eval:
            if episode_mode == 'traj':
                if sample_length > 0 and step % sample_length == 0:
                    assist_info = env.get_AssistInfo()
                    
                    submit_ctr = step // sample_length
                    submit_max_num = rollout_length // sample_length
                    
                    s_idx = sample_length * (submit_ctr - 1)
                    e_idx = sample_length * submit_ctr
                    
                    submit_traj(
                        data_server, step_data_list, step_data, rollout_desc,
                        s_idx, e_idx,
                        credit_reassign_cfg=kwargs.get("credit_reassign_cfg"),
                        assist_info=assist_info
                    )
                    
                    if submit_ctr != submit_max_num:
                        behavior_policies = pull_policies(rollout_worker, policy_ids)
        
        # Check if env ends
        if env.is_terminated():
            stats = env.get_episode_stats()
            
            if is_hierarchical:
                # Add policy_switches to the main agent's stats
                if rollout_desc.agent_id in stats:
                    stats[rollout_desc.agent_id]['policy_switches'] = episode_switch_count
            
            result = {
                "main_agent_id": rollout_desc.agent_id,
                "policy_ids": policy_ids,
                "stats": stats,
            }
            
            if is_hierarchical:
                result["meta_transitions"] = len(meta_step_data_list)
                result["final_sub_policy"] = current_sub_policy_idx
                if any_needs_reencoding:
                    result["multi_encoder_mode"] = True
                
            results.append(result)
            
            # Reset environment
            step_data = env_reset(env, behavior_policies, custom_reset_config)
            
            if is_hierarchical:
                # Reset hierarchical state
                main_policy.reset_commitment_state()
                event_detector.reset()
                current_sub_policy_idx = None
                steps_since_switch = 0
                accumulated_reward = 0.0
                meta_decision_obs = None
                episode_switch_count = 0
    
    # Submit remaining data
    if not eval and sample_length <= 0:
        if episode_mode == 'traj':
            if is_hierarchical and len(meta_step_data_list) > 0:
                # For hierarchical policies, submit meta-level transitions
                # These contain meta-actions (0-3) for training the meta-policy
                # Add final transition if we have pending meta-state
                if current_sub_policy_idx is not None and meta_decision_obs is not None:
                    final_transition = {
                        EpisodeKey.CUR_OBS: meta_decision_obs,
                        EpisodeKey.ACTION: np.array([current_sub_policy_idx] * num_players),
                        EpisodeKey.REWARD: np.array([[accumulated_reward]] * num_players),
                        EpisodeKey.DONE: np.ones((num_players, 1), dtype=bool),  # Episode ended
                        EpisodeKey.ACTION_MASK: meta_decision_state.get(EpisodeKey.ACTION_MASK),
                        EpisodeKey.ACTOR_RNN_STATE: np.repeat(
                            meta_decision_state.get(EpisodeKey.ACTOR_RNN_STATE), 
                            num_players, axis=0
                        ),
                        EpisodeKey.CRITIC_RNN_STATE: np.repeat(
                            meta_decision_state.get(EpisodeKey.CRITIC_RNN_STATE), 
                            num_players, axis=0
                        ),
                        # Required for PPO loss computation
                        EpisodeKey.ACTION_LOG_PROB: meta_decision_state.get(EpisodeKey.ACTION_LOG_PROB),
                        EpisodeKey.STATE_VALUE: meta_decision_state.get(EpisodeKey.STATE_VALUE),
                    }
                    
                
                # Submit meta-level trajectory for hierarchical policy training
                # Construct meta_last_step_data for bootstrapping
                meta_last_step_data = {
                    rollout_desc.agent_id: {
                        EpisodeKey.NEXT_OBS: step_data[rollout_desc.agent_id][EpisodeKey.NEXT_OBS],
                        EpisodeKey.DONE: step_data[rollout_desc.agent_id][EpisodeKey.DONE],
                        # Use current meta RNN states for bootstrapping
                        EpisodeKey.ACTOR_RNN_STATE: np.repeat(meta_actor_rnn_state, num_players, axis=0),
                        EpisodeKey.CRITIC_RNN_STATE: np.repeat(meta_critic_rnn_state, num_players, axis=0),
                    }
                }
                
                submit_traj(data_server, meta_step_data_list, meta_last_step_data, rollout_desc)
            else:
                # Standard (non-hierarchical) submission
                submit_traj(data_server, step_data_list, step_data, rollout_desc)
    
    results = {"results": results}
    
    if is_hierarchical:
        results["meta_step_count"] = len(meta_step_data_list)
        if any_needs_reencoding:
            results["multi_encoder_mode"] = True
        
    return results
