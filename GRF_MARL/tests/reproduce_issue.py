
import unittest
import numpy as np

# Mocking the classes/functions for isolation

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

class MockPolicy:
    def __init__(self, commitment_config):
        self.commitment_config = commitment_config
        self.commitment_mode = self.commitment_config.get('mode', 'both')
        self.min_commitment_steps = self.commitment_config.get('min_steps', 50)
        self.commitment_events = self.commitment_config.get('events', ['possession_change', 'game_mode_change'])
        self._current_sub_policy_idx = None # Initial state
        
    def should_switch_policy(self, steps_since_switch: int, event_occurred: bool) -> bool:
        # This mirrors the NEW implementation in policy.py
        if self._current_sub_policy_idx is None:
            # First step, must select a policy
            return True
            
        # Enforce minimum commitment steps as a hard constraint
        if steps_since_switch < self.min_commitment_steps:
            return False
            
        if self.commitment_mode == 'steps':
            return True  # Already checked min_steps above
        elif self.commitment_mode == 'events':
            return event_occurred
        elif self.commitment_mode == 'both':
            return steps_since_switch >= self.min_commitment_steps and event_occurred
        else:
            # Default: always allow switching
            return True

class TestHierarchicalLogic(unittest.TestCase):
    def test_min_steps_enforcement(self):
        config = {'mode': 'both', 'min_steps': 50}
        policy = MockPolicy(config)
        policy._current_sub_policy_idx = 0 # Simulate initialized
        
        # Case 1: < 50 steps, event occurred -> Should be False (New Logic)
        self.assertFalse(policy.should_switch_policy(10, True), "Should NOT switch before min_steps even with event")
        
        # Case 2: >= 50 steps, event occurred -> Should be True
        self.assertTrue(policy.should_switch_policy(50, True), "Should switch after min_steps with event")
        
        # Case 3: >= 50 steps, NO event -> Should be False (New Logic: both required)
        self.assertFalse(policy.should_switch_policy(50, False), "Should NOT switch if no event in 'both' mode")

    def test_rollout_fix_simulation(self):
        # Simulate rollout_func behavior with the FIX
        config = {'mode': 'both', 'min_steps': 50}
        policy = MockPolicy(config)
        
        # Rollout loop simulation
        # Step 0: Initial selection
        current_sub_policy_idx = None
        should_switch = current_sub_policy_idx is None or policy.should_switch_policy(0, False)
        self.assertTrue(should_switch)
        
        # Rollout selects policy 0
        current_sub_policy_idx = 0
        # FIX: Update internal state
        policy._current_sub_policy_idx = current_sub_policy_idx
        
        # Step 1: 
        steps_since_switch = 1
        should_switch = current_sub_policy_idx is None or policy.should_switch_policy(steps_since_switch, False)
        
        # This should now be False
        self.assertFalse(should_switch, "Fix confirmed: Policy does not switch immediately")

if __name__ == '__main__':
    unittest.main()
