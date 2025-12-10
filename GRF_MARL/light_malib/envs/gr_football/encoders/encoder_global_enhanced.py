"""
Global Feature Encoder (Enhanced)
Outputs a unified global state vector for the entire team.
Includes rich features: Match State, Cards, Offside, and Sticky Actions (aggregated).
"""

import numpy as np
from gym.spaces import Box, Discrete

class FeatureEncoder:
    def __init__(self, **kwargs):
        self.num_players = kwargs['num_players'] # total players on pitch (e.g. 10 for 5v5)
        self.num_left_players = self.num_players // 2 # Assuming symmetric teams for now
        self.num_right_players = self.num_players - self.num_left_players
        self.action_n = 19
        
    def encode(self, states):
        # We compute the global state ONCE using information from all players
        # Then we return copies of it for each player
        
        if not states:
            return []
            
        # Use the first state for shared info
        base_obs = states[0].obs
        
        # --- Shared Features (Ball & Match) ---
        
        # Ball State (15 dims)
        ball_x, ball_y, ball_z = base_obs["ball"]
        ball_x_speed, ball_y_speed, _ = base_obs["ball_direction"]
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        
        ball_owned = 0.0
        if base_obs["ball_owned_team"] != -1:
            ball_owned = 1.0
            
        ball_owned_by_us = 0.0
        if base_obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
            
        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y) # 6 dims
        
        ball_state = np.concatenate([
            np.array(base_obs["ball"]), # 3
            np.array(ball_which_zone), # 6
            np.array(base_obs["ball_direction"]) * 20, # 3
            np.array([ball_speed * 20]), # 1
            np.array([ball_owned, ball_owned_by_us]) # 2
        ]) # Total 15
        
        # Match State (10 dims)
        steps_left = base_obs["steps_left"]
        half_steps_left = steps_left
        if half_steps_left > 1500:
            half_steps_left -= 1501
        half_steps_left = 1.0 * min(half_steps_left, 300.0)
        half_steps_left /= 300.0
        
        score_ratio = base_obs["score"][0] - base_obs["score"][1]
        score_ratio /= 5.0
        score_ratio = min(score_ratio, 1.0)
        score_ratio = max(-1.0, score_ratio)
        
        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[base_obs["game_mode"]] = 1
        
        match_state = np.concatenate([
            np.array([1.0 * steps_left / 3001, half_steps_left, score_ratio]), # 3
            game_mode # 7
        ]) # Total 10
        
        # --- Player States ---
        
        # Pre-compute offsides (requires state object)
        # We use the first state to compute offsides for everyone
        l_o, r_o = states[0].get_offside(base_obs)
        
        # Left Team (My Team)
        left_team_states = []
        for i in range(self.num_left_players):
            # Shared info
            pos = base_obs["left_team"][i]
            direction = np.array(base_obs["left_team_direction"][i])
            speed = np.linalg.norm(direction)
            role = base_obs["left_team_roles"][i]
            role_onehot = self._encode_role_onehot(role)
            tired = base_obs["left_team_tired_factor"][i]
            yellow = base_obs["left_team_yellow_card"][i]
            active = base_obs["left_team_active"][i]
            offside = l_o[i]
            
            # Private info (Sticky Actions) - Aggregated from individual states
            # states[i] corresponds to player i
            if i < len(states):
                sticky = states[i].obs["sticky_actions"] # 10 dims
            else:
                # Fallback if we don't control this player (shouldn't happen for full team control)
                sticky = np.zeros(10)
                
            # Derived info (Ball Distance)
            dist = np.linalg.norm(pos - base_obs["ball"][:2])
            
            # Feature Vector:
            # Pos(2), Dir(2), Speed(1), Role(10), Tired(1), Sticky(10), Yellow(1), Active(1), Offside(1), Dist(1)
            # Total: 30 dims
            p_state = np.concatenate([
                pos, # 2
                direction * 100, # 2
                [speed * 100], # 1
                role_onehot, # 10
                [tired], # 1
                sticky, # 10
                [yellow, active, offside, dist] # 4
            ])
            left_team_states.append(p_state)
            
        left_team_flat = np.concatenate(left_team_states)
        
        # Right Team (Opponents)
        right_team_states = []
        for i in range(self.num_right_players):
            pos = base_obs["right_team"][i]
            direction = np.array(base_obs["right_team_direction"][i])
            speed = np.linalg.norm(direction)
            tired = base_obs["right_team_tired_factor"][i]
            yellow = base_obs["right_team_yellow_card"][i]
            active = base_obs["right_team_active"][i]
            offside = r_o[i]
            
            # Derived info (Ball Distance)
            dist = np.linalg.norm(pos - base_obs["ball"][:2])
            
            # Feature Vector:
            # Pos(2), Dir(2), Speed(1), Tired(1), Yellow(1), Active(1), Offside(1), Dist(1)
            # Total: 10 dims
            p_state = np.concatenate([
                pos * 2, # 2
                direction * 100, # 2
                [speed * 100], # 1
                [tired, yellow, active, offside, dist] # 5
            ])
            right_team_states.append(p_state)
            
        right_team_flat = np.concatenate(right_team_states)
        
        # Combine all
        global_state = np.concatenate([
            ball_state,      # 15
            match_state,     # 10
            left_team_flat,  # N_left * 30
            right_team_flat  # N_right * 10
        ])
        
        # Return copies for each agent
        return [global_state for _ in states]
    
    @property
    def observation_space(self):
        # Calculate dimension:
        # Ball: 15
        # Match: 10
        # Left Team: N_left * 30
        # Right Team: N_right * 10
        
        ball_dim = 15
        match_dim = 10
        left_player_dim = 30
        right_player_dim = 10
        
        total_dim = ball_dim + match_dim + \
                    (self.num_left_players * left_player_dim) + \
                    (self.num_right_players * right_player_dim)
                    
        return Box(low=-1000, high=1000, shape=[total_dim])
    
    @property
    def global_observation_space(self):
        return self.observation_space
    
    @property
    def action_space(self):
        return Discrete(19)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def _encode_role_onehot(self, role_num):
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[role_num] = 1.0
        return np.array(result)
    
    def get_available_actions(self, obs, ball_distance, his_actions):
        return np.ones(19)
