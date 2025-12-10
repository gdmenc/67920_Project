"""
Global Feature Encoder (Basic)
Outputs a unified global state vector for the entire team.
Eliminates redundancy by including shared features (ball, match state) only once.
"""

import numpy as np
from gym.spaces import Box, Discrete

class FeatureEncoder:
    def __init__(self, **kwargs):
        self.num_players = kwargs['num_players'] # total players on pitch (e.g. 10 for 5v5)
        self.num_left_players = self.num_players // 2 # Assuming symmetric teams
        self.num_right_players = self.num_players - self.num_left_players
        self.action_n = 19
        
    def encode(self, states):
        feats = []
        for state in states:
            feat = self.encode_each(state)
            feats.append(feat)
        return feats
    
    @property
    def observation_space(self):
        # Calculate dimension:
        # Ball: 15
        # Match State (Game Mode): 7
        # Left Team (N_left players): N_left * 16
        # Right Team (N_right players): N_right * 6
        
        ball_dim = 15
        match_dim = 7
        left_player_dim = 16
        right_player_dim = 6
        
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

    def encode_each(self, state):
        obs = state.obs
        
        # --- Shared Features (Ball & Match) ---
        
        # Ball State (15 dims)
        ball_x, ball_y, ball_z = obs["ball"]
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        
        ball_owned = 0.0
        if obs["ball_owned_team"] != -1:
            ball_owned = 1.0
            
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
            
        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y) # 6 dims
        
        ball_state = np.concatenate([
            np.array(obs["ball"]), # 3
            np.array(ball_which_zone), # 6
            np.array(obs["ball_direction"]) * 20, # 3
            np.array([ball_speed * 20]), # 1
            np.array([ball_owned, ball_owned_by_us]) # 2
        ]) # Total 15
        
        # Match State (7 dims)
        game_mode = np.zeros(7, dtype=np.float32)
        game_mode[obs["game_mode"]] = 1
        
        # --- Player States ---
        
        # Left Team (My Team) - All players absolute states
        left_team_states = []
        for i in range(self.num_left_players):
            pos = obs["left_team"][i]
            direction = np.array(obs["left_team_direction"][i])
            speed = np.linalg.norm(direction)
            role = obs["left_team_roles"][i]
            role_onehot = self._encode_role_onehot(role)
            tired = obs["left_team_tired_factor"][i]
            
            p_state = np.concatenate([
                pos, # 2
                direction * 100, # 2
                [speed * 100], # 1
                role_onehot, # 10
                [tired] # 1
            ])
            left_team_states.append(p_state)
            
        left_team_flat = np.concatenate(left_team_states)
        
        # Right Team (Opponents)
        right_team_states = []
        for i in range(self.num_right_players):
            pos = obs["right_team"][i]
            direction = np.array(obs["right_team_direction"][i])
            speed = np.linalg.norm(direction)
            tired = obs["right_team_tired_factor"][i]
            
            # Features: Pos(2), Dir(2), Speed(1), Tired(1). Total 6.
            p_state = np.concatenate([
                pos * 2, # 2 (scaling matches basic encoder)
                direction * 100, # 2
                [speed * 100], # 1
                [tired] # 1
            ])
            right_team_states.append(p_state)
            
        right_team_flat = np.concatenate(right_team_states)
        
        # Combine all
        global_state = np.concatenate([
            ball_state,      # 15
            game_mode,       # 7
            left_team_flat,  # N * 16
            right_team_flat  # N * 6
        ])
        
        return global_state

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
        # This encoder is for the meta-policy which outputs sub-policy indices.
        # It doesn't need low-level available actions.
        # But if the interface requires it, we can return all ones or implement it.
        # For now, we don't include it in the global state vector.
        return np.ones(19)
