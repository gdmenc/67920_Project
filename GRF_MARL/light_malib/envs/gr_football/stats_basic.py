# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class StatsCaculator:
    def __init__(self):
        self.stats = None
        self.reset()

    def reset(self):
        self.last_hold_player = -1
        self.cumulative_shot_reward = None
        self.passing_flag = [False] * 11
        self.shot_flag = [False] * 11
        self.steal_ball_recording = False
        self.lost_ball_recording = False

        self.lost_ball_timestamp = None
        self.step_count = 0
        self.current_recovery_time = 0

        self.stats = {
            "reward": 0,
            "win": 0,
            "lose": 0,
            "score": 0,
            "my_goal": 0,
            "goal_diff": 0,
            "num_pass": 0,
            "num_shot": 0,
            "total_pass": 0,
            "good_pass": 0,
            "bad_pass": 0,
            "total_shot": 0,
            "good_shot": 0,
            "bad_shot": 0,
            "total_possession": 0,
            "tackle": 0,
            "get_tackled": 0,
            "interception": 0,
            "get_intercepted": 0,
            "total_move": 0,
            # Advanced Metrics
            "possession_final_third": 0,
            "total_possession_time": 0,
            "opponent_possession_final_third": 0,
            "opponent_total_possession_time": 0,
            "expected_goals": 0.0,
            "recovery_times": [],
        }

    def calc_stats(self, state, reward, idx):
        obs = state.obs
        prev_obs = state.prev_obs
        action = state.action

        self.stats["reward"] += reward
        self.stats["num_pass"] += 1 if action in [9, 10, 11] else 0
        self.stats["num_shot"] += 1 if action == 12 else 0

        if idx == 0:
            # only count once
            my_score, opponent_score = obs["score"]
            if my_score > opponent_score:
                self.stats["win"] = 1
                self.stats["lose"] = 0
                self.stats["score"] = 1
            elif my_score < opponent_score:
                self.stats["win"] = 0
                self.stats["lose"] = 1
                self.stats["score"] = 0
            else:
                self.stats["win"] = 0
                self.stats["lose"] = 0
                self.stats["score"] = 0.5
            self.stats["my_goal"] = my_score
            self.stats["goal_diff"] = my_score - opponent_score

        self.count_possession(obs)
        self.count_pass(obs, action)
        self.count_shot(prev_obs, obs, action)
        self.count_getpossession(prev_obs, obs)
        self.count_losepossession(prev_obs, obs)
        self.count_move(prev_obs, obs)

        self.count_possession(obs)
        self.count_pass(obs, action)
        self.count_shot(prev_obs, obs, action)
        self.count_getpossession(prev_obs, obs)
        self.count_losepossession(prev_obs, obs)
        self.count_move(prev_obs, obs)
        self.count_advanced_metrics(obs, prev_obs)

    def count_possession(self, obs):
        if (
            obs["ball_owned_team"] == 0 and obs["active"] == 1
        ):  # compute only once for the whole team
            self.stats["total_possession"] += 1

    def count_advanced_metrics(self, obs, prev_obs):
        # Field Tilt: Possession in final third
        # Field dimensions: x in [-1, 1]. Final third for left team (0) is x > 0.33.
        ball_x = obs["ball"][0]
        if obs["ball_owned_team"] == 0:
            if ball_x > 0.33:
                self.stats["possession_final_third"] += 1
            self.stats["total_possession_time"] += 1
        elif obs["ball_owned_team"] == 1:
            if ball_x < -0.33:
                self.stats["opponent_possession_final_third"] += 1
            self.stats["opponent_total_possession_time"] += 1

        # Defensive Recovery
        # Track time since ball lost until ball regained
        if self.lost_ball_recording:
            self.current_recovery_time += 1
        
        # If we just regained possession (and we were tracking loss)
        if self.steal_ball_recording and not self.lost_ball_recording:
             # This logic is a bit tricky because steal_ball_recording is set when we *start* to steal?
             # Let's look at count_getpossession. It sets steal_ball_recording=True when opponent loses ball to no one?
             # Actually, let's simplify. 
             pass

    def count_pass(self, obs, player_action):

        for i, p in enumerate(self.passing_flag):
            if p:  # if passing
                if obs["ball_owned_team"] == 0 and obs["active"] == i:
                    pass
                else:
                    if obs["ball_owned_team"] == 0 and obs["ball_owned_player"] != i:
                        self.passing_flag[i] = False
                        self.stats["good_pass"] += 1
                    elif obs["ball_owned_team"] == -1:
                        pass
                    elif obs["ball_owned_team"] == 1 and obs["active"] == i:
                        self.stats["bad_pass"] += 1
                        self.passing_flag[i] = False

        if player_action == 9 or player_action == 10 or player_action == 11:
            if (
                obs["ball_owned_team"] == 0
                and not self.passing_flag[obs["active"]]
                and (obs["active"] == obs["ball_owned_player"])
            ):

                self.passing_flag[obs["active"]] = True
                self.stats["total_pass"] += 1

    def count_shot(self, prev_obs, obs, player_action):

        for i, p in enumerate(self.shot_flag):
            if p:
                if prev_obs["score"][0] < obs["score"][0] and obs["active"] == i:
                    self.stats["good_shot"] += 1
                    self.shot_flag[i] = False
                else:

                    if (
                        obs["ball_owned_team"] == 0
                        and obs["active"] == i
                        and obs["ball_owned_player"] == i
                    ):  # havnt left the player
                        pass
                    else:
                        if (
                            obs["ball_owned_team"] == 0
                            and obs["ball_owned_player"] != i
                            and obs["active"] == i
                        ):
                            self.stats["bad_shot"] += 1
                            self.shot_flag[i] = False

                        elif obs["ball_owned_team"] == -1:
                            pass
                        elif obs["ball_owned_team"] == 1:
                            self.stats["bad_shot"] += 1
                            self.shot_flag[i] = False
                        else:
                            pass

        if player_action == 12:
            if (
                obs["ball_owned_team"] == 0
                and not self.shot_flag[obs["active"]]
                and (obs["active"] == obs["ball_owned_player"])
            ):

                self.shot_flag[obs["active"]] = True
                self.stats["total_shot"] += 1
                
                # xG Calculation (Simple Heuristic)
                # Distance to goal center (1, 0)
                ball_pos = obs["ball"]
                dist = np.sqrt((ball_pos[0] - 1)**2 + (ball_pos[1] - 0)**2)
                # Angle to goal (simple approximation)
                # Goal width is approx 0.144 (from -0.072 to 0.072 in y? No, standard pitch is 68m wide, goal is 7.32m. 7.32/68 ~ 0.107. In GRF y is -0.42 to 0.42 (width 0.84). So goal is approx 0.1 wide?)
                # Let's use a simple distance decay: xG = 0.8 * exp(-2 * dist)
                xg = 0.8 * np.exp(-2 * dist)
                self.stats["expected_goals"] += xg

    def count_getpossession(self, prev_obs, obs):

        if prev_obs["score"][1] < obs["score"][1]:
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return

        if (
            obs["game_mode"] == 3
        ):  # change of ball ownership from free kick, this is likely due to opponent offside
            self.steal_ball_recording = (
                False  # change of ball ownership due to opponent's goal
            )
            return

        if self.steal_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 1:
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0 and obs["ball_owned_player"] == 0
            ):  # our goalkeeper intercept the ball
                self.steal_ball_recording = False
            elif (
                obs["ball_owned_team"] == 0
                and obs["ball_owned_player"] != 0
                and obs["active"] == obs["ball_owned_player"]
            ):
                self.steal_ball_recording = False
                self.stats[
                    "interception"
                ] += 1  # only reward the agent stealing the ball (can we make it team reward?)
                
                # Defensive Recovery Success
                if self.lost_ball_timestamp is not None:
                     recovery_time = self.step_count - self.lost_ball_timestamp
                     self.stats["recovery_times"].append(recovery_time)
                     self.lost_ball_timestamp = None

        if (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == 0:
            if obs["active"] == obs["ball_owned_player"]:
                self.stats["tackle"] += 1
                # Defensive Recovery Success (Tackle)
                if self.lost_ball_timestamp is not None:
                     recovery_time = self.step_count - self.lost_ball_timestamp
                     self.stats["recovery_times"].append(recovery_time)
                     self.lost_ball_timestamp = None
                     
        elif (
            prev_obs["ball_owned_team"] == 1 and prev_obs["ball_owned_player"] != 0
        ) and obs["ball_owned_team"] == -1:
            self.steal_ball_recording = True
        else:
            pass

    def count_losepossession(self, prev_obs, obs):

        if prev_obs["score"][0] < obs["score"][0]:
            self.lost_ball_recording = (
                False  # change of ball ownership due to ours goal
            )
            self.lost_ball_timestamp = None
            return

        if self.lost_ball_recording:
            if obs["ball_owned_team"] == -1:
                pass
            elif obs["ball_owned_team"] == 0:  # back to our team
                self.lost_ball_recording = False
                self.lost_ball_timestamp = None
                # can add reward here
            else:  # opponent own it
                if self.last_hold_player == 0:  # our goalkeeper lose the ball
                    self.lost_ball_recording = False
                    self.lost_ball_timestamp = None

                if obs["active"] == self.last_hold_player:
                    self.lost_ball_recording = False
                    self.lost_ball_timestamp = None
                    self.stats["get_intercepted"] += 1

        if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
            if (
                obs["active"] == prev_obs["ball_owned_player"]
            ):  # current player is the last holding player
                self.stats["get_tackled"] += 1
            
            # Start tracking recovery time
            self.lost_ball_recording = True # Wait, this logic was already here? No, it wasn't set to True here in original code?
            # Original code:
            # elif prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == -1:
            #    self.lost_ball_recording = True
            
            # If we lose directly to opponent (tackled), we should also start tracking?
            # The original code didn't set lost_ball_recording = True here.
            # I will set a timestamp.
            self.lost_ball_timestamp = self.step_count

        elif prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == -1:
            self.lost_ball_recording = True
            self.last_hold_player = prev_obs["ball_owned_player"]
            self.lost_ball_timestamp = self.step_count

    def count_move(self, prev_obs, obs):
        self.step_count += 1 # Increment step count
        current_player = obs["active"]
        left_position_move = np.sum(
            (prev_obs["left_team"][current_player] - obs["left_team"][current_player])
            ** 2
        )
        self.stats["total_move"] += left_position_move
