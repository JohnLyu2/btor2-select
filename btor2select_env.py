import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Btor2SelectMDP_Discrete(gym.Env):
    def __init__(self, instance_embed_size, res_dict, tool_config_dict, early_timeouts, embd_dict, timeout, render_mode=None):
        # print(f"gym version: {gym.__version__}")
        self.tool_config_num = len(tool_config_dict)
        self.instance_embed_size = instance_embed_size
        assert len(res_dict) > 0
        self.res_dict = res_dict
        self.tool_config_dict = tool_config_dict
        self.early_timeouts = early_timeouts
        self.timeout_option_num = len(early_timeouts) + 1 # always include commit the remaining time
        self.embd_dict = embd_dict
        self.timeout = timeout
        self.min_assign_time = 1.0
        # Observations are (all scaled to 0 - 1): Instance Embedding + ToolConfigAttempts + RemainingTime
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.instance_embed_size + self.tool_config_num + 1,), dtype=np.float32)
        # action space: solver0_timeout0, solver0_timeout1, ..., solver0_alltime, solver1_timeout0, ..., solver1_alltime, ...
        self.action_space = spaces.Discrete(self.tool_config_num * self.timeout_option_num)
        assert render_mode is None

    def _get_obs(self):
        sc_instance_embed_float32 = np.array(self._scaled_instance_embed, dtype=np.float32)
        attemps_float32 = np.array(self._attempts, dtype=np.float32)
        sc_remaining_time_float32 = np.array([self._remaining_time/self.timeout], dtype=np.float32)
        # concatenate the instance embedding, tool config attempts, and remaining time
        obs = np.concatenate([sc_instance_embed_float32, attemps_float32, sc_remaining_time_float32])
        return obs

    # def _get_info(self):
    #     return {"instance": self._instance_yml}

    # def _get_info(self):
    #     info_dict = {}
    #     info_dict["instance"] = self._instance_yml
    #     info_dict["remaining_time"] = self._remaining_time
    #     info_dict["history"] = self._history
    #     return info_dict

    # def print_info(self, info_dict):
    #     info_str = f"Instance: {info_dict['instance']}, Remaining Time: {info_dict['remaining_time']:2f}"
    #     for i, (tool_config, is_solved, is_correct, status_str, used_time, assign_time) in enumerate(info_dict['history']):
    #         info_str += f"\nAttempt {i+1}: ToolConfig {self.tool_config_dict[tool_config][1]} ({tool_config}), Solved: {is_solved}, Correct: {is_correct}, Status: {status_str}, RunTime: {used_time:.2f}, AssignTime: {assign_time:.2f}"
    #     print(info_str)

    # def legal_actions(self):
    #     # whether self._attempts is 0, if > 0, then it is illegal, return a list of 0s and 1s
    #     return [1 if attempt == 0 else 0 for attempt in self._attempts]

    def seed(self, seed=None):
        # Initialize the random number generator
        self.np_random, self.seed_value = gym.utils.seeding.np_random(seed)
        return [self.seed_value]

    def reset(self, seed=None):
        self.seed(seed)
        # randomly select an instance from the keys of self.res_dict
        self._instance_yml = self.np_random.choice(list(self.res_dict.keys()))
        obs, info = self._reset_setup()
        return obs, info
        
    def _reset_setup(self):
        self._instance_embed = self.embd_dict[self._instance_yml][1]
        self._scaled_instance_embed = self.embd_dict[self._instance_yml][2]
        self._embd_time = self.embd_dict[self._instance_yml][0]
        self._res_lst = self.res_dict[self._instance_yml]
        self._attempts = np.zeros(self.tool_config_num, dtype=int)
        self._remaining_time = self.timeout - self._embd_time
        self._history = []
        observation = self._get_obs()
        self._legal_actions = [1] * self.action_space.n
        return observation, {"is_success": False, "time": 0}

    def get_legal_actions(self):
        return self._legal_actions

    def step(self, action):
        # if not self._legal_actions[action]:
        #     raise ValueError(f"Action {action} is not legal")
        tool_config = action // self.timeout_option_num
        timeout_option = action % self.timeout_option_num
        assert timeout_option < self.timeout_option_num
        # assigned time is what the agent decides to do
        assign_time = self.early_timeouts[timeout_option] if timeout_option < len(self.early_timeouts) else self.timeout
        # allocated time is adjusted according to the remaining time
        alloc_time = assign_time if assign_time <= self._remaining_time else self._remaining_time
        res_tuple = self._res_lst[tool_config]
        return_time = res_tuple[3]
        assert return_time > 0
        is_interrupted = return_time > alloc_time
        if is_interrupted:
            res_tuple = (False, True, "INTERRUPTED", alloc_time)
        is_solved, is_correct, status_str, used_time = res_tuple
        self._attempts[tool_config] = 1
        # if the tool with timeout X is used, then the tool with timeout <= X is not legal for future
        for i in range(timeout_option+1):
            self._legal_actions[action - i] = 0
        exceed_penalty = self._remaining_time / self.timeout # may check these are implemented correctly later
        self._remaining_time = self._remaining_time - used_time
        assert self._remaining_time >= 0, f"Remaining Time: {self._remaining_time} < 0"
        self._history.append((tool_config, is_solved, is_correct, status_str, used_time, assign_time))
        # An episode is done if self._remaining_time is 0, or is_solved, or steps exceed the tool_config_num
        is_exceed = True if len(self._history) >= self.tool_config_num else False
        terminated = self._remaining_time == 0 or is_solved or is_exceed
        used_time_scaled = used_time / self.timeout
        if is_solved:
            reward = 1 - used_time_scaled
        elif is_exceed:
            reward = -exceed_penalty
        else:
            reward = -used_time_scaled
        observation = self._get_obs()
        total_used_time = self.timeout - self._remaining_time
        info = {"is_success": is_solved, "time": total_used_time}
        return observation, reward, terminated, False, info

    def render(self):
        # may print out sth
        pass

    def close(self):
        pass

class Btor2SelectMDP_D_test(Btor2SelectMDP_Discrete):
    def __init__(self, instance_embed_size, res_dict, tool_config_dict, early_timeouts, embd_dict, timeout, render_mode=None):
        super().__init__(instance_embed_size, res_dict, tool_config_dict, early_timeouts, embd_dict, timeout, render_mode)
        self.tobe_tested = 0

    def is_test_done(self):
        return self.tobe_tested >= len(self.res_dict)

    def reset_test(self):
        assert not self.is_test_done()
        self._instance_yml = list(self.res_dict.keys())[self.tobe_tested]
        self.tobe_tested += 1
        obs, info = self._reset_setup()
        return obs, info