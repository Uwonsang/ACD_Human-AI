import gym
from gym import spaces

import cv2
import pygame
import copy
import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv as OriginalEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action
import torch

# def _convert_action(joint_action) -> list:
#     action_set = []
#     for _action in joint_action:
#         action_set.append(Action.INDEX_TO_ACTION[int(_action)])
#
#     return action_set


class OverCookedEnv_Play(gym.Env):
    
    def __init__(self,
                 scenario="sample",
                 episode_length=400,
                 time_limits=60,
                 tutorial=False
                 ):

        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }


        if tutorial:
            self.mdp_params = {"layouts_dir": './overcooked_layout/original/', 'start_order_list': None, "rew_shaping_params": rew_shaping_params}
        else:
            self.mdp_params = {"layouts_dir": './overcooked_layout/eval/', 'start_order_list': None, "rew_shaping_params": rew_shaping_params}

        self.scenario = scenario
        self.episode_length = episode_length
        self.time_limits = time_limits
        self.mdp_params.update({'layout_name': self.scenario})
        self.agent_idx = 0


        self.mdp_fn = lambda: OvercookedGridworld.from_layout_name(**self.mdp_params)
        self.base_mdp = self.mdp_fn()
        self.base_env = OriginalEnv(self.mdp_fn, start_state_fn=None, horizon=episode_length)
        self.featurize_fn = lambda state: self.base_mdp.lossless_state_encoding(state)
        self.featurize_fn_bc = lambda state: self.base_mdp.featurize_state(state)
        self.observation_space = self._setup_observation_space()
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

        self.visualizer = StateVisualizer()
        self.score = 0

        self._available_actions = Action.ALL_ACTIONS


    def _setup_observation_space(self):
        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_T = self.featurize_fn(dummy_state)[0].transpose(2, 1, 0)
        obs_shape = obs_T.shape
        high = np.ones(obs_shape) * float("inf")
        low = np.zeros(obs_shape)
        # high = np.ones(obs_shape) * max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)
        return gym.spaces.Box(low, high, dtype=np.float32)


    def step(self, action):
        assert all(self.action_space.contains(a) for a in action), "%r (%s) invalid" % (action, type(action))
        agent_action, other_agent_action = [Action.INDEX_TO_ACTION[a] for a in action]

        if self.agent_idx == 0:
            joint_action = [agent_action, other_agent_action]
        else:
            joint_action = [other_agent_action, agent_action]

        joint_action = tuple(joint_action)

        next_state, reward, done, info = self.base_env.step(joint_action)
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)

        both_agents_ob = np.asarray(both_agents_ob)


        obs = {"both_agent_obs": both_agents_ob,
               "overcooked_state": next_state,
               "other_agent_env_idx": 1 - self.agent_idx}

        obs["both_agent_obs"] = obs["both_agent_obs"].transpose(0, 3, 2, 1)
        obs["both_agent_obs"] = torch.from_numpy(obs["both_agent_obs"]).float()
        obs["both_agent_obs"] = torch.unsqueeze(obs["both_agent_obs"], dim=0)
        # obs["other_agent_env_idx"] = torch.from_numpy(obs["other_agent_env_idx"]).float()

        self.score += reward

        return obs, reward, done, info

    def reset(self):
        self.base_env.reset()

        self.mdp = self.base_env.mdp
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)

        both_agents_ob = np.asarray(both_agents_ob)

        obs = {"both_agent_obs": both_agents_ob,
                "overcooked_state": self.base_env.state,
                "other_agent_env_idx": 1 - self.agent_idx}

        obs["both_agent_obs"] = obs["both_agent_obs"].transpose(0, 3, 2, 1)
        obs["both_agent_obs"] = torch.from_numpy(obs["both_agent_obs"]).float()
        obs["both_agent_obs"] = torch.unsqueeze(obs["both_agent_obs"], dim=0)
        # obs["other_agent_env_idx"] = torch.from_numpy(obs["other_agent_env_idx"])

        return obs

    def render(self, time=0, mode='rgb_array'):
        timeleft = self.time_limits - time
        image = self.visualizer.render_state(state=self.base_env.state, grid=self.base_env.mdp.terrain_mtx,
                                             hud_data=StateVisualizer.default_hud_data(self.base_env.state, timeleft=timeleft, score=self.score))

        buffer = pygame.surfarray.array3d(image)
        image = copy.deepcopy(buffer)
        image = np.flip(np.rot90(image, 3), 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (int(528*1.5), int(464*1.5)))

        return image


