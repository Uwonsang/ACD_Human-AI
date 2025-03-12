# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial

import torch
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize, SubprocVecEnv

from level_replay.level_sampler import LevelSampler
from overcooked_ai_py.mdp.overcooked_env import Overcooked
from overcooked_ai.baselines_utils import RewardShapingEnv
import numpy as np

class SeededSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        super(SubprocVecEnv, self).__init__(env_fns, )

    def seed_async(self, seed, index):
        self._assert_not_closed()
        self.remotes[index].send(('seed', seed))
        self.waiting = True

    def seed_wait(self, index):
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def seed(self, seed, index):
        self.seed_async(seed, index)
        return self.seed_wait(index)

    def observe_async(self, index):
        self._assert_not_closed()
        self.remotes[index].send(('observe', None))
        self.waiting = True

    def observe_wait(self, index):
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def observe(self, index):
        self.observe_async(index)
        return self.observe_wait(index)

    def level_seed_async(self, index):
        self._assert_not_closed()
        self.remotes[index].send(('level_seed', None))
        self.waiting = True

    def level_seed_wait(self, index):
        self._assert_not_closed()
        level_seed = self.remotes[index].recv()
        self.waiting = False
        return level_seed

    def level_seed(self, index):
        self.level_seed_async(index)
        return self.level_seed_wait(index)

class VecOvercooked(SeededSubprocVecEnv):
    def __init__(self, seeds, all_args, **_params):

        layout_list = self._make_layout_dir(all_args, _params)
        if seeds is None:
            if _params['obp_eval_map']:
                seed_list = list(range(5))
            elif "start_seed" in _params:
                seed_list = [int(seed.split('_')[0]) for seed in layout_list[_params["start_seed"]:_params["start_seed"]+all_args.num_processes_test]]
            else:
                seed_list = [int(seed.split('_')[0]) for seed in layout_list[-50:]]
            seeds = seed_list
        else:
            seeds = [int(s) for s in np.random.choice(seeds, _params["sim_threads"])]

        all_args.activate_planner = _params["activate_planner"]
        all_args.human_proxy_num = _params["human_proxy_num"]
        all_args.obp_eval_map = _params["obp_eval_map"]

        env_fn = [partial(self._gym_env_fn, layout_list, all_args, _params, seeds[i], i) for i in range(_params["sim_threads"])]

        super(SeededSubprocVecEnv, self).__init__(env_fn)

    @staticmethod
    def _make_layout_dir(all_args, _params):

        if _params['obp_eval_map']:
            layout_path = all_args.eval_layout_path
        else:
            layout_path = os.path.join(all_args.layouts_dir, all_args.layouts_type)

        layout_list = [file for file in os.listdir(layout_path) if os.path.isfile(os.path.join(layout_path, file))]
        layout_list = sorted(layout_list, key=lambda x: int(x.split('_')[0]))
        layout_name_list = []

        for i in layout_list:
            layout_name = i.split('.')[0]
            layout_name_list.append(layout_name)

        return layout_name_list

    @staticmethod
    def _gym_env_fn(layout_list, all_args, _params, seed, thread_num):
        ENV = Overcooked
        gym_env = ENV(all_args, layout_list, seed, thread_num)
        return gym_env


class VecPyTorchOvercooked(RewardShapingEnv):
    def __init__(self, venv, device, level_sampler=None):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchOvercooked, self).__init__(venv)
        self.device = device
        self.is_first_step = False

        self.level_sampler = level_sampler

        m, n, c = venv.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [c, m, n],
            dtype=self.observation_space.dtype)

    @property
    def raw_venv(self):
        rvenv = self.venv
        while hasattr(rvenv, 'venv'):
            rvenv = rvenv.venv
        return rvenv

    def reset(self):
        if self.level_sampler:
            seeds = torch.zeros(self.venv.num_envs, dtype=torch.long)
            for e in range(self.venv.num_envs):
                seed = self.level_sampler.sample('sample')
                seeds[e] = seed
                self.venv.seed(seed, e)

        obs = self.venv.reset()
        if obs["both_agent_obs"].shape[2] != 20:
            obs["both_agent_obs"] = obs["both_agent_obs"].transpose(0, 1, 4, 3, 2)
        obs["both_agent_obs"] = torch.from_numpy(obs["both_agent_obs"]).float().to(self.device)
        obs["other_agent_env_idx"] = torch.from_numpy(obs["other_agent_env_idx"]).float().to(self.device)
        # obs = torch.from_numpy(obs).float().to(self.device) / 255.

        if self.level_sampler:
            return obs, seeds
        else:
            return obs

    def step_async(self, actions):
        actions = torch.tensor(actions, device='cpu')
        actions = np.array(actions)
        # if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
        #     # Squeeze the dimension for discrete actions
        #     actions = actions.squeeze(1)
        # actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        # reset environment here
        for e in done.nonzero()[0]:
            if self.level_sampler:
                seed = self.level_sampler.sample(info[e]['level_seed'], 'sample')
                obs[e] = self.venv.seed(seed, e)
            else:
                # seed = int.from_bytes(os.urandom(4), byteorder="little")
                obs[e] = self.venv.reset()
            # obs[e] = self.venv.seed(seed, e)  # seed resets the corresponding level

        if obs["both_agent_obs"].shape[2] != 20:
            obs["both_agent_obs"] = obs["both_agent_obs"].transpose(0, 1, 4, 3, 2)
        obs["both_agent_obs"] = torch.from_numpy(obs["both_agent_obs"]).float().to(self.device)
        obs["other_agent_env_idx"] = torch.from_numpy(obs["other_agent_env_idx"]).float().to(self.device)
        # obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(np.array(reward)).unsqueeze(dim=1).float()

        return obs, reward, done, info

# Makes a vector environment
def make_lr_venv(num_envs, seeds, device, all_args, **kwargs):

    level_sampler = kwargs.get('level_sampler')
    level_sampler_args = kwargs.get('level_sampler_args')

    ret_normalization = not kwargs.get('no_ret_normalization', False)
    print("Making {} envs...".format(num_envs))

    overcooked_params["sim_threads"] = num_envs
    if "human_proxy_num" in kwargs:
        overcooked_params["human_proxy_num"] = kwargs["human_proxy_num"]
    if "obp_eval_map" in kwargs:
        overcooked_params["obp_eval_map"] = kwargs["obp_eval_map"]
    if "activate_planner" in kwargs:
        overcooked_params["activate_planner"] = kwargs["activate_planner"]
    if "start_seed" in kwargs:
        overcooked_params["start_seed"] = kwargs["start_seed"]


    params = overcooked_params

    gym_env = VecOvercooked(seeds, all_args, **params)
    gym_env = VecMonitor(venv=gym_env, filename=None, keep_buf=100)
    gym_env = VecNormalize(venv=gym_env, ob=False, ret=ret_normalization)

    gym_env = RewardShapingEnv(gym_env)
    if level_sampler_args:
        level_sampler = LevelSampler(
            seeds,
            gym_env.observation_space, gym_env.action_space,
            **level_sampler_args)

    elif seeds:
        level_sampler = LevelSampler(
            seeds,
            gym_env.observation_space, gym_env.action_space,
            strategy='random')

    envs = VecPyTorchOvercooked(gym_env, device, level_sampler=level_sampler)
    gym_env.update_reward_shaping_param(1 if params["mdp_params"]["rew_shaping_params"] != 0 else 0)
    envs.self_play_randomization = 0 if params["SELF_PLAY_HORIZON"] is None else 1
    envs.trajectory_sp = params["TRAJECTORY_SELF_PLAY"]

    return envs, level_sampler

##################
# GENERAL PARAMS #
##################

# Number of parallel environments used for simulating rollouts
sim_threads = 30

human_proxy_num = 0

obp_eval_map = False

##################
# MDP/ENV PARAMS #
##################

# Mdp params
layout_name = "simple"
start_order_list = None

rew_shaping_params = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}

# Env params
horizon = 400

SELF_PLAY_HORIZON = 1

# Whether mixing of self play policies# happens on a trajectory or on a single-timestep level
# Recommended to keep to true
TRAJECTORY_SELF_PLAY = True

overcooked_params = {
    "mdp_params": {
        "layout_name": layout_name,
        "start_order_list": start_order_list,
        "rew_shaping_params": rew_shaping_params
    },
    "env_params": {
        "horizon": horizon
    },

    "SELF_PLAY_HORIZON": SELF_PLAY_HORIZON,
    "TRAJECTORY_SELF_PLAY": TRAJECTORY_SELF_PLAY,
    "sim_threads": sim_threads,
    "human_proxy_num": human_proxy_num,
    "obp_eval_map": obp_eval_map
}