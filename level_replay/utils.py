# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import collections
import timeit
import random

import numpy
import torch
import matplotlib.pyplot as plt
import PIL
import numpy as np
import pickle
from collections import defaultdict


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir, pattern='*'):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, pattern))
        for f in files:
            os.remove(f)


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(save_path):
    # This code will be moved to utils.py in future
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def save_score(run_dir, mean_train, mean_test, level_seeds_count, total_values):
    path_test = run_dir + '/' + 'test_mean.txt'
    np.savetxt(path_test, mean_test, fmt='%2f', delimiter=',')
    path_train = run_dir + '/' + 'train_mean.txt'
    np.savetxt(path_train, mean_train, fmt='%2f', delimiter=',')
    seeds_count = run_dir + '/' + 'seeds_count.txt'
    np.savetxt(seeds_count, level_seeds_count, fmt='%2f', delimiter=',')
    path_values = run_dir + '/' + 'total_values.txt'
    np.savetxt(path_values, total_values, fmt='%2f', delimiter=',')


def checkpoint(args, agent, actor_critic, checkpointpath):

    objs = {
            "model_state_dict": actor_critic.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "args": vars(args),
        }
    
    torch.save(objs, checkpointpath + ".tar")
    
    with open(checkpointpath + ".pkl", 'wb') as f:
        pickle.dump(objs,  f)


def save_index_plot(seeds, seeds_count):
    plt.figure(figsize=(16, 12))
    plt.title("Index count")
    plt.bar(seeds, torch.sum(seeds_count, 0), color='blue', label="index count")
    plt.legend()
    plt.tight_layout()

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    plt.close('all')

    return pil_image


def save_score_plot(seeds, seeds_score, seeds_count):

    vis_process_num = 5
    num_process_label = [f"process {i}" for i in range(0, vis_process_num)]
    fig, axes = plt.subplots(nrows=vis_process_num, ncols=1, figsize=(16, 12))

    for i, ax in enumerate(axes):
        basic_colors = ['tab:blue'] * len(seeds_score)
        basic_colors[int(seeds_count[i])] = 'tab:orange'
        ax.bar(seeds, seeds_score, color=basic_colors, label=num_process_label[i])
        ax.legend()
    plt.tight_layout()

    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    plt.close('all')

    return pil_image


def get_target_list(overcooked_result_dir, targets):
    target_list = []
    setting_seed_group = defaultdict(lambda: {'num_files': 0, 'files': []})

    for root, dirs, files in os.walk(overcooked_result_dir):
        path_part = root.split(os.sep)
        setting_name = "_".join(path_part[-4:-1])
        seed_name = path_part[-1]

        for file in files:
            # targets 리스트 내의 패턴 중 하나라도 일치하는지 확인
            if file == targets:
                full_path = os.path.join(root, file)
                target_list.append((setting_name + '_' + seed_name, full_path))
                setting_seed_group[setting_name]['files'].append(full_path)
                setting_seed_group[setting_name]['num_files'] += 1

    group_list = []
    for experiment, data in setting_seed_group.items():
        group_list.append((experiment, data['num_files'], data['files']))

    return target_list, group_list

class OvercookedController:

    def __init__(self, envs, model, args):
        self.envs = envs
        self.model = model
        self.args = args

    def act(self, rollouts, repeat_num, step):
        obs = rollouts.obs[repeat_num][step]
        other_agent_idx = rollouts.other_agent_idx[repeat_num][step]
        agent_idx = 1 - other_agent_idx
        obs0 = obs[:, 0, :, :]
        obs1 = obs[:, 1, :, :]

        ego_value, ego_actions, ego_action_log_dist, ego_recurrent_hidden_states, ego_recurrent_cell_states = self.model.act(obs0, rollouts.recurrent_hidden_states[repeat_num][step], rollouts.recurrent_cell_states[repeat_num][step], rollouts.masks[repeat_num][step])
        ego_action_log_prob = ego_action_log_dist.gather(-1, ego_actions)

        sp_value, other_agent_actions, sp_action_log_dist, sp_recurrent_hidden_states, sp_recurrent_cell_states = self.model.act(obs1, rollouts.recurrent_hidden_states[repeat_num][step], rollouts.recurrent_cell_states[repeat_num][step], rollouts.masks[repeat_num][step])
        sp_action_log_prob = sp_action_log_dist.gather(-1, other_agent_actions)

        joint_action = [(ego_actions[i], other_agent_actions[i]) for i in range(len(ego_actions))]

        agent_infos = {"joint_action": joint_action,
                       "ego": {"rnn_hidden": ego_recurrent_hidden_states, "rnn_cell": ego_recurrent_cell_states,
                               "action": ego_actions, "a_log_prob": ego_action_log_prob,
                               "a_log_dist": ego_action_log_dist, "value": ego_value}}

        return agent_infos

class OvercookedController_pbt:

    def __init__(self, envs, ego_agent, co_player_agent, args):
        self.envs = envs
        self.ego_model = ego_agent.model
        self.ego_rollouts = ego_agent.rollouts
        self.co_player_model = co_player_agent.model
        self.co_player_rollouts = co_player_agent.rollouts
        self.args = args

    def act(self, repeat_num, step):
        self.ego_model.train()
        self.co_player_model.train()

        obs = self.ego_rollouts.obs[repeat_num][step]
        other_agent_idx = self.ego_rollouts.other_agent_idx[repeat_num][step]
        agent_idx = 1 - other_agent_idx
        obs0 = obs[:, 0, :, :]
        obs1 = obs[:, 1, :, :]

        ego_v, ego_a, ego_a_log_dist, ego_rnn_hidden, ego_rnn_cells = self.ego_model.act(obs0, self.ego_rollouts.recurrent_hidden_states[repeat_num][step], self.ego_rollouts.recurrent_cell_states[repeat_num][step], self.ego_rollouts.masks[repeat_num][step])
        ego_a_log_prob = ego_a_log_dist.gather(-1, ego_a)

        co_player_v, co_player_a, co_player_a_log_dist, co_player_rnn_hidden, co_player_rnn_cells = self.co_player_model.act(obs1, self.co_player_rollouts.recurrent_hidden_states[repeat_num][step], self.co_player_rollouts.recurrent_cell_states[repeat_num][step], self.co_player_rollouts.masks[repeat_num][step])
        co_player_a_log_prob = co_player_a_log_dist.gather(-1, co_player_a)

        joint_action = [(ego_a[i], co_player_a[i]) for i in range(len(ego_a))]

        agent_infos = {
            "joint_action": joint_action,
            "ego": {"rnn_hidden": ego_rnn_hidden, "rnn_cell": ego_rnn_cells, "action": ego_a,
                    "a_log_prob": ego_a_log_prob, "a_log_dist": ego_a_log_dist, "value": ego_v},
            "co_player": {"rnn_hidden": co_player_rnn_hidden, "rnn_cell": co_player_rnn_cells, "action": co_player_a,
                          "a_log_prob": co_player_a_log_prob, "a_log_dist": co_player_a_log_dist, "value": co_player_v}}

        return agent_infos

