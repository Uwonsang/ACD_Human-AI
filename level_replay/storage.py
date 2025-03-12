# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/model.py

import torch
import numpy as np
from torch.utils.data.sampler import \
    BatchSampler, SubsetRandomSampler, SequentialSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, split_ratio=0.05):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.action_log_dist = torch.zeros(num_steps, num_processes, action_space.n)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)

        self.num_steps = num_steps
        self.step = 0
        
        self.split_ratio = split_ratio

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.level_seeds = self.level_seeds.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, action_log_dist,
               value_preds, rewards, masks, bad_masks, level_seeds=None):
        if len(rewards.shape) == 3: rewards = rewards.squeeze(2)
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.action_log_dist[self.step].copy_(action_log_dist)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + gamma * self.value_preds[
                step + 1] * self.masks[step +
                                        1] - self.value_preds[step]
            gae = delta + gamma * gae_lambda * self.masks[step +
                                                            1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
     
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                            self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ


class OvercookedRolloutStorage(object):
    def __init__(self, num_steps, num_processes, num_repeat, obs_shape, action_space,
                 recurrent_hidden_state_size, split_ratio=0.05):
        self.obs = torch.zeros(num_repeat, num_steps + 1, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_repeat, num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.recurrent_cell_states = torch.zeros(
            num_repeat, num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_repeat, num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_repeat, num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_repeat, num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_repeat, num_steps, num_processes, 1)
        self.action_log_dist = torch.zeros(num_repeat, num_steps, num_processes, action_space.n)
        ## for overcooked
        self.other_agent_idx = torch.zeros((num_repeat, num_steps + 1, num_processes))
        self.curr_state = np.zeros((num_repeat, num_steps + 1, num_processes), dtype=object)

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_repeat, num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_repeat, num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_repeat, num_steps + 1, num_processes, 1)

        self.level_seeds = torch.zeros(num_repeat, num_steps, num_processes, 1, dtype=torch.int)

        self.num_repeat = num_repeat
        self.num_steps = num_steps
        self.step = 0

        self.split_ratio = split_ratio

        self.num_process = num_processes

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.recurrent_cell_states = self.recurrent_cell_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.level_seeds = self.level_seeds.to(device)
        # self.curr_state = self.curr_state.to(device)
        self.other_agent_idx = self.other_agent_idx.to(device)

    def co_to(self, device):
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.recurrent_cell_states = self.recurrent_cell_states.to(device)
        self.masks = self.masks.to(device)

    def obs_init(self, obs):

        for repeat in range(self.num_repeat):
            self.obs[repeat][0].copy_(obs["both_agent_obs"])
            self.other_agent_idx[repeat][0].copy_(obs["other_agent_env_idx"])
            self.curr_state[repeat][0] = obs["overcooked_state"]

    def next_value(self, model, agent_idx):

        next_value_list = []

        for repeat in range(self.num_repeat):
            obs_id = self.obs[repeat][-1]
            obs = obs_id[:, agent_idx, :, :]
            args = [obs, self.recurrent_hidden_states[repeat][-1], self.recurrent_cell_states[repeat][-1], self.masks[repeat][-1]]

            next_value = model.get_value(*args).detach()
            next_value_list.append(next_value)

        next_value_list = torch.stack(next_value_list, dim=0)

        return next_value_list


    def _insert_common(self, repeat_step, obs, recurrent_hidden_states, recurrent_cell_states, actions, action_log_probs, action_log_dist,
               value_preds, rewards, masks, bad_masks, curr_state, other_agent_idx, level_seeds):
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)
        self.obs[repeat_step][self.step + 1].copy_(obs)
        self.recurrent_hidden_states[repeat_step][self.step + 1].copy_(recurrent_hidden_states)
        self.recurrent_cell_states[repeat_step][self.step + 1].copy_(recurrent_cell_states)
        self.actions[repeat_step][self.step].copy_(actions)
        self.action_log_probs[repeat_step][self.step].copy_(action_log_probs)
        self.action_log_dist[repeat_step][self.step].copy_(action_log_dist)
        self.value_preds[repeat_step][self.step].copy_(value_preds)
        self.rewards[repeat_step][self.step].copy_(rewards)
        self.masks[repeat_step][self.step + 1].copy_(masks)
        self.bad_masks[repeat_step][self.step + 1].copy_(bad_masks)
        self.curr_state[repeat_step][self.step + 1] = curr_state
        self.other_agent_idx[repeat_step][self.step + 1].copy_(other_agent_idx)

        if level_seeds is not None:
            self.level_seeds[repeat_step][self.step].copy_(level_seeds)

    def insert(self, repeat_step, obs, rnn_hidden, rnn_cell, actions, action_log_probs, action_log_dist,
               value_preds, rewards, masks, bad_masks, curr_state, other_agent_idx, level_seeds=None):
        self._insert_common(repeat_step, obs, rnn_hidden, rnn_cell, actions, action_log_probs, action_log_dist, value_preds, rewards, masks, bad_masks, curr_state, other_agent_idx, level_seeds)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self, num_repeat):

        for repeat in range(num_repeat):

            self.obs[repeat][0].copy_(self.obs[repeat][-1])
            self.recurrent_hidden_states[repeat][0].copy_(self.recurrent_hidden_states[repeat][-1])
            self.recurrent_cell_states[repeat][0].copy_(self.recurrent_cell_states[repeat][-1])
            self.masks[repeat][0].copy_(self.masks[repeat][-1])
            self.bad_masks[repeat][0].copy_(self.bad_masks[repeat][-1])
            self.curr_state[repeat][0] = self.curr_state[repeat][-1]
            self.other_agent_idx[repeat][0].copy_(self.other_agent_idx[repeat][-1])

    def compute_returns(self,
                        next_value,
                        gamma,
                        gae_lambda):

        self.value_preds[:, -1, :, :] = next_value

        deltas = self.rewards + gamma * self.value_preds[:, 1:, :, :] * self.masks[:, 1:, :, :] - self.value_preds[:, :-1, :, :]
        gaes = torch.zeros_like(self.rewards)

        gae = 0
        for step in reversed(range(deltas.size(1))):
            gae = deltas[:, step, :, :] + gamma * gae_lambda * self.masks[:, step + 1, :, :] * gae
            gaes[:, step, :, :] = gae

        self.returns[:, :-1, :, :] = gaes + self.value_preds[:, :-1, :, :]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               ratio_mini_batch=None,
                               mini_batch_size=None):
        num_repeat, num_steps, num_processes = self.rewards.size()[0:3]
        batch_size = num_processes * num_steps * num_repeat

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            
            assert ratio_mini_batch <= 1.0, (
                "The value of ratio_mini_batch({}) is greatuer than 1.0. "
                "It must be a real number between 0.0 and 1.0."
                "".format(ratio_mini_batch) 
            )
            if ratio_mini_batch > 0:
                mini_batch_size = int(batch_size * ratio_mini_batch)
            else:   
                mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:

            obs0 = self.obs[:, :, :, 0, :, :]

            obs_batch = obs0[:, :-1, :, :, :].contiguous().view(-1, obs0.shape[3], obs0.shape[4], obs0.shape[5])[indices]

            recurrent_hidden_states_batch = self.recurrent_hidden_states[:, :-1, :, :].reshape(-1, self.recurrent_hidden_states.size(-1))[indices]
            recurrent_cell_states_batch = self.recurrent_cell_states[:, :-1, :, :].reshape(-1, self.recurrent_cell_states.size(-1))[indices]

            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:, :-1, :, :].reshape(-1, 1)[indices]
            return_batch = self.returns[:, :-1, :, :].reshape(-1, 1)[indices]
            masks_batch = self.masks[:, :-1, :, :].reshape(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, recurrent_cell_states_batch, actions_batch, \
              value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            recurrent_cell_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                recurrent_cell_states_batch.append(
                    self.recurrent_cell_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)
            recurrent_hidden_cell_batch = torch.stack(
                recurrent_cell_states_batch, 1).view(N, -1)
            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                                                         old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, recurrent_hidden_cell_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

