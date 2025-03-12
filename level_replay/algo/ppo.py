# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim


class PPO():
    """
    Vanilla PPO
    """
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 ratio_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 num_repeat,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.ratio_mini_batch = ratio_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.num_repeat = num_repeat
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.ppo_update_step = 0

    def update(self, rollouts, sample_unseen_bool, step, base_policy="ppo"):

        # '''Scheduler'''
        # if step > 1 and step % 4 == 0:
        #     self.entropy_coef = max(0.1 - (0.095 / 1250) * step, 0.005)

        total_advantages = []
        total_value = []
        for repeat in range(self.num_repeat):
            advantages = rollouts.returns[repeat][:-1] - rollouts.value_preds[repeat][:-1]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            total_advantages.append(advantages)

            values = rollouts.value_preds[repeat][:-1]
            total_value.append(values.abs().mean())

        total_advantages = torch.stack(total_advantages, dim=0)
        total_value = torch.stack(total_value, dim=0)

        num_repeat, num_step, num_processes, _ = total_advantages.shape
        total_advantages = total_advantages.reshape(num_repeat * num_step, num_processes, -1)
        total_value = total_value.mean()

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    total_advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    total_advantages, self.num_mini_batch, self.ratio_mini_batch)

            for sample in data_generator:
                self.ppo_update_step += 1

                if base_policy == "e3t":
                    obs_batch, recurrent_hidden_states_batch, recurrent_cell_states_batch, actions_batch, \
                        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, obs0_pre_batch, action1_batch = sample

                    values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, recurrent_cell_states_batch, masks_batch,
                        actions_batch, obs0_pre_batch, action1_batch)

                else:
                    obs_batch, recurrent_hidden_states_batch, recurrent_cell_states_batch, actions_batch, \
                        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                    values, action_log_probs, dist_entropy, _, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, recurrent_cell_states_batch, masks_batch,
                        actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (
                    value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                self.optimizer.zero_grad()
                loss = (value_loss*self.value_loss_coef + action_loss - dist_entropy*self.entropy_coef)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                if sample_unseen_bool is False:
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        if self.ratio_mini_batch > 0:
            new_num_mini_batch = len(total_advantages.flatten()) // len(obs_batch)
            num_updates = self.ppo_epoch * new_num_mini_batch
        else:
            num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_value, self.ppo_update_step, num_updates
