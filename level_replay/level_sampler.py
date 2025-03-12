# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from collections import deque
from level_replay import utils


class LevelSampler():
    def __init__(
        self, seeds, obs_space, action_space, num_actors=1, 
        strategy='random', replay_schedule='fixed', score_transform='power',
        temperature=1.0, eps=0.05,
        rho=0.2, nu=0.5, alpha=1.0, 
        staleness_coef=0, staleness_transform='power', staleness_temperature=1.0, layouts_type='big_4',
        num_repeat=1, similarity_coef=0.1, name="ego"):
        self.obs_space = obs_space
        self.action_space = action_space
        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.similarity_coef = similarity_coef
        self.tmp_number = 0
        self.sample_num = 0
        self.sample_unseen_bool = True
        self.seed_deque = deque(maxlen=5)  # 시드별 점수 담는 fifo 구조
        self.index_deque = deque(maxlen=5)  # 시드 중 뽑은 인덱스 담는 fifo 구조
        self.total_index = []

        self.layout_type = layouts_type
        self.name = name
        self.top_k_buffer = dict()
        self.num_actors = num_actors
        self.num_repeat = num_repeat

        # Track seeds and scores as in np arrays backed by shared memory
        self._init_seed_index(seeds)

        self.unseen_seed_weights = np.array([1.]*len(seeds))
        self.seed_scores = np.array([0.]*len(seeds), dtype=np.float)
        self.partial_seed_scores = np.zeros((self.num_repeat, self.num_actors, len(seeds)), dtype=np.float)
        self.partial_seed_steps = np.zeros((self.num_repeat, self.num_actors, len(seeds)), dtype=np.int64)
        self.seed_staleness = np.array([0.]*len(seeds), dtype=np.float)

        self.next_seed_index = 0  # Only used for sequential strategy
        self.max_seed_returns = np.full(len(seeds), np.nan, dtype=np.float)

    def seed_range(self):
        return (int(min(self.seeds)), int(max(self.seeds)))

    def _init_seed_index(self, seeds):
        self.seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

    def update_with_rollouts(self, rollouts, num_repeat):
        if self.strategy == 'random':
            return

        # Update with a RolloutStorage object
        if self.strategy == 'policy_entropy':
            score_function = self._average_entropy
        elif self.strategy == 'least_confidence':
            score_function = self._average_least_confidence
        elif self.strategy == 'min_margin':
            score_function = self._average_min_margin
        elif self.strategy == 'gae':
            score_function = self._average_gae
        elif self.strategy == 'value_l1':
            score_function = self._average_value_l1
        elif self.strategy == 'positive_value_loss':
            score_function = self._average_positive_value_loss
        elif self.strategy == 'one_step_td_error':
            score_function = self._one_step_td_error
        elif self.strategy == 'return':
            score_function = self._return
        elif self.strategy == 'threshold':
            score_function = self._threshold_reward
        elif self.strategy == 'reward':
            score_function = self._reward
        else:
            raise ValueError(f'Unsupported strategy, {self.strategy}')

        seed_scores = self._update_with_rollouts(rollouts, score_function, num_repeat)

        return seed_scores

    def update_seed_score(self, repeat_index, actor_index, seed_idx, score, num_steps):
        score = self._partial_update_seed_score(repeat_index, actor_index, seed_idx, score, num_steps, done=True)

        self.unseen_seed_weights[seed_idx] = 0.  # No longer unseen

        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha)*old_score + self.alpha*score

    def _partial_update_seed_score(self, repeat_index, actor_index, seed_idx, score, num_steps, done=False):
        partial_score = self.partial_seed_scores[repeat_index][actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[repeat_index][actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)

        if done:
            self.partial_seed_scores[repeat_index][actor_index][seed_idx] = 0.  # zero partial score, partial num_steps
            self.partial_seed_steps[repeat_index][actor_index][seed_idx] = 0
        else:
            self.partial_seed_scores[repeat_index][actor_index][seed_idx] = merged_score
            self.partial_seed_steps[repeat_index][actor_index][seed_idx] = running_num_steps

        return merged_score

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions

        return (-torch.exp(episode_logits)*episode_logits).sum(-1).mean().item()/max_entropy

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        return (1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])).mean().item()

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        return 1 - (top2_confidence[:, 0] - top2_confidence[:, 1]).mean().item()

    def _average_gae(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        return advantages.mean().item()

    def _average_value_l1(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        return advantages.abs().mean().item()

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs['rewards']
        value_preds = kwargs['value_preds']

        max_t = len(rewards)
        td_errors = (rewards[:-1] + value_preds[:max_t-1] - value_preds[1:max_t]).abs()

        return td_errors.abs().mean().item()

    def _return(self, **kwargs):
        returns = kwargs['returns']

        return returns.mean().item()

    def _average_positive_value_loss(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        clipped_advantages = (returns - value_preds).clamp(0)

        return clipped_advantages.mean().item()

    def _threshold_reward(self, **kwargs):
        rewards = kwargs['rewards']
        total_reward = torch.sum(rewards)
        if total_reward < 5 or total_reward > 80:
            total_reward = 0
        return total_reward

    def _reward(self, **kwargs):
        rewards = kwargs['rewards']
        total_reward = torch.sum(rewards)

        return total_reward


    @property
    def requires_value_buffers(self):
        return self.strategy in ['gae', 'value_l1', 'one_step_td_error', 'return', 'positive_value_loss', 'threshold', 'reward']

    def _update_with_rollouts(self, rollouts, score_function, num_repeat):

        for repeat_index in range(num_repeat):

            level_seeds = rollouts.level_seeds[repeat_index]
            policy_logits = rollouts.action_log_dist[repeat_index]
            done = ~(rollouts.masks[repeat_index] > 0)
            total_steps, num_actors = policy_logits.shape[:2]
            num_decisions = len(policy_logits)

            for actor_index in range(num_actors):
                done_steps = done[:, actor_index].nonzero()[:total_steps, 0]
                start_t = 0

                for t in done_steps:
                    if not start_t < total_steps: break

                    if t == 0:  # if t is 0, then this done step caused a full update of previous seed last cycle
                        continue

                    seed_t = level_seeds[start_t, actor_index].item()
                    seed_idx_t = self.seed2index[seed_t]

                    current_returns = rollouts.returns[repeat_index][start_t:t, actor_index].sum().item()

                    # In overcooked, we set the fixed horizon time
                    if np.isnan(self.max_seed_returns[seed_idx_t]):
                        self.max_seed_returns[seed_idx_t] = current_returns
                    else:
                        self.max_seed_returns[seed_idx_t] = max(self.max_seed_returns[seed_idx_t], current_returns)

                    score_function_kwargs = {}
                    episode_logits = policy_logits[start_t:t, actor_index]
                    score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                    score_function_kwargs['seed_idx'] = seed_idx_t

                    if self.requires_value_buffers:
                        score_function_kwargs['returns'] = rollouts.returns[repeat_index][start_t:t, actor_index]
                        score_function_kwargs['rewards'] = rollouts.rewards[repeat_index][start_t:t, actor_index]
                        score_function_kwargs['value_preds'] = rollouts.value_preds[repeat_index][start_t:t, actor_index]

                    score = score_function(**score_function_kwargs)
                    num_steps = len(episode_logits)
                    self.update_seed_score(repeat_index, actor_index, seed_idx_t, score, num_steps)

                    start_t = t.item()

                if start_t < total_steps:
                    seed_t = level_seeds[start_t, actor_index].item()
                    seed_idx_t = self.seed2index[seed_t]

                    score_function_kwargs = {}
                    episode_logits = policy_logits[start_t:, actor_index]
                    score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                    score_function_kwargs['seed_idx'] = seed_idx_t

                    if self.requires_value_buffers:
                        score_function_kwargs['returns'] = rollouts.returns[repeat_index][start_t:, actor_index]
                        score_function_kwargs['rewards'] = rollouts.rewards[repeat_index][start_t:, actor_index]
                        score_function_kwargs['value_preds'] = rollouts.value_preds[repeat_index][start_t:, actor_index]

                    score = score_function(**score_function_kwargs)
                    num_steps = len(episode_logits)
                    self._partial_update_seed_score(repeat_index, actor_index, seed_idx_t, score, num_steps)

        return self.seed_scores

    def after_update(self):
        # Replace multiple "for" statements with fast computations using numpy
        # Reset partial updates, since weights have changed, and thus logits are now stale
        non_zero_indices = np.argwhere(self.partial_seed_scores != 0)
        for repeat_idx, actor_index, seed_idx in non_zero_indices:
            self.update_seed_score(repeat_idx, actor_index, seed_idx, 0, 0)
            
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def _sample_replay_level(self, past_level_seed):
        sample_weights = self.sample_weights(past_level_seed)

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float)/len(sample_weights)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        self.index_deque.append(seed)  # 뽑은 seed append 해주는 코드
        # Code that causing severe bottleneck based on num_train_seeds
        # if self.tmp_number > 63 and self.tmp_number % 64 == 0:
        #     self.save_plot()
        # self.tmp_number += 1

        return int(seed)

    def _sample_unseen_level(self):
        sample_weights = self.unseen_seed_weights/self.unseen_seed_weights.sum()
        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def sample(self, past_level_seed, strategy=None):
        self.sample_num += 1

        if not strategy:
            strategy = self.strategy
        if strategy == 'sample':
            strategy = self.strategy

        if strategy == 'random':
            seed_idx = np.random.choice(range((len(self.seeds))))
            seed = self.seeds[seed_idx]

            return int(seed)

        if strategy == 'sequential':
            seed_idx = self.next_seed_index
            self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
            seed = self.seeds[seed_idx]
            return int(seed)

        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen)/len(self.seeds)

        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho: 
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level(past_level_seed)

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else:  # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                self.sample_unseen_bool = False
                return self._sample_replay_level(past_level_seed)
            else:
                self.sample_unseen_bool = True
                return self._sample_unseen_level()

    def sample_weights(self, past_level_seed):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1-self.unseen_seed_weights)  # zero out unseen levels

        if self.top_k_buffer:
            top_k_array = np.zeros(len(self.seed_scores))
            for key in self.top_k_buffer.keys():
                top_k_array[key] = 1
            weights = weights * top_k_array

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1-self.unseen_seed_weights)

            if self.top_k_buffer:
                staleness_weights = staleness_weights * top_k_array

            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z

            weights = (1 - self.staleness_coef) * weights + (self.staleness_coef * staleness_weights)

        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        if transform == 'max':
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -1e5  # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.
        elif transform == 'min':
            weights = np.zeros_like(scores)
            scores = scores[:]
            self.seed_deque.append(scores)
            scores[self.unseen_seed_weights > 0] = 1e5  # only argmin over seen levels
            argmin = np.random.choice(np.flatnonzero(np.isclose(scores, scores.min())))
            weights[argmin] = 1.
        elif transform == 'eps_greedy':
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1. - self.eps
            weights += self.eps/len(self.seeds)
        elif transform == 'rank':
            self.seed_deque.append(scores)
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == 'rank_low':
            self.seed_deque.append(scores)
            temp = scores.argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks ** (1. / temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)

        return weights

    def top_k_insert(self, seed, seed_scores, max_size, env_sample_method):
        replace = False

        # update buffer
        if seed in self.top_k_buffer.keys():
            self.top_k_buffer[seed] = seed_scores[seed]

        if len(self.top_k_buffer) < max_size:
            self.top_k_buffer[seed] = seed_scores[seed]
        else:
            if env_sample_method == "top":
                idx = min(self.top_k_buffer, key=self.top_k_buffer.get)
                replace = self.top_k_buffer[idx] < seed_scores[seed]
            elif env_sample_method == "low":
                idx = max(self.top_k_buffer, key=self.top_k_buffer.get)
                replace = self.top_k_buffer[idx] > seed_scores[seed]
            else:
                raise ValueError(f"Invalid env_sample_method: {env_sample_method}")

        if replace:
            del self.top_k_buffer[idx]
            self.top_k_buffer[seed] = seed_scores[seed]
