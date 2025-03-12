import torch
from level_replay import utils
from level_replay import algo
from level_replay.model import OvercookedPolicy
from level_replay.storage import OvercookedRolloutStorage
from level_replay.level_sampler import LevelSampler
import numpy as np


class PBTAgent(object):

    def __init__(self, name, args, device, gym_env=None):
        self.args = args
        self.name = name
        self.gym_env = gym_env
        self.device = device

        self.model = OvercookedPolicy(gym_env.observation_space.shape, gym_env.action_space.n, args)

        self.agent = algo.PPO(self.model, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.ratio_mini_batch,
                              args.value_loss_coef, args.entropy_coef, args.num_repeat, lr=args.lr, eps=args.eps,
                              max_grad_norm=args.max_grad_norm)


    def env_buffer_init(self, seeds, level_sampler_args):
        self.level_sampler = LevelSampler(seeds, self.gym_env.observation_space, self.gym_env.action_space,
                                          **level_sampler_args)


    def set_rollouts(self, obs):
        self.rollouts = OvercookedRolloutStorage(self.args.num_steps, self.args.num_processes, self.args.num_repeat,
                                                 obs["both_agent_obs"].shape, self.gym_env.action_space,
                                                 self.model.recurrent_hidden_state_size)
        self.rollouts.obs_init(obs)
        self.rollouts.to(self.device)

    def set_co_rollouts(self, obs):
        self.rollouts = OvercookedRolloutStorage(self.args.num_steps, self.args.num_processes, self.args.num_repeat,
                                                 obs["both_agent_obs"].shape, self.gym_env.action_space,
                                                 self.model.recurrent_hidden_state_size)
        self.rollouts.obs_init(obs)
        self.rollouts.co_to(self.device)

    def insert_rollouts(self, repeat_step, agent_infos, obs, reward, masks, bad_masks, level_seeds):

        both_obs, curr_state, other_agent_idx = obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]
        self.rollouts.insert(repeat_step, both_obs, agent_infos["rnn_hidden"], agent_infos["rnn_cell"], agent_infos["action"],
                                agent_infos["a_log_prob"], agent_infos["a_log_dist"], agent_infos["value"], reward, masks, bad_masks, curr_state,
                                other_agent_idx, level_seeds)

    def compute_rollouts(self, agent_idx):

        with torch.no_grad():
            next_value = self.rollouts.next_value(self.model, agent_idx)
        self.rollouts.compute_returns(next_value, self.args.gamma, self.args.gae_lambda)

    def update(self, rollouts, sample_unseen_bool, step):
        """update agent model and parameters"""
        value_loss, action_loss, dist_entropy, value, ppo_update_step, ppo_num_updates = self.agent.update(rollouts, sample_unseen_bool, step)
        rollouts.after_update(self.args.num_repeat)

        return value_loss, action_loss, dist_entropy, value, ppo_update_step, ppo_num_updates

    def save(self, save_folder):
        """Save agent model and parameters"""
        """model, log, parameter"""
        utils.make_dir(save_folder)
        save_path = save_folder + "/model.tar"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                "args": vars(self.args),
            }, save_path
        )

    def load(self, load_model_path, device):
        check_point = torch.load(load_model_path, map_location=device)
        self.model.load_state_dict(check_point["model_stata_dict"])


def prioritized_agent_sampling(population, pbt_lambda, sample_method="max"):

    N = len(population)

    if sample_method == "max":
        score_pool = population_score(population, sample_method)
        idx = np.argmax(score_pool)
    elif sample_method == "min":
        score_pool = population_score(population, sample_method)
        idx = np.argmin(score_pool)
    elif sample_method == "avg":
        score_pool = population_score(population, sample_method)
        idx = np.argmin(score_pool)

    weight = np.full(N, pbt_lambda/N)
    weight[idx] = (N - pbt_lambda * (N-1)) / N

    return weight


def population_score(population, sample_method):

    score_pool = []
    for agent in population:
        env_buffer = agent.level_sampler.top_k_buffer

        if not env_buffer:
            score_pool.append(0)
            continue

        if sample_method == "max":
            score = max([score for seed, score in env_buffer.items()])
        elif sample_method == "min":
            score = min([score for seed, score in env_buffer.items()])
        elif sample_method == "avg":
            total_score = sum([score for seed, score in env_buffer.items()])
            score = total_score / len(env_buffer)

        score_pool.append(score)

    return score_pool