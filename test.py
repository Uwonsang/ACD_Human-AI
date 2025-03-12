# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import logging

import numpy as np
import torch
from tqdm import tqdm

from level_replay.model import OvercookedPolicy
from level_replay.level_sampler import LevelSampler
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, BASE_REW_SHAPING_PARAMS
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from level_replay.envs import make_lr_venv


def evaluate(
    args, 
    actor_critic, 
    num_episodes, 
    device,
    eval_envs=None,
    use_render=False,
    store_traj=False,
    num_processes=5,
    deterministic=False,
    seeds=None,
    level_sampler=None, 
    progressbar=None,
    isFinalEval=False,
    co_player=None,
    params=None,
    activate_planner=True):
    
    actor_critic.eval()
    args.use_render = use_render
    args.store_traj = store_traj
    if not eval_envs:
        eval_envs, level_sampler = make_lr_venv(
            num_envs=num_processes, seeds=seeds, device=device, all_args=args,
            no_ret_normalization=args.no_ret_normalization, level_sampler=level_sampler,
            activate_planner=activate_planner,
            obp_eval_map=True)
        
    eval_player1_episode_rewards = np.zeros((num_episodes, num_processes))
    eval_player2_episode_rewards = np.zeros((num_episodes, num_processes))
    eval_episode_rewards = np.zeros((num_episodes, num_processes))
    eval_episode_sparse_reward = np.zeros((num_episodes, num_processes))
    
    if level_sampler:
        obs, _ = eval_envs.reset()
    else:
        obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_recurrent_cell_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)
    
    if co_player is not None:
        co_player.eval()
        co_eval_recurrent_hidden_states = torch.zeros(
            num_processes, co_player.recurrent_hidden_state_size, device=device)
        co_eval_recurrent_cell_states = torch.zeros(
            num_processes, co_player.recurrent_hidden_state_size, device=device)
        co_eval_masks = torch.ones(num_processes, 1, device=device)
    if params is not None:
        mdp_fn = lambda: OvercookedGridworld.from_layout_name(**params)
        base_mdp = mdp_fn()
        mlp = MediumLevelPlanner.from_pickle_or_compute(base_mdp, NO_COUNTERS_PARAMS, force_compute=True)
        featurize_fn_bc = lambda state: base_mdp.featurize_state(state, mlp)  # Encoding obs for BC

    eval_epi_num = 0
    while eval_epi_num < num_episodes:
        with torch.no_grad():
            both_obs = obs["both_agent_obs"]
            obs0 = both_obs[:, 0, :, :]
            obs1 = both_obs[:, 1, :, :]
            eval_value, eval_action, eval_action_log_dist, eval_recurrent_hidden_states, eval_recurrent_cell_states = actor_critic.act(obs0, eval_recurrent_hidden_states, eval_recurrent_cell_states, eval_masks, deterministic=deterministic)
            if co_player is not None:
                if params is not None:
                    curr_state = obs["overcooked_state"][0]
                    obs0_bc, obs1_bc = featurize_fn_bc(curr_state)
                    obs1_bc = torch.from_numpy(obs1_bc).float()
                    obs1_bc = obs1_bc.to(device)
                    value, other_agent_actions, action_log_dist, co_eval_recurrent_hidden_states, co_eval_recurrent_cell_states = co_player.act(obs1_bc, co_eval_recurrent_hidden_states, co_eval_recurrent_cell_states, co_eval_masks, deterministic=deterministic)
                else:
                    value, other_agent_actions, action_log_dist, co_eval_recurrent_hidden_states, co_eval_recurrent_cell_states = co_player.act(obs1, co_eval_recurrent_hidden_states, co_eval_recurrent_cell_states, co_eval_masks, deterministic=deterministic)
            else:
                value, other_agent_actions, action_log_dist, recurrent_hidden_states, recurrent_cell_states = actor_critic.act(obs1, eval_recurrent_hidden_states, eval_recurrent_cell_states, eval_masks, deterministic=deterministic)

            joint_action = [(eval_action[i], other_agent_actions[i]) for i in range(len(eval_action))]

            obs, _, done, infos = eval_envs.step(joint_action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        if co_player is not None:
            co_eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                eval_episode_rewards[eval_epi_num][i] = info['episode']['r']
                eval_player1_episode_rewards[eval_epi_num][i] = info['episode']['ep_sparse_r_by_agent'][0] + info['episode']['ep_shaped_r_by_agent'][0]
                eval_player2_episode_rewards[eval_epi_num][i] = info['episode']['ep_sparse_r_by_agent'][1] + info['episode']['ep_shaped_r_by_agent'][1]
                eval_episode_sparse_reward[eval_epi_num][i] = info['episode']['ep_sparse_r_by_agent'][0] + info['episode']['ep_sparse_r_by_agent'][1]
                if progressbar:
                    progressbar.update(1)

        if all(done):
            eval_epi_num += 1

    if progressbar:
        progressbar.close()

    if isFinalEval:
        eval_envs.close()

    return np.mean(eval_episode_rewards, axis=0), np.mean(eval_player1_episode_rewards, axis=0), np.mean(eval_player2_episode_rewards, axis=0), np.mean(eval_episode_sparse_reward, axis=0)


def evaluate_e3t(args, actor_critic, num_episodes, device, eval_envs=None, use_render=False, store_traj=False,
                 num_processes=5, deterministic=False, seeds=None, level_sampler=None, progressbar=None,
                 isFinalEval=False, co_player=None, activate_planner=True):

    actor_critic.eval()
    args.use_render = use_render
    args.store_traj = store_traj
    if not eval_envs:
        eval_envs, level_sampler = make_lr_venv(
            num_envs=num_processes, seeds=seeds, device=device, all_args=args,
            no_ret_normalization=args.no_ret_normalization, level_sampler=level_sampler,
            activate_planner=activate_planner,
            obp_eval_map=True)

    eval_player1_episode_rewards = np.zeros((num_episodes, num_processes))
    eval_player2_episode_rewards = np.zeros((num_episodes, num_processes))
    eval_episode_rewards = np.zeros((num_episodes, num_processes))
    eval_episode_sparse_reward = np.zeros((num_episodes, num_processes))

    if level_sampler:
        obs, _ = eval_envs.reset()
    else:
        obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_recurrent_cell_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.ones(num_processes, 1, device=device)

    if co_player is not None:
        co_player.eval()
        co_eval_recurrent_hidden_states = torch.zeros(
            num_processes, co_player.recurrent_hidden_state_size, device=device)
        co_eval_recurrent_cell_states = torch.zeros(
            num_processes, co_player.recurrent_hidden_state_size, device=device)
        co_eval_masks = torch.ones(num_processes, 1, device=device)

    eval_epi_num = 0
    while eval_epi_num < num_episodes:
        with torch.no_grad():
            all_envs_done = False
            flag = 0
            while not all_envs_done:
                both_obs = obs["both_agent_obs"]
                obs0, obs1 = both_obs[:, 0, :, :], both_obs[:, 1, :, :]

                if not flag:
                    action_reward0 = 4 + torch.zeros((num_processes, args.past_length), dtype=torch.long, device=device)
                    action_reward1 = 4 + torch.zeros((num_processes, args.past_length), dtype=torch.long, device=device)
                    pre_obs0 = torch.unsqueeze(obs0, 1).repeat(1, args.past_length, 1, 1, 1)
                    pre_obs1 = torch.unsqueeze(obs1, 1).repeat(1, args.past_length, 1, 1, 1)
                    flag = 1

                eval_v, eval_a, eval_a_log_dist, eval_rnn_hidden, eval_rnn_cells = actor_critic.act(
                    obs0, pre_obs0, action_reward1, eval_recurrent_hidden_states, eval_recurrent_cell_states, eval_masks, deterministic=deterministic)

                if co_player is not None:
                    co_player_v, co_player_a, co_player_a_log_dist, co_player_rnn_hidden, co_player_rnn_cells = co_player.act(
                        obs1, pre_obs1, action_reward0, co_eval_recurrent_hidden_states, co_eval_recurrent_cell_states, co_eval_masks,
                        deterministic=deterministic)
                else:
                    co_player_v, co_player_a, co_player_a_log_dist, co_player_rnn_hidden, co_player_rnn_cells = actor_critic.act(
                        obs1, pre_obs1, action_reward0, eval_recurrent_hidden_states, eval_recurrent_cell_states, eval_masks,
                        deterministic=deterministic)

                joint_action = [(eval_a[i], co_player_a[i]) for i in range(len(eval_a))]

                pre_obs0 = torch.cat((pre_obs0[:, 1:, :, :, :], obs0.unsqueeze(1)), dim=1)
                pre_obs1 = torch.cat((pre_obs1[:, 1:, :, :, :], obs1.unsqueeze(1)), dim=1)

                action_reward0 = torch.cat((action_reward0[:, 1:], eval_a), dim=1)
                action_reward1 = torch.cat((action_reward1[:, 1:], co_player_a), dim=1)

                obs, _, done, infos = eval_envs.step(joint_action)

                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32, device=device)

                if co_player is not None:
                    co_eval_masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done],
                        dtype=torch.float32, device=device)

                all_envs_done = all(done)

                for i, info in enumerate(infos):
                    if 'episode' in info.keys():
                        eval_episode_rewards[eval_epi_num][i] = info['episode']['r']
                        eval_player1_episode_rewards[eval_epi_num][i] = info['episode']['ep_sparse_r_by_agent'][0] + info['episode']['ep_shaped_r_by_agent'][0]
                        eval_player2_episode_rewards[eval_epi_num][i] = info['episode']['ep_sparse_r_by_agent'][1] + info['episode']['ep_shaped_r_by_agent'][1]
                        eval_episode_sparse_reward[eval_epi_num][i] = info['episode']['ep_sparse_r_by_agent'][0] + info['episode']['ep_sparse_r_by_agent'][1]
                        if progressbar:
                            progressbar.update(1)

                if all_envs_done:
                    eval_epi_num += 1

    if progressbar:
        progressbar.close()

    if isFinalEval:
        eval_envs.close()

    # return 전체 결과, player 1 결과, player 2 결과
    return np.mean(eval_episode_rewards, axis=0), np.mean(eval_player1_episode_rewards, axis=0), np.mean(eval_player2_episode_rewards, axis=0), np.mean(eval_episode_sparse_reward, axis=0)


def evaluate_saved_model(
        args,
        result_dir,
        xpid,
        num_episodes=10,
        seeds=None,
        verbose=False,
        progressbar=False,
        num_processes=1):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')

    if verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.xpid is None:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(os.path.join(result_dir, "latest", "model.tar"))
        )
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser(os.path.join(result_dir, xpid, "model.tar"))
        )

    # Set up level sampler
    if seeds is None:
        seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(num_episodes)]

    dummy_env, _ = make_lr_venv(
        num_envs=num_processes,
        seeds=None, device=device,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        human_proxy_num=1)

    level_sampler = LevelSampler(
        seeds, 
        dummy_env.observation_space, dummy_env.action_space,
        strategy='sequential')

    model = OvercookedPolicy(dummy_env.observation_space.shape, dummy_env.action_space.n, args)

    pbar = None
    if progressbar:
        pbar = tqdm(total=num_episodes)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    checkpoint = torch.load(checkpointpath, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_processes = min(num_processes, num_episodes)
    eval_episode_rewards = evaluate(args, model, num_episodes, device=device, num_processes=num_processes,
                                    level_sampler=level_sampler, progressbar=pbar)

    mean_return = np.mean(eval_episode_rewards)
    median_return = np.median(eval_episode_rewards)

    logging.info(
        "Average returns over %i episodes: %.2f", num_episodes, mean_return
    )
    logging.info(
        "Median returns over %i episodes: %.2f", num_episodes, median_return
    )

    return mean_return, median_return
