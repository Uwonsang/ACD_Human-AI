import os
import sys
from collections import deque
import timeit
import logging

import numpy as np
import torch
from level_replay import utils
from level_replay.envs import make_lr_venv
from level_replay.arguments import parser
from test import evaluate, evaluate_e3t
from tqdm import tqdm
import time
import wandb
import yaml
import warnings
from baselines.logger import HumanOutputFormat
from level_replay.pbt_agent import PBTAgent
from level_replay.pbt_agent import prioritized_agent_sampling
import pickle
import copy

warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args, seeds):
    timer = timeit.default_timer
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')

    torch.set_num_threads(1)
    utils.seed(args.seed)

    stdout_logger = HumanOutputFormat(sys.stdout)

    if args.eval_sp:
        eval_activate_planner = False
    else:
        eval_activate_planner = True

    # Configure actor envs
    level_sampler_args = dict(
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
        replay_schedule=args.level_replay_schedule,
        score_transform=args.level_replay_score_transform,
        temperature=args.level_replay_temperature,
        eps=args.level_replay_eps,
        rho=args.level_replay_rho,
        nu=args.level_replay_nu,
        alpha=args.level_replay_alpha,
        staleness_coef=args.staleness_coef,
        staleness_transform=args.staleness_transform,
        staleness_temperature=args.staleness_temperature,
        layouts_type=args.layouts_type,
        num_repeat=args.num_repeat,
        name="ego",
    )

    envs, _ = make_lr_venv(
        num_envs=args.num_processes,
        seeds=seeds, device=device,
        all_args=args,
        no_ret_normalization=args.no_ret_normalization,
        level_sampler_args=level_sampler_args,
        activate_planner=False,
        obp_eval_map=False)

    train_eval_envs, train_eval_level_sampler = make_lr_venv(
        num_envs=args.num_processes_test,
        seeds=seeds, device=device,
        all_args=args,
        no_ret_normalization=args.no_ret_normalization,
        level_sampler=None,
        activate_planner=eval_activate_planner,
        obp_eval_map=False)

    eval_envs, _ = make_lr_venv(
        num_envs=args.num_processes_test,
        seeds=None, device=device,
        all_args=args,
        no_ret_normalization=args.no_ret_normalization,
        level_sampler=None,
        activate_planner=eval_activate_planner,
        obp_eval_map=True)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes // args.num_repeat // args.population_size
    print("num_env_steps : {}, num_processes : {}, num_repeat : {}, num_train_seed: {}".format(int(args.num_env_steps), args.num_processes, args.num_repeat, args.num_train_seeds, args.population_size))
    print("save_model :{}".format(args.save_model))

    batch_size = args.num_steps * args.num_processes * args.num_repeat
    print("batch: {}, mini_batch: {}".format(batch_size, batch_size // args.num_mini_batch if args.ratio_mini_batch < 0 else int(batch_size * args.ratio_mini_batch)))

    '''upload ego_model'''
    ego_agent = PBTAgent('ego', args, device, gym_env=envs)
    ego_agent.model.to(device)

    if args.eval_sp:
        co_model = ego_agent.model
    else:
        co_model = None

    co_player_init = []
    for pbt_iter in range(args.population_size):
        co_player_agent = PBTAgent('co_player_' + str(pbt_iter), args, device, gym_env=envs)
        level_sampler_args["name"] = 'co_player_' + str(pbt_iter)
        co_player_agent.env_buffer_init(seeds, level_sampler_args)
        co_player_init.append(co_player_agent)

    level_seeds = torch.zeros(args.num_processes)
    if co_player_init[0].level_sampler:
        envs.level_sampler = co_player_init[0].level_sampler
        level_sampler = envs.level_sampler
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)
    level_seeds_count = torch.zeros(args.num_processes, len(seeds), dtype=torch.long)

    "init rollouts"
    ego_agent.set_rollouts(obs)
    for co_player_agent in co_player_init:
        co_player_agent.set_co_rollouts(obs)

    total_update = 0
    step_level_seeds = {k: [] for k in ["total_update", "co_player", "level_seed"]}
    co_player_buffer = {k: [] for k in ["total_update", "ppo_update", "co_player", "buffer"]}
    co_player_population = []
    for pbt_iter in tqdm(range(args.population_size), desc="population training"):

        epi_rewards = deque(maxlen=100)
        episode_rewards_p1 = deque(maxlen=100)
        episode_rewards_p2 = deque(maxlen=100)
        update_start_time = timer()
        for j in tqdm(range(num_updates), desc="training ego"):

            if pbt_iter == 0:
                co_player_agent = co_player_init[0]
            else:
                '''prioritized sample with co_player score'''
                prioritized_weight = prioritized_agent_sampling(co_player_population, args.pbt_lambda, sample_method=args.agent_sample_method)
                if args.co_player_random:
                    co_player_agent = np.random.choice(co_player_population)
                else:
                    co_player_agent = np.random.choice(co_player_population, p=prioritized_weight)
                envs.level_sampler = co_player_agent.level_sampler
                level_sampler = envs.level_sampler

            '''upload_co_player_model'''
            co_player_agent.model.to(device)

            for repeat_step in range(args.num_repeat):
                for step in range(args.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        overcooked_controller = utils.OvercookedController_pbt(envs, ego_agent, co_player_agent, args)
                        agent_infos = overcooked_controller.act(repeat_step, step)
                        joint_action, ego_infos, co_player_infos = agent_infos["joint_action"], agent_infos['ego'], agent_infos["co_player"]

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(joint_action)

                    # Reset all done levels by sampling from level sampler
                    for i, info in enumerate(infos):
                        if 'episode' in info.keys():
                            epi_rewards.append(info['episode']['r'])
                            episode_rewards_p1.append(info['episode']['ep_sparse_r_by_agent'][0] + info['episode']['ep_shaped_r_by_agent'][0])
                            episode_rewards_p2.append(info['episode']['ep_sparse_r_by_agent'][1] + info['episode']['ep_shaped_r_by_agent'][1])
                        if level_sampler:
                            level_seeds[i][0] = info['level_seed']

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0]
                         for info in infos])

                    '''insert'''
                    ego_agent.insert_rollouts(repeat_step, ego_infos, obs, reward, masks, bad_masks, level_seeds)

            all_level_seeds = [ego_agent.rollouts.level_seeds[repeat][0].cpu().numpy() for repeat in range(args.num_repeat)]
            step_level_seeds["total_update"].append(total_update)
            step_level_seeds["co_player"].append(int(co_player_agent.name.split("_")[-1]))
            step_level_seeds["level_seed"].append(all_level_seeds)

            for repeat_step in range(args.num_repeat):
                for i, seed in enumerate(all_level_seeds[repeat_step]):
                    level_seeds_count[i, seed] += 1

            '''compute_rollouts'''
            ego_agent.compute_rollouts(agent_idx=0)

            # Update level sampler
            if level_sampler:
                seed_scores = level_sampler.update_with_rollouts(ego_agent.rollouts, args.num_repeat)
                '''robust_plr'''
                sample_unseen_bool = level_sampler.sample_unseen_bool

            '''ego_update'''
            ego_value_loss, ego_action_loss, ego_dist_entropy, ego_values, ppo_update_step, ppo_num_updates = ego_agent.update(ego_agent.rollouts, sample_unseen_bool, step=j)

            '''co_player_envs_buffer'''
            for repeat_step in range(args.num_repeat):
                for seed in all_level_seeds[repeat_step]:
                    level_sampler.top_k_insert(int(seed), seed_scores, max_size=1000, env_sample_method=args.env_sample_method)

            "reset ego_buffer"
            ego_agent.set_rollouts(obs)

            if level_sampler:
                level_sampler.after_update()

            'unload co-player_model'
            co_player_agent.model.to('cpu')

            total_num_steps = total_update * args.num_repeat * args.num_processes * args.num_steps
            update_end_time = timer()
            elapsed_time = update_end_time - update_start_time
            sps = (args.num_processes * args.num_steps) / elapsed_time
            remain_n_updates = (num_updates * args.population_size) - total_update
            remain_time = elapsed_time * remain_n_updates
            update_start_time = update_end_time

            stats = {
                "time/step": total_num_steps, "time/n_update": total_update, "time/remain_n_updates": remain_n_updates,
                "time/remain_time": remain_time, "time/sps": sps,
                "train/pg_loss": ego_action_loss, "train/value_loss": ego_value_loss, "train/dist_entropy": ego_dist_entropy,
                "rollouts/epi_mean_reward": np.mean(epi_rewards),
                "rollouts/co_player": co_player_agent.name, "rollouts/pbt_iter": pbt_iter
            }

            total_ppo_update_num = args.population_size * num_updates * ppo_num_updates

            # Log stats every log_interval updates or if it is the last update
            if (ppo_update_step % (ppo_num_updates * args.log_interval) == 0 and len(epi_rewards) > 1) or total_ppo_update_num == (ppo_update_step * (args.population_size * (num_updates - 1))):

                eval_epi_rewards, eval_epi_rewards_p1, eval_epi_rewards_p2, eval_episode_sparse_reward = evaluate_e3t(args, ego_agent.model, args.num_test_seeds, device, eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False, co_player=co_model)
                train_eval_epi_rewards, train_eval_epi_rewards_p1, train_eval_epi_rewards_p2, train_eval_episode_sparse_reward = evaluate_e3t(
                    args, ego_agent.model, args.num_test_seeds, device, train_eval_envs, level_sampler=train_eval_level_sampler,
                    num_processes=args.num_processes_test, use_render=False, store_traj=False, co_player=co_model)

                epi_mean_reward_gap = np.mean(train_eval_epi_rewards) - np.mean(eval_epi_rewards)
                stats.update({"train/epi_mean_reward_gap": epi_mean_reward_gap, "eval/epi_mean_reward": np.mean(eval_epi_rewards)})

                level_seeds_count_np = level_seeds_count.numpy()
                seed_save_dir = os.path.expanduser(f'{args.run_dir}/level_seeds_count')
                utils.make_dir(seed_save_dir)
                np.save(seed_save_dir + f'/{j}th_update_seeds.npy', level_seeds_count_np)

                if args.use_wandb:
                    # seed_count_image = utils.save_index_plot(seeds, level_seeds_count)
                    # wandb.log({"index_count": [wandb.Image(seed_count_image)]}, step=j)
                    wandb.log(
                        {"train/pg_loss": ego_action_loss, "train/value_loss": ego_value_loss, "train/dist_entropy": ego_dist_entropy,
                         "train/epi_mean_reward_gap": epi_mean_reward_gap,
                         "rollouts/co_player_name": int(co_player_agent.name.split("_")[-1]),
                         "rollouts/epi_mean_reward": np.mean(epi_rewards),
                         "rollouts/epi_mean_reward(player_1)": np.mean(episode_rewards_p1),
                         "rollouts/epi_mean_reward(player_2)": np.mean(episode_rewards_p2),
                         "train_eval/epi_mean_reward": np.mean(train_eval_epi_rewards),
                         "train_eval/epi_mean_reward(player_1)": np.mean(train_eval_epi_rewards_p1),
                         "train_eval/epi_mean_reward(player_2)": np.mean(train_eval_epi_rewards_p2),
                         "eval/epi_mean_reward": np.mean(eval_epi_rewards),
                         "eval/epi_mean_reward(player_1)": np.mean(eval_epi_rewards_p1),
                         "eval/epi_mean_reward(player_2)": np.mean(eval_epi_rewards_p2),
                         "eval_env/env1_reward": eval_epi_rewards[0], "eval_env/env2_reward": eval_epi_rewards[1], "eval_env/env3_reward": eval_epi_rewards[2],
                         "eval_env/env4_reward": eval_epi_rewards[3], "eval_env/env5_reward": eval_epi_rewards[4]}, step=ppo_update_step)

                'evaluate in after last update'
                if j == num_updates - 1 and pbt_iter == args.population_size - 1:
                    logging.info(f"\nLast update: Evaluating on {args.num_test_seeds} test levels...\n  ")
                    final_eval_epi_rewards = evaluate(args, ego_agent.model, args.final_num_test_seeds, device,
                                                      num_processes=args.num_processes_test,
                                                      use_render=False, store_traj=True, co_player=co_model, activate_planner=eval_activate_planner)

                    mean_final_eval_epi_rewards = np.mean(final_eval_epi_rewards)

                    if args.use_wandb:
                        wandb.log({'final:mean_epi_return': mean_final_eval_epi_rewards}, step=ppo_update_step)

            '''save co_player buffer'''
            if ppo_update_step % (ppo_num_updates * args.co_player_interval) == 0:
                for co_player in co_player_population:
                    co_player_buffer["total_update"].append(total_update)
                    co_player_buffer["ppo_update"].append(ppo_update_step)
                    co_player_buffer["co_player"].append(int(co_player.name.split("_")[-1]))
                    co_player_buffer["buffer"].append(copy.deepcopy(co_player.level_sampler.top_k_buffer))

            if args.verbose:
                stdout_logger.writekvs(stats)

        '''make dir_co_player buffer'''
        co_buffer_save_dir = os.path.expanduser(f'{args.run_dir}/co_player_buffer')
        utils.make_dir(co_buffer_save_dir)
        co_buffer_save_path = co_buffer_save_dir + '/' + 'co_player_buffer.pkl'

        '''update num'''
        total_update += 1

        '''Add co_player to population_buffer'''
        co_player_init[pbt_iter].model.load_state_dict(ego_agent.model.state_dict())
        co_player_population.append(co_player_init[pbt_iter])

        if args.save_model:
            co_player_save_dir = os.path.expanduser(f'{args.run_dir}/co_player_model')
            utils.make_dir(co_player_save_dir)
            co_player_save_path = os.path.expandvars(os.path.expanduser("%s/%s" % (co_player_save_dir, pbt_iter)))
            utils.checkpoint(args, co_player_init[pbt_iter].agent, co_player_init[pbt_iter].model, co_player_save_path)


    if args.save_model:
        checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s" % (args.run_dir, "model")))
        utils.checkpoint(args, ego_agent.agent, ego_agent.model, checkpointpath)

    step_seed_save_dir = os.path.expanduser(f'{args.run_dir}/step_level_seeds')
    utils.make_dir(step_seed_save_dir)
    step_seed_save_path = step_seed_save_dir + '/' + 'step_level_seed.pkl'
    pickle.dump(step_level_seeds, open(step_seed_save_path, 'wb'))
    pickle.dump(co_player_buffer, open(co_buffer_save_path, 'wb'))
    wandb.finish()


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds]


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    project_name = "Overcooked" + '_' + args.layouts_type + '_' + "JAAMAS(PPO_fixed)"
    if args.use_wandb:
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        wandb.init(project=project_name, entity='overcooked_ai', group='pbt',
                   name='"strategy : {{{}}}, score_transform : {{{}}}, run_seed : {{{}}}'
                   .format(args.level_replay_strategy, args.level_replay_score_transform, args.seed))
        wandb.config.update(args)

    layout_path = os.path.join(args.layouts_dir, args.layouts_type)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    # Configure logging
    args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.expandvars(os.path.expanduser(
        "%s/%s/%s" % (args.log_dir, args.xpid, args.level_replay_strategy + "_" + args.level_replay_score_transform)))
    utils.make_dir(run_dir)
    args.run_dir = run_dir

    train(args, train_seeds)
