import os
import sys
from collections import deque
import timeit
import logging

import numpy as np
import torch
from level_replay import algo, utils
from level_replay.model import OvercookedPolicy
from level_replay.storage import OvercookedRolloutStorage
from level_replay.envs import make_lr_venv
from level_replay.arguments import parser
from test import evaluate, evaluate_e3t
from tqdm import tqdm
import time
import wandb
import yaml
import warnings
from baselines.logger import HumanOutputFormat
import pickle
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args, seeds):

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    if 'cuda:0' in device.type:
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
    )

    envs, level_sampler = make_lr_venv(
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

    eval_envs, _level_sampler = make_lr_venv(
        num_envs=args.num_processes_test,
        seeds=None, device=device,
        all_args=args,
        no_ret_normalization=args.no_ret_normalization,
        level_sampler=None,
        activate_planner=eval_activate_planner,
        obp_eval_map=True)

    actor_critic = OvercookedPolicy(envs.observation_space.shape, envs.action_space.n, args)

    actor_critic.to(device)

    if args.load_model:
        check_point = torch.load(args.load_model_path, map_location=device)
        actor_critic.load_state_dict(check_point["model_state_dict"])

    if args.eval_sp:
        co_model = actor_critic
    else:
        co_model = None

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.ratio_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.num_repeat,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()

    rollouts = OvercookedRolloutStorage(args.num_steps, args.num_processes, args.num_repeat,
                                        obs["both_agent_obs"].shape,
                                        envs.action_space, actor_critic.recurrent_hidden_state_size)

    level_seeds = level_seeds.unsqueeze(-1)
    rollouts.obs_init(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)
    episode_rewards_p1 = deque(maxlen=100)
    episode_rewards_p2 = deque(maxlen=100)
    total_episode_rewards = []
    total_test_rewards = []
    total_values = []

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes // args.num_repeat

    print("num_env_steps : {}, num_steps : {}, num_processes : {}, num_repeat : {}, num_train_seed: {}".format(int(args.num_env_steps), args.num_steps, args.num_processes, args.num_repeat, args.num_train_seeds))
    print("save_model :{}".format(args.save_model))

    batch_size = args.num_steps * args.num_processes * args.num_repeat
    print("batch: {}, mini_batch: {}".format(batch_size, batch_size//args.num_mini_batch if args.ratio_mini_batch < 0 else int(batch_size*args.ratio_mini_batch)))

    sample_unseen_bool = False
    level_seeds_count = torch.zeros(args.num_processes, len(seeds), dtype=torch.long)

    total_update = 0
    step_level_seeds = {k: [] for k in ["total_update", "level_seed"]}
    timer = timeit.default_timer
    update_start_time = timer()
    for j in tqdm(range(num_updates)):
        total_update += 1

        actor_critic.train()
        for repeat_step in range(args.num_repeat):
            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    overcooked_controller = utils.OvercookedController(envs, actor_critic, args)

                    agent_infos = overcooked_controller.act(rollouts, repeat_step, step)
                    joint_action, ego_infos = agent_infos["joint_action"], agent_infos['ego']

                    obs, reward, done, infos = envs.step(joint_action)
                    both_obs, curr_state, other_agent_idx = obs["both_agent_obs"], obs["overcooked_state"], obs["other_agent_env_idx"]
                    agent_idx = 1 - other_agent_idx
                    agent_idx = agent_idx.tolist()

                    if args.ignore:
                        sums = [info['sparse_r_by_agent'][int(agent_idx[index])] + info['shaped_r_by_agent'][int(agent_idx[index])] for index, info in enumerate(infos)]
                        reward = torch.tensor(sums).view(-1, 1)

                # Reset all done levels by sampling from level sampler
                for i, info in enumerate(infos):
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                        episode_rewards_p1.append(info['episode']['ep_sparse_r_by_agent'][0]+info['episode']['ep_shaped_r_by_agent'][0])
                        episode_rewards_p2.append(info['episode']['ep_sparse_r_by_agent'][1]+info['episode']['ep_shaped_r_by_agent'][1])
                    if level_sampler:
                        level_seeds[i][0] = info['level_seed']

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])

                rollouts.insert(repeat_step, both_obs, ego_infos["rnn_hidden"], ego_infos["rnn_cell"],
                                ego_infos["action"], ego_infos["a_log_prob"], ego_infos["a_log_dist"],
                                ego_infos["value"], reward, masks, bad_masks, curr_state, other_agent_idx,
                                level_seeds)

        all_level_seeds = [rollouts.level_seeds[repeat][0].cpu().numpy() for repeat in range(args.num_repeat)]
        step_level_seeds["total_update"].append(total_update)
        step_level_seeds["level_seed"].append(all_level_seeds)

        for repeat_step in range(args.num_repeat):
            for i, seed in enumerate(all_level_seeds[repeat_step]):
                level_seeds_count[i, seed] += 1
                
        with torch.no_grad():
            full_next_value = rollouts.next_value(actor_critic, agent_idx=0)

        rollouts.compute_returns(full_next_value, args.gamma, args.gae_lambda)

        # Update level sampler
        if level_sampler:
            seed_scores = level_sampler.update_with_rollouts(rollouts, args.num_repeat)

        if args.level_replay_strategy == 'positive_value_loss':
            sample_unseen_bool = level_sampler.sample_unseen_bool

        value_loss, action_loss, dist_entropy, tmp_values, ppo_update_step, ppo_num_updates = agent.update(rollouts, sample_unseen_bool, step=j)
        rollouts.after_update(args.num_repeat)

        if level_sampler:
            level_sampler.after_update()

        total_num_steps = (j + 1) * args.num_repeat * args.num_processes * args.num_steps
        update_end_time = timer()
        elapsed_time = update_end_time - update_start_time
        sps = (args.num_processes * args.num_steps) / elapsed_time
        remain_n_updates = num_updates - j
        remain_time = elapsed_time * remain_n_updates
        update_start_time = update_end_time

        total_values.append(tmp_values.cpu().numpy())
        total_episode_rewards.append(np.mean(episode_rewards))

        stats = {"time/step": total_num_steps, "time/n_update": j, "time/remain_n_updates": remain_n_updates,
                 "time/remain_time": remain_time, "time/sps": sps, "train/pg_loss": action_loss,
                 "train/value_loss": value_loss, "train/dist_entropy": dist_entropy, "rollouts/epi_mean_reward": np.mean(episode_rewards)}

        total_ppo_update_num = num_updates * ppo_num_updates

        # Log stats every log_interval updates or if it is the last update
        if (ppo_update_step % (ppo_num_updates * args.log_interval) == 0 and len(episode_rewards) > 1) or total_ppo_update_num == (ppo_update_step * (num_updates - 1)):
            eval_episode_rewards, eval_episode_rewards_p1, eval_episode_rewards_p2, eval_episode_sparse_reward = evaluate(args, actor_critic, args.num_test_seeds, device, eval_envs, num_processes=args.num_processes_test, use_render=False, store_traj=False, co_player=co_model)
            train_eval_episode_rewards, train_eval_episode_rewards_p1, train_eval_episode_rewards_p2, _ = evaluate(args, actor_critic, args.num_test_seeds, device, train_eval_envs, level_sampler=train_eval_level_sampler, num_processes=args.num_processes_test, use_render=False, store_traj=False, co_player=co_model)

            epi_mean_reward_gap = np.mean(train_eval_episode_rewards) - np.mean(eval_episode_rewards)
            total_test_rewards.append(np.mean(eval_episode_rewards))

            stats.update({"train/epi_mean_reward_gap": epi_mean_reward_gap, "eval/epi_mean_reward": np.mean(eval_episode_rewards)})

            level_seeds_count_np = level_seeds_count.numpy()
            seed_save_dir = os.path.expanduser(f'{args.run_dir}/level_seeds_count')
            utils.make_dir(seed_save_dir)
            np.save(seed_save_dir + f'/{j}th_update_seeds.npy', level_seeds_count_np)

            if args.use_wandb:
                # seed_count_image = utils.save_index_plot(seeds, level_seeds_count)
                # wandb.log({"index_count": [wandb.Image(seed_count_image)]}, step=j)
                wandb.log({"train/pg_loss": action_loss, "train/value_loss": value_loss, "train/dist_entropy": dist_entropy, "train/epi_mean_reward_gap": epi_mean_reward_gap,
                           "rollouts/epi_mean_reward": np.mean(episode_rewards), "rollouts/epi_mean_reward(player_1)": np.mean(episode_rewards_p1), "rollouts/epi_mean_reward(player_2)": np.mean(episode_rewards_p2),
                           "train_eval/epi_mean_reward": np.mean(train_eval_episode_rewards), "train_eval/epi_mean_reward(player_1)": np.mean(train_eval_episode_rewards_p1), "train_eval/epi_mean_reward(player_2)": np.mean(train_eval_episode_rewards_p2),
                           "eval/epi_mean_reward": np.mean(eval_episode_rewards), "eval/epi_mean_reward(player_1)": np.mean(eval_episode_rewards_p1), "eval/epi_mean_reward(player_2)": np.mean(eval_episode_rewards_p2),
                           "eval_env/env1_reward": eval_episode_rewards[0], "eval_env/env2_reward": eval_episode_rewards[1], "eval_env/env3_reward": eval_episode_rewards[2],
                           "eval_env/env4_reward": eval_episode_rewards[3], "eval_env/env5_reward": eval_episode_rewards[4]},  step=ppo_update_step)

            'evaluate last_update'
            if j == num_updates - 1:
                logging.info(f"\nLast update: Evaluating on {args.num_test_seeds} test levels...\n  ")
                final_eval_episode_rewards = evaluate(args, actor_critic, args.final_num_test_seeds, device,
                                                      num_processes=args.num_processes_test,
                                                      use_render=False, store_traj=True, co_player=co_model,
                                                      activate_planner=eval_activate_planner)

                mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)

                if args.use_wandb:
                    wandb.log({'final:mean_episode_return': mean_final_eval_episode_rewards}, step=ppo_update_step)

        if args.verbose:
            stdout_logger.writekvs(stats)

    if args.save_model:
        checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s" % (args.run_dir, "model")))
        utils.checkpoint(args, agent, actor_critic, checkpointpath)

    total_values = np.array(total_values)
    total_episode_rewards = np.array(total_episode_rewards)
    total_test_rewards = np.array(total_test_rewards)
    utils.save_score(args.run_dir, total_episode_rewards, total_test_rewards, level_seeds_count, total_values)

    step_seed_save_dir = os.path.expanduser(f'{args.run_dir}/step_level_seeds')
    utils.make_dir(step_seed_save_dir)
    step_seed_save_path = step_seed_save_dir + '/' + 'step_level_seed.pkl'
    pickle.dump(step_level_seeds, open(step_seed_save_path, 'wb'))

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
        wandb.init(project=project_name, entity='overcooked_ai', group='plr',
                   name='strategy : {{{}}}, score_transform : {{{}}}, run_seed : {{{}}}'
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
