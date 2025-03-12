# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/arguments.py

import argparse

parser = argparse.ArgumentParser(description='RL')

# PPO Arguments.
parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='learning rate')
parser.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
parser.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')
parser.add_argument(
    '--gamma',
    type=float,
    default=0.99,
    help='discount factor for rewards')
parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.98,
    help='gae lambda parameter')
parser.add_argument(
    '--entropy_coef',
    type=float,
    default=0.1,
    help='entropy term coefficient')
parser.add_argument(
    '--value_loss_coef',
    type=float,
    default=0.1,
    help='value loss coefficient (orignal ppo default: 0.5)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.1,
    help='max norm of gradients)')
parser.add_argument(
    '--no_ret_normalization',
    action='store_true',
    help='Whether to use unnormalized returns')
parser.add_argument(
    '--seed', 
    type=int, 
    default=1, 
    help='random seed')
parser.add_argument(
    '--num_processes',
    type=int,
    default=1,
    help='how many training CPU processes to use')
parser.add_argument(
    '--num_processes_test',
    type=int,
    default=5,
    help='how many training CPU processes to use in test')
parser.add_argument(
    '--num_steps',
    type=int,
    default=400,
    help='number of forward steps in A2C')
parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=8,
    help='number of ppo epochs')
parser.add_argument(
    '--num_mini_batch',
    type=int,
    default=6,
    help='number of batches for ppo')
parser.add_argument(
    '--ratio_mini_batch',
    type=float,
    default=-0.05,
    help='ratio of minibatch for ppo (Enable when entering values between 0.0 and 1.0)'
)
parser.add_argument(
    '--clip_param',
    type=float,
    default=0.05,
    help='ppo clip parameter')
parser.add_argument(
    '--num_env_steps',
    type=int,
    default=3e8,
    help='number of environment steps to train')
parser.add_argument(
    '--xpid',
    default='latest',
    help='name for the run - prefix to log files')
parser.add_argument(
    '--log_dir',
    default='/app/result',
    help='directory to save agent logs')
parser.add_argument(
    '--result_log_dir',
    default='/app/overcooked_result_log',
    help='directory to save result evalute logs')
parser.add_argument(
    '--hidden_size',
    type=int,
    default=256,
    help='state embedding dimension')

# Procgen arguments.
parser.add_argument(
    '--num_train_seeds',
    type=int,
    default=6000,
    help='number of Procgen levels to use for training')
parser.add_argument(
    "--num_test_seeds",
    type=int,
    default=10,
    help="Number of test seeds")
parser.add_argument(
    "--final_num_test_seeds",
    type=int,
    default=100,
    help="Number of test seeds")
parser.add_argument(
    '--seed_path',
    type=str,
    default=None,
    help='Path to file containing specific training seeds')

# Level Replay arguments.
parser.add_argument(
    "--level_replay_score_transform",
    type=str, 
    default='softmax',
    choices=['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax', 'min', 'rank_low'],
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_temperature", 
    type=float,
    default=1.0,
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_strategy", 
    type=str,
    default='random',
    choices=['off', 'random', 'sequential', 'policy_entropy', 'least_confidence', 'min_margin', 'gae', 'value_l1',
             'one_step_td_error', 'return', 'positive_value_loss', 'threshold', 'reward'],
    help="Level replay scoring strategy")
parser.add_argument(
    "--level_replay_eps", 
    type=float,
    default=0.05,
    help="Level replay epsilon for eps-greedy sampling")
parser.add_argument(
    "--level_replay_schedule",
    type=str,
    default='proportionate',
    help="Level replay schedule for sampling seen levels")
parser.add_argument(
    "--level_replay_rho",
    type=float, 
    default=0.5,
    help="Minimum size of replay set relative to total number of levels before sampling replays.")
parser.add_argument(
    "--level_replay_nu", 
    type=float,
    default=0.5,
    help="Probability of sampling a new level instead of a replay level.")
parser.add_argument(
    "--level_replay_alpha",
    type=float, 
    default=1.0,
    help="Level score EWA smoothing factor")
parser.add_argument(
    "--staleness_coef",
    type=float, 
    default=0.5,
    help="Staleness weighing")
parser.add_argument(
    "--staleness_transform",
    type=str, 
    default='power',
    choices=['max', 'eps_greedy', 'rank', 'power', 'softmax'], 
    help="Staleness normalization transform")
parser.add_argument(
    "--staleness_temperature",
    type=float, 
    default=0.1,
    help="Staleness normalization temperature")

# Logging arguments
parser.add_argument(
    "--verbose", 
    action="store_true",
    help="Whether to print logs")
parser.add_argument(
    '--log_interval',
    type=int,
    default=10,
    help='log interval, one log per n updates')
parser.add_argument(
    "--save_interval", 
    type=int, 
    default=60,
    help="Save model every this many minutes.")
parser.add_argument(
    "--weight_log_interval", 
    type=int, 
    default=1,
    help="Save level weights every this many updates")
parser.add_argument(
    "--co_player_interval",
    type=int,
    default=125,
    help="Save co_player_buffer every this many updates")
parser.add_argument(
    "--use_wandb",
    action="store_true",
    default=False,
    help="use_wandb_program")

# Policy Model arguments
parser.add_argument(
    "--use_lstm",
    action="store_true",
    help="use_lstm_in_model")
parser.add_argument(
    "--num_convs_layer",
    type=int,
    default=3,
    help="set the convs_layer number")
parser.add_argument(
    "--num_hidden_layer",
    type=int,
    default=3,
    help="set the hidden layer number")
parser.add_argument(
    "--num_lstm_layer",
    type=int,
    default=2,
    help="set the lstm layer number")
parser.add_argument(
    "--latent_dim",
    type=int,
    default=64,
    help="E3T latent_dim"
)
# Overcooked arguments
parser.add_argument(
    "--episode_length",
    type=int,
    default=400,
    help="Overcooked_env_horizon_num")
parser.add_argument(
    "--num_repeat",
    type=int,
    default=5,
    help="Overcooked_env_horizon_nu")
parser.add_argument(
    "--random_index",
    action="store_true",
    default=False,
    help='use agent random_index')
parser.add_argument(
    "--load_model",
    action="store_true",
    default=False
)
parser.add_argument(
    "--load_model_path",
    type=str,
    default='/app/result/'
)
parser.add_argument(
    "--save_model",
    action="store_true",
    default=True
)
parser.add_argument(
    "--layouts_dir",
    type=str,
    default="/data/overcooked_layout",
    help="overcooked_layout_path")
parser.add_argument(
    "--layouts_type",
    type=str,
    default="big_4_fix",
    help="layouts_type")
parser.add_argument(
    "--use_render",
    type=bool,
    default=False,
    help="map_render"
)
parser.add_argument(
    "--ignore",
    action="store_true",
    default=False,
    help="ignore script agent reward")
parser.add_argument(
    "--population_size",
    type=int,
    default=8,
    help="set the population_size")
parser.add_argument(
    '--pbt_lambda',
    type=float,
    default=0.2,
    help='co-player sampling parameter in pbt')
parser.add_argument(
    '--agent_sample_method',
    type=str,
    default="max",
    choices=["max", "min"],
    help='co-player sampling methods in pbt')
parser.add_argument(
    '--env_sample_method',
    type=str,
    default="top",
    help='co-player_env sampling parameter in pbt')
parser.add_argument(
    "--model_name",
    type=str,
    default='ours',
    help="save check_model name"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="/app/overcooked_result/IJCAI2024_special/3.pbt/return_usesim/seed2/model.tar",
    help="load check_model to eval"
)
# Greedy Human Arguments
parser.add_argument(
    "--human_proxy_num",
    type=int,
    default=0,
    help = "0 : ego vs ego / ego vs co_player , 1 : ego vs human proxy, 2  : human proxy vs human proxy"
)
parser.add_argument(
    "--is_save_model_test",
    type=bool,
    default=False,
)
parser.add_argument(
    '--wait_prob',
    type=float,
    default=0.5,
    help='Probability of a geedy human agent not taking action'
)
parser.add_argument(
    '--eval_sp',
    action="store_true",
    default=False,
    help="evaluation sp model"
)
parser.add_argument(
    '--eval_layout_path',
    type=str,
    default="/app/overcooked_layout/eval",
    help="eval_layout_path"
)
parser.add_argument(
    "--store_traj",
    type=bool,
    default=False,
    help="map_render_in_final_test"
)
parser.add_argument(
    "--co_player_random",
    action="store_true",
    default=False,
    help="Random sample co_player"
)
