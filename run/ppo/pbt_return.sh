#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_pbt.py --level_replay_strategy=return --level_replay_score_transform=rank_low --num_processes=50  \
--verbose --num_env_steps=300000000 --ratio_mini_batch 0.05 --save_model --agent_sample_method min \
--env_sample_method=low --seed 1 --staleness_coef 0.3