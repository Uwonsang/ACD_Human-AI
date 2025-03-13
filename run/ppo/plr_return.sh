#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --level_replay_strategy=return --level_replay_score_transform=rank_low --num_processes=50  \
--verbose --num_env_steps=300000000 --ratio_mini_batch 0.05 --save_model --staleness_coef 0.3 --seed 1