#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py --level_replay_strategy=positive_value_loss --level_replay_score_transform=rank --num_processes=50  \
--verbose --num_env_steps=300000000 --ratio_mini_batch 0.05 --use_wandb --save_model --staleness_coef 0.3 --seed 1