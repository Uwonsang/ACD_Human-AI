#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_pbt.py --level_replay_strategy=positive_value_loss --level_replay_score_transform=rank --num_processes=50  \
--verbose --num_env_steps=300000000 --ratio_mini_batch 0.05 --save_model --seed 3 --staleness_coef 0.3