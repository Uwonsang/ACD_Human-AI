#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py --level_replay_strategy=random --num_processes=50  \
--verbose --num_env_steps=300000000 --ratio_mini_batch 0.05 --save_model --staleness_coef 0.3  --seed 1