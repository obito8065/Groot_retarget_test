#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=8

dataset_list=(
    "/export/intent/linzhao/code/datasets/rm_lerobot/day1/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day2/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day3/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day4/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day5/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day6/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day7/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day8/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day9/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/day10/"
    "/export/intent/linzhao/code/datasets/rm_lerobot/take_cup_62/"
)

set -x

# 每次实验应该修改这里的output name.
OUT_NAME=../outputs/raw_ft_day10

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size 200 \
--output-dir $OUT_NAME \
--max-steps 50000 \
--data-config realman_rightarm \
--video-backend torchvision_av \
--dataloader_num_workers 16 \
--base_model_path ../GR00T-N1.5-3B \
--report_to tensorboard \
--no-balance-trajectory-weights \
--no-balance-dataset-weights \
--action_horizon 16 \
--save_steps 5000
# realman_rightarm只支持action_horizon=16