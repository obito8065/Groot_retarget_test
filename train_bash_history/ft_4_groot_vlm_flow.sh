#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=8

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

dataset_list=(
    "/vla/users/zhaolin/datasets/rm_lerobot_merge/"
)

set -x

# 训练LLM
OUT_NAME=../outputs/ft_llm_flow_day10
python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size 40 \
--output-dir $OUT_NAME \
--max-steps 70000 \
--data-config realman_rightarm \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path ../GR00T-N1.5-3B \
--report_to tensorboard \
--no-balance-trajectory-weights \
--no-balance-dataset-weights \
--action_horizon 16 \
--save_steps 5000 \
--tune-llm

# 训练ViT
# OUT_NAME=../outputs/ft_vit_flow_day10
# python scripts/gr00t_finetune.py \
# --dataset_path ${dataset_list[@]} \
# --num-gpus $NUM_GPUS \
# --batch-size 40 \
# --output-dir $OUT_NAME \
# --max-steps 70000 \
# --data-config realman_rightarm \
# --video-backend torchvision_av \
# --dataloader_num_workers 8 \
# --base_model_path ../GR00T-N1.5-3B \
# --report_to tensorboard \
# --no-balance-trajectory-weights \
# --no-balance-dataset-weights \
# --action_horizon 16 \
# --save_steps 5000 \
# --tune-visual

# 训练ViT和LLM
# # 每次实验应该修改这里的output name.
# OUT_NAME=../outputs/ft_vlm_flow_day10

# python scripts/gr00t_finetune.py \
# --dataset_path ${dataset_list[@]} \
# --num-gpus $NUM_GPUS \
# --batch-size 40 \
# --output-dir $OUT_NAME \
# --max-steps 70000 \
# --data-config realman_rightarm \
# --video-backend torchvision_av \
# --dataloader_num_workers 8 \
# --base_model_path ../GR00T-N1.5-3B \
# --report_to tensorboard \
# --no-balance-trajectory-weights \
# --no-balance-dataset-weights \
# --action_horizon 16 \
# --save_steps 5000 \
# --tune-visual --tune-llm

# \
# --ddp_find_unused_parameters


# realman_rightarm只支持action_horizon=16