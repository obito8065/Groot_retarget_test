#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=8
SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

dataset_list=(
    "/vla/users/zhaolin/datasets/libero/lerobot_libero/libero_10_no_noops/"
)

set -x
# pip install av==15.0.0 修复内存爆炸的问题.

##################################################################
########################## Only Ego Image #######################
##################################################################
# 训练LLM+DiT,Ego-Image,随机初始化DiT参数
OUT_NAME=../outputs/ft_flow_libero_long_llm_single_random_DiT
python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size 40 \
--output-dir $OUT_NAME \
--max-steps 40000 \
--data-config libero_single \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path ../GR00T-N1.5-3B \
--report_to tensorboard \
--no-balance-trajectory-weights \
--no-balance-dataset-weights \
--action_horizon 16 \
--save_steps 5000 \
--no-instr-use-episode-index \
--tune-llm \
--reinitialize-action-head

# 训练LLM+DiT, Ego-Image
#OUT_NAME=../outputs/ft_flow_libero_long_llm_single
#python scripts/gr00t_finetune.py \
#--dataset_path ${dataset_list[@]} \
#--num-gpus $NUM_GPUS \
#--batch-size 40 \
#--output-dir $OUT_NAME \
#--max-steps 40000 \
#--data-config libero_single \
#--video-backend torchvision_av \
#--dataloader_num_workers 8 \
#--base_model_path ../GR00T-N1.5-3B \
#--report_to tensorboard \
#--no-balance-trajectory-weights \
#--no-balance-dataset-weights \
#--action_horizon 16 \
#--save_steps 5000 \
#--no-instr-use-episode-index \
#--tune-llm


##################################################################
########################## Ego+Wrist Image #######################
##################################################################
# bash ft_on_libero_long.sh >> ../outputs/ft_flow_libero_long_llm/outs_07_23.log 2>&1 &
# 训练LLM+Action Head
#  OUT_NAME=../outputs/ft_flow_libero_long_llm
#  python scripts/gr00t_finetune.py \
#  --dataset_path ${dataset_list[@]} \
#  --num-gpus $NUM_GPUS \
#  --batch-size 40 \
#  --output-dir $OUT_NAME \
#  --max-steps 40000 \
#  --data-config libero_spatial \
#  --video-backend torchvision_av \
#  --dataloader_num_workers 8 \
#  --base_model_path ../GR00T-N1.5-3B \
#  --report_to tensorboard \
#  --no-balance-trajectory-weights \
#  --no-balance-dataset-weights \
#  --action_horizon 16 \
#  --save_steps 1000 \
#  --no-instr-use-episode-index \
#  --tune-llm

# 只训练action head
# OUT_NAME=../outputs/ft_flow_libero_long
# python scripts/gr00t_finetune.py \
# --dataset_path ${dataset_list[@]} \
# --num-gpus $NUM_GPUS \
# --batch-size 40 \
# --output-dir $OUT_NAME \
# --max-steps 40000 \
# --data-config libero_spatial \
# --video-backend torchvision_av \
# --dataloader_num_workers 8 \
# --base_model_path ../GR00T-N1.5-3B \
# --report_to tensorboard \
# --no-balance-trajectory-weights \
# --no-balance-dataset-weights \
# --action_horizon 16 \
# --save_steps 1000 \
# --no-instr-use-episode-index \
# --resume