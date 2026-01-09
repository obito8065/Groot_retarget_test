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

# OUT_NAME=../outputs/ft_flow_libero_long_llm_single_with_latent 0729:使用了last VL Embedding Tokens as action tokens.
# INFO:root:Total success rate: 0.822
# INFO:root:Total episodes: 500


# fix error: use action tokens instead of vl embed tokens
OUT_NAME=../outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token

mkdir -p $OUT_NAME
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
--enable_latent_embedding
