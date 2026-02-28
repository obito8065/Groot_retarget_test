#!/bin/bash

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH
# source /vla/miniconda3/bin/activate groot

set -x
export DEFAULT_EAGLE_PATH=/vla/users/zhaolin/Qwen2.5-VL-3B-Instruct
dataset_list=(
    "/vla/users/zhaolin/datasets/libero/sample_lerobot_libero/libero_spatial_no_noops_sample/"
)
data_config=(
  "libero_image_wrist"
)
NUM_GPUS=8
BATCH_SIZE=8
MAX_STEPS=8000


OUTPUT_DIR=/vla/users/lijiayi/code/GR00T_QwenVLA/outputs_qwen/libero_spatial/n1.5_pretrain_finetune_on_sample_spatial_v2.1

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

# MODELPATH=../GR00T-N1.5-3B
MODELPATH=/vla/users/zhaolin/outputs/pretrain_v1.6_qwenvla_tune_vit_timeactionhead/checkpoint-60000

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir ${OUTPUT_DIR} \
--max-steps $MAX_STEPS \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path $MODELPATH \
--report_to wandb \
--action_horizon 16 \
--save_steps 1000 \
--no-instr-use-episode-index \
--data-config "${data_config[@]}" \
--select_layer 12 \
--backbone_type qwen2_5_vl \
--backbone_model_name_or_path /vla/users/zhaolin/Qwen2.5-VL-3B-Instruct \
--tune_visual \
--use-time-aware-action-head \
--load_pretrained \
--no-update_backbone # 使用预训练模型,不更新backbone和action head.

# --no-load_pretrained \
# --update_backbone --update_action_head
