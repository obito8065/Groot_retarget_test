#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH
source /vla/miniconda3/bin/activate groot

set -x
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
dataset_list=(
    "/vla/users/zhaolin/datasets/libero/lerobot_libero/libero_10_no_noops/"
)
data_config=(
  "libero_image_wrist"
)
NUM_GPUS=8
BATCH_SIZE=8
MAX_STEPS=50000

# new-exp: 加入新的TimeAwareActionHead
OUTPUT_DIR=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_5w
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir ${OUTPUT_DIR} \
--max-steps $MAX_STEPS \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path ../GR00T-N1.5-3B \
--report_to tensorboard \
--action_horizon 16 \
--save_steps 5000 \
--no-instr-use-episode-index \
--data-config "${data_config[@]}" \
--select_layer 12 \
--update_backbone \
--backbone_type qwen2_5_vl \
--backbone_model_name_or_path ../Qwen2.5-VL-3B-Instruct \
--tune_visual \
--use-time-aware-action-head
