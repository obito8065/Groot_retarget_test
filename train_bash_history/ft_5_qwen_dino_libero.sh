#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH
source /vla/miniconda3/bin/activate groot

set -x
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
export DEFAULT_DINO_PATH=/vla/users/zhaolin/pretrained/dinov3-vitl16-pretrain-lvd1689m

dataset_list=(
    "/vla/users/zhaolin/datasets/libero/lerobot_libero/libero_10_no_noops/"
)
data_config=(
  "libero_image_wrist"
)

BATCH_SIZE=8
MAX_STEPS=50000 # num-train-epochs 32, 步长50000

# exp-2 使用dino features,训练ViT ../outputs/Qwen_ft_on_libero_tune_visual_random_dit_dino_5w_25000
# --tune_visual \
#OUTPUT_DIR=../outputs/Qwen_ft_on_libero_tune_visual_random_dit_dino_5w

# exp-4 使用dino features,不训练ViT
OUTPUT_DIR=../outputs/Qwen_ft_on_libero_no_tune_visual_random_dit_dino_5w_bs8

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
--update_action_head \
--backbone_type qwen2_5_vl \
--backbone_model_name_or_path ../Qwen2.5-VL-3B-Instruct \
--max-steps $MAX_STEPS \
--use-dino

# \
# --num-train-epochs 40 \
# --tune_visual \
# 2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

