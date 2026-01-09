#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

# 使用 bash pretrain_3_qwen_bridge_brdbench.sh 训练过的ckpt

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

# ../GR00T-N1.5-3B
PRETRAIN_MODEL=../outputs/pretrain_qwen_bridge_brdbench_v1.2_freeze_vlm_random_dit/checkpoint-35000

# 08-27 exp-1: Qwen_ft_on_libero_tune_visual_use_pretrainvla_5w # 使用预训练qwen, 训ViT, --tune_visual

# 08-28 exp-2: Qwen_ft_on_libero_no_tune_visual_use_pretrainvla_5w # 使用预训练qwen, 不训ViT
OUTPUT_DIR=../outputs/Qwen_ft_on_libero_no_tune_visual_use_pretrainvla_5w # do not random dit
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
--base_model_path $PRETRAIN_MODEL \
--report_to tensorboard \
--action_horizon 16 \
--save_steps 5000 \
--no-instr-use-episode-index \
--data-config "${data_config[@]}" \
--select_layer 12 \
--backbone_type qwen2_5_vl
