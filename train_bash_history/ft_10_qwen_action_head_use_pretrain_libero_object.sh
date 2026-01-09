#!/bin/bash
# export CUDA_VISIBLE_DEVICES=8
SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH
source /vla/miniconda3/bin/activate groot

set -x
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
dataset_list=(
    # "/vla/users/zhaolin/datasets/libero/lerobot_libero/libero_10_no_noops/"
    # "/vla/users/zhaolin/datasets/libero/lerobot_libero/libero_goal_no_noops"
    "/vla/users/zhaolin/datasets/libero/lerobot_libero/libero_object_no_noops"
)
data_config=(
  "libero_image_wrist"
)
NUM_GPUS=8
BATCH_SIZE=8
MAX_STEPS=50000

# new-exp: 加入新的TimeAwareActionHead
# OUTPUT_DIR=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_5w 与预训练使用相同的embodiment_tag,性能较差
# OUTPUT_DIR=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w # 初始化使用不同的embodiment_tag
# OUTPUT_DIR=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w # 使用oxe预训练过的checkpoint.
# OUTPUT_DIR=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_125000_tag_5w
OUTPUT_DIR=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_object

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

# MODELPATH=../GR00T-N1.5-3B
# MODELPATH=../outputs/pretrain_qwen_bridge_brdbench_v1.3_freeze_vlm_timeactionhead/checkpoint-90000
# MODELPATH=../outputs/pretrain_qwen_oxe_all_v1.4_freeze_vlm_timeactionhead/checkpoint-20000 微调性能 libero-long 0.92
# MODELPATH=../outputs/pretrain_qwen_oxe_all_v1.4_freeze_vlm_timeactionhead/checkpoint-125000 性能下滑
MODELPATH=../outputs/pretrain_qwen_oxe_all_v1.5_tune_vit_timeactionhead/backup/checkpoint-190000

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir ${OUTPUT_DIR} \
--max-steps $MAX_STEPS \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path $MODELPATH \
--report_to tensorboard \
--action_horizon 16 \
--save_steps 5000 \
--no-instr-use-episode-index \
--data-config "${data_config[@]}" \
--select_layer 12 \
--backbone_type qwen2_5_vl \
--backbone_model_name_or_path ../Qwen2.5-VL-3B-Instruct \
--tune_visual \
--use-time-aware-action-head \
--load_pretrained \
--no-update_backbone
