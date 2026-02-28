#!/bin/bash

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

# 指定GPU
export CUDA_VISIBLE_DEVICES=0
set -x
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA
dataset_list=(
      "/mnt/workspace/datasets/robocasa_24tasks_datasets/pick_and_place_lerobot_task24_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000_ee"
)
data_config=(
  "fourier_gr1_arms_waist"
)
embodiment_tag_list=(
   "robocasa"
)
NUM_GPUS=1
BATCH_SIZE=64
max_steps=30000
WORKERS=4
SAVE_STEPS=2000

OUTPUT_DIR=/mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robocasa_ckpt_1tasks_1000ep/n1.5_nopretrain_finetuneVL_on_robocasa_task2_v0.1

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"


MODELPATH=/mnt/workspace/users/lijiayi/checkpoints/GR00T-N1.5-3B

python scripts/gr00t_finetune.py \
--dataset_path "${dataset_list[@]}" \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir $OUTPUT_DIR \
--max-steps $max_steps \
--no-instr-use-episode-index \
--video-backend torchvision_av \
--dataloader_num_workers $WORKERS \
--base_model_path  $MODELPATH \
--action_horizon 16 \
--report_to tensorboard \
--save_steps $SAVE_STEPS \
--data-config "${data_config[@]}" \
--tune_visual \
--embodiment_tag "${embodiment_tag_list[@]}" \
--update_action_head 

