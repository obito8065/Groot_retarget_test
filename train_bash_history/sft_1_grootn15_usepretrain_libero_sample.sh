#!/bin/bash

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH
# source /vla/miniconda3/bin/activate groot
DEBUG=false
set -x
# export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
dataset_list=(
    "/vla/users/zhaolin/datasets/libero/sample_lerobot_libero/libero_object_no_noops_sample/"
)
data_config=(
  "libero_image_wrist"
)

# ==== debug or normal mode switch ====
if [ "$DEBUG" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Debug mode"
    NUM_GPUS=1
    BATCH_SIZE=8
    WORKERS=0
else
    echo "Normal mode"
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
    NUM_GPUS=6
    BATCH_SIZE=8
    WORKERS=8
fi

MAX_STEPS=8000


OUTPUT_DIR=/vla/users/lijiayi/code/GR00T_QwenVLA/outputs/n1.5_pretrain_finetune_on_sample_object_v2.1

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

# MODELPATH=../GR00T-N1.5-3B
MODELPATH=/vla/users/lijiayi/code/GR00T_QwenVLA/outputs/n1.5_finetune_egohuman_v2.0/checkpoint-10000

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir ${OUTPUT_DIR} \
--max-steps $MAX_STEPS \
--video-backend torchvision_av \
--dataloader_num_workers $WORKERS \
--base_model_path $MODELPATH \
--report_to wandb \
--action_horizon 16 \
--save_steps 500 \
--no-instr-use-episode-index \
--data-config "${data_config[@]}" \
--backbone_type eagle2_5 \
--tune_visual \
--no-tune-llm 




