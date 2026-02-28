#!/bin/bash
# source /vla/miniconda3/bin/activate gr00t_n15

set -x 
echo -e "\n############## start pretrain ###############\n"
echo -e "\n########## New Action Head ##########\n"
DEBUG=false
# export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct

# ==== debug or normal mode switch ====
if [ "$DEBUG" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Debug mode"
    NUM_GPUS=1
    BATCH_SIZE=96
    WORKERS=0
else
    echo "Normal mode"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    NUM_GPUS=8
    # GPU: 87046MiB/97871MiB, 754/16382 [24:10<7:15:08,  1.67s/it]
    BATCH_SIZE=64 # 64
    WORKERS=8
fi
SAVE_STEPS=5000 # 20000
EPOCHS=20
SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

# DATASET_DIR=/export1/vla/datasets/lerobot_oxe

dataset_list=(
    /mnt/workspace/datasets/egodex30w_datasets/egodex_merge30w_AlohaGripper
)
data_config=(
  "ego_dex_aloha_standard"
)



OUTPUT_DIR=/mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/n1.5_pretrain_egodex_aloha_v0.0
# OUTPUT_DIR=../outputs/n1.5_finetune_egohuman_v0

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

#########################################################
# 预训练阶段: freeze VLM backbone, 微调阶段再训练ViT或LLM
#########################################################
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir $OUTPUT_DIR \
--num-train-epochs $EPOCHS \
--no-instr-use-episode-index \
--video-backend torchvision_av \
--dataloader_num_workers $WORKERS \
--base_model_path /mnt/workspace/users/lijiayi/checkpoints/GR00T-N1.5-3B \
--action_horizon 16 \
--save_steps $SAVE_STEPS \
--data-config "${data_config[@]}" \
--no-tune-llm \
--update_action_head \
--tune_visual
# --backbone_type eagle2_5 \

