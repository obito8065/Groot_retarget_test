#!/bin/bash

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

# 指定GPU
set -x
cd /vla/users/lijiayi/code/groot_retarget
dataset_list=(
      "/vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_300_keypoints_v5"
)
data_config=(
  "robocasa_retarget_50_horizon"
)
embodiment_tag_list=(
   "robocasa"
)

# ==== debug or normal mode switch ====
if [ "$DEBUG" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Debug mode"
    NUM_GPUS=1
    BATCH_SIZE=96
    WORKERS=0
else
    echo "Normal mode"
    # 每个节点使用的GPU（根据节点实际GPU数量调整）
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
    NUM_GPUS_PER_NODE=6  # 每个节点的GPU数量
    NUM_GPUS=$NUM_GPUS_PER_NODE  # 补充定义：单节点的GPU数（用于条件判断）
    TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 总GPU数量（多节点）
    # GPU: 87046MiB/97871MiB, 754/16382 [24:10<7:15:08,  1.67s/it]
    BATCH_SIZE=64  # 每个GPU的batch size
    WORKERS=8
fi



num_epochs=200
SAVE_STEPS=3000

OUTPUT_DIR=/vla/users/lijiayi/code/groot_retarget/output_ckpt/n1.5_nopretrain_finetuneALL_on_robocasa_task2_retarget_v5_bs384_horizon50

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"


MODELPATH=/vla/users/lijiayi/code/GR00T_QwenVLA/gr00t/checkpoint/GR00T-N1.5-3B

python scripts/gr00t_finetune.py \
--dataset_path "${dataset_list[@]}" \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir $OUTPUT_DIR \
--num_train_epochs $num_epochs \
--no-instr-use-episode-index \
--video-backend torchvision_av \
--dataloader_num_workers $WORKERS \
--base_model_path  $MODELPATH \
--action_horizon 50 \
--action_dim 64 \
--report_to tensorboard \
--save_steps $SAVE_STEPS \
--data-config "${data_config[@]}" \
--tune_visual \
--embodiment_tag "${embodiment_tag_list[@]}" \
--update_action_head \
--learning_rate 3e-5 \

