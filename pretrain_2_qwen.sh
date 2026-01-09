#!/bin/bash

echo -e "\n############## start pretrain ###############\n"
source /vla/miniconda3/bin/activate groot
DEBUG=false
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct

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
    BATCH_SIZE=64 
    WORKERS=4
fi
SAVE_STEPS=5000 # 20000
EPOCHS=10
SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

DATASET_DIR=/export1/vla/datasets/lerobot_oxe

dataset_list=(
    $DATASET_DIR/austin_buds_dataset_converted_externally_to_rlds_lerobot
    $DATASET_DIR/austin_sailor_dataset_converted_externally_to_rlds_lerobot
    $DATASET_DIR/bc_z_lerobot
    $DATASET_DIR/berkeley_autolab_ur5_lerobot
    $DATASET_DIR/berkeley_fanuc_manipulation_lerobot
    $DATASET_DIR/brdbench_20_lerobot
    $DATASET_DIR/bridge_orig_lerobot
    $DATASET_DIR/dlr_edan_shared_control_converted_externally_to_rlds_lerobot
    $DATASET_DIR/droid_lerobot
    $DATASET_DIR/fractal20220817_data_lerobot
    $DATASET_DIR/iamlab_cmu_pickup_insert_converted_externally_to_rlds_lerobot
    $DATASET_DIR/jaco_play_lerobot
    $DATASET_DIR/roboset_lerobot
    $DATASET_DIR/roboturk_lerobot # 单独再次生成
    $DATASET_DIR/stanford_hydra_dataset_converted_externally_to_rlds_lerobot
    $DATASET_DIR/taco_play_lerobot
    $DATASET_DIR/toto_lerobot
    $DATASET_DIR/ucsd_kitchen_dataset_converted_externally_to_rlds_lerobot
    $DATASET_DIR/utaustin_mutex_lerobot
    $DATASET_DIR/viola_lerobot
)

OUTPUT_DIR=../outputs/pretrain_qwen_v1.1
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

#########################################################
# 预训练阶段: freeze VLM backbone, 微调阶段再训练ViT或LLM
#########################################################

python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS \
--batch-size $BATCH_SIZE \
--output-dir $OUTPUT_DIR \
--num-train-epochs $EPOCHS \
--video-backend torchvision_av \
--dataloader_num_workers $WORKERS \
--base_model_path ../GR00T-N1.5-3B \
--action_horizon 16 \
--report_to tensorboard \
--save_steps $SAVE_STEPS \
--no-instr-use-episode-index \
--select_layer 12 \
--update_backbone \
--update_action_head \
--backbone_type qwen2_5_vl \
--backbone_model_name_or_path ../Qwen2.5-VL-3B-Instruct \
--pretrain \
--no-tune-llm \
--no-tune-visual \
2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"