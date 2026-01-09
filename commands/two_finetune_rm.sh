#!/bin/bash 

# 多阶段训练睿尔曼机器人:
# - 第一阶段: finetune_multi.sh:  ../outputs/resize_img_backup_checkpoint-40000/
# - 第二阶段: two_finetune_rm.sh

NUM_GPUS=8

dataset_list=(
    "/export/intent/linzhao/code/datasets/day_5/take_cup_62/"
)

set -x

# 每次实验应该修改这里的output name.
OUT_NAME=../outputs/resize_img_two_finetune_take_cup
MODEL_PATH=../outputs/resize_img_backup_checkpoint-40000

# realman_multiimg
# 单图进行裁剪
python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS --batch-size 100 \
--output-dir $OUT_NAME \
--max-steps 5000 \
--data-config realman_multiimg \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path $MODEL_PATH \
--report_to tensorboard \
--no-balance-trajectory-weights \
--no-balance-dataset-weights

# 单图不更换尺寸
# 输出结果: outputs/raw_img_instr_checkpoint-40000
#python scripts/gr00t_finetune.py \
#--dataset_path ${dataset_list[@]} \
#--num-gpus $NUM_GPUS --batch-size 100 \
#--output-dir ../outputs/multi_take \
#--max-steps 40000 \
#--data-config realman_rightarm \
#--video-backend torchvision_av \
#--dataloader_num_workers 8 \
#--base_model_path ../GR00T-N1.5-3B \
#--report_to tensorboard \
#--no-balance-trajectory-weights \
#--no-balance-dataset-weights