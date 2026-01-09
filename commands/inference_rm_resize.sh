#!/bin/bash
set -x
source /home/jd/miniconda3/bin/activate groot

export PYTHONPATH=$(pwd):$PYTHONPATH
MODEL_PATH=/mnt2/finetune_models/zhaolin/groot/resize_img_chunk50_50000steps # 训练数据增多.
# /mnt2/finetune_models/zhaolin/groot/resize_img_first_finetune_multi # 第一阶段训练的多任务模型 - 50000步
# /mnt2/finetune_models/zhaolin/groot/resize_img_two_finetune_take_cup-5000 # 两阶段训练,第二阶段拿杯子
# /mnt2/finetune_models/zhaolin/groot/multi_take_checkpoint-40000 # /mnt2/finetune_models/zhaolin/checkpoint-43000

python gr00t/inference/inference.py --plot \
--embodiment-tag new_embodiment \
--data-config realman_multiimg \
--model_path $MODEL_PATH \
--prompt "Pick up the cup from the table. Place the cup in the green area." \
--modality_keys arm_right_position_action arm_right_axangle_action hand_right_pose_action \
--steps 10 \
--action_horizon 50