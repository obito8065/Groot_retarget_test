#!/bin/bash

# 这里修改为训练好的模型路径
MODEL_PATH=../outputs/resize_img_chunk32_70000_day8/checkpoint-70000

# 动作块 action_horizon

python scripts/eval_policy.py --plot --action_horizon 32 \
--dataset-path ../datasets/day_8/day1 \
--embodiment-tag new_embodiment \
--data-config realman_multiimg \
--model_path $MODEL_PATH \
--modality_keys arm_right_position_action arm_right_axangle_action hand_right_pose_action \
--trajs 1         # 测试轨迹数