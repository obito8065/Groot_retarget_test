#!/bin/bash


python gr00t/inference/inference.py \
--plot --dataset-path /home/robot/pi/datasets/take_cup_lerobot62 \
--embodiment-tag new_embodiment --data-config realman_rightarm \
--model_path ../outputs/checkpoint-1000 --simulate --prompt "Take the cup" --modality_keys arm_right_position_action arm_right_axangle_action hand_right_pose_action