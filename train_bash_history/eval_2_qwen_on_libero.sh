#!/bin/bash

######################################################
# run the simulator
# (libero) robot@robot:~/pi/outputs$ source ~/miniconda3/bin/activate libero
# (libero) robot@robot:~/pi/outputs$ python /home/robot/pi/env/openpi/libero_env.py --args.task-suite-name libero_10
######################################################

set -x
# 环境变量控制载入Qwen2.5VL模型
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
export DEFAULT_DINO_PATH=/vla/users/zhaolin/pretrained/dinov3-vitl16-pretrain-lvd1689m
python scripts/groot_eval_libero.py \
--server \
--model_path ../outputs/Qwen_ft_on_libero_no_tune_visual_random_dit_dino_5w_bs8_step50000 \
--embodiment-tag new_embodiment \
--data-config libero_image_wrist \
--denoising-steps 4 \
--backbone-type qwen2_5_vl \
--use_dino

# Qwen_ft_on_libero_no_tune_visual_random_dit_dino_5w_bs8_step50000 | 添加 --use_dino 参数 | 0.672 | 非常糟糕.
# Qwen_ft_on_libero_no_tune_visual_use_pretrainvla_5w_step50000 | 0.78 | 0.778
# qwen-random_action_head-freeze_llm-tune_visual-libero_10_no_noops-max_steps_50000-bs_16 | 0.9
# Qwen_ft_on_libero_tune_visual_random_dit_5w_step3w | 0.862
# Qwen_ft_on_libero_tune_visual_random_dit_5w_step5w | 0.9
