#!/bin/bash


######################## 环境设置 ############################
# /vla/users/zhaolin/env/libero_sim# python libero_env.py --args.task-suite-name libero_10

######################## 评估脚本 ############################
source /vla/miniconda3/bin/activate groot

set -x

export CUDA_VISIBLE_DEVICES=0
SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
python scripts/groot_eval_libero.py \
--server \
--model_path ../outputs/Qwen_ft_on_libero_tune_visual_use_pretrainvla_5w/checkpoint-50000 \
--embodiment-tag new_embodiment \
--data-config libero_image_wrist \
--denoising-steps 4 \
--backbone-type qwen2_5_vl

