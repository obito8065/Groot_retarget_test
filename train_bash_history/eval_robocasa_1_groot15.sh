#!/bin/bash

# server of groot 3B model for robocasa

set -x
source /vla/miniconda3/bin/activate gr00t_n15
SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH


cd /vla/users/lijiayi/code/GR00T_QwenVLA 

DEFAULT_PORT=8281
# 检查是否提供了参数
if [ -z "$1" ]; then
  PORT=$DEFAULT_PORT
else
  PORT=$1
fi
echo "Using port: $PORT"

DEFAULT_GPU=1
# 检查是否提供了参数
if [ -z "$2" ]; then
  GPU=$DEFAULT_GPU
else
  GPU=$2
fi
echo "Using GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU


# DEFAULT_MODEL_PATH="/vla/users/lijiayi/code/GR00T_QwenVLA/gr00t/checkpoint/GR00T-N1.5-3B"
DEFAULT_MODEL_PATH="/vla/users/lijiayi/code/GR00T_QwenVLA/outputs_robocasa/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands/n1.5_nopretrain_finetune_on_robocasa_tunevl_v3.1"
# 检查是否提供了参数
if [ -z "$3" ]; then
  MODEL_PATH=$DEFAULT_MODEL_PATH
else
  MODEL_PATH=$3
fi
echo "Using MODEL_PATH: $MODEL_PATH"


python3 scripts/inference_service.py \
    --server     \
    --model_path "$MODEL_PATH"  \
    --embodiment-tag gr1 \
    --data_config fourier_gr1_arms_waist  \
    --denoising-steps 4 \
    --port "$PORT"